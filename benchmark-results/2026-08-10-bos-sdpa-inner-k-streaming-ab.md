# BOS SDPA K256 inner-K128 streaming A/B

## 결론

K256 전체가 도착할 때까지 QK compute가 기다리는 문제를 줄이기 위해 K만 128-token 두 block으로
publish하는 opt-in을 구현했다. 출력은 baseline과 bit-exact였다. 그러나 225-call 가중 SDPA latency는
`2.257493→2.271822 ms`, `0.635%` 느려졌다. stable로 승격하지 않고 default-off 실험 경로로 둔다.

현재 병목은 첫 K256 publish wait 하나만이 아니다. half마다 QK matmul 초기화와 CB publish/pop이 한 번씩
추가된다. 숨긴 K read tail보다 추가 compute/handshake 고정비가 더 컸다.

## 대상과 설정

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- active SDPA compute/readers: 16 (`8 KV heads × 2 cores/head`)
- DRAM: 3 physical banks, 2 worker endpoints/bank, 6 endpoints
- endpoint loads: `3/2/3/3/3/2`; NoC loads: `8/8`
- context/curpos: 65,536 / 65,535
- outer K chunk: 256 tokens (`Sk_chunk_t=8`)
- KV page block: 32 tokens; paged causal decode
- Q/KV dtype: BF16/BFLOAT8_B
- tagged async, six-reader relay, reduce-only helper, TurboQuant: off
- profiler: off

## 구현

새 opt-in:

```text
TT_METAL_SDPA_DECODE_INNER_K_STREAMING=1
```

reader는 K256을 `[DHt=4][inner sequence tiles=4]` layout의 K128 두 block으로 만든다. 각 block은 기존
`row→col` DRAM tile read 순서를 유지하며 CB에 별도로 publish한다. compute는 QK matmul을 K128마다
시작하고 두 결과를 같은 8-tile QK CB에 이어 붙인다. 그 뒤 max, exp, sum, PV, cross-chunk online-softmax
merge는 기존 K256 순서를 그대로 사용한다. 따라서 chunk-level softmax update 수는 늘지 않는다.

지원 범위는 BH runtime, B=1, S=64K, K256, Sq=32, paged causal 8-KV-head, 16 active cores, dual-NoC
6-endpoint, no mask/sink/sliding-window/MLA다. tagged async와 six-reader relay가 함께 요청되면 inner path가
우선하며 두 경로는 비활성화된다.

변경 파일:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
- `models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py`
- `README.md`

## 검증

`ttnncpp`와 `ttnn` 증분 build 성공. runtime이 load하는
`build_home_release/lib/_ttnncpp.so`에 새 build를 배포하고 SHA-256 일치를 확인했다.

첫 실제 opt-in run은 reader preprocessor branch의 closing brace 누락으로 JIT compile exit 1이었다.
device kernel launch 전 compile failure였고 device close는 정상 완료됐다. brace 수정 뒤 JIT compile,
warmup, measured call, close 모두 정상 완료됐다. timeout, exit 124/137, device hang은 없었다.

baseline과 inner output 비교:

- shape: `[1,1,32,3072]`
- exact: true
- max/mean absolute delta: `0 / 0`
- PCC: `1.0`

## 성능 결과

양수 delta는 inner가 느림을 뜻한다.

| 측정 | calls/path | baseline SDPA ms | inner SDPA ms | SDPA delta | baseline layer ms | inner layer ms | layer delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| short matched | 25 | 2.242028 | 2.237332 | -0.209% | 5.398923 | 5.375189 | -0.440% |
| long matched | 200 | 2.259427 | 2.276133 | +0.739% | 5.384160 | 5.431293 | +0.875% |
| call-weighted | 225 | 2.257493 | 2.271822 | +0.635% | - | - | - |

short run의 작은 개선은 long run에서 재현되지 않았다. inner streaming은 현 구현에서 neutral-to-negative다.

## 재현

공통 환경:

```bash
export TT_METAL_LLAMA32_SDPA_DECODE_K_CHUNK_SIZE=256
export TT_METAL_SDPA_DECODE_DUAL_NOC=1
export TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6
export TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1
export TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1
export TT_METAL_SDPA_DECODE_TAGGED_ASYNC=0
export TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0
export TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0
export TT_METAL_TURBOQUANT=0
```

각 process에서 baseline은 `TT_METAL_SDPA_DECODE_INNER_K_STREAMING=0`, inner는 `=1`로 두고 실행한다.

```bash
timeout --signal=INT --kill-after=15s 180s \
  python_env/bin/python models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py \
  --context-len 65536 --sdpa-k-chunk-size 256 --kv-block-size 32 \
  --warmup 3 --iterations 20 --repeats 10 --precision-mode performance
```

## Artifact와 patch

- run root: `/home/iris_hb4/benchmark_runs/sdpa_inner_k_streaming_2026_08_10`
- implementation: `/home/iris_hb4/tmp/codex-patches/20260810-120500-sdpa-inner-k-streaming.patch`
- reader locality: `/home/iris_hb4/tmp/codex-patches/20260810-122000-sdpa-inner-k-reader-locality.patch`
- brace fix: `/home/iris_hb4/tmp/codex-patches/20260810-123500-sdpa-inner-k-brace-fix.patch`
- metadata: `/home/iris_hb4/tmp/codex-patches/20260810-125000-sdpa-inner-k-metadata.patch`

## 다음 수정 후보

K matmul을 두 번 초기화하지 않고 한 matmul invocation 내부에서 N-subblock readiness를 소비해야 한다.
즉 CB 전체 `K*N` wait를 subblock wait로 내리는 방향이다. 현재 helper를 두 번 호출하는 방식으로는
고정비가 이득을 상쇄한다. 이 변경은 matmul LLK input indexing과 CB pop 계약을 다시 설계해야 한다.

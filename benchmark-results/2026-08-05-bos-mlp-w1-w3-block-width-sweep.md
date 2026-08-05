# BOS MLP W1/W3 block-width sweep

날짜: 2026-08-05

## 결론

`TT_METAL_MLP_W1_W3_IN0_BLOCK_W=2`는 isolated MLP 평균 latency를 기본값 6보다
약 1.44% 줄였다. 64K 전체 decode 개선은 두 번 평균 0.1812%뿐이다. fixed-input full-model
logits top-1은 같지만 PCC는 0.9992749이며 bit-exact하지 않다. 따라서 기본값으로 승격하지 않고
opt-in 후보로 보존한다.

## 장치와 topology

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 cores
- 이 MLP data path의 active readers/compute workers: 12/12
- physical DRAM: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime-selected DRAM-interface workers: 6

로그의 P100 heuristic warning은 이 custom BOS 보드의 authoritative SKU 판정이 아니다.

## 공통 설정

- model: Llama 3.2 3B Instruct, 28 layers, batch 1
- MLP: DRAM-sharded weights, W2 `in0_block_w=16`, 16 KiB read-page cap
- reader: fanout-2, tagged depth 2, balanced endpoints, 12 readers
- helper, fanout-3, endpoint-local, fused gate/up, fused epilogue: off
- SDPA: 64K paged KV, chunk 256, dual-NoC, 6 endpoints, pair/bank balanced
- TurboQuant: off
- profiler/Watcher: off
- timeout: `timeout --signal=INT --kill-after=15s 300s`

변경점은 W1/W3 program config의 `in0_block_w`뿐이다. 새 환경변수는 양의 정수만 허용한다.

## Isolated MLP sweep

각 구성은 correctness 검사 뒤 20회 측정했다. 기본값 6과 후보 2는 각각 두 번 반복했다.

| W1/W3 block | PCC | mean ms | median ms | 판정 |
|---:|---:|---:|---:|---|
| 6, repeat 1 | 0.9996410623 | 1.438703 | 1.434310 | baseline |
| 6, repeat 2 | 0.9996410623 | 1.443063 | 1.435894 | baseline |
| 3 | 0.9996423032 | 1.461394 | 1.460377 | 느림 |
| 2, repeat 1 | 0.9996240145 | 1.421633 | 1.418915 | 후보 |
| 2, repeat 2 | 0.9996240145 | 1.418548 | 1.415148 | 후보 |
| 1 | 0.9995606285 | 1.431164 | 1.428630 | block 2보다 느리고 PCC 낮음 |

두 반복 평균:

| 구성 | mean ms | median ms | baseline 대비 |
|---|---:|---:|---:|
| block 6 | 1.440883 | 1.435102 | - |
| block 2 | 1.420091 | 1.417032 | mean -1.443%, median -1.259% |

block 2의 read page 로그는 W1/W3 12672 bytes, W2 8704 bytes였다. 기본값 로그도 같은
full-model run에서 W1/W3 12672 bytes, W2 8704 bytes였다. 이득은 단순 DRAM page 확대가 아니라
K-block publication 및 reader/compute cadence 변화로 해석한다.

## 64K full-model decode A/B

warmup 3 tokens 뒤 50 tokens를 측정했다. synthetic zero-initialized paged KV를 사용했다.

| 구성 | run 1 tok/s | run 2 tok/s | 평균 tok/s | 평균 ms/token |
|---|---:|---:|---:|---:|
| block 6 | 7.638155 | 7.638942 | 7.638548 | 130.914926 |
| block 2 | 7.652201 | 7.652576 | 7.652389 | 130.678153 |

block 2 개선은 평균 `+0.1812%`, latency `-0.1809%`다. 반복 간 편차보다 방향은 안정적이지만
절대 이득이 작다. isolated 개선 대부분이 전체 layer의 SDPA, norm, LM head 및 기타 비용에 희석됐다.

## Fixed-input logits

입력은 token id 1, position 65535, seed 0, zero-initialized 64K paged KV다.

| 항목 | 결과 |
|---|---:|
| baseline top-1 | 320 |
| block 2 top-1 | 320 |
| PCC | 0.9992749095 |
| max absolute difference | 0.3125 |
| mean absolute difference | 0.05626357 |
| RMSE | 0.07094941 |
| bit-exact | false |

block width가 accumulation grouping을 바꾸므로 bit-exact 결과를 기대할 수 없다. top-1 보존만으로
정확성 승격 조건이 되지 않는다. 더 넓은 prompt/token regression 없이 기본값으로 바꾸지 않는다.

## 재현과 artifact

- throughput runner: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/benchmark_llama32_3b_64k_hang_wa.py`
- logits runner: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_llama32_3b_64k_full_logits.py`
- logits artifact: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_mlp_w1w3_block_ab_2026_08_05/block6.pt`
- logits artifact: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_mlp_w1w3_block_ab_2026_08_05/block2.pt`
- source patch: `/home/iris_hb4/tmp/codex-patches/20260805-094000-mlp-w1-w3-block-width-override.patch`
- source patch SHA-256: `494f4ef50c1dbc2e360430367e9337a142557d8bf9a6c8779e363618b5e29e3a`

모든 device run은 exit 0과 `DEVICE_CLOSED`를 확인했다. throughput raw console log는 별도 파일로
보존하지 않았다. 위 tensor artifact와 본 문서의 수치가 남은 재현 자료다.

## 다음 단계

1. block 2는 opt-in으로 유지한다.
2. 다음 실험은 block 크기 추가 sweep가 아니라 W1/W3 reader의 read issue, CB publication,
   compute wait 구간을 host-safe counter로 분리한다.
3. NoC profiler 전 isolated correctness와 짧은 latency를 다시 통과시킨다.
4. 새로운 dataflow 변경은 fixed-input full logits와 여러 token/prompt regression을 먼저 통과시킨다.

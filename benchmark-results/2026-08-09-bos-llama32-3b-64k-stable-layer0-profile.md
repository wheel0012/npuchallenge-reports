# BOS Llama 3.2 3B 64K stable layer-0 decode profile

## 결론

Stable SDPA와 MLP opt-in을 함께 적용한 TransformerBlock layer 0의 measured region은 TTNN device
operation 23개다. device FW duration 합은 4,661.931 us이며 layer-equivalent 역수는 214.50/s다.
이 값은 28-layer full-model tokens/s가 아니다.

SDPA decode 단일 op는 2,046.243 us로 합계의 43.89%다. attention sublayer 전체는
2,883.419 us다. FFN norm부터 residual add까지 MLP sublayer는 1,778.512 us다.

## Artifact

Run root:

`/home/iris_hb4/profiler_runs/llama32_3b_64k_stable_layer0_visualizer_2026_08_09_14_27_41`

TTNN Visualizer 전달 구조:

```text
llama32_3b_64k_stable_layer0_visualizer_2026_08_09_14_27_41/
├── memory_report_visualizer/
│   ├── config.json
│   └── db.sqlite
└── perf_report_visualize/
    ├── ops_perf_result.csv
    ├── profile_log_device.csv
    └── tracy_profile_log_host.tracy
```

`ops_perf_result.csv`는 시작 signpost 뒤와 종료 signpost 앞의 full-column 행만 담는다.
warmup, model load, JIT compile은 포함하지 않는다. `profile_log_device.csv`와 Tracy trace는 원본
capture artifact다.

Memory DB integrity는 `ok`다. DB에는 operations 121개, tensors 111개, errors 0개가 기록됐다.
상세 buffer logging run의 시간은 성능값으로 사용하지 않는다.

## 측정 조건

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 workers
- model: `meta-llama/Llama-3.2-3B-Instruct`
- scope: TransformerBlock layer 0, batch 1, decode 1회
- context: 65,536 tokens, `current_pos=65535`
- paged KV: 2,048 blocks x 32 tokens
- KV 내용: synthetic zero K/V, 실제 prefill 없음
- precision preset: `DecodersPrecision.performance`
- source branch/HEAD: `iris-new`, `8c3fc9e953271b673035799d9523542d2ac43eff`
- runtime extension SHA-256: `f0d1868f260876e9c663df0a525a2ca3395e258179ba7538f601fda0973d86cf`
- `attention.py` SHA-256: `b5e10a96f3ad84161f3840c6793a3e217e73684f6893a97df73fb6163e5d2fa1`
- `mlp.py` SHA-256: `a84b50225ded61d1e58d7826d36457169936374c5105e3a45b4f13991aaeed64`

두 source 파일은 HEAD 대비 각각 9줄 추가된 working-tree stable 구성이다. 따라서 HEAD hash만으로
동일 source를 재현할 수 없고 위 checksum도 함께 필요하다.

## Stable opt-in

SDPA는 K chunk 256, dual-NoC, 6 endpoints, pair/bank-balanced mapping, grouped concat 및 exact
Wo-input 경로다. log에서 endpoint load `3/2/3/3/3/2`, NoC0/NoC1 load `8/8`을 확인했다.
reduce-only helper, six-reader sharded, L1 Wo program, gather Wo는 껐다.

MLP는 DRAM-sharded weights, W2 input block width 16, 16 KiB read cap, fanout-2 tagged reader,
balanced endpoints를 사용한다. log에서 DRAM interface workers 6, readers 12, compute workers 12,
NOC1 endpoint groups `4:4:4`를 확인했다. helpers, fanout-3, endpoint-local, fused gate/up은 껐다.
TurboQuant도 껐다.

## 절차와 안전 결과

1. 동일 runner로 profiler-free operational gate를 먼저 실행했다.
2. gate는 warmup, measured, device close marker를 모두 남기고 exit 0이었다.
3. Tracy는 `-p -r -v --check-exit-code`로 실행했다.
4. runner가 warmup 뒤 synchronize와 `ReadDeviceProfiler`를 수행했다.
5. `llama32_3b_64k_single_layer_decode` signpost 뒤 TransformerBlock을 정확히 한 번 실행했다.
6. 종료 signpost, measured profiler drain, 정상 device close 뒤 Tracy가 exit 0으로 report를 생성했다.
7. memory DB는 Tracy와 분리한 detailed-buffer run으로 생성했고 exit 0이었다.

timeout, signal 종료, exit 124/137은 없었다. active Tracy/Python child도 남지 않았다. 기존 PID 1
소유 zombie는 새 run의 child가 아니며 signal하지 않았다.

## Device performance

| 구간 | Device FW duration 합 | 전체 합 비중 |
|---|---:|---:|
| TransformerBlock layer 0 | 4,661.931 us | 100.00% |
| attention norm부터 attention residual add | 2,883.419 us | 61.85% |
| SDPA decode op | 2,046.243 us | 43.89% |
| FFN norm부터 MLP residual add | 1,778.512 us | 38.15% |
| MLP W1/W3/SwiGLU/W2 중심 5 ops | 1,400.620 us | 30.04% |

측정 device-cycle 범위는 2,538,071 cycles다. FW duration 합은 op별 duration을 더한 값이라 op
overlap이 있으면 wall latency와 다르다. host signpost 간격도 asynchronous enqueue 시간이므로 device
wall latency로 쓰지 않는다.

주요 measured op:

| Op | FW duration | CSV CORE COUNT |
|---|---:|---:|
| QKV Matmul | 434.414 us | 16 |
| SDPA decode | 2,046.243 us | 20 |
| Wo Matmul | 240.626 us | 20 |
| MLP Matmul call 1 | 247.408 us | 20 |
| MLP Matmul call 2 | 434.018 us | 20 |
| SwiGLU Binary | 236.522 us | 20 |
| MLP Matmul call 3 | 431.615 us | 20 |

CSV의 op-level `CORE COUNT=20`과 MLP program log의 `compute workers=12`는 다른 계층의 수치다.
전자는 operation program이 보고한 참여 core 범위이고, 후자는 DRAM-sharded matmul data path의 실제
reader/compute worker 역할 수다. 20-core MLP compute로 해석하지 않는다.

## 한계

- 실제 65,535-token prefill 결과가 아니라 synthetic zero paged KV다.
- 입력도 random residual이다. output reference/PCC를 이 run에서 계산하지 않았다.
- performance preset과 stable 경로의 기능 검증은 기존 correctness run에 의존한다.
- NoC trace/NPE를 수집하지 않았다. CSV의 NPE congestion 및 DRAM BW util 열은 비어 있다.
- layer-equivalent 214.50/s를 full-model decode throughput으로 외삽하지 않는다.
- memory capture에는 setup, warmup, measured call이 모두 기록된다. signpost 성능 CSV만 warmup 제외다.

## 재현 핵심

Runner:

`/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32_3b_64k_single_layer_profile.py`

Performance capture는 timeout의 direct child를 Tracy Python process로 두고 실행했다. raw ops CSV를
`llama32_3b_64k_single_layer_decode`와 다음 signpost 사이로 필터했다. Memory capture는
`enable_fast_runtime_mode=false`, `enable_logging=true`, `enable_detailed_buffer_report=true`로 별도
실행했다.

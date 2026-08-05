# BOS Llama 3.2 3B 64K SDPA grouped concat A/B

- Date: 2026-08-05
- Source revision: `142354481e02999186df74ef691bfb12f8d6d17e`
- Build: `build_home_release`, `ttnn-runtime`, `tt_pybinds`
- Device: custom 20-core BOS NPU
- Runtime/code architecture: Blackhole
- Status: verified opt-in experiment

## Question

SDPA decode 출력 뒤의 head concat가 불필요한 DRAM 및 layout 왕복을 만든다. Llama 3.2 3B의
24 query heads를 12 worker cores에 2 heads/core로 배치하고 L1에서 직접 concat하면 64K single-layer
decode latency가 줄어드는지 확인했다.

이 실험은 reducer와 Wo matmul을 하나의 device program으로 fuse한 결과가 아니다. SDPA output은 여전히
DRAM에 기록되고, 이후 height-sharded L1으로 이동한다. 제거 대상은 그 뒤 generic concat가 수행하던
`L1 shard → DRAM interleaved → row-major → reshape → tile` 경로다.

## Hardware와 active cores

- Board identity: custom 20-core BOS NPU
- Runtime/code architecture: Blackhole
- Available worker grid: `5×4 = 20 cores`
- SDPA active compute cores: 8 KV heads × 2 cores/head = 16 cores
- Grouped concat output cores: 24 query heads ÷ 2 heads/core = 12 cores
- Physical DRAM: 3 banks
- Worker NoC endpoints: bank당 2개, 총 6개
- 이 A/B에서는 `TT_METAL_SDPA_DECODE_ENDPOINT_COUNT`가 설정되지 않았다.
- 로그의 P100/P150 이름은 runtime heuristic이다. board identity로 사용하지 않는다.

## 기존 dataflow

Baseline의 post-SDPA 경로는 다음과 같다.

```text
SDPA decode
  → DRAM TILE [1, B, 32 padded heads, 128]
  → to_memory_config: height-sharded L1
  → generic nlp_concat_heads_decode helper
      → sharded_to_interleaved: DRAM
      → TILE → ROW_MAJOR
      → batch를 32로 pad
      → reshape [1, 1, 32, 24×128]
      → ROW_MAJOR → TILE
  → Wo matmul
```

`to_memory_config`로 만든 L1 shard를 helper가 즉시 DRAM interleaved로 되돌린다. 따라서 앞의 reshard가
concat 준비에 직접 활용되지 않는다.

## 변경 dataflow

`ttnn.experimental.nlp_concat_heads_decode`에 `heads_per_core`를 추가했다. 기본값은 1이므로 기존 호출은
변하지 않는다. BOS 실험은 `heads_per_core=2`를 사용한다.

```text
SDPA decode
  → DRAM TILE [1, B, 32 padded heads, 128]
  → to_memory_config: height-sharded L1
  → grouped nlp_concat_heads_decode
      → 12 output cores
      → core i가 heads [2i, 2i+1]을 NoC read
      → core-local 32×256 TILE shard에 직접 기록
      → 전체 logical output [1, 1, 32, 3072]
  → Wo matmul
```

Reader와 writer data-movement RISCs는 기존 two-phase subtile read 구조를 유지한다. 각 output core는
`head_start = core_index × heads_per_core`를 runtime argument로 받는다. Kernel은 각 local head의 padded-head
tile offset을 계산하고, 두 head를 같은 width shard의 연속 영역에 기록한다.

Grouped mode는 subcore-grid factory와 함께 사용하지 못하도록 validation을 추가했다. `num_heads`는
`heads_per_core`로 나누어떨어져야 한다.

## Opt-in과 변경 위치

Opt-in 환경변수:

```bash
TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1
```

주요 위치:

- `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tt/attention.py`
- `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/`
- dataflow kernel:
  `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp`

TurboQuant 경로는 사용하지 않았다. 아래 실험 flag도 비활성화했다.

```text
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER
TT_METAL_SDPA_DECODE_SIX_READER_SHARDED
TT_METAL_SDPA_DECODE_BYPASS_OUTPUT_RESHARD
```

## Configuration

```text
model: meta-llama/Llama-3.2-3B-Instruct
layer: 0
mode: cur_pos-only single-layer decode
context_len: 65536
decode_cur_pos: 65535
chunk_size: 2048
sdpa_k_chunk_size: 256
cores_per_kv_head: 2
kv_layout: paged
precision_mode: performance
warmup: 3
iterations_per_repeat: 5
repeats: 7
profiler: disabled
```

Runner의 model-config log는 `SDPA decode K chunk size: 128`을 출력하지만, A/B result JSON과 runner argument는
`sdpa_k_chunk_size=256`이다. 이 문서는 측정 명령과 result JSON의 값을 실험값으로 기록한다.

## Build와 사전 검증

```bash
cmake --build build_home_release --target ttnn -j 8
cmake --install build_home_release --component ttnn-runtime
cmake --install build_home_release --component tt_pybinds
python3 -m py_compile models/bos_model/llama32/tt/attention.py
git diff --check
```

`ttnn` C++ 및 pybind build는 성공했다. 시스템에 `clang-format` executable이 없어 별도 formatter dry-run은
수행하지 못했다.

## Correctness

먼저 warmup 1회, measured call 1회로 grouped output tensor를 저장했다. 실제 batch row 0을 기존 baseline과
비교한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| shape | `[1, 1, 32, 3072]` |
| actual row exact equality | `true` |
| actual row nonzero differences | 0 |
| actual row max absolute error | 0.0 |
| actual row mean absolute error | 0.0 |
| actual row PCC | 1.0 |

전체 32 rows 비교는 exact가 아니다. Batch는 실제로 1이고 rows 1–31은 padding 영역이다. 이 영역은 두
경로에서 정의되지 않은 값이 남을 수 있으므로 correctness 판정에서 제외했다. 실제 consumer가 사용하는
row 0은 bit-exact다.

## Performance results

동일 설치 binary, 동일 runner, 동일 측정 횟수에서 opt-in만 변경했다.

| 경로 | mean layer latency | median | min | mean SDPA wall |
|---|---:|---:|---:|---:|
| Baseline generic concat | 6.656809 ms | 6.664521 ms | 6.614429 ms | 3.479778 ms |
| 12-core grouped concat | 6.477166 ms | 6.492559 ms | 6.397765 ms | 3.465896 ms |
| 변화 | **-2.70%** | **-2.58%** | -3.28% | -0.40% |

Grouped samples:

```text
6.3977648, 6.4636814, 6.4925588, 6.5000508,
6.4990022, 6.4946768, 6.4924288 ms
```

Baseline samples:

```text
6.6645214, 6.6785184, 6.6554088, 6.6503340,
6.6666890, 6.6677626, 6.6144286 ms
```

두 run 모두 exit code 0과 device close를 확인했다. Timeout, signal 종료, Watcher abort는 없었다.

## Interpretation

### 관측 사실

- 실제 output row는 bit-exact다.
- 전체 layer mean latency는 2.70% 감소했다.
- 측정 범위가 SDPA 호출만 감싼 `sdpa_wall_mean_ms`는 0.40% 차이다.
- grouped path는 24 one-head output cores 대신 12 two-head output cores를 사용한다.
- generic helper의 intermediate DRAM/layout 변환은 grouped path에서 실행되지 않는다.

### 추론

Layer 개선 대부분은 flash-decode compute나 KV DRAM reader 자체가 아니라 SDPA 이후 head concat 및 layout
변환 비용 감소에서 나온다. SDPA wall이 거의 그대로이고 layer wall만 2.70% 줄어든 것이 이 해석을
지지한다.

이 결과는 “12 cores가 24 cores보다 compute가 빠르다”는 뜻이 아니다. BOS에는 24 worker cores가 없어서
기존 specialized one-head/core op를 직접 쓸 수 없었다. 2 heads/core grouping이 20-core 한도 안에서
specialized L1 concat를 가능하게 만든 것이 핵심이다.

### 미검증 가설

- SDPA writer가 grouped concat의 height-sharded L1 input을 직접 생산하면 현재 남은
  `SDPA DRAM → L1` 왕복도 제거할 수 있다.
- Grouped concat output shard를 Wo reader의 기대 배치와 더 가깝게 배치하면 추가 개선 가능성이 있다.
- Padded rows를 명시적으로 초기화하면 전체 tensor exact도 얻을 수 있으나 실제 batch=1 성능에는 불필요한
  write가 추가될 수 있다.

## 이전 출력 경로 실험과 관계

같은 artifact 디렉터리에서 다음 선행 경로를 확인했다.

- interleaved direct-L1 output: exact였지만 baseline보다 약 1.17% 느렸다.
- sharded direct-L1 output: actual row PCC 약 0.999968, max abs 약 0.00293으로 exact가 아니어서 rollback했다.
- output-reshard bypass: exact, same-time mean 약 1.91% 개선.
- 이번 grouped concat: exact, same-time mean 2.70% 개선.

Direct-L1 변경은 현행 source에서 제거했다. `TT_METAL_SDPA_DECODE_BYPASS_OUTPUT_RESHARD`는 별도 opt-in으로
남아 있지만 grouped mode가 켜지면 사용하지 않는다.

## Reproduction commands

Grouped:

```bash
TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1 \
TT_METAL_SDPA_DECODE_PROFILE_WALL=1 \
timeout --signal=INT --kill-after=15s 300s \
/home/iris_hb4/tt-metal-hb4/python_env/bin/python \
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py \
  --context-len 65536 --chunk-size 2048 --sdpa-k-chunk-size 256 \
  --cores-per-kv-head 2 --kv-layout paged --precision-mode performance \
  --warmup 3 --iterations 5 --repeats 7 \
  --result-json /home/iris_hb4/profiler_runs/sdpa_l1_output_ab_2026_08_05/grouped_concat_2heads_repeated.json
```

Baseline은 위 명령에서 `TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1`만 제거하고 result path를
`grouped_concat_baseline_repeated.json`으로 바꾼다.

재현 전 shell에 위험한 과거 experimental flag가 남아 있지 않은지 확인한다. 특히
`TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1`은 사용하지 않는다.

## Artifact paths

Run directory:

```text
/home/iris_hb4/profiler_runs/sdpa_l1_output_ab_2026_08_05/
```

주요 파일:

- `baseline.pt`
- `grouped_concat_2heads.pt`
- `grouped_concat_2heads.json`
- `grouped_concat_2heads_repeated.json`
- `grouped_concat_baseline_repeated.json`
- `bypass_output_reshard_repeated.json`
- `baseline_repeated_after_bypass.json`

Audit patches:

- `/home/iris_hb4/tmp/codex-patches/20260805-112900-sdpa-grouped-concat-2heads.patch`
  - SHA-256: `4446ead0993a233c99be1ff219b3da31df88471baff27bb0e9c7da1c9cbf67e8`
- `/home/iris_hb4/tmp/codex-patches/20260805-113400-probe-grouped-concat-program.patch`
  - SHA-256: `84e99ab06d0422ce63e31d9b49e8d3816eef7e9e8352d3e2d865f711b7a1150d`

Correctness tensors:

- `baseline.pt`: `473342d690d545359bd18e403590c8b84bcc40335df6806e093969d1c839563c`
- `grouped_concat_2heads.pt`: `bb023bd1b935719a94341c8e961c9a59fd49e520325afb5f05ce35845f257b08`

## Limitations and next steps

- Cur-pos-only single-layer decode다. 실제 65,535-token prefill은 실행하지 않았다.
- Full 28-layer model tokens/s 개선을 직접 측정하지 않았다.
- NoC profiler와 DRAM bandwidth capture를 사용하지 않았다.
- 2.70%는 single-layer wall 개선이다. Full-model 개선폭으로 그대로 곱하지 않는다.
- Grouped opt-in 여부는 현재 result JSON 내부 field로 저장되지 않는다. Exact command와 artifact filename으로
  구분했다.
- 다음 안전한 단계는 grouped path의 isolated NoC capture가 아니라, 먼저 full-model correctness와 짧은
  decode latency를 profiler 없이 검증하는 것이다.
- 그 다음 후보는 SDPA writer → concat input의 direct L1 producer-consumer 계약이다. 이는 새 handshake를
  포함하므로 기존 grouped path와 별도 opt-in으로 구현해야 한다.

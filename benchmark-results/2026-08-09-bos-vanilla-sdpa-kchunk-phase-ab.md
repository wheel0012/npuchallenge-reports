# BOS vanilla SDPA K-chunk 128/256/512 phase A/B

날짜: 2026-08-09

## 결론

64K isolated vanilla paged SDPA에서 K chunk를 128 token에서 256 token으로 키우자 계측 kernel이
3.482728 ms에서 3.330791 ms로 4.362% 감소했다. 그러나 compute의 K/V CB wait와 reader의 read
barrier time은 감소하지 않았다. 따라서 이번 4.36%는 DRAM 요청량 감소나 input-ready 개선보다
chunk-level QK/PV 호출, CB handoff, online-softmax merge 수 감소에서 나온 것으로 해석한다.
K512는 3.324249 ms로 K256보다 0.196%만 빨랐다. CB wait는 더 증가했다. 계측 범위에서는 K256부터
성능이 plateau에 들어갔다.

## 하드웨어와 실행 경로

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 cores
- SDPA active cores: 16 readers/compute cores. 5x4 설정값과 구분한다.
- DRAM: 3 physical banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- vanilla KV layout: interleaved paged KV cache
- vanilla reader route: NOC1, 실제 endpoint trace의 raw destination은 `(2,1)`, `(3,1)`, `(4,1)` 세 곳
- 이 실험은 six-endpoint reader opt-in, route-overlap opt-in, TurboQuant를 사용하지 않았다.

## 설정

| 항목 | 값 |
|---|---:|
| sequence/current position | 65,536 tokens |
| batch | 1 |
| Q heads / KV heads | 24 / 8 |
| head dimension | 128 |
| page block | 128 tokens |
| Q chunk | 128 tokens |
| K/V dtype | BFLOAT8_B |
| Q dtype | BFLOAT16 |
| K chunk A/B | 128 / 256 / 512 tokens |
| active split | 2 cores/KV head, 16 cores |
| profiler clock | 650 MHz |

각 구성은 profiler-free correctness/JIT gate 뒤 Tracy device-zone capture 1회를 수행했다. capture는
warmup과 measured invocation을 분리했고 measured run host ID 1024만 집계했다. reader zone은 NCRISC,
compute CB wait zone은 TRISC_0의 `ZONE_TOTAL` cycle을 650 MHz로 환산했다.

## 정적 작업량 계산

각 core는 32,768 KV tokens를 담당한다. BFLOAT8_B tile은 1,088 bytes다.

| 항목 | K128 | K256 | K512 |
|---|---:|---:|---:|
| chunks/core | 256 | 128 | 64 |
| K tiles/chunk | 16 | 32 | 64 |
| V tiles/chunk | 16 | 32 | 64 |
| K tiles/core | 4,096 | 4,096 | 4,096 |
| V tiles/core | 4,096 | 4,096 | 4,096 |
| K+V CB handoffs/core | 512 | 256 | 128 |
| online-softmax merges/core | 255 | 127 | 63 |
| K+V reader barriers/core | 512 | 512 | 512 |

K128은 16-tile chunk 끝에서 barrier 1회다. K256은 18-tile threshold와 마지막 14 tiles에서 barrier
2회다. K512는 chunk당 barrier 4회, chunk 수 64다. 총 barrier 호출은 동일하다. 16 cores의 K+V payload는 세 구성 모두
131,072 tiles, 142,606,336 bytes, 정확히 136 MiB다. page-table traffic은 이 payload 계산에서 제외했다.

## 측정 결과

### Correctness gate

| 구성 | PCC | max abs | completion |
|---|---:|---:|---|
| K128 | 0.9998791595 | 0.04526168 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |
| K256 | 0.9999178293 | 0.02732441 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |
| K512 | 0.9999187055 | 0.02732441 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |

### Kernel duration

| metric | K128 | K256 | K512 | K512 vs K256 |
|---|---:|---:|---:|---:|
| device kernel | 3.482728 ms | 3.330791 ms | 3.324249 ms | -0.196% |
| NCRISC kernel | 3.450845 ms | 3.293785 ms | 3.303038 ms | +0.281% |
| TRISC0 kernel | 3.479691 ms | 3.327788 ms | 3.321254 ms | -0.196% |

K128/K256 kernel ratio는 1.0456이다. K128/K512는 1.0477이다.

### Reader read-barrier 누적 시간

K와 V zone 합, core별 통계다. reader barrier는 compute와 겹치므로 kernel time에 더하면 안 된다.

| 통계 | K128 | K256 | K512 | K512 vs K256 |
|---|---:|---:|---:|---:|
| core mean | 1.784925 ms | 1.891567 ms | 1.919115 ms | +1.456% |
| core median | 2.005882 ms | 2.040145 ms | 1.997924 ms | -2.069% |
| critical core max | 2.314026 ms | 2.448168 ms | 2.480860 ms | +1.335% |

K256/K512는 동일한 총 barrier 호출과 payload를 유지했고 barrier time도 줄지 않았다. DRAM service 개선의
증거가 없다.

### Compute K/V CB wait 누적 시간

K와 V wait 합, TRISC_0 core별 통계다.

| 통계 | K128 | K256 | K512 | K512 vs K256 |
|---|---:|---:|---:|---:|
| core mean | 0.587502 ms | 1.114655 ms | 1.466564 ms | +31.571% |
| core median | 0.516082 ms | 1.228109 ms | 1.559831 ms | +27.011% |
| critical core max | 1.115974 ms | 1.658969 ms | 2.042380 ms | +23.130% |
| critical max / kernel | 32.04% | 49.81% | 61.44% | +11.63 pp |

K wait core mean은 0.248035 ms에서 0.631652 ms로 154.66% 증가했다. V wait core mean은
0.339467 ms에서 0.483003 ms로 42.28% 증가했다. K256은 wait call 수가 절반인데도 누적 wait가
증가했다. K512의 K/V wait mean은 0.890846/0.575718 ms로 더 커졌다. 큰 CB reservation이 ready가
될 때까지 한 번에 더 오래 막히는 형태다.

## 판정

### 관측 사실

1. K256은 measured kernel을 151.937 us, 4.362% 줄였다.
2. K512는 K256보다 6.542 us, 0.196%만 줄였다.
3. K/V payload, tile 수, reader barrier 호출 수는 세 구성에서 동일하다.
4. K512의 reader barrier mean은 K256보다 1.456% 증가했다.
5. K512의 compute CB wait는 K256보다 mean 31.571%, critical core 23.130% 증가했다.
6. 모든 gate/capture가 정상 completion, close, exit 0으로 끝났다.

### 강한 추론

K256의 이득은 DRAM bandwidth나 CB readiness 개선이 아니다. chunks/core가 256에서 128로 줄면서
QK matmul, PV matmul, CB handoff, loop bookkeeping, online-softmax merge가 절반 가까이 감소한 효과가
input wait 증가보다 컸다. K512에서는 overhead 추가 감소와 wait 추가 증가가 거의 상쇄됐다.

### 미검증 가설

- K256의 더 긴 단일 QK/PV 구간이 reader prefetch와 compute를 더 길게 겹쳐 전체 kernel을 줄였을 수 있다.
- profiler zone 호출 자체도 K256에서 절반이므로 계측 overhead 일부가 4.36%에 포함될 수 있다.
- core별 wait 편차는 endpoint route와 head/core assignment가 합쳐진 결과일 수 있다. 이번 capture만으로
  각 원인의 비율은 분리하지 못했다.

## 다음 정량 실험

1. 임시 zone 없는 production kernel에서 K128/K256/K512 각각 20회 latency 분포를 얻는다.
2. K64 capture를 추가해 작은 chunk 쪽 기울기를 확인한다.
3. K256을 유지한 채 read threshold만 sweep한다. 총 barrier 수와 CB-ready latency를 분리할 수 있다.
4. core-to-endpoint mapping별 critical wait를 묶어 route 효과와 chunk 효과를 분리한다.

## 재현과 artifact

실험 runner는 vanilla 저장소에 임시 추가해 K chunk 128, 256, 512로 실행했다.
Tracy는 timeout의 직접 child로 실행했고 profiler artifact root를 구성별로 분리했다. 임시 source patch는
측정 뒤 역적용했다.

- artifact root:
  `/home/iris_hb4/profiler_runs/sdpa_vanilla_kchunk_phase_ab_2026_08_09_12_15_00`
- K128 device CSV:
  `k128_capture/reports/sdpa_vanilla_k128_phase/2026_08_09_12_27_55/profile_log_device.csv`
- K128 ops CSV:
  `k128_capture/reports/sdpa_vanilla_k128_phase/2026_08_09_12_27_55/ops_perf_results_sdpa_vanilla_k128_phase.csv`
- K256 device CSV:
  `k256_capture/reports/sdpa_vanilla_k256_phase/2026_08_09_12_28_35/profile_log_device.csv`
- K256 ops CSV:
  `k256_capture/reports/sdpa_vanilla_k256_phase/2026_08_09_12_28_35/ops_perf_results_sdpa_vanilla_k256_phase.csv`
- K512 device CSV:
  `k512_capture/reports/sdpa_vanilla_k512_phase/2026_08_09_12_45_16/profile_log_device.csv`
- K512 ops CSV:
  `k512_capture/reports/sdpa_vanilla_k512_phase/2026_08_09_12_45_16/ops_perf_results_sdpa_vanilla_k512_phase_2026_08_09_12_45_16.csv`
- gate/capture logs: `k128_gate.log`, `k256_gate.log`, `k512_gate.log`, `k128_profile.log`, `k256_profile.log`, `k512_profile.log`
- instrumentation patch:
  `/home/iris_hb4/tmp/codex-patches/20260809-121000-vanilla-sdpa-kchunk-phase-profile.patch`
- instrumentation patch SHA-256:
  `15f8d2361ed176e9ca091ac7baa7fee8d3ac348c5b4a33106dfe34868031e20e`
- K512 runner extension patch:
  `/home/iris_hb4/tmp/codex-patches/20260809-125500-vanilla-sdpa-kchunk512-runner.patch`

## 한계

- 각 구성 measured capture 1회다. 통계적 latency A/B가 아니다.
- device-zone instrumentation이 production kernel timing을 교란한다.
- NoC trace를 동시에 켜지 않았다. endpoint별 byte/time은 이전 endpoint trace를 참조했다.
- UMD의 P100 문자열은 heuristic warning이다. board identity 근거로 사용하지 않았다.
- 결과는 isolated SDPA다. full-model tokens/s 개선폭으로 직접 환산하지 않는다.

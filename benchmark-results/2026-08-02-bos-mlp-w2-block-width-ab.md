# BOS MLP DRAM-sharded 및 W2 block-width A/B

측정일: 2026-08-02 UTC

## 요약

재부팅 후 32×32 BF16 add smoke가 정상 종료된 상태에서 Llama 3.2 3B의 한 MLP layer를 비교했다.
interleaved baseline 대비 DRAM-sharded 구성은 중앙 latency를 14.83% 줄였고 처리율은 17.41%
높였다. DRAM-sharded 상태에서 W2 `in0_block_w`를 자동값 8에서 16으로 늘린 추가 효과는 중앙
latency 1.42% 감소, 처리율 1.44% 증가에 그쳤다.

따라서 이번 실험에서 주효한 변수는 W2 block-width보다 DRAM-sharded weight data path다.

## 장치와 실행 조건

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- MLP program grid 설정: 4×4
- operation active-core 수: 이번 direct latency run에서는 kernel trace로 검증하지 않음
- physical DRAM topology: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime log: `Dram Interface Workers: 6`
- 모델: `meta-llama/Llama-3.2-3B-Instruct`, layer 0 MLP
- 입력: `[1, 1, 32, 3072]`, BFLOAT8_B device input
- weight: W1/W2/W3 BFLOAT8_B
- correctness: PyTorch reference 대비 PCC threshold 0.99
- 측정: correctness/JIT call 1회 뒤 10회; 각 sample은 MLP 호출과 device synchronize를 포함
- 제외: input 생성 및 host-to-device 준비, Tracy, NoC capture
- 모든 run: 외부 `SIGINT` timeout 180초, `--kill-after=15s`, 명시적 device close

`4×4` 설정만으로 active compute core가 정확히 16개였다고 단정하지 않는다. 또한 interface-worker
6개라는 log는 해당 matmul data path의 선택이며 tensor shard 수나 physical bank 수 자체가 아니다.
사용한 DRAM-sharded helper는 generic mapping이므로 BOS 전용 3-bank/6-endpoint route 최적화로
해석하지 않는다.

## 결과

| 구성 | PCC | mean (ms) | median (ms) | min (ms) | baseline 대비 처리율 |
|---|---:|---:|---:|---:|---:|
| interleaved, W2 auto | 0.999641 | 2.213819 | 2.229688 | 2.178289 | 기준 |
| DRAM-sharded, W2 auto=8 | 0.999660 | 1.899828 | 1.899062 | 1.855579 | +17.41% |
| DRAM-sharded, W2 block=16 | 0.999641 | 1.878791 | 1.872031 | 1.863055 | +19.11% |

DRAM-sharded block 16은 block 8 대비 mean 기준 1.12%, median 기준 1.44% 높은 처리율이었다. W2의
per-core K는 16 tiles이며 auto block 8은 2 block/core, block 16은 1 block/core가 된다. 전체 W2
K-loop 관점에서는 32회에서 16회로 줄지만, 전체 MLP latency 개선은 작았다. 이는 W2 block 경계만이
현재 layer의 지배 병목은 아니라는 관측과 일치한다.

세 BFP8 weight의 tile 저장량을 합친 약 80.216 MB를 전체 MLP 중앙 latency로 단순 나눈 참고값은
interleaved 35.98 GB/s, DRAM-sharded block 8 42.24 GB/s, block 16 42.85 GB/s다. 이 값에는 W1/W3/W2
compute, SiLU, elementwise 및 collective 시간이 섞여 있으므로 직접 측정한 DRAM bandwidth가 아니라
`effective aggregate weight-byte rate`다.

## 재현

runner:

```text
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

공통 환경과 명령:

```bash
cd /home/iris_hb4/tt-metal-hb4
env \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  MLP_AB_ITERATIONS=10 \
  timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

DRAM-sharded auto=8에는 `TT_METAL_MLP_DRAM_SHARDED=1`을 추가한다. block 16에는 여기에
`TT_METAL_MLP_W2_IN0_BLOCK_W=16`도 추가한다. interleaved baseline에서는 두 변수를 모두 unset한다.

## 한계와 다음 판단

- sample 10회의 단일 세션 결과이며 장시간 분포나 온도 변동을 포함하지 않는다.
- whole-layer direct timing이므로 W1/W3/W2별 latency와 NoC/DRAM stall을 분리하지 못한다.
- 실제 end-to-end token/s 개선은 다른 layer와 attention 비중 때문에 이 수치보다 작다.
- 이번 결과만 보면 block 16은 유지 후보지만 1.4%는 작다. 반복 세션에서 재현된 뒤 기본값 승격을
  판단한다.
- 다음 profiler는 safety contract에 따라 동일 binary의 correctness/latency가 먼저 통과한 isolated
  single operation으로만 수행한다.

## 2026-08-02 NoC saturation 측정 시도

DRAM-sharded + W2 block 16을 새 baseline으로 두고 warmup/correctness 1회와 measured 1회만 수집하는
isolated NoC capture를 시작했다. Tracy parent가 device child를 시작하기 전에 `/usr/bin/python3`의
`No module named tracy`로 exit code 4를 반환했다. device open, model load, MLP 및 kernel launch에는
진입하지 않았고 raw NoC trace, ops CSV와 device trace도 생성되지 않았다. 따라서 포화 여부에 대한
측정값은 아직 없다.

실패 run은 다음 위치에 보존했다.

```text
/home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_w2_block16_noc_2026_08_02_09_15_00
```

원인은 Tracy가 내부 child를 bare `python3`로 시작하는데 shell PATH가 venv보다 system Python을 먼저
선택한 것이다. venv `bin`을 PATH 선두에 두면 host-only import 검증은 통과한다. 그러나 incomplete
report 뒤에는 같은 session에서 장치 작업을 반복하지 않는 safety contract에 따라 즉시 재시도하지
않았다.

## NoC capture 성공 및 포화도 결론

사용자가 이후 장비 가동 성공을 확인했고, 첫 32×32 BF16 add가 결과 2.0과 정상 close로 통과했다.
venv PATH를 고정한 새 isolated capture는 PCC 0.999641, `MLP_COMPLETED`, 정상 device close 및 exit
code 0으로 완료됐다. ops CSV, device profile과 raw NoC trace 12개가 생성됐다.

measured W1/W3/W2의 raw DRAM read bytes를 device kernel duration으로 나눈 결과는 각각 45.94,
45.22, 45.04 GB/s였다. 세 projection 합산은 45.40 GB/s로, BOS DRAM saturation microbenchmark
86.83 GB/s의 52.28%다. 따라서 DRAM-sharded + W2 block 16도 DRAM 전체를 포화하지 않는다.

각 projection은 20-core row와 6 interface readers를 사용했다. 여섯 reader의 개별 request 수는
동일했지만 세 destination coordinate에 reader 3개, 2개, 1개가 연결됐다. 이에 따라 traffic도
50%:33.3%:16.7%로 갈렸다. aggregate destination rate는 약 22.70, 15.13, 7.57 GB/s다. 현 병목은
reader 여섯 개의 개별 작업량 부족보다 generic DRAM-sharded mapping의 destination 불균형에 더
가깝다.

상세 raw 계산과 artifact는 다음 run에 있다.

```text
/home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_w2_block16_noc_2026_08_02_09_10_00
```

NPE가 출력한 generic `DRAM BW Util=8.4%`는 device-sync warning과 잘못된 P100 topology metadata가
있어 BOS 절대 utilization으로 사용하지 않았다. 결론은 raw NoC read byte, device kernel duration 및
BOS microbenchmark의 86.83 GB/s 실측 peak를 기준으로 한다.

## Full-demo 적용 제한

후속 Llama 3.2 3B greeting smoke에서 DRAM-sharded + W2 block 16 구성은 첫 prefill MLP W1의 host
validation에서 실패했다. weight는 모든 mode에서 DRAM width-sharded로 load되지만 DRAM-sharded
program config는 decode에만 전달되기 때문이다. generic prefill matmul은 input B에 interleaved
layout을 요구했다.

따라서 이 구성은 현재 `isolated decode MLP baseline`으로만 분류하며 full Llama baseline으로 채택하지
않는다. full demo에는 prefill/decode별 weight layout 분리 또는 정식 DRAM-sharded prefill 지원이
필요하다. 상세 incident는
`incidents/2026-08-02-bos-llama32-dram-sharded-prefill-validation-failure.md`에 기록했다.

## 2026-08-03 reader locality 및 read-page cap A/B

`TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY`는 weight-read NoC 기준으로 6 interface reader의 좌표를
맞춘다. kernel NoC, CB 계약과 output reshard write는 바꾸지 않는다. 별도 16 KiB page-cap opt-in은
기본 8 KiB 상한을 Blackhole의 단일 NoC packet 상한인 16 KiB로 넓히되 tagged triple buffer는 유지한다.

| 구성 | mean | median | min | PCC |
|---|---:|---:|---:|---:|
| locality A/B baseline: locality=0, 8 KiB cap (n=3) | 1.876325 ms | 1.880904 ms | 1.852806 ms | 0.9996410623 |
| locality=1, 8 KiB cap (n=3) | 1.889436 ms | 1.868732 ms | 1.861963 ms | 0.9996410623 |
| page-cap A/B baseline: locality=0, 8 KiB cap (n=5) | 1.911376 ms | 1.905811 ms | 1.896639 ms | 0.9996410623 |
| locality=0, 16 KiB cap (n=5) | 1.898638 ms | 1.890470 ms | 1.882830 ms | 0.9996410623 |
| locality=1, 16 KiB cap (n=5) | 1.902761 ms | 1.899199 ms | 1.888148 ms | 0.9996410623 |

locality-only 행은 직전 별도 3-sample session 결과라 절대 latency를 다른 5-sample 행과 직접 비교하지
않는다. 같은 session의 locality=0 baseline과 비교하면 평균은 0.70% 느렸고 median 차이는 noise 수준이다.

16 KiB cap은 tile 정렬 후 W1/W3 page 6,528 B에는 변화를 주지 않았고 W2 page만 4,352 B에서 8,704 B로
증가시켰다. 같은 새 binary의 5-sample A/B에서 mean은 0.67%, median은 0.80%, min은 0.73% 감소했다.

- 모든 run은 `Dram Interface Workers: 6`을 확인했다.
- 모든 run은 PCC 0.9996410623, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit code 0을 통과했다.
- profiler와 Watcher는 사용하지 않았으며 새 profiler artifact는 없다.

opt-in은 다음 환경변수로 재현한다.

```bash
TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1
```

Blackhole 이외 architecture에서는 host validation으로 거절한다. 변경 patch는
`/home/iris_hb4/tmp/codex-patches/20260803-064900-mlp-dram-sharded-16k-read-page.patch`이며 실제
page 병합 효과가 1% 미만이므로 기본값은 계속 8 KiB로 유지하고 16 KiB cap은 opt-in 후보로만 유지한다.

## 2026-08-03 BRISC/TRISC span 및 math-active core 분석

### 실행 상태와 artifact

16 KiB read-page opt-in의 isolated decode MLP를 profiler 없이 먼저 검증한 뒤, correctness/JIT 1회와
measured 1회만 포함한 기본 device-span capture와 performance-counter capture를 각각 한 번 수행했다.
두 run 모두 PCC 0.9996410623, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit code 0을 통과했다. NoC trace와
Watcher는 사용하지 않았다.

```text
/home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_w2_block16_page16k_engine_2026_08_03_07_03_30
/home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_w2_block16_page16k_counters_2026_08_03_07_05_00
```

두 디렉터리 모두 ops CSV, `profile_log_device.csv`와 `tracy_profile_log_host.tracy`가 완성됐다.

### 관측 사실

| Projection | Device kernel | BRISC | NCRISC | TRISC max |
|---|---:|---:|---:|---:|
| W1 | 579.502 us | 579.089 us | 526.275 us | 575.866 us |
| W3 | 583.205 us | 583.189 us | 532.498 us | 580.691 us |
| W2 | 563.248 us | 563.240 us | 529.088 us | 561.985 us |

BRISC와 TRISC의 종료 시점이 1.6--3.6 us 이내라 단순 kernel span만 보면 reader와 compute가 같은
critical path에 있다. 다만 긴 TRISC span은 지속적인 유효 math 실행을 뜻하지 않는다.

counter capture에서는 20개 program core 가운데 `(0,2) (0,4) (1,4) (2,4) (3,4) (4,4)`의 6개
core에서만 FPU/MATH counter가 0이 아니었다. 이 좌표는 기존 NoC capture의 6 interface-reader
source와 정확히 같다. 활성 6개 core의 FPU utilization은 W1 39.04--41.47%, W3 39.35--41.28%,
W2 35.91--37.00%였고 나머지 14개 core는 0%였다. SFPU는 세 projection 모두 0%였다.

요청은 `--profiler-capture-perf-counters=all`이었지만 이 Blackhole 경로의 raw device log에는
TRISC1의 FPU/SFPU/MATH counter만 생성됐다. pack, unpack, L1 및 instruction counter는 생성되지
않았으므로 해당 stall 비율은 이번 run으로 정량화할 수 없다. ops CSV의 full-grid 평균은 generic
`Max Compute Cores: 24` normalization을 포함하므로 custom 20-core BOS utilization로 사용하지 않는다.

factory 소스도 이 동작을 명시한다. compute kernel 자체는 20-core bounding rectangle에 생성되지만,
`all_worker_cores`에 포함되지 않은 core에는 runtime argument `is_worker_core=false`가 전달된다.
`all_worker_cores`의 크기는 DRAM-interface worker 수인 6과 같다.

### 해석과 다음 설계

현재 DRAM-sharded MLP는 20-core compute가 아니라 **6 reader+compute core** 구조다. destination
3:2:1 불균형뿐 아니라 14개 math-idle core가 더 큰 병렬성 제한이다. 활성 6개 core에서도 FPU가 약
36--41%만 유효하므로 DRAM/CB 공급 대기로 인한 bubble이 남아 있다고 해석할 수 있다. 정확한 bubble
종류는 pack/unpack/L1 counter 부재 때문에 아직 미검증이다.

다음 구현 후보는 6 DRAM reader와 compute ownership을 분리하는 것이다. 여섯 reader가 weight를
burst-read한 뒤 20 compute core의 더 작은 N slice로 전달하고, input activation multicast와 output
reshard를 20 compute owner에 맞춰 다시 구성해야 한다. 현재 in1 kernel은 `SKIP_MCAST`이고 output N
partition도 6 worker에 묶여 있으므로 compute runtime flag만 20개에 켜면 CB 계약과 output ownership이
깨진다. 별도 opt-in path에서 reader-to-compute multicast, semaphore와 reshard mapping을 함께 설계한다.

기존 8 KiB span은 NoC instrumentation이 포함된 다른 capture라 이번 16 KiB span과 직접 A/B하지 않는다.
read-page 개선의 신뢰 가능한 수치는 profiler 없는 paired run의 mean 0.67%, median 0.80%다.

### 재현 명령

```bash
PATH=/home/iris_hb4/tt-metal-hb4/python_env/bin:/usr/local/bin:/usr/bin:/bin \
TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=1 \
TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
timeout --signal=INT --kill-after=15s 120s \
/home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy -p -r --sync-host-device \
--check-exit-code \
-o /home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_w2_block16_page16k_engine_2026_08_03_07_03_30/perf_capture \
-n mlp_decode_dram_sharded_w2_block16_page16k_engine \
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

counter capture는 같은 명령에 `--profiler-capture-perf-counters=all`을 추가하고 output/name을
`mlp_decode_dram_sharded_w2_block16_page16k_counters_2026_08_03_07_05_00`으로 바꿨다.

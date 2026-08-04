# BOS MLP 6-endpoint fanout-2 및 row-burst 실험

측정일: 2026-08-03 UTC

## 요약

Llama 3.2 3B layer 0의 isolated decode MLP에서 기존 6-interface-worker DRAM-sharded 경로를
endpoint당 두 compute worker가 weight shard를 절반씩 담당하는 12-worker fanout 구조로 확장했다.
첫 구현처럼 BFP8 tile마다 NoC read를 발행하면 평균 latency가 2.249223 ms로 악화됐다. 같은 layout과
worker mapping을 유지하고 한 K-row를 연속 burst로 읽자 1.875653 ms로 회복됐으며, 기존 6-worker
baseline 1.898638 ms보다 평균 latency가 1.21% 낮고 역수 처리율은 1.23% 높았다.

따라서 이번 결과의 핵심은 worker 수 자체가 아니라 **분할된 shard에서도 연속 DRAM request를 유지하는
dataflow 계약**이다. 최종 개선폭은 작으므로 fanout-2는 아직 opt-in 실험 경로로 유지한다.

## 장치와 범위

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- physical DRAM topology: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime이 선택한 DRAM interface workers: 6
- fanout-2 compute/read workers: 12
- 모델: `meta-llama/Llama-3.2-3B-Instruct`, layer 0 MLP
- 입력: `[1, 1, 32, 3072]`, BFLOAT8_B device input
- weight: W1/W2/W3 BFLOAT8_B, DRAM width-sharded
- W2 `in0_block_w`: 16
- DRAM read-page cap: 16 KiB opt-in
- reader locality: off
- correctness: PyTorch reference 대비 PCC
- profiler/Watcher: 사용하지 않음

`Dram Interface Workers: 6`은 이 matmul data path가 선택한 interface source 수다. fanout-2는 DRAM
endpoint를 12개로 늘리지 않는다. 6개 endpoint 각각의 logical weight shard를 두 half-shard로 나누고,
endpoint당 두 worker가 서로 다른 N 범위를 읽고 계산하는 구조다. 20개 available worker 전체가 활성인
구성도 아니다.

이 측정은 isolated decode MLP라 attention의 64K curpos에 의존하지 않는다. 64K full-layer 결과에
미치는 영향은 기존 64K layer 측정에 직접 합산하지 않았으며 별도 end-to-end 검증이 필요하다.

## 구현

### TTNN/model 계층

- `model_config.py`: fanout-2일 때 weight width를 `tile × 6 endpoints × 2 workers`에 맞춰 padding한다.
  W1/W3 shard width는 1,408 elements(44 tiles), W2는 512 elements(16 tiles)다.
- `mlp.py`: fanout layout 전용 `.dram_sharded.fanout2` cache suffix를 사용한다. 이 분리가 없으면 기존
  6-worker cache가 재사용돼 실행은 끝나도 PCC가 0.000487 수준으로 무너졌다.
- 환경변수 `TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1`일 때만 새 경로를 선택한다.

### matmul program factory

- 기존 6 DRAM reader core와 activation storage core에서 고른 6 partner를 endpoint별로 interleave한다.
- 12 worker에 logical reader ID를 전달하고 physical endpoint는 `logical_id / 2`로 선택한다.
- 각 endpoint shard의 앞/뒤 half를 두 worker가 나누며 output N ownership과 reshard loop도 12-way로
  맞춘다.
- `per_core_N_compute=ceil(N/12)`를 사용하되 fanout 경로에서는 wider subblock padding을 끈다.
  reader width와 compute/output ownership이 같아야 한다.
- Blackhole, no-bias, normal compute/mcast/write 경로에서만 opt-in을 허용한다.

### dataflow reader

초기 correctness 구현은 half-shard의 각 tile에 `noc_async_read_tile`을 호출했다. 최종 구현은 동일한
row-strided 주소 계산을 유지하면서 한 K-row의 연속 bytes를 단일 `noc_async_read`로 요청한다. 실제
호출 크기는 W1/W3가 22 BFP8 tiles = 23,936 B, W2가 8 tiles = 8,704 B다. runtime log의 W1/W3
`13,056 B`는 `get_max_page_size_and_num_pages`가 계산한 기존 page size이며 새 row request 크기가 아니다.
두 호출 모두 Blackhole hardware burst 한도 8,160 B보다 크므로 하위 `ncrisc_noc_fast_read_any_len`이
여러 transaction으로 나눈다.

## 실행 안전성과 correctness 이력

사용자가 서버 재부팅을 확인한 뒤 계약상 첫 32×32 BF16 add가 result 2.0, 정상 device close, exit 0으로
통과했다. 이후 실행은 모두 profiler 없이 외부 `SIGINT` timeout 180초와 `--kill-after=15s`를 사용했다.

구현 중 관측한 실패는 모두 정상 device close 뒤 host validation 또는 PCC 실패로 끝났으며 timeout,
signal 종료, exit 124/137은 없었다.

1. 12-way로 나누어지지 않는 N을 거부한 host guard: kernel launch 전 `TT_FATAL`, exit 1
2. compute width padding과 reader ownership 불일치: PCC 실패, 정상 close
3. 기존 6-worker tensor cache 재사용: PCC 0.000487, 정상 close
4. fanout 전용 cache와 정확한 compute width 적용: PCC 0.999641, 정상 close, exit 0
5. row-burst 적용: PCC 0.999641, 정상 close, exit 0

## 결과

모든 표의 sample은 correctness/JIT call 뒤 MLP 호출과 device synchronize를 포함한다.

| 구성 | n | PCC | mean (ms) | median (ms) | min (ms) |
|---|---:|---:|---:|---:|---:|
| 기존 6-worker, locality off, 16 KiB cap | 5 | 0.999641 | 1.898638 | 1.890470 | 1.882830 |
| fanout-2 12-worker, tile read | 5 | 0.999641 | 2.249223 | 2.242075 | 2.232436 |
| fanout-2 12-worker, row burst | 5 | 0.999641 | 1.875653 | 1.869043 | 1.862983 |

row-burst sample은 `1.907145, 1.874082, 1.865014, 1.869043, 1.862983 ms`다.

| 비교 | mean latency 변화 | 역수 처리율 변화 |
|---|---:|---:|
| 기존 6-worker → fanout tile read | +18.47% | -15.59% |
| fanout tile read → fanout row burst | -16.61% | +19.92% |
| 기존 6-worker → fanout row burst | -1.21% | +1.23% |

단발 row-burst correctness run도 1.875247 ms, PCC 0.999641로 종료됐다.

## 해석

관측 사실은 다음과 같다.

- 12-way compute ownership과 fanout 전용 weight layout은 correctness를 만족한다.
- tile별 read는 worker를 두 배로 늘렸는데도 6-worker baseline보다 18.47% 느렸다.
- 연속 K-row burst만 바꾸자 같은 12-worker mapping에서 latency가 16.61% 줄었다.
- 최종 baseline 대비 이득은 1.21%로 작다.

따라서 tile별 command 발행, address setup 및 짧은 transaction이 fanout 이점을 대부분 상쇄했다고
해석한다. row burst는 이상적인 DRAM benchmark의 긴 연속 request에 조금 더 가까워졌지만, 최종 1.21%
이득만으로 DRAM 포화나 compute bubble 제거를 주장할 수 없다. 아래 isolated profile에서 compute
TOPS보다 DRAM/NoC 공급과 destination 불균형이 우선 병목임을 추가로 확인했다.

## 2026-08-03 performance-counter 및 NoC profile

### 실행 상태

동일 binary의 profiler-free correctness와 latency가 먼저 통과한 뒤 두 pass를 분리해 실행했다.

1. performance counter: `fpu,pack,unpack,l1,instrn`
2. NoC trace: `--collect-noc-traces`, counter와 동시 사용하지 않음

각 pass는 correctness/JIT 1회와 measured 1회만 포함했다. 두 run 모두 PCC 0.9996410623,
`MLP_COMPLETED`, `DEVICE_CLOSED`, exit code 0을 통과했고 ops CSV, device CSV와 host Tracy artifact가
완성됐다. timeout, signal 종료와 Watcher는 없었다.

```text
/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_counters_2026_08_03_08_30_00
/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_noc_2026_08_03_08_35_00
```

NoC pass의 `tt-npe` import는 실패해 NPE timeline은 생성되지 않았지만 raw
`noc_trace_dev0_ID*.json`, ops CSV와 device CSV는 정상 생성됐다. 따라서 성공한 raw capture로
분류하되 generic NPE utilization은 사용하지 않는다.

### Math-engine 결과

measured W1/W3/W2의 global call count는 각각 7168, 8192, 11264다. 세 projection 모두 20개 program
core 가운데 정확히 12개 core에서 FPU counter가 0보다 컸다.

| Projection | Device kernel | FPU-active cores | active-core FPU util 범위 |
|---|---:|---:|---:|
| W1 | 568.457 us | 12 | 18.33--19.46% |
| W3 | 569.620 us | 12 | 18.38--19.41% |
| W2 | 554.143 us | 12 | 18.26--19.43% |

기존 6-worker path의 active-core FPU utilization은 W1/W3 약 39--41%, W2 약 36--37%였다. fanout-2는
동일한 총 math 작업을 두 배의 core에 절반씩 나눠 active-core utilization이 거의 절반이 됐다. 12개
core가 모두 실제 계산하지만 약 19%만 FPU-active이므로 TOPS ceiling에 도달한 상태가 아니다.

CLI는 다섯 counter group을 모두 받았지만 이 checkout의 raw device log에는 `FPU_COUNTER`,
`SFPU_COUNTER`, `MATH_COUNTER`만 각각 160개 생성됐다. pack, unpack, L1, instruction counter는
생성되지 않아 해당 stall을 이번 pass로 정량화할 수 없다.

### DRAM read bandwidth

NoC profiler의 local event는 payload를 32 B chunk의 `uint8_t`로 저장하므로 한 event가 최대
`255 × 32 = 8,160 B`까지만 표현된다. 이에 따라 23,936 B 및 8,704 B `noc_async_read`가 raw JSON에서
모두 8,160 B로 보인다. JSON의 `num_bytes`를 단순 합산하면 W1/W3 9.40 MB, W2 25.07 MB로
과소계상된다.

요청 수, compile-time row width와 BFP8 tile size 1,088 B로 실제 requested bytes를 복원했다.

- W1/W3: `12 workers × 96 K rows × 22 tiles × 1,088 B = 27,574,272 B`
- W2: `12 workers × 256 K rows × 8 tiles × 1,088 B = 26,738,688 B`

| Projection | NoC-instrumented kernel | Reconstructed read bytes | Direct bandwidth | 기존 6-worker 대비 |
|---|---:|---:|---:|---:|
| W1 | 569.725 us | 27,574,272 | 48.40 GB/s | +5.35% |
| W3 | 567.080 us | 27,574,272 | 48.63 GB/s | +7.54% |
| W2 | 547.977 us | 26,738,688 | 48.80 GB/s | +8.34% |
| 합산 | 1.684782 ms | 81,887,232 | 48.60 GB/s | +7.07% |

기존 6-worker projection 합산은 45.40 GB/s 및 1.776219 ms였다. fanout-2에서 projection kernel 합산
시간은 5.15% 줄었지만 profiler-free full MLP 평균 개선은 1.21%였다. padding으로 W1/W3 read bytes가
기존보다 각각 626,688 B 늘었으므로 단순 latency만 비교한 수치와 bandwidth 증가율을 구분한다.

### Destination 분포와 병목 판정

각 source는 projection별로 동일한 request 수를 냈고 모든 weight read는 `NOC_1`이었다. 그러나 12개
source가 향한 세 DRAM destination 좌표 `(3,1)`, `(2,1)`, `(4,1)`의 source 수는 `6:5:1`, 즉
traffic `50%:41.67%:8.33%`였다. 합산 48.60 GB/s를 비례 배분하면 약 24.30, 20.25, 4.05 GB/s다.

기존 6-worker mapping은 `3:2:1`이었다. fanout이 이를 단순 복제했다면 `6:4:2`여야 하지만 실제 trace는
`6:5:1`이다. host에서 `logical_reader_id / 2`를 넘기더라도 source별
`get_noc_addr_from_bank_id` 결과가 기대한 pair와 일치하는지는 아직 미검증이다. 다음 변경은 reader 수를
더 늘리는 것보다 runtime log에 logical reader ID, bank ID와 resolved destination을 함께 남겨 12개
mapping을 고정하고 세 destination을 `4:4:4`에 가깝게 만드는 것이 우선이다.

결론적으로 이번 구성은 TOPS가 따라오지 못하는 상태가 아니다. 12개 core의 낮은 FPU active 비율,
microbenchmark peak 86.83 GB/s의 55.98%인 aggregate read rate, `6:5:1` destination 불균형을 함께 보면
DRAM/NoC→CB 공급이 우선 병목이라는 신뢰도가 높다.

## 재현 명령

```bash
cd /home/iris_hb4/tt-metal-hb4
env \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  MLP_AB_ITERATIONS=5 \
  TT_METAL_MLP_DRAM_SHARDED=1 \
  TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 \
  timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## Artifact와 한계

- profiler-free latency run에는 raw terminal output을 보존한 timestamped artifact 디렉터리가 없다.
  profiler의 raw JSON/CSV/Tracy artifact는 위 두 run 디렉터리에 보존했다.
- 기존 baseline은 같은 binary 계열과 runner의 별도 5-sample session 결과라 paired process A/B는 아니다.
- full Llama prefill에는 기존 DRAM-sharded weight-layout validation 제한이 남아 있다.
- 64K full-layer token/s는 측정하지 않았다.
- NoC event의 8,160 B 표현 한도 때문에 bandwidth는 raw `num_bytes` 합이 아니라 compile-time requested
  bytes로 복원했다.
- pack/unpack/L1/instruction counter가 raw log에 생성되지 않아 세부 CB stall 분류는 남아 있다.
- `6:5:1` destination mapping의 `4:4:4` 교정은 아래 후속 실험에서 완료했다.

## 관련 patch

- `/home/iris_hb4/tmp/codex-patches/20260803-073200-mlp-dram-fanout2-weight-padding.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-074200-mlp-dram-fanout2-core-roles.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-082500-mlp-fanout2-cache-key.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-083500-mlp-fanout2-row-burst.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-093200-mlp-balanced-endpoint-partners.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-091000-mlp-balanced-physical-coordinates.patch` (compile 실패 경로)
- `/home/iris_hb4/tmp/codex-patches/20260803-091600-mlp-balanced-full-grid.patch`

## 후속: NOC1 destination 4:4:4 교정

이 절의 결과가 위 `6:5:1` 결론과 다음 구현 후보를 대체한다. 환경변수
`TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1`을 추가한 opt-in이며, fanout-2와 NOC1 weight read에서만
허용한다.

### Host-side topology 확인

관측한 주소 및 route 계약은 다음과 같다.

- UMD Blackhole 물리 topology는 3 DRAM banks × bank당 2 NoC ports다.
- BOS `blackhole_140_arch.yaml`은 이를 6개 DRAM view로 노출하고 view 0..5에
  `0x1200000000`..`0x1700000000` address offset을 준다.
- allocator는 6개 view를 동일 폭 logical DRAM shard로 취급한다.
- BOS `get_noc_addr_from_bank_id<true>`에서 `bank_id`는 view offset 선택에 쓰이지만, NOC1 destination
  endpoint는 reader core의 x 좌표 그룹으로 정해진다.
- 기존 base reader 6개는 endpoint 그룹별 `3:2:1`이다. 4×4 input-storage grid에서 앞의 non-reader
  6개를 partner로 고른 구현은 `+3:+3:+0`이 되어 raw trace의 `6:5:1`과 일치한다.

따라서 첫 교정은 weight storage prepack이 아니라 reader placement다. balanced opt-in은 BOS 전체 5×4
worker grid에서 partner를 고르고 base `3:2:1`에 `+1:+2:+3`을 보태 최종 `4:4:4`를 만든다. 기존
fanout-2의 bank/view ID, half-shard address 계산, weight cache 및 output ownership은 바꾸지 않았다.

첫 구현은 logical core를 virtual NoC 좌표로 바꾼 뒤 그룹화해 partner를 3개만 찾았고, device kernel
launch 전 host `TT_FATAL`로 종료됐다. `DEVICE_CLOSED`, exit 1이며 timeout/signal/124/137은 없었다.
runtime이 출력한 logical x→physical x `{0,1,2,3,4}`와 기존 raw trace를 대조해 logical x 분류 및 5×4
후보로 교정했다.

### Profiler-free 결과

correctness/JIT 뒤 5회 표본은 다음과 같다.

| 구성 | n | PCC | mean (ms) | median (ms) | min (ms) |
|---|---:|---:|---:|---:|---:|
| fanout-2 row burst, destination `6:5:1` | 5 | 0.999641 | 1.875653 | 1.869043 | 1.862983 |
| fanout-2 row burst, destination `4:4:4` | 5 | 0.999641 | 1.472280 | 1.461554 | 1.457888 |

새 sample은 `1.516636, 1.461554, 1.466939, 1.458381, 1.457888 ms`다. 기존 fanout row-burst 대비
mean latency는 21.51% 감소했고 역수 처리율은 27.40% 증가했다. 기존 6-worker baseline
1.898638 ms 대비 mean latency는 22.46% 감소했다. 두 성공 run 모두 `MLP_COMPLETED`, `DEVICE_CLOSED`,
exit 0이다.

### NoC capture 결과

동일 binary의 profiler-free correctness와 latency를 먼저 통과한 뒤 correctness/JIT 1회와 measured 1회로
분리 capture했다.

```text
/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_balanced_noc_2026_08_03_09_15_00
```

PCC 0.9996410623, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0이며 raw NoC JSON, device CSV, ops CSV와 host
Tracy artifact가 완성됐다. `tt-npe` import 실패로 NPE timeline은 없지만 raw capture는 정상이다.

| Projection | Device kernel | Reconstructed read bytes | Direct bandwidth |
|---|---:|---:|---:|
| W1 | 437.009 us | 27,574,272 | 63.10 GB/s |
| W3 | 440.292 us | 27,574,272 | 62.63 GB/s |
| W2 | 423.948 us | 26,738,688 | 63.07 GB/s |
| 합산 | 1.301249 ms | 81,887,232 | 62.93 GB/s |

W1/W3/W2 모두 raw source→destination 수가 `(3,1):(2,1):(4,1) = 4:4:4`이고 모든 weight read는
`NOC_1`이다. aggregate bandwidth는 기존 불균형 fanout의 48.60 GB/s보다 29.47% 높고, direct DRAM
microbenchmark peak 86.83 GB/s의 72.48%다. 이 결과는 이번 MLP에서 storage 재배치보다 reader endpoint
균형이 먼저였음을 실측으로 확인한다.

다만 이는 isolated decode MLP 결과다. 64K full layer 및 전체 tokens/s 개선폭은 아직 측정하지 않았고,
약 24 GB/s의 microbenchmark gap에는 CB/reshard/compute pipeline과 실제 matmul access pattern 차이가
남아 있다.

## 후속: 6 compute owner + 6 prefetch helper

12 compute worker의 활성-core FPU utilization이 약 19%였다는 결과를 바탕으로, 12개 reader는 유지하면서
compute/output ownership만 6개 core로 줄이는 opt-in을 구현했다.

- 환경변수: `TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=1`
- compute/output owner 6 cores + dedicated prefetch helper 6 cores
- total DRAM readers: 12
- NOC1 destination 배치: `4:4:4` 유지
- owner/helper는 같은 physical DRAM shard의 앞/뒤 절반을 각각 읽음
- owner→helper credit 및 helper→owner valid semaphore로 CB slot을 동기화

### 실행 이력

stale shared library를 읽은 첫 run은 기존 `readers: 6, compute workers: 12`였으므로 실험에서
제외했다. 새 runtime을 명시한 뒤 다음 순서로 교정했다.

1. 6-way shard width 43 tiles: launch 전 host `TT_FATAL`, exit 1, 정상 close
2. helper runtime args 1→7 count 변경: launch 전 host `TT_FATAL`, exit 1, 정상 close
3. shard width 44 tiles와 고정 7-argument shape 적용
4. Watcher run에서 실제 `readers: 12, compute workers: 6`, endpoint `4:4:4` 확인 뒤 device abort

Watcher는 core `(4,0)` BRISC가 reader kernel 종료 시 pending non-posted NoC write가 남았다고 판정했다.
마지막 waypoint는 `NKFW, W, W, W, W`였다. Python은 abort 뒤 `D` 상태에 머물렀고 외부 180초 timeout
cleanup도 끝나지 않아 SIGKILL, 최종 exit 137이 됐다. PCC, latency, bandwidth와 정상 device close는
얻지 못했으므로 이 구성의 성능 결과는 없다. artifact는 아래에 보존했다.

`/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_correctness_2026_08_03_13_30_00/run.log`

### 수정 상태와 다음 게이트

helper는 credit 전에 다음 block을 prefetch하므로 각 block의 remote row write 뒤
`noc_async_write_barrier()`를 수행하고, 그 다음 valid semaphore와 CB pop을 수행하도록 수정했다.
patch 적용과 host `ttnncpp` build는 통과했지만 dataflow-kernel JIT와 device correctness는 미검증이다.
exit 137 이후 장치는 격리 상태다. 사용자 재시작 확인 뒤 32×32 add가 통과해야만 isolated correctness,
5-sample latency, warmup 1 + measured 1 NoC 순으로 진행한다. barrier가 overlap 이득을 줄일 수 있으므로
성공 결과 전에는 6-compute 구성이 12-compute보다 빠르다고 주장하지 않는다.

- patch: `/home/iris_hb4/tmp/codex-patches/20260803-141500-mlp-helper-write-barrier.patch`

### 재부팅 후 barrier 경로 결과

사용자가 재부팅을 확인한 뒤 32×32 add 안전 게이트, isolated correctness, profiler-free latency,
NoC capture와 counter capture를 순서대로 완료했다. 모든 run은 정상 close와 exit 0을 통과했다.

| 구성 | n | PCC | mean (ms) | median (ms) | min (ms) |
|---|---:|---:|---:|---:|---:|
| 12-compute balanced fanout-2 | 5 | 0.999641 | 1.472280 | 1.461554 | 1.457888 |
| 6-compute + 6-helper, per-block barrier | 5 | 0.999641 | 1.556066 | 1.560501 | 1.523887 |

6-compute 경로는 latency가 5.69% 높고 역수 처리율은 5.38% 낮다. 기존 6-reader/6-compute baseline
1.898638 ms보다는 latency가 18.04% 낮지만, 목표 대조군인 12-compute balanced 경로를 이기지 못했다.

### NoC 및 counter 비교

| Projection | 12-compute kernel | 6-compute helper kernel | helper read BW | duration 변화 |
|---|---:|---:|---:|---:|
| W1 | 437.009 us | 462.003 us | 59.68 GB/s | +5.72% |
| W3 | 440.292 us | 466.106 us | 59.16 GB/s | +5.86% |
| W2 | 423.948 us | 446.362 us | 59.90 GB/s | +5.29% |
| 합산 | 1.301249 ms | 1.374471 ms | 59.58 GB/s | +5.63% |

DRAM reads는 두 구성 모두 12 source, NOC1 destination `4:4:4`, 총 81,887,232 B로 동일했다.
helper 경로는 W1/W3 각 13,787,136 B와 W2 13,369,344 B, 합계 40,943,616 B를 owner L1로 추가
전송하고 block마다 write barrier를 수행한다. 이에 따라 aggregate read rate가 62.93에서 59.58 GB/s로
5.33% 낮아졌다.

FPU-active core는 정확히 6개였고 active-core utilization은 W1/W3 46.91--49.29%, W2
44.31--46.63%였다. 6-core compute가 포화된 것은 아니지만 helper relay 비용을 상쇄할 여유도 없었다.

결론적으로 현재 6-helper offload는 compute bubble 제거가 아니라 추가 on-chip copy와 synchronization을
만든다. opt-in은 분석용으로 유지하되 기본값은 기존 12-compute balanced fanout-2로 둔다.

artifacts:

- correctness/latency: `mlp_prefetch_helpers_barrier_correctness_2026_08_03_15_45_00`, `mlp_prefetch_helpers_barrier_latency_2026_08_03_15_50_00`
- NoC: `mlp_prefetch_helpers_barrier_noc_2026_08_03_15_55_00`
- counters: `mlp_prefetch_helpers_barrier_counters_2026_08_03_16_05_00`

### 기존 6-compute/no-helper 재측정

불필요한 새 분기를 만들지 않고 기존 `FANOUT2=0`, `PREFETCH_HELPERS=0` 경로를 같은 binary에서
재사용했다. runtime log로 `readers: 6, compute workers: 6`을 확인했다.

| 구성 | readers | compute | mean (ms) | helper 대비 |
|---|---:|---:|---:|---:|
| balanced direct fanout-2 | 12 | 12 | 1.472280 | -5.38% latency |
| prefetch helper | 12 | 6 | 1.556066 | 기준 |
| 기존 no-helper | 6 | 6 | 1.879179 | +20.76% latency |

no-helper samples는 `1.938916, 1.861170, 1.852085, 1.894039, 1.849683 ms`이고 PCC는
0.999641, median 1.861170 ms, 정상 close, exit 0이다. 기존 별도 baseline 1.898638 ms와도 1.02% 이내다.

helper는 no-helper 대비 latency를 17.19% 줄이고 역수 처리율을 20.76% 높였으므로 reader fanout 이득은
실재한다. 그러나 12-compute direct 경로보다 5.69% 느리므로 relay를 production 경로에 추가할 이유는
없다. 현재 최선은 helper도 새 6-compute 분기도 아닌 기존 balanced 12-compute direct 경로다.

artifact: `/home/iris_hb4/profiler_runs/mlp_existing_six_compute_no_helper_2026_08_03_16_20_00/run.log`

## 후속: 18-compute direct fanout-3

helper relay를 더 늘리는 대신 6 logical DRAM shards를 shard당 3 reader/compute가 직접 나누어 읽고
계산하는 opt-in TT_METAL_MLP_DRAM_SHARDED_FANOUT3=1을 추가했다. fanout-2와 fanout-3는 상호
배타적이고 helper는 fanout-3에서 금지된다. 기본 경로는 기존 balanced fanout-2다.

custom BOS의 5×4 worker grid를 NOC1 endpoint group으로 분류하면 수용량은 8:8:4다. 따라서 18
reader를 6:6:6으로 배치할 수 없으며 capacity-aware 최선은 7:7:4다. fanout-2는 동일
알고리즘에서 기존 4:4:4를 유지한다. 성공 run의 host log는 다음을 직접 확인했다.

    DRAM-sharded fanout-3 balanced endpoints: true, NOC1 endpoint groups: 7:7:4
    DRAM-sharded fanout factor: 3, prefetch helpers: false, readers: 18, compute workers: 18

첫 시도는 새 build_home_release/ttnn/_ttnncpp.so 대신 stale
build_home_release/lib/_ttnncpp.so를 로드해 기존 fanout-2-only host fatal에서 종료됐다. 두 번째
시도는 균등 6:6:6 limit 때문에 partner를 10개만 선택해 host fatal에서 종료됐다. 둘 다 device
kernel launch 전 실패, DEVICE_CLOSED, exit 1이며 timeout이나 격리 사건은 아니다. runtime library를
동일 checksum으로 배포하고 endpoint capacity를 교정한 세 번째 correctness와 후속 latency run은
모두 정상 close와 exit 0을 통과했다.

| 구성 | readers | compute | endpoint groups | n | PCC | mean (ms) | median (ms) | min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 no-helper | 6 | 6 | 기존 배치 | 5 | 0.999641 | 1.879179 | 1.861170 | 1.849683 |
| balanced fanout-2 | 12 | 12 | 4:4:4 | 5 | 0.999641 | 1.472280 | 1.461554 | 1.457888 |
| capacity-aware fanout-3 | 18 | 18 | 7:7:4 | 5 | 0.999641 | 1.703471 | 1.687755 | 1.686237 |

fanout-3 samples는 1.742215, 1.686237, 1.687755, 1.713754, 1.687396 ms다. 6-compute
baseline보다는 latency가 9.35% 낮지만, 목표 대조군인 12-compute fanout-2보다 15.70% 높고 역수
처리율은 13.57% 낮다.

Llama 3.2 3B shape에서 fanout-3 정렬은 W1/W3 output width를 8192→8640(+5.47%), W2 output
width를 3072→3456(+12.50%)으로 pad한다. fanout-2의 해당 width는 8448과 3072이므로 fanout-3는
특히 W2에 새 read/compute 낭비를 만든다. 여기에 7:7:4 endpoint 비대칭과 18-way input
multicast/output reshard 비용이 더해진다. 12→18 확대가 명확히 느려 profiler/NoC capture는 수행하지
않았다. 현재 production 후보는 계속 balanced 12-compute fanout-2다.

artifacts:

- stale runtime host failure:
  /home/iris_hb4/profiler_runs/mlp_fanout3_18_compute_correctness_2026_08_03_17_15_00/run.log
- equal-share capacity host failure:
  /home/iris_hb4/profiler_runs/mlp_fanout3_18_compute_correctness_2026_08_03_17_25_00/run.log
- correctness:
  /home/iris_hb4/profiler_runs/mlp_fanout3_18_compute_correctness_2026_08_03_17_35_00/run.log
- 5-sample latency:
  /home/iris_hb4/profiler_runs/mlp_fanout3_18_compute_latency_2026_08_03_17_40_00/run.log

## 후속: 물리 6-endpoint × 3-reader dual-NoC

기존 fanout-3의 `7:7:4`는 NOC1 logical endpoint group 분류였고 실제 6개 DRAM worker endpoint에
3 reader씩 붙인 구성이 아니었다. 이를 분리하기 위해 분석용 opt-in
`TT_METAL_MLP_DRAM_SHARDED_FANOUT3_DUAL_NOC=1`을 추가했다.

- board: custom 20-core BOS NPU, Blackhole runtime/code architecture
- available worker grid: 5×4, 이번 program의 reader/compute 목표: 18
- DRAM: 3 physical banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- endpoint x→NoC: `{0:NOC0, 1:NOC1, 2:NOC1, 3:NOC1, 4:NOC0, 5:NOC0}`
- endpoint별 reader 수: `3:3:3:3:3:3`
- reader kernel 수: NOC0 9 cores + NOC1 9 cores
- weight page cap: 16 KiB, W2 `in0_block_w=16`, helper 없음

host는 각 endpoint reader를 서로 다른 3개 physical worker row에 놓고, endpoint x와 DRAM view를
runtime topology API로 연결한다. dataflow kernel은 generic bank helper 대신 microbenchmark와 같은
`bank_to_dram_offset[dram_view] + buffer_address`와 explicit endpoint NoC coordinate로 source address를
만든다. host build와 Python runtime library 배포 checksum은
`87dbd2cc6d8f8f4770f23022ea875c8391e079ea2bb1b8bf2b986289df9a3950`으로 일치한다.

### 실행 결과

첫 run은 `HF_MODEL` 누락으로 model init host assertion, 두 번째 run은 순차 greedy가 마지막 endpoint에
서로 다른 worker row를 남기지 못해 host `TT_FATAL`로 끝났다. 둘 다 device kernel launch 전
`DEVICE_CLOSED`와 driver close가 확인됐으므로 device timeout 사건으로 분류하지 않는다.

row-balanced 배치로 교정한 세 번째 profiler-free correctness run은 host log에서 다음을 확인했다.

    DRAM-sharded fanout-3 explicit endpoints: 3:3:3:3:3:3, reader NoCs: 9:9, compute workers: 18
    DRAM-sharded fanout factor: 3, prefetch helpers: false, readers: 18, compute workers: 18

W1과 W2 program 생성 로그 뒤 `MLP_PCC`, `MLP_COMPLETED`, `DEVICE_CLOSED`가 나오지 않았다. 외부
`timeout --signal=INT --kill-after=15s 180s`의 SIGINT cleanup도 끝나지 않아 SIGKILL, exit 137로
종료됐다. PID 11104 Python child는 종료 뒤 PID 1 아래 `Z/<defunct>`로 남았다. 따라서 correctness,
latency 및 bandwidth 결과는 없으며 이 구성을 성공이나 성능 개선으로 분류하지 않는다.

- artifact: `/home/iris_hb4/profiler_runs/mlp_fanout3_dual_noc_3x6_correctness_2026_08_03_19_10_00/run.log`
- artifact SHA-256: `9f42c5f0cea9cdfc5d3ddd9b144db70b2cb125ae8afb0ddef332d9e00b0bd79f`

### 원인 가설과 수정 상태

성공한 20-core/6-endpoint DRAM microbenchmark는 같은 endpoint의 edge와 같은 `(NoC, worker-row)`
route의 edge가 VC를 공유하지 않도록 4-VC edge coloring을 수행한다. timeout 당시 MLP 구현은 endpoint별
고정 VC 하나를 사용해 endpoint당 3 reader가 같은 VC를 공유했다. 이는 확인된 producer/NoC 계약 차이이며
read completion 정지의 가장 유력한 원인이다. 다만 Watcher waypoint가 없는 run이므로 root cause 확정은
아니다.

timeout 뒤 host-side에서 microbenchmark와 같은 endpoint/route conflict-free 4-VC coloring으로 수정하고
`ttnncpp` build 및 library 배포까지 완료했다. 이 교정본은 device 미검증이다. exit 137로 현재 장치는
격리 상태이며, 사용자 재부팅 확인 뒤 첫 workload인 timeout 보호 32×32 add가 통과하기 전에는 corrected
dual-NoC correctness를 실행하지 않는다. correction이 통과하더라도 1-iteration correctness 뒤에만
5-sample latency를 수행하며, 기존 fanout-2 1.472280 ms를 이기지 못하면 NoC profile은 수행하지 않는다.

현재 production 후보는 계속 balanced fanout-2 12-reader/12-compute 경로다.

### 재시작 후 VC 교정본 재검증

사용자가 서버 재시작을 확인했다. 첫 workload인 32×32 BF16 add는 `SMOKE_VALUE 2.0`,
`DEVICE_CLOSED`, 정상 driver close로 통과했다. 이어 profiler/Watcher 없이 VC edge-coloring 교정본의
isolated correctness를 1회 실행했다.

host log는 다시 endpoint `3:3:3:3:3:3`, reader NoC `9:9`, reader/compute 18을 확인했지만 W1/W2
program 생성 뒤 completion이 없었다. 180초 SIGINT와 15초 cleanup 뒤 SIGKILL, exit 137이며 PCC,
`MLP_COMPLETED`, `DEVICE_CLOSED`는 없다. Python PID 7036은 PID 1 아래 zombie로 남았다.

- add artifact SHA-256: `bac03263a91383ed6ec616133902f67067af23a93c597b1c14ba095319d87b0f`
- retry artifact: `/home/iris_hb4/profiler_runs/mlp_fanout3_dual_noc_3x6_vc_correctness_2026_08_03_20_35_00/run.log`
- retry artifact SHA-256: `b17c0d21d2801ca6f4a4d3e47c3410900b9f89f922bb98633da4577f5b808ad4`

따라서 VC 공유는 실제 계약 위반이었지만 단독 root cause가 아니다. NOC0으로 분리된 RISCV0
reader/writer가 output reshard write까지 NOC0으로 수행하는 결합 계약을 다음 우선 가설로 둔다. 장치는
다시 격리했고 latency 및 NoC profile은 수행하지 않았다. SDPA TurboQuant는 별도 opt-in operation이며 이번 MLP 실행에서 호출하거나 활성화하지 않았다.

### 독립 writer NoC 실패와 fanout-2 재확인

재부팅 후 add gate를 통과하고 dual fanout-3 reader의 반대 NoC로 output reshard write와 write barrier를 보내도록 분리했다. build와 runtime library checksum은 모두 `4a9bbbece0a8c6fed1f31c7779f647d87d875999950ab44782ca86466d5800f3`로 일치했다. Watcher 100ms와 `timeout --signal=INT --kill-after=15s 180s` 아래 1-iteration correctness를 실행했으나 W1/W2 program 생성 뒤 완료되지 않았다.

- add artifact: `/home/iris_hb4/profiler_runs/post_reboot_add_smoke_2026_08_03_13_21_04/run.log`
- add SHA-256: `97b8d4bf749be6bc4e27ecdebc0bbd996568264f9e0a629eeba4c6d39e029fa4`
- failed artifact: `/home/iris_hb4/profiler_runs/mlp_fanout3_dual_noc_3x6_separate_writer_correctness_2026_08_03_13_31_24/run.log`
- failed SHA-256: `927b33cfed92bb76b6208d5b5593d5bbd9b510ed9e7357c61f52c6de581a219e`
- process result: `MLP_PCC`, `MLP_COMPLETED`, `DEVICE_CLOSED` 없음, exit 137, Python PID 3802 zombie
- Watcher result: 주기적 device check는 지속했으나 명시적 error 또는 stable kernel waypoint는 출력하지 않음

사용자가 다시 재부팅한 뒤 add gate를 통과하고 같은 build에서 기존 balanced fanout-2를 재실행했다. 환경은 `FANOUT2=1`, `FANOUT3=0`, `FANOUT3_DUAL_NOC=0`, `BALANCED_ENDPOINTS=1`, `PREFETCH_HELPERS=0`, profiler/Watcher off, 1 iteration이다.

- topology: NOC1 endpoint groups `4:4:4`, readers 12, compute workers 12
- PCC: `0.9996410623374821`
- latency sample: `1.487526 ms` (기존 5-sample mean `1.472280 ms`와 1.04% 차이)
- completion: `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0, 잔류 workload 없음
- add artifact: `/home/iris_hb4/profiler_runs/post_reboot_add_smoke_2026_08_03_13_40_59/run.log`
- add SHA-256: `860225ebe144c12caa5d20b1ecc354c9355cd18d758aade205673f00ae339b91`
- fanout-2 artifact: `/home/iris_hb4/profiler_runs/mlp_existing_fanout2_balanced_correctness_2026_08_03_13_41_50/run.log`
- fanout-2 SHA-256: `7f4b7479714c216f2222851f331fbf129873f466b7d4d2a7a9036e43023452e9`

따라서 현재 host library, 공통 DRAM-sharded factory와 기존 fanout-2 dataflow는 정상이다. hang 범위는 fanout-3 dual-NoC 전용 core-set/kernel instantiation 또는 reader-side completion 계약으로 좁혀지며, output writer NoC 결합도 단독 root cause가 아니다.

### Split-kernel-only 원인 분리 결과

가장 강한 남은 가설을 분리하기 위해
`TT_METAL_MLP_DRAM_SHARDED_FANOUT3_SPLIT_KERNEL_ONLY=1` opt-in을 추가했다. 이 구성은 성공했던
standard fanout-3의 core mapping, NOC1 logical endpoint group `7:7:4`, generic bank address helper와
18 reader/compute를 그대로 유지한다. explicit physical endpoint address, custom `3:3:3:3:3:3`
placement, VC coloring, 독립 writer NoC는 사용하지 않는다. 유일한 핵심 차이는 RISCV0의 동일 in1
reader/writer source를 서로 겹치지 않는 core set의 NOC0/NOC1 kernel handle 두 개로 생성해 reader
NoC를 `9:9`로 나눈 것이다.

Watcher 100 ms와 외부 `timeout --signal=INT --kill-after=15s 180s` 아래 1-iteration correctness는
W1/W2 program 생성 뒤 완료되지 않았다. 약 194초까지 주기적인 Watcher dump는 계속됐지만 PCC,
`MLP_COMPLETED`, `DEVICE_CLOSED`가 없었고 timeout cleanup 상한 뒤 Python PID 4737이 PID 1 아래
`Z/<defunct>`로 남았다. 따라서 exit 137 timeout 실패로 분류하며 latency와 NoC profile은 수행하지
않았다.

- artifact: `/home/iris_hb4/profiler_runs/mlp_fanout3_split_kernel_only_correctness_2026_08_03_14_02_23/run.log`
- artifact SHA-256: `de7cc49e6bcf6490250a5bb9f281d13e97d9c44e92dc5fe5a9b3bc5786748115`
- Watcher SHA-256: `f1b9146f4f194c5eb931275f9d891a5a2bf599d22359913b8aa478ecb8dd713f`
- build/runtime `_ttnncpp.so` SHA-256: `4c9ef08f182dc036193f438d69516d69b32fbc7c6f48187b54e4a1f01e2a9360`

Watcher의 마지막 안정 dump에는 동일 source의 RISCV0 kernel ID 5와 6이 서로 다른 worker에 배치된
상태가 반복된다. 명시적 Watcher error나 정확한 barrier waypoint는 없으므로 어느 instruction에서
막혔는지는 아직 확정할 수 없다. 그러나 성공한 standard fanout-3에서 mapping/addressing을 유지한 채
split handle만 추가해 같은 hang이 재현됐으므로, explicit endpoint 주소나 custom placement보다
`같은 RISCV0 processor에 서로 다른 NoC 설정의 두 kernel handle을 생성하는 방식`을 원인 범위로
좁힐 수 있다. 이 방식은 실패 구성으로 유지하며 production 후보는 single-NOC1 balanced fanout-2다.
SDPA TurboQuant opt-in은 호출하거나 활성화하지 않았다. exit 137 뒤 장치는 다시 격리 상태다.

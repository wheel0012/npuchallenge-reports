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

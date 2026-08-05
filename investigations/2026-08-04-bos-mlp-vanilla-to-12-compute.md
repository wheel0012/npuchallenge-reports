# BOS MLP vanilla에서 12-compute 확정까지

날짜: 2026-08-04 UTC

## 결론

현재 decode MLP의 compute 수는 12로 확정한다. 근거는 단순히 6 DRAM endpoints × 2라는 산술이 아니다.
6-compute, 12-compute, 18-compute를 실제 correctness와 latency로 비교했고, 12-compute balanced
fanout-2가 가장 빨랐다.

| 구성 | readers | compute | mean latency | 12-compute 대비 |
|---|---:|---:|---:|---:|
| 기존 direct DRAM-sharded | 6 | 6 | 1.879179 ms | +27.64% |
| helper relay | 12 | 6 | 1.556066 ms | +5.69% |
| balanced direct fanout-2 | 12 | 12 | 1.472280 ms | 기준 |
| capacity-aware fanout-3 | 18 | 18 | 1.703471 ms | +15.70% |

12는 하드웨어 최대 core 수가 아니다. BOS에는 20 workers가 있다. 이 operation에서 12가 최적인 이유는
6 logical DRAM views 각각을 두 direct reader/compute가 나눠 맡아 endpoint traffic을 4:4:4로 만들면서,
18-way padding·route·multicast 비용과 6-way compute 직렬화를 피하기 때문이다.

## 용어와 장치

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- physical DRAM: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime-selected DRAM-interface workers: 6
- 모델: Llama 3.2 3B decode MLP, W1/W3/W2 BFP8

`20-core program grid`, `Dram Interface Workers: 6`, `active compute cores`는 서로 다른 값이다. Factory가
20-core rectangle에 kernel을 만들어도 runtime argument `is_worker_core=false`인 core는 계산하지 않는다.
따라서 grid 설정만으로 active 20-core라고 쓰지 않았다.

## 1. Vanilla interleaved baseline

초기 MLP는 W1/W3/W2 weight를 interleaved DRAM에서 읽었다. Isolated layer-0 결과는 mean
2.213819 ms, median 2.229688 ms, PCC 0.999641이었다. 이 run은 active-core trace 전이므로 정확한
compute core 수를 주장하지 않는다. 이후 비교의 기능 baseline이다.

DRAM-sharded weight와 W2 block 16을 적용하자 mean 1.878791 ms, median 1.872031 ms가 됐다.
Interleaved 대비 median latency -16.04%, 역수 처리율 +19.11%다. 이 단계에서 성능은 올랐지만
`4×4 MLP grid`를 보고 16 compute라고 오해할 여지가 남았다.

## 2. 실제 active core는 6개

Performance-counter capture에서 20 program cores 중 아래 6개만 FPU/MATH counter가 0보다 컸다.

```text
(0,2) (0,4) (1,4) (2,4) (3,4) (4,4)
```

이 좌표는 6 interface-reader source와 일치했다. W1/W3 active-core FPU utilization은 약 39--41%,
W2는 약 36--37%였다. 나머지 14 cores는 0%였다. 따라서 초기 DRAM-sharded 경로는
`6 reader + 6 compute owner` 결합 구조로 확정했다.

Raw NoC 기준 W1/W3/W2 bandwidth는 45.94/45.22/45.04 GB/s, 합산 45.40 GB/s였다. Direct DRAM
microbenchmark 86.83 GB/s의 52.28%다. Reader별 request 수는 같았지만 destination은 3:2:1로 갈렸다.
문제는 계산량보다 DRAM/NoC 공급과 endpoint 불균형에 가까웠다.

## 3. 왜 compute flag만 20개 켜지 않았나

기존 dataflow에서 weight reader와 compute/output owner는 같은 core다. In1 kernel은 `SKIP_MCAST`이고,
output N partition과 reshard mapping도 worker 수에 묶인다. Compute runtime flag만 20 cores에 켜면 다음
계약이 깨진다.

1. Reader가 publish하는 in1 CB tile 수와 compute owner 수.
2. Activation multicast destination과 semaphore count.
3. Per-core output N ownership.
4. Output reshard source mapping.

그래서 20-core 강제 활성화 대신 reader width, compute width, output ownership을 함께 바꾸는 fanout 경로를
별도 opt-in으로 만들었다.

## 4. 12-compute fanout-2 도입

6 logical DRAM shards마다 두 worker를 배치했다. 각 worker는 원래 shard width의 절반을 직접 읽고 같은
core에서 계산한다. 총 readers/compute는 12/12다. Weight width를 `tile × 6 shards × 2 workers`에 맞춰
pad하고 fanout 전용 cache 이름을 사용했다.

초기 tile-read 구현은 mean 2.249223 ms였다. 기존 6-worker 1.898638 ms보다 18.47% 느렸다. Reader를
늘리는 것만으로는 이득이 없었다. 한 K-row를 연속 burst로 바꾸자 1.875653 ms로 회복됐다. Tile command,
address setup, 짧은 transaction이 fanout 병렬성을 상쇄했던 것이다.

이 시점 12 active FPU cores는 trace로 확인했다. 각 core utilization은 약 18--19%였다. 같은 math를 두 배
core에 나눠 6-core 대비 utilization이 절반이 됐지만 kernel은 빨라졌다. TOPS 포화가 아니라 input
starvation 상태라는 증거다.

## 5. 12-compute를 빠르게 만든 핵심: 4:4:4

첫 fanout-2 trace의 NOC1 destination source 수는 6:5:1이었다. 합산 bandwidth는 48.60 GB/s로
6-worker의 45.40 GB/s보다 7.07%만 높았다. 12 readers 중 상당수가 같은 destination에 몰렸다.

Endpoint assignment를 4:4:4로 교정하자 결과가 크게 바뀌었다.

| 구성 | mean | median | PCC |
|---|---:|---:|---:|
| fanout-2, destination 6:5:1 | 1.875653 ms | 1.869043 ms | 0.999641 |
| fanout-2, destination 4:4:4 | 1.472280 ms | 1.461554 ms | 0.999641 |

Mean latency는 21.51% 줄었다. W1/W3/W2 bandwidth는 63.10/62.63/63.07 GB/s, 합산 62.93 GB/s가
됐다. 6:5:1의 48.60 GB/s 대비 +29.47%다. 따라서 12-compute의 이득은 core 수 자체보다
`두 reader per DRAM view + 균등 destination + direct compute ownership` 조합에서 나왔다.

## 6. 6-compute 재검증

### 6 direct readers, 6 compute

같은 binary에서 fanout/helper를 끄고 재측정했다. Mean 1.879179 ms, median 1.861170 ms, PCC
0.999641이었다. 과거 1.898638 ms와 1.02% 이내다. 6-compute baseline이 정상 재현됐다.

### 12 readers, 6 compute + 6 helpers

Reader 부족만 풀기 위해 6 helpers가 weight 절반을 읽고 owner L1로 전달했다. Mean 1.556066 ms로
6-direct보다 17.19% 빨랐다. Reader fanout 효과는 실재했다. 그러나 12-direct 1.472280 ms보다 5.69%
느렸다.

Helper는 총 40,943,616 B를 owner L1로 추가 전송하고 block마다 write barrier를 수행했다. DRAM read
rate도 62.93→59.58 GB/s로 낮아졌다. Helper DRAM 절감 3--5 us를 remote write 약 5 us와 residual wait
약 6 us가 소모했다. 6 compute owner의 FPU utilization은 약 44--49%였지만 전체 kernel은 더 느렸다.

## 7. 18-compute 검증

6 logical views마다 3 workers를 붙인 fanout-3를 capacity-aware 7:7:4로 배치했다. Correctness를 통과한
latency run의 mean은 1.703471 ms였다. 12-compute보다 latency가 15.70% 길고 역수 처리율은 13.57%
낮았다. 18-way width padding, route fanout, activation multicast와 output ownership 비용이 추가 parallelism을
상쇄했다. 따라서 available 20 workers를 모두 쓰는 방향은 채택하지 않았다.

### 20-core 활성화 가능성과 비채택 이유

20 workers를 모두 active compute로 만드는 것은 기능적으로 가능하다. 다만 6 logical DRAM views에 20
workers를 균등하게 대응시킬 수 없다. Direct-reader 방식이면 `4+4+3+3+3+3` 같은 비대칭 fanout이
필요하고, 두 endpoint를 묶은 physical-bank 부하는 최소 `7:7:6`이 된다. 이에 맞춰 weight width,
per-core output N, activation multicast와 output reshard mapping도 함께 변경해야 한다.

현재 병목에서는 active-core 수 증가가 그대로 성능 증가로 이어지지 않는다. Balanced 12-compute의
core별 FPU utilization은 약 18--19%다. 18-compute 실측도 12-compute보다 느렸다. 20-way 분할은 다음
비용을 더 늘릴 가능성이 높다.

1. DRAM endpoint당 reader 경합.
2. 비대칭 output-column partition과 tile padding.
3. Activation multicast destination 증가.
4. Core별 작은 작업량에 대한 CB, route와 launch 고정비.
5. W2 입력 reshard와 output ownership 복잡도.

따라서 다음 20-core 후보는 한 matmul을 20조각으로 자르는 방식보다 12 matmul owners를 유지하고 남은
8 workers에 독립 작업을 겹치는 방식이다. W1과 W3는 같은 X에 의존하지만 서로 독립이므로 별도 core
group에서 동시 실행할 수 있다. SwiGLU 또는 reshard를 선행 완료 구간과 pipeline하는 것도 후보이다.
단, W1과 W3가 같은 6 endpoints를 동시에 사용하면 DRAM 경합이 늘고, W2는 SwiGLU 완료에 의존한다.
Helper relay 실측도 추가 L1 왕복 때문에 12-direct보다 느렸다. 따라서 idle core 사용 자체를 이득으로
간주하지 않는다.

20-core 후보의 채택 gate는 다음과 같다.

- Counter에서 실제 active compute 20개 확인.
- 세 physical-bank destination과 6 endpoint traffic을 별도 확인.
- PCC가 현재 0.999641 이상 유지.
- Balanced 12-compute의 1.472280 ms보다 end-to-end MLP latency 단축.
- Padding, L1 relay와 reshard bytes를 포함한 전체 traffic 감소 또는 동일 수준 유지.

## 8. 12-compute 확정 기준

12-compute는 다음 조건을 모두 만족한 유일한 구성이다.

1. PCC 0.999641로 기능 정확성을 유지했다.
2. 6-direct, 6-helper, 18-direct보다 mean latency가 낮았다.
3. Counter에서 12 active FPU cores를 확인했다.
4. 세 physical-bank destination을 4:4:4로 균등화했다.
5. Helper relay 없이 weight를 consumer core가 직접 읽어 추가 L1 왕복을 피했다.
6. 재검증 run이 completion과 device close까지 정상 종료했다.

이는 12가 보편적 최적값이라는 뜻이 아니다. 현재 tensor shape, BFP8 weight, block 설정과 BOS route에 대한
결론이다. Shape나 dtype이 바뀌면 fanout과 padding 손익을 다시 측정해야 한다.

## 원시 artifact 교차확인

중앙 보고서 밖의 `profiler_runs`도 직접 대조했다.

- 6 direct: `/home/iris_hb4/profiler_runs/mlp_existing_six_compute_no_helper_2026_08_03_16_20_00/run.log`
  (`MLP_MEAN_MS 1.879179`).
- 12 readers/6 compute helper: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_latency_2026_08_03_15_50_00/run.log`
  (`readers:12 compute workers:6`, `MLP_MEAN_MS 1.556066`).
- 12 direct 재검증: `/home/iris_hb4/profiler_runs/mlp_existing_fanout2_balanced_correctness_2026_08_03_13_41_50/run.log`
  (`readers:12 compute workers:12`, PCC 0.999641, completion/close 확인).
- 18 direct: `/home/iris_hb4/profiler_runs/mlp_fanout3_18_compute_latency_2026_08_03_17_40_00/run.log`
  (`MLP_MEAN_MS 1.703471`).
- 12-compute balanced NoC artifact:
  `/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_balanced_noc_2026_08_03_09_15_00`.
- Helper NoC artifact:
  `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_noc_2026_08_03_15_55_00`.

`mlp_prefetch_helpers_correctness_2026_08_03_12_50_00`은 로그에 `readers:6 compute workers:12`와 PCC가
남지만 stale `_ttnncpp.so`를 사용한 run이다. Helper 성능 또는 12-compute 확정 근거에서 제외했다. 이후
13:05, 13:15, 13:30 helper 시도도 각각 odd per-core width, runtime-argument mismatch, Watcher assert로
실패했으므로 성능 표본에 넣지 않았다.

## 한계와 현재 상태

- Vanilla interleaved run에는 active-core counter가 없어 기능 baseline으로만 사용했다.
- `86.83 GB/s`는 compute 없는 direct DRAM microbenchmark 상한이다. MLP가 그대로 달성할 값은 아니다.
- 12-compute에서도 약 62.93 GB/s다. 다음 병목은 단순 endpoint 수보다 compute/CB cadence와 route bubble이다.
- 현재 장치는 별도 experimental matmul epilogue timeout exit 137 뒤 격리 상태다. 이 문서 작성 중 device
  workload는 실행하지 않았다.

## 관련 문서

- `benchmark-results/2026-08-02-bos-mlp-w2-block-width-ab.md`
- `benchmark-results/2026-08-03-bos-mlp-six-endpoint-fanout2-row-burst.md`
- `benchmark-results/2026-08-03-bos-mlp-input-readiness-profile.md`
- `benchmark-results/2026-08-04-bos-mlp-compute-block-cadence.md`
- `investigations/2026-08-04-bos-mlp-optimization-investigation.md`

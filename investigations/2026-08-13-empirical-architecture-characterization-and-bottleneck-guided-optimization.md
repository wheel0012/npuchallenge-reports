# Empirical Architecture Characterization and Bottleneck-Guided Optimization

날짜: 2026-08-13 UTC

## 결론

BOS 최적화는 무작위 parameter sweep이 아니다. DRAM, NoC, compute의 empirical ceiling을 먼저
측정하고 실제 matmul이 각 ceiling에서 얼마나 떨어졌는지 확인한다. 이후 가장 가까운 병목 축만
통제 A/B한다.

방법론:

1. architecture resource와 active resource를 구분한다.
2. subsystem별 empirical roofline을 만든다.
3. throughput, latency, backpressure를 함께 측정한다.
4. 독립 변수 하나씩 바꾸어 saturation boundary를 찾는다.
5. isolated 개선을 layer와 end-to-end에서 재검증한다.

## 장치와 범위

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- worker grid: 5×4 = 20 cores
- DRAM: 3 physical banks
- worker NoC endpoints: bank당 2개, 총 6개
- NoC: NOC0, NOC1

Program grid와 active compute-core 수는 다르다. Physical bank, endpoint, logical DRAM grid,
DRAM-interface worker 수도 서로 다른 값이다.

## DRAM empirical roofline

### Progressive DRAM saturation

#### Spatial resource utilization

- 20 readers
- 3 banks 동일 payload
- 6 endpoints 모두 활성화
- NOC0/NOC1 동일 payload
- start skew와 endpoint finish skew 확인

#### Transaction-size sweep

동일 32 KiB burst에서 packet 크기만 변경했다.

| Packet 구성 | Bandwidth |
|---|---:|
| 4 KiB × 8 | 86.181 GB/s |
| 8 KiB × 4 | 93.925 GB/s |
| 16 KiB × 2 | 93.110 GB/s |

8 KiB packet은 4 KiB 대비 약 9.0% 빠르다. 16 KiB는 추가 이득이 없다.

#### Burst-length sweep

8 KiB packet을 고정했다.

| Burst | Bandwidth |
|---:|---:|
| 8 KiB | 93.654 GB/s |
| 16 KiB | 93.719 GB/s |
| 32 KiB | 93.925 GB/s |
| 64 KiB | 95.501 GB/s |
| 128 KiB | 92.340 GB/s |

현재 sweet spot은 64 KiB다. 128 KiB 하락의 정확한 controller 원인은 미검증이다.

#### Request-concurrency sweep

8 KiB packet, 64 KiB burst에서 tagged depth를 비교했다.

| Metric | Depth 2 | Depth 3 |
|---|---:|---:|
| Bandwidth | 96.678 GB/s | 96.048 GB/s |
| Issue | 4.72% | 13.29% |
| Retire wait | 93.91% | 85.24% |
| Latency mean | 16,417 cycles | 24,257 cycles |
| Finish skew | 976,749 cycles | 704,177 cycles |

Depth 3은 barrier wait 일부를 issue backpressure로 이동시켰다. Observed latency는 47.8% 증가하고
bandwidth는 0.65% 감소했다. Depth 2에서 outstanding traffic은 충분하다.

Retire wait는 순수 idle이 아니다. 유효 payload 전송 시간도 포함한다.

### 현재 판정

> Balanced six-endpoint, dual-NoC, packed sequential read-only empirical roofline:
> approximately 96–98 GB/s.

다음을 동시에 확인했다.

- 모든 endpoint와 bank에 동일 byte
- NOC0/NOC1 동일 byte
- packet, burst, depth 확대에서 bandwidth plateau 또는 regression
- correctness 통과
- 정상 device close

120 GB/s는 DRAM timing과 controller efficiency가 검증되지 않은 추정치다. 실측 utilization 계산에
사용하지 않는다.

### 추가 memory-benchmark 병목 확인

현재 depth 계측은 request-concurrency 부족을 기각했다. 그러나 96–98 GB/s가 DRAM controller 자체
한계인지, NoC injection 또는 특정 endpoint가 제한하는지는 아직 분리되지 않았다.

#### 1. Reader injection scaling

Reader 수를 6, 12, 20으로 바꾸되 bank, endpoint, 총 payload를 동일하게 유지한다.

측정:

- aggregate bandwidth
- reader별 issue cycles
- issue call 내부 backpressure
- endpoint별 finish timestamp

판정:

- reader 증가와 함께 bandwidth 증가: request injection 부족
- bandwidth 고정, issue stall 증가: shared service 또는 NoC command path 포화
- 일부 endpoint만 늦음: route 또는 endpoint service 불균형

#### 2. Bank와 endpoint scaling

1 bank/2 endpoints, 2 banks/4 endpoints, 3 banks/6 endpoints를 같은 per-endpoint workload로 비교한다.
Bank당 endpoint 하나만 사용한 경우와 두 개 모두 사용한 경우도 분리한다.

판정:

- bank 수에 비례해 증가: bank-level parallelism 정상
- 두 번째 endpoint 추가가 무효: physical-bank controller 또는 media bandwidth 제한
- 특정 bank만 낮음: bank별 service asymmetry
- 모든 bank가 함께 정체: shared NoC 또는 memory-controller 경로 제한

#### 3. Endpoint service distribution

평균 bandwidth만으로 slow endpoint를 숨기지 않는다.

측정:

- endpoint별 latency p50/p95/max
- endpoint별 completion timestamp
- per-core finish skew
- 동일 endpoint 내부 reader variance
- 동일 physical bank의 두 endpoint 차이

Byte가 같아도 completion time이 다르면 load balance가 완료된 것이 아니다. 현재 depth-2 finish skew
976,749 cycles는 추가 분해 대상이다.

#### 4. NoC route와 injection 분리

같은 DRAM endpoint와 payload를 유지하면서 reader 위치, NOC0/NOC1, VC만 변경한다.

- near core와 far core
- 단일 NoC와 dual NoC
- route overlap 최소/최대 placement
- VC 고정과 분산

Placement에 따라 bandwidth와 latency가 크게 변하면 DRAM media보다 NoC route가 병목이다. Placement
변화가 없고 endpoint latency가 유지되면 DRAM service 한계 가능성이 커진다.

#### 5. Address locality와 controller behavior

Hardware row-buffer counter가 없으면 access pattern으로 간접 판정한다.

- sequential
- fixed stride
- page permutation
- randomized page
- aligned와 boundary-crossing 시작 주소
- working-set size sweep

Sequential만 빠르면 row locality, prefetch 또는 controller scheduling 의존성이 크다. Random access도
비슷하면 packet service와 shared bandwidth가 더 강한 제한 후보가 된다.

#### 6. Read/write mode

현재 ceiling은 read-only다. 다음을 별도 roofline으로 측정한다.

- read-only
- write-only
- read/write split-NoC
- alternating read/write
- read-heavy와 write-heavy ratio

Read/write turnaround와 writer acknowledgement 때문에 mixed ceiling은 read-only ceiling과 다를 수
있다. 하나의 DRAM peak로 합치지 않는다.

#### 7. Working-set과 steady state

- DRAM allocation 크기 sweep
- warmup 횟수 고정
- measured duration 확대
- run별 min/median/max/p95
- clock과 thermal 상태 기록

작은 working set이나 짧은 run만 빠르면 sustained DRAM ceiling으로 채택하지 않는다.

#### 병목 판정표

| 관측 | 지지되는 병목 |
|---|---|
| reader 증가 시 bandwidth 증가 | request injection 부족 |
| depth 증가 시 issue stall만 증가 | command queue 또는 service 포화 |
| bank 수에 비례해 증가하지 않음 | shared path 또는 controller scaling |
| 두 번째 endpoint가 무효 | physical-bank service 한계 |
| endpoint finish skew가 큼 | endpoint/route imbalance |
| near/far core 차이가 큼 | NoC hop 또는 route contention |
| stride/random에서 급락 | address locality 또는 row-buffer sensitivity |
| read는 빠르고 mixed는 급락 | read/write turnaround |
| 긴 run에서만 하락 | thermal, clock 또는 steady-state 문제 |

#### 현재 근거로 기각·유지되는 가설

기각:

- depth 2의 outstanding request 수가 절대적으로 부족하다.
- depth를 3으로 늘리면 bandwidth가 증가한다.

유지:

- endpoint별 service 또는 route 차이가 남아 있다.
- sequential access가 controller locality 이득을 받고 있다.
- physical bank의 두 endpoint가 media bandwidth를 공유한다.
- read/write turnaround ceiling이 read-only보다 낮다.

## NoC empirical roofline

L1-to-L1 benchmark로 DRAM과 compute를 제거한다.

통제 변수:

- source/destination 좌표와 hop
- NOC0/NOC1
- virtual channel
- unicast/multicast
- packet 크기와 fanout
- route overlap

측정값:

- aggregate와 link별 bandwidth
- sender issue stall
- receiver completion skew

## Compute empirical roofline

Operand를 L1에 resident시켜 DRAM을 제거한다.

통제 변수:

- active core 수
- data format와 math fidelity
- tile/block geometry
- K accumulation length
- unpack/math/pack overlap
- padding
- output writeback

측정값:

- sustained TOPS
- matrix-engine utilization
- unpack/math/pack stall
- core scaling efficiency
- correctness 또는 PCC

기존 GEMM 결과는 shape, format, fidelity, residency가 다르다. 아직 BOS compute peak 하나로 합치지
않는다.

## Bottleneck-guided matmul optimization

Matmul arithmetic intensity:

    AI = useful operations /
         (weight bytes + activation bytes + output bytes + mandatory relay bytes)

근사 상한:

    attainable performance =
        min(compute ceiling,
            matching DRAM ceiling × arithmetic intensity,
            matching NoC ceiling × communication intensity)

Matching은 format, direction, locality, packet geometry와 active resource가 같아야 한다.

최적화 순서:

1. weight, activation, partial sum, output residency 확인
2. shard ownership과 endpoint mapping 확인
3. transaction, burst, prefetch depth 조정
4. per-core M/N/K block과 padding 조정
5. producer/consumer, unpack/math/pack overlap 계측
6. core scaling과 communication overhead 함께 검증
7. isolated, layer, end-to-end 순서로 성능과 정확도 검증

## Controlled A/B 원칙

1. baseline command와 commit 고정
2. payload, iteration, warmup, measured run 수 고정
3. 독립 변수 하나만 변경
4. latency, bandwidth, stall, correctness 동시 기록
5. 최소 5회 반복
6. plateau와 regression도 결과로 기록
7. incomplete profiler artifact는 성공에서 제외

Sweep은 brute-force optimization이 아니다. 증가, plateau, regression 경계를 찾아 saturation을
확인하는 도구다.

## 관측과 가설

### 관측

- balanced read-only benchmark는 96–98 GB/s에 도달했다.
- request coalescing은 약 9% 개선했다.
- 과도한 burst와 depth 증가는 성능을 낮췄다.
- depth 3은 wait를 issue backpressure로 이동시켰다.

### 추론

- 현재 benchmark에서 command generation 부족은 주 병목이 아니다.
- depth 2가 충분한 outstanding traffic을 제공한다.
- 다음 우선순위는 address locality와 endpoint route 분석이다.

### 미검증

- 128 KiB regression 원인이 head-of-line blocking이다.
- endpoint finish skew 원인이 bank 내부 scheduling이다.
- 96–98 GB/s가 모든 access pattern의 공통 peak다.
- write-only와 mixed ceiling도 read-only와 같다.

## 다음 실험

1. sequential/strided/random address A/B
2. alignment와 boundary sweep
3. core placement, VC, route overlap A/B
4. write-only와 mixed read/write roofline
5. NoC-only L1-to-L1 roofline
6. L1-resident TOPS roofline
7. matmul을 matching roofline에 투영

## 재현 정보

DRAM benchmark:

    /home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/
    12_dram_20_core_6_noc_read/

Timestamp patch:

    /home/iris_hb4/tmp/codex-patches/20260813-133000-dram-service-metrics.patch

관련 문서:

- benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md
- benchmark-results/2026-07-26-llama32-3b-gemm-20core.md
- guides/gemm-benchmark-measurement.md

## 발표용 문장

> We established empirical ceilings for DRAM, NoC, and compute, then used controlled A/B experiments to
> identify the limiting subsystem.

> Parameter sweeps located saturation boundaries; they were not unconstrained brute-force optimization.

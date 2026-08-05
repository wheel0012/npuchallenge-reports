# BOS Llama 3.2 3B MLP 개선 조사

날짜: 2026-08-04 UTC

## 결론

현재 검증된 주 경로는 `DRAM-sharded weight + W2 block 16 + 12-reader fanout-2 + endpoint 4:4:4 +
tagged two-block`이다. 가장 큰 개선은 interleaved weight를 DRAM-sharded로 바꾼 것과 fanout-2
destination을 4:4:4로 균등화한 것이다. Reader-packed layout, 6-compute helper, K-block merge,
pending depth-3은 이득이 없거나 느렸다.

남은 병목은 단순 DRAM peak 부족이 아니다. 실제 matmul consumer는 kernel 시간의 약 67%를 input CB에서
기다린다. W1은 주로 weight-late지만 W3/W2는 activation과 weight의 phase가 core/block마다 바뀐다.
따라서 weight prefetch만 더 늘리거나 activation만 credit화하는 단일 처방은 부족하다.

검증된 추가 개선은 W3/W1 matmul을 합치고 SwiGLU를 L1에서 수행한 core-local 경로다. Isolated MLP는
2.02%, 최적화된 64K full decode는 0.305% 빨라졌다. 그러나 fixed-input full-model logits는 bitwise
exact가 아니며 PCC 0.999288이다. 이 경로는 계속 opt-in이다.

진짜 matmul epilogue에서 SiLU와 multiply까지 수행하는 새 경로는 아직 실패 상태다. Host validation과
device JIT compile은 교정됐지만 첫 실제 device launch가 180초 뒤 exit 137로 끝났다. 현재 장치는
격리 상태이며 재부팅 확인 전 추가 device workload를 실행하지 않는다.

## 장치와 범위

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- 주 MLP 경로: 20 program cores, 12 readers, 12 active compute workers
- physical DRAM: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime-selected DRAM-interface workers: 6
- 모델: `meta-llama/Llama-3.2-3B-Instruct`, decode MLP
- TurboQuant, fanout-3, prefetch helper: 주 결과에서 disabled

`Dram Interface Workers: 6`은 선택된 data path의 interface-worker 수다. Physical bank 수, tensor shard 수,
active compute-core 수와 같은 뜻이 아니다. Runtime의 P100/P150 추정 문자열도 BOS board identity가 아니다.

## 성능 개선 이력

| 단계 | 비교 | 결과 | 판정 |
|---|---|---:|---|
| DRAM-sharded weight | interleaved median 2.229688 ms → 1.899062 ms | latency -14.83%, 처리율 +17.41% | 채택 후보 |
| W2 block 16 | DRAM-sharded block 8 → 16 | median -1.42%, 처리율 +1.44% | 유지 |
| fanout-2 row burst | 6-worker baseline 1.898638 ms 대비 | mean -1.21% | 단독 효과 작음 |
| endpoint balance | destination 6:5:1 → 4:4:4 | mean 1.875653 → 1.472280 ms | 핵심 개선 |
| tagged two-block | full barrier → tagged | mean 1.472701 → 1.439071 ms | +2.28%, opt-in |
| fused W3/W1 + DRAM SwiGLU | fanout-2 baseline 대비 | mean +1.88% | 폐기 |
| fused W3/W1 + local SwiGLU | fanout-2 baseline 대비 | mean -2.02% | opt-in 유지 |
| 64K full decode local SwiGLU | 7.643404 → 7.666719 tok/s | +0.305% | 작지만 재현됨 |

Endpoint balance profile의 W1/W3/W2 direct bandwidth는 63.10/62.63/63.07 GB/s, 합산 62.93 GB/s다.
기존 6:5:1 fanout의 48.60 GB/s보다 29.47% 높지만 direct DRAM microbenchmark peak 86.83 GB/s의
72.48%다. 나머지 차이는 activation multicast, CB cadence, compute, reshard와 실제 matmul access pattern이
섞인 결과다.

## 병목 조사

### Consumer cadence

| projection | kernel | input wait 합 | wait/kernel | compute block | 다음 input 구간 |
|---|---:|---:|---:|---:|---:|
| W1 | 431.935 us | 288.311 us | 66.75% | 7.772 us | 17.278 us |
| W3 | 441.600 us | 295.628 us | 66.94% | 7.770 us | 17.595 us |
| W2 | 419.628 us | 282.611 us | 67.35% | 7.486 us | 17.016 us |

관측: matrix engine service보다 다음 input 준비 구간이 길다. 12 compute를 6으로 줄일 근거가 없다.
현재는 TOPS ceiling보다 producer-consumer cadence가 먼저 막힌다.

### Activation과 weight

Wait order를 `in0→in1`에서 `in1→in0`으로 바꿔도 총 wait는 거의 유지됐다. W1은 order-independent
weight-late다. W3는 양쪽이 거의 균형이고 W2는 backpressure에 따라 late side가 바뀐다. MLP 중간
activation은 이전 projection 결과에 의존하지만, 같은 projection 내부에서는 activation과 weight를
overlap할 수 있다. 필연적 전체 barrier가 아니라 다음 K block을 제때 publish하지 못하는 phase 문제다.

Tagged two-block은 BRISC weight barrier를 W1/W3/W2에서 20.90/18.52/36.82% 줄였다. 그래도 consumer
critical wait는 약 70%로 유지됐다. Pending request 하나를 유지하는 것은 효과가 있지만 충분하지 않다.

### Helper와 layout

6 owner + 6 helper는 direct보다 느렸다. Helper DRAM read 절감은 약 3--5 us였지만 remote-L1 write가
약 5 us, owner residual wait가 약 6 us 추가됐다. Compute workers도 12→6으로 줄어 이득을 상쇄했다.

Reader-packed multiblock layout도 채택하지 않는다. W2 변화는 +0.39%뿐이고 W1 sustained 4× cadence는
-2.46%였다. 기존 strided layout도 microbenchmark에서 86.560 GB/s에 도달했다. DRAM 내부 주소
불연속성이 현재 full MLP wait의 단독 원인은 아니다.

## 채택·유지·폐기

### 유지

1. Decode weight DRAM sharding.
2. W2 `in0_block_w=16`.
3. 12-reader/12-compute fanout-2.
4. BOS 6 endpoints의 destination 4:4:4 균형.
5. Tagged two-block opt-in.
6. Core-local SwiGLU opt-in과 별도 accuracy 검증.

### 폐기 또는 보류

1. 6-compute + helper: remote hop과 compute 감소로 느림.
2. 18-compute fanout-3 및 dual-NoC fanout-3: 성능 열세 또는 timeout 이력.
3. Reader-packed weight layout: sustained 이득 없음.
4. K-block merge-2: mean +1.324% 악화.
5. Tagged depth-3: mean +0.224% 악화.
6. Activation credit depth-3: mean +1.921% 악화.
7. Fused gate/up 뒤 DRAM SwiGLU: mean +1.88% 악화.

## Core-local SwiGLU 정확도 한계

Isolated PCC는 baseline 0.9996410623, local SwiGLU 0.9996954416이다. 그러나 full-model fixed-input logits는
bitwise equal이 아니며 98,518/128,256 elements가 다르다. Max/mean absolute difference는
0.3125/0.044205, PCC/cosine은 0.999288/0.999414다. Top-1은 양쪽 모두 320이고 top-5 overlap은 5/5다.
Accumulation grouping과 연산 순서 차이 후보가 있다. 실제 KV/token accuracy suite 전에는 exact 또는
accuracy-preserving이라고 부르지 않는다.

## Matmul epilogue 최신 시도

목표는 W3/W1 projection, W1 SiLU, W3 multiply를 같은 matmul compute kernel에서 끝내는 것이다.
Gate/up activation을 별도 L1 op로 왕복하지 않고 core-local `c7`에 W3 partial을 두었다가 W1 결과와
결합해 `c4` output으로 publish한다.

### 교정된 항목

1. Logical fused width 16,384와 physical padded width 16,896을 분리했다.
2. Factory가 logical `N=512 tiles` 대신 DRAM shard physical `N=528 tiles`를 사용하게 했다.
3. 6 DRAM shards × 88 tiles, 12 compute readers × 44 raw tiles, worker당 22 output tiles로 맞췄다.
4. `silu_tile_init`/`silu_tile` JIT 선언 누락에 `compute_kernel_api.h`를 추가했다.
5. Host build, runtime install, Python import는 성공했다.

## 2026-08-05 tagged NoC pending-gap 계측

재부팅 뒤 add gate와 profiler-free isolated MLP는 정상 통과했다. 동일 12-reader/12-compute,
fanout-2, tagged two-block, NOC1 destination 4:4:4 구성의 NoC capture는 MLP completion 뒤 profiler
후처리에서 `Invalid NoC transfer type on device: 0`으로 abort했다. 외부 timeout exit 124와 D-state
child가 발생했으므로 raw NoC capture는 실패이며 장치를 격리했다.

불완전한 device CSV에서 `MLP_IN1_ISSUED`→`MLP_IN1_DRAM_DONE` 구간만 추출했다. 이는 실제 link
utilization이 아닌 outstanding DRAM read 근사치다.

| projection | 평균 pending | 최대 pending | pending ≥12 시간 | 내부 pending-empty 시간 |
|---|---:|---:|---:|---:|
| W1 | 18.43 | 24 | 94.56% | 0% |
| W3 | 18.52 | 24 | 95.07% | 0% |
| W2 | 16.37 | 22 | 94.27% | 0% |

Projection 경계 gap은 W1→W3 15.185 µs, W3→W2 70.122 µs다. 합계 85.307 µs는 measured
1.507990 ms의 5.66%다. 따라서 약 67% consumer input wait는 global NoC-empty 시간이 아니다.
현재 reader는 projection 내부에서 이미 계속 요청을 outstanding으로 유지한다.

현재 fused gate/up output은 16번째 K block의 `last_out`에서만 publish된다. 출력도 12 workers ×
22 tiles의 physical 264-tile 배치이고, W2 input은 16×16의 256-tile 배치다. 현행 4.042 µs L1
reshard가 이 재배치를 담당한다. 12 producer→8 W2 consumer pipeline에는 22→16 reblocking과 W2
activation K-block multicast가 새로 필요하며 W2 reader 수는 12에서 8로 감소한다.

결론: pipeline은 FlashAttention처럼 DRAM byte를 크게 제거하지 않는다. phase gap 일부를 숨기는
상한은 약 5.7%이며 실제 이득은 그보다 작을 가능성이 높다. 우선순위는 pipeline 구현보다 DRAM
request service latency, endpoint별 queueing, W2 phase 시작 gap의 원인 분해다.

## 2026-08-05 DRAM admission depth-1 및 6-compute 대조

Tagged fanout-2의 reader당 pending depth를 2에서 1로 줄이는 opt-in을 구현했다. 최대 outstanding
block은 24에서 12로 감소한다. 동일 isolated MLP 20회에서 depth-2 mean/median은
1.467559/1.468924 ms, depth-1은 1.493915/1.495990 ms였다.

| variant | readers/compute | mean ms | depth-2 대비 |
|---|---:|---:|---:|
| tagged depth-2 | 12/12 | 1.467559 | 기준 |
| tagged depth-1 | 12/12 | 1.493915 | +1.796% |
| fanout off | 6/6 | 측정 불가 | exit 137 |

Depth-1 제한은 overlap을 줄여 성능을 악화시켰다. 12-reader request count 자체가 주 병목이라는 가설은
지지되지 않는다. 6-compute 20회는 90초 timeout과 exit 137로 실패해 장치를 격리했다. 현행 선택은
12-compute tagged depth-2 유지다. 다음 후보는 global depth 감소가 아니라 endpoint별 작은 phase
stagger이며, 재부팅 뒤 1 measured call부터 검증해야 한다.

## 2026-08-05 endpoint lane start stagger 결과

재부팅과 add gate 뒤 fanout-2 lane 1의 최초 request만 256 cycles 지연했다. 12 readers/compute,
tagged depth-2, NOC1 destination 4:4:4는 바꾸지 않았다.

W2 block16과 16 KiB cap을 log로 확인한 20회 A/B에서 stagger/no-stagger mean은
1.473408/1.440942 ms, median은 1.474435/1.436232 ms였다. Stagger가 mean 2.253%, median 2.660%
느리다. PCC는 양쪽 모두 0.9996410623이고 정상 close/exit 0이다.

### 관측 사실

- Reader 수 12→6은 이전 run에서 completion 없이 exit 137이었다.
- Pending depth 2→1은 mean 1.796% 악화했다.
- Lane-1 start stagger 256 cycles는 mean 2.253% 악화했다.

### 결론

Request 개수나 endpoint 동시 시작을 줄이는 admission 제어는 현재 병목을 개선하지 않는다. Reader가
projection 내부에서 pending request를 유지한다는 기존 marker 관측과도 일치한다. 기본 구성은
12-compute tagged depth-2, no stagger다. 다음 조사는 reader admission보다 compute/consumer가 이미
도착한 block을 소비하는 cadence와 projection 경계 W3→W2 70.122 µs gap에 집중한다.

## 2026-08-05 physical endpoint-local fanout-2

Microbenchmark의 native six-endpoint mapping을 12-compute MLP opt-in으로 이식했다. Physical endpoint
x=0..5에 reader 두 개씩 배치하고 x={0,4,5}는 NOC0, x={1,2,3}은 NOC1을 사용했다. Host log에서
endpoint `2:2:2:2:2:2`, reader NoC `6:6`, compute 12, conflict-free VC를 확인했다.

Profiler/Watcher 없는 첫 correctness 1회는 W1/W3와 W2 mapping log 뒤 completion 없이 정지했다.
90초 SIGINT와 15초 cleanup 상한 뒤 exit 137이었다. PCC와 device close는 없다. 장치를 격리했다.

따라서 physical proximity 자체의 성능은 측정하지 못했다. 실패 signature는 기존 fanout-3 explicit
six-endpoint split-kernel 경로와 같다. Endpoint별 reader 수와 VC를 교정해도 정지했으므로 다음 원인
후보는 split NOC0/NOC1 reader kernel의 CB producer/consumer 또는 output writer 계약이다. 이 opt-in은
재실행하지 않는다. 기본 선택은 NOC1 4:4:4, 12-compute tagged depth-2다.

## 2026-08-05 endpoint-local single-kernel 반증 실험

Split RISCV0 kernel handle을 제거하고 단일 RISCV0 kernel에서 endpoint x로 reader NoC을 runtime
선택했다. Dynamic endpoint address, `noc_async_read`, TRID 설정과 barrier 모두 같은 runtime NoC을
사용하고 output writer는 반대 NoC을 사용했다. Host build와 runtime install은 성공했다.

재부팅 뒤 add gate를 통과한 상태에서 profiler/Watcher 없는 isolated MLP correctness 1회를 실행했다.
W1/W3와 W2 모두 endpoint `2:2:2:2:2:2`, reader NoC `6:6`, compute 12, single-kernel log를 확인했다.
결과는 PCC/completion/close 없는 timeout, exit 137이다. 장치는 다시 격리했다.

### 해석

- split NOC0/NOC1 kernel handle은 hang의 필요조건이 아니다.
- Reader 수 12, fanout-2 tagged depth-2, VC coloring만으로도 설명되지 않는다.
- 실패 경로의 남은 고유 요소는 physical explicit-endpoint address와 runtime NoC 전환, 반대-NoC writer다.
- 정확한 wait 지점은 미측정이다. Writer barrier, output reshard destination 및 CB ownership 계약을
  baseline NOC1 4:4:4와 host-side로 비교해야 한다.

성능은 측정하지 못했다. 안전한 기본 선택은 계속 NOC1 4:4:4, 12-compute tagged depth-2다.
Endpoint-local flag는 다음 재부팅 뒤에도 계약 수정과 별도 correctness 계획 전에는 실행하지 않는다.

## 2026-08-05 microbenchmark 대비 구조 제약

두 번째 재부팅 뒤 add gate는 정상 통과했다. 이후 device workload 없이 source 계약만 비교했다.

### 확인된 사실

- 정상 six-endpoint microbenchmark는 NOC0 reader를 RISCV0 기본 NoC, NOC1 reader를 RISCV1 기본 NoC에
  배치한다. 두 processor의 core set은 분리된다.
- MLP matmul은 RISCV1이 in0 activation multicast를 맡고 RISCV0이 weight reader와 output writer를
  함께 맡는다.
- MLP에서 같은 RISCV0의 core set을 NOC0/NOC1 kernel handle로 나눈 경로와 단일 RISCV0에서 runtime
  NoC을 바꾼 경로가 모두 completion 없이 멈췄다.
- Explicit endpoint의 `view_for_endpoint_x`와 address 식은 microbenchmark와 같다.
- Writer runtime-arg index와 output reshard vector layout은 baseline과 같다.
- 선택된 x=4 worker를 포함하도록 program bounding box와 CB 범위가 확장된다.

### 결론

현재 증거는 DRAM view 주소, x=4 CB 누락, split handle 하나보다 data-movement processor와 NoC 역할
결합을 가리킨다. Microbenchmark의 dual-NoC 방식을 fused matmul의 RISCV0 weight-reader에 그대로
이식할 수 없다고 보는 것이 안전하다.

현행 선택은 NOC1 4:4:4, 12-compute tagged depth-2다. Six endpoint를 다시 쓰려면 RISCV1 in0 역할을
재배치하거나 dedicated producer core가 읽고 compute core에 전달하는 새 handshake가 필요하다. 이는
단순 reader placement가 아니라 dataflow 재설계로 분류한다.

## 2026-08-05 fixed writer와 ordering 분해 결과

Endpoint-local writer를 GEMM처럼 고정 kernel NoC으로 바꾸자 동일 explicit reader 구성이 device
completion과 close까지 진행했다. PCC는 `0.028334455265917317`로 실패했다.

### 관측

- Opposite-NoC writer 제거 전: completion 없는 exit 137.
- Opposite-NoC writer 제거 후: clean close, exit 1, PCC 실패.
- Reader 배열은 endpoint x 순서라 DRAM view가 `0,0,3,3,4,4,5,5,1,1,2,2`였다.
- 이를 `0,0,1,1,...,5,5`로 단순 재정렬하자 다시 completion 없는 exit 137이 발생했다.

### 결론

Opposite-NoC output reshard writer는 hang 원인으로 확인됐다. 그러나 fixed writer만으로 correctness는
복구되지 않는다. `compute_worker_cores_ordered`는 다음 세 의미를 동시에 가진다.

1. Physical compute/reader placement
2. Weight shard와 output-column partition 순서
3. Output reshard destination/ownership 진행 순서

따라서 endpoint locality를 위해 이 배열을 재정렬하면 weight/output correctness 또는 reshard 진행
계약 중 하나가 깨진다. 다음 설계는 physical core order를 유지한 채 별도의 logical partition index와
writer ownership permutation을 전달해야 한다. 현재 장치는 exit 137로 다시 격리 상태다.

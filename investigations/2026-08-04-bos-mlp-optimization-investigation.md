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

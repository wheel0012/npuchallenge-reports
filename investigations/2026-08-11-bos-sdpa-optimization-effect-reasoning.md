# BOS SDPA optimization effect reasoning

날짜: 2026-08-11

## 결론

BOS 64K decode의 SDPA 개선은 두 축이다.

1. `dual-NoC + 6 endpoints`: 같은 K/V를 더 병렬로 전달해 reader critical path를 줄인다.
2. `K chunk 128 → 256`: 같은 K/V byte를 더 큰 algorithmic iteration으로 처리해 online-softmax와
   chunk-level 호출 횟수를 줄인다.

둘은 독립 합산되지 않는다. Reader가 빨라지면 compute/softmax 비중이 커지고, chunk가 커지면 긴 CB wait와
큰 QK/PV 구간이 생긴다. 전체 시간은 reader와 compute의 합이 아니라 느린 경로와 reducer tail이 정한다.

Post-SDPA concat, QKV projection, Wo projection은 attention sublayer 최적화지만 SDPA kernel 내부 최적화는
아니다. 발표와 성능 귀속에서 분리한다.

## 범위와 하드웨어

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 cores
- 64K SDPA active readers/compute: 16; 8 KV heads, 2 cores/head
- DRAM: 3 physical banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- stable endpoint loads: `3/2/3/3/3/2`
- stable NoC reader loads: `8/8`
- KV: paged, DRAM interleaved, BFLOAT8_B

5x4 program grid를 20 active SDPA cores로 표현하지 않는다. 이 구성의 실제 active core는 16개다.

## Critical-path model

```text
reader:   K read ───────── V read ───────── next K/V
                    ╲             ╲
compute:             QK → local softmax → PV → online merge
                                                    ╲
pair tail:                              worker/reducer merge
                                                    ╲
post-SDPA:                         output layout → concat → Wo
```

근사식:

```text
T_SDPA_kernel ≈ max(T_reader_delivery, T_compute_pipeline)
                + T_slowest_head_tail
                + T_cross_core_reduce

T_attention ≈ T_QKV + T_SDPA_kernel + T_post_SDPA_layout + T_Wo + gaps
```

Reader zone과 compute zone은 겹친다. 둘을 더하면 안 된다. 변화 전후의 device kernel span, slowest core,
head-pair completion tail을 우선 사용한다.

## 효과별 reasoning

### 1. Dual-NoC와 6-endpoint reader

Vanilla K128은 약 3.47 ms, effective K/V 41.12 GB/s였다. Dual-NoC 6-endpoint K128은 2.519 ms,
56.61 GB/s였다. 같은 실제 SDPA에서 effective bandwidth는 37.7% 증가했다.

직접 바뀐 것:

- 16 readers를 세 destination이 아니라 6 worker endpoints에 분산
- reader NoC를 한쪽에 몰지 않고 `8/8`로 분할
- 한 physical bank의 두 worker endpoint를 모두 사용

강한 판정: endpoint service parallelism과 NoC 경로 분산은 유효하다.

제한: `dual-NoC`, endpoint 수, assignment가 함께 바뀌었다. Dual-NoC 단독 효과로 37.7%를 말하면 안 된다.
보존된 3-endpoint dual-NoC K128은 약 3.129 ms, 45.57 GB/s지만 단일 profiler row 재구성값이다. 따라서
다음처럼 말한다.

> We distributed 16 SDPA readers across both NoCs and all six worker endpoints, increasing effective K/V
> bandwidth from 41.12 to 56.61 GB/s at K128.

### 2. K chunk 128 → 256

K/V payload는 변하지 않는다. 줄어드는 것은 iteration 수다.

| 항목/core | K128 | K256 |
|---|---:|---:|
| chunks | 256 | 128 |
| QK/PV calls | 256/256 | 128/128 |
| current-softmax updates | 256 | 128 |
| online merges | 255 | 127 |

Vanilla phase 계측의 empty-overhead 보정 결과:

| phase | K256 - K128 |
|---|---:|
| QK | +366.986 us |
| PV | +122.057 us |
| current softmax | -219.566 us |
| online merge | -419.653 us |
| net | **-150.176 us** |

독립 vanilla kernel 감소 151.937 us의 98.84%를 설명한다. 핵심은 DRAM byte 감소가 아니라
online-softmax state update 감소다. 큰 QK/PV 구간은 489.043 us 비싸져 절약 대부분을 상쇄한다.

6-endpoint 환경에서는 2.51933 → 2.03641 ms, -19.17%였다. Vanilla의 -4.36%보다 크다. 이는 두 실험의
context가 달라 K256 효과가 reader service와 상호작용함을 뜻한다. 가능한 설명은 빨라진 reader가
chunk-level compute/softmax overhead를 더 크게 노출하고, 34 KiB K/V phase가 더 긴 useful burst를 만든다는
것이다. 이 상호작용은 아직 직교 2x2 A/B로 분리하지 않았다.

발표 문장:

> We doubled the K-chunk from 128 to 256 tokens, halving chunk-level online-softmax updates while keeping
> total K/V traffic unchanged.

`halving max/softmax arithmetic`라고 말하면 틀린다. Token 전체에 대한 max/exp/sum 산술은 남는다.

### 3. K512 plateau

6-endpoint에서 K512는 2.00242 ms, 71.22 GB/s다. K256보다 1.67%만 개선됐다. K256에서 iteration 고정비
대부분을 제거했고, 이후에는 큰 CB reservation, QK/PV shape cost, reader/head tail이 지배한다.

K512를 기본값으로 승격할 근거는 약하다. K256이 성능과 자원 위험의 sweet spot이다.

### 4. Tagged prefetch와 barrier

실제 SDPA의 cross-chunk tagged prefetch는 56.500 → 56.374 GB/s로 중립이었다. Synthetic tile packet도
tagged와 full barrier 차이가 약 1.4%였다. Full barrier 자체는 큰 미회수 성능의 주원인이 아니다.

따라서 `double buffering이 없어서 느렸다`고 발표하지 않는다. 현재 CB/compute overlap이 reader gap을 이미
가리거나, tile별 paged/interleaved address와 service latency가 병목이다.

### 5. Endpoint balance와 route overlap

K256 endpoint balance 후보는 기존 assignment보다 0.18~0.38% 느렸다. Directed-link overlap 최소화도
SDPA wall을 0.362% 악화시켰다. Static reader count, Manhattan distance, shared-edge count는 현재 critical
head를 예측하지 못한다.

Stable `3/2/3/3/3/2`, `8/8`은 검증된 동작 구성이다. 그러나 pair/bank balance 자체를 독립 성능 개선으로
주장하지 않는다.

### 6. Grouped concat

Vanilla post-SDPA 경로:

```text
SDPA DRAM output → L1 shard → DRAM interleaved → ROW_MAJOR → reshape → TILE → Wo
```

Exact grouped 경로:

```text
SDPA DRAM output → height-sharded L1 → 12-core, 2-heads/core concat
→ DRAM interleaved TILE → baseline-compatible Wo
```

동일-process A/B에서 layer mean은 4.469103 → 4.352791 ms, -2.60%다. SDPA wall은 -0.03%로 같다.
따라서 개선은 KV reader나 flash-decode compute가 아니라 post-SDPA layout movement 제거다.

완전한 L1→Wo fusion은 아니다. Bit-exact를 위해 DRAM interleaved TILE bridge 하나를 유지한다. L1-sharded
Wo/gather prototype은 bridge를 제거했지만 full-model bit exact를 잃었다.

### 7. QKV와 Wo projection

DRAM-sharded balanced fanout-2를 적용하면 QKV kernel은 433.448 → 278.943 us, -35.64%; Wo는
236.042 → 165.835 us, -29.74%였다. Layer makespan은 -1.41%, full-model throughput은 +5.67%였다.

이 변화는 weight reader와 matmul partition 최적화다. SDPA K/V streaming 효과에 합치지 않는다. 또한
full-model PCC 0.9992725, top-1/top-5 동일이나 bitwise exact는 아니다.

## 증거 강도

| 주장 | 강도 | 이유 |
|---|---|---|
| K256이 online-merge/update 비용을 줄임 | 높음 | phase delta가 kernel delta의 98.84% 설명 |
| dual-NoC+6-endpoint bundle이 reader path 개선 | 높음 | 실제 SDPA latency/BW A/B |
| dual-NoC 단독 기여 | 낮음 | 동등 반복 A/B 없음 |
| pair/bank balance가 성능 개선 | 반증 | 0.18~0.38% 악화 |
| route-overlap 최소화가 성능 개선 | 반증 | 0.362% 악화 |
| tagged prefetch가 성능 개선 | 반증 | 실제 SDPA 중립 |
| grouped concat가 post-SDPA movement 개선 | 높음 | SDPA wall 동일, layer -2.60% |
| QKV/Wo sharding이 projection 개선 | 높음 | 개별 kernel A/B; bit exact 아님 |

## End-to-end 수치 해석

Waterfall의 vanilla K128 → `K256 + dual-NoC + 6 endpoints` 단계는 5.124290 → 6.414458 tok/s,
+25.18%다. 이 값은 bundle의 full-model incremental 효과다. K256 phase 효과와 endpoint 효과를 곱하거나
더해 25.18%를 재구성하지 않는다.

현재 stable layer-0 profile에서 SDPA decode는 2,046.243 us, layer device-duration 합의 43.89%다. Attention
sublayer 전체는 2,883.419 us다. 남은 최적화는 SDPA 내부와 projection/layout을 분리해 평가한다.

## 다음 reasoning 실험

1. 동일 runner에서 `K128/K256 × vanilla-reader/6-endpoint-reader` 2x2, 각 5회. K와 endpoint interaction 계산.
2. K128 고정, `single-NoC 3EP → dual-NoC 3EP → dual-NoC 6EP` 반복 A/B. NoC와 endpoint 기여 분리.
3. 6EP K256에서 per-head reader completion, QK/PV, reducer wait를 같은 core ID로 결합. Slowest-head 원인 확인.
4. Exact grouped concat OFF/ON을 layer makespan과 DRAM bytes로 재측정. 발표용 data-movement waterfall 생성.

## 발표 구조

1. Algorithmic granularity: K128→K256, merge/update 감소.
2. Memory service parallelism: 16 readers를 two NoCs와 six endpoints로 분산.
3. Post-attention movement: generic concat/layout pipeline 축소.
4. Projection streaming: QKV/Wo DRAM-sharded reader.
5. Negative controls: tagged prefetch, static balancing, route-overlap 최적화는 효과 없음.

핵심 표현:

> The speedup came from reducing chunk-level online-softmax work and improving K/V service parallelism,
> not from reducing total K/V bytes.

## 근거 문서

- `benchmark-results/2026-08-10-bos-sdpa-kchunk-fixed-cost-decomposition.md`
- `investigations/2026-08-01-sdpa-dram-performance-optimization-history.md`
- `investigations/2026-08-01-vanilla-sdpa-vs-dram-saturation-gap.md`
- `benchmark-results/2026-08-05-bos-sdpa-grouped-concat-exact-wo-input-ab.md`
- `benchmark-results/2026-08-09-bos-attention-qkv-wo-dram-sharded-ab.md`
- `benchmark-results/2026-08-09-bos-sdpa-route-overlap-ab.md`
- `benchmark-results/2026-08-09-bos-llama32-3b-64k-optimization-waterfall.md`

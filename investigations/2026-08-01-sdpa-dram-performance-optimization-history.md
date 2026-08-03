# BOS 64K SDPA 성능 최적화 이력: vanilla에서 six-reader POC까지

- 작성일: 2026-08-01
- 대상: Blackhole/P100, Llama 3.2 3B decode SDPA
- 주 workload: batch 1, context 65,536, Q heads 24, KV heads 8, head dim 128
- 데이터 형식: BF16 Q, BF8 K/V, paged interleaved DRAM KV cache
- 프로그램 grid: 5×4
- 상태: 측정 결과와 현재 소스를 바탕으로 한 회고

## 1. 결론

이번 최적화에서 실제 SDPA 성능을 가장 크게 올린 변경은 두 가지다.

1. reader를 NOC0/NOC1과 6개 DRAM endpoint에 분산한 것
2. K/V chunk를 128 token에서 256 token으로 키운 것

비교 가능한 effective K/V bandwidth 기준으로 vanilla의 약 41.12 GB/s에서
6-endpoint, K-chunk 256 구성의 약 70.03 GB/s까지 올라갔다. 이는 약 70.3% 향상이며,
kernel critical-path latency는 대략 3.47 ms 수준에서 2.04 ms 수준으로 줄었다.

512-token chunk는 71.22 GB/s로 최고 수치였지만 256 대비 추가 이득이 1.7%뿐이다.
따라서 현재까지의 실용적인 후보는 6 endpoint, dual NoC, K-chunk 256이다. 다만 이 결과는
cur-pos-only 단일-layer 실험이며 production 기본값으로 바꾸기 전에 실제 page table과 모델
정확도 검증이 더 필요하다.

반대로 다음 변경은 성능 개선으로 확인되지 않았다.

- tagged cross-chunk K/V prefetch: 측정 오차 범위에서 중립
- reducer/worker pair 단위 NoC balancing: critical head를 줄이지 못해 소폭 악화
- 명목상 5/5/6 bank balancing: 소폭 악화
- paging 제거: 약 0.4% 개선에 그침
- 일반 DRAM-sharded tensor로의 교체: 큰 폭으로 악화
- 6 owner가 전체 block을 fanout: relay와 ACK 비용으로 악화

6개 endpoint-local reader가 서로 다른 sequence shard를 읽는 microbenchmark는
약 77.37 GB/s를 보여 유망했다. 그러나 실제 SDPA 통합은 유효한 성능 결과를 얻지 못했다.
마지막 통합 시도 당시 발생한 서버 freeze는 kernel 실행 이후의 deadlock이 아니라 별도의
worker-FW 장애와 직접 PCIe-link reset 뒤에 발생했으므로, six-reader 알고리즘의 성능이나
정확성을 입증하거나 반증하는 자료로 사용하면 안 된다.

## 2. 측정값을 읽는 방법

### 2.1 실제 활성 코어는 20개가 아니라 16개

5×4는 사용 가능한 program grid다. 이 shape에서 batch가 1이고 KV head가 8개이며
`max_cores_per_head_batch=16`이므로 각 KV head에 reducer 1개와 worker 1개, 총 2개 core가
배정된다. 따라서 실제 SDPA active core는 `8 × 2 = 16`개다. 나머지 4개 grid 위치는 이
SDPA 호출의 compute-active core가 아니다.

20-reader saturation microbenchmark의 20개 reader와 실제 SDPA의 16 active core를 같은
것으로 해석하면 안 된다.

### 2.2 endpoint와 physical DRAM bank

BOS에서는 3개의 physical DRAM bank에 대해 worker가 접근할 수 있는 endpoint가 두 개씩
노출되어 총 6 endpoint를 사용한다. 이 보고서의 “6 endpoint”는 DRAM bank가 6개라는 뜻이
아니다. endpoint 쌍은 `{x0,x1}`, `{x2,x5}`, `{x3,x4}`이며 각각 하나의 physical bank에
대응한다.

### 2.3 bandwidth 정의

주 비교값은 64K에서 SDPA가 논리적으로 읽는 K/V payload를 최대 core kernel span으로 나눈
effective K/V bandwidth다. NPE의 `DRAM BW UTIL (%)`, 실제 controller counter, 그리고
saturation microbenchmark의 raw DRAM throughput은 byte accounting이 다르므로 직접 같은
수치로 취급하지 않는다.

초기 vanilla NPE 문서의 38.72 GB/s는 다른 논리 traffic 정의와 3.466786 ms device duration을
사용했다. 이후 A/B 표의 41.12 GB/s는 padded/effective K/V 정의를 통일한 값이다. 이 보고서의
상대 비교는 후자인 41.12 GB/s를 vanilla 기준으로 사용한다.

## 3. 단계별 변화

| 단계 | 구성 | 대표 latency/span | Effective K/V 또는 DRAM BW | vanilla 대비 | 판정 |
|---|---|---:|---:|---:|---|
| 0 | Vanilla paged SDPA, K=128 | 약 3.47 ms | 41.12 GB/s | 기준 | 기준점 |
| 1 | Dual NoC, 3 endpoint, K=128 | 3.129 ms | 약 45.57 GB/s | 약 +10.8% | 개선, 중간 단계 |
| 2 | Dual NoC, 6 endpoint, K=128 | 2.519 ms | 56.61 GB/s | 약 +37.7% | 유효 |
| 3 | 6 endpoint + cross-chunk tagged prefetch | 2.522 ms | 56.37 GB/s | 약 +37.1% | 중립/미채택 |
| 4 | 6 endpoint + K=256 | 2.036 ms | 70.03 GB/s | 약 +70.3% | 가장 실용적인 후보 |
| 5 | 6 endpoint + K=512 | 2.002 ms | 71.22 GB/s | 약 +73.2% | 최고 수치, 한계효용 작음 |
| 6 | K=256 + pair-balanced endpoints | 2.037 ms | 70.00 GB/s | 약 +70.2% | baseline보다 0.38% 느림 |
| 7 | K=256 + nominal 5/5/6 bank balance | 2.033 ms | 70.14 GB/s | 약 +70.6% | baseline보다 0.18% 느림 |

3-endpoint bandwidth는 보존된 profiler row의 3.129072 ms max-core duration에 이후 A/B와 같은
effective K/V byte 수를 적용해 산출한 약 45.57 GB/s다. 단일 row에서 재구성한 값이므로
5-run 평균과 같은 강도로 해석하지 않는다. 5-endpoint 실제-prefill 성공 run도 보존되어 있지만
SDPA NPE bandwidth 열이 비어 있어 이 표의 동등 비교에서는 제외했다.

### 3.1 Vanilla 기준점

Vanilla reader는 paged/interleaved K/V를 tile 단위로 읽는다. 128-token chunk마다 K를 읽고
`noc_async_read_barrier()`로 완전히 drain한 뒤 K CB를 publish하며, V도 같은 순서를 반복한다.
reader는 `TensorAccessor`가 계산한 interleaved 주소를 따라 endpoint를 바꾸어 방문한다.

관측된 특징은 다음과 같다.

- real SDPA effective K/V bandwidth: 41.12 GB/s
- synthetic three-endpoint saturation reference: 66.47 GB/s
- 20-reader, dual-NoC, six-endpoint saturation reference: 86.83 GB/s
- physical x별 reader span과 barrier 비중에 큰 차이
- 15개 완전한 reader trace에서 x0 평균 span 2.226M cycles, x4 1.553M cycles
- x0 barrier/span 47.8%, x4 14.7%
- tt-npe modeled congestion impact는 0%였으므로 aggregate NoC link saturation만으로 설명되지 않음

즉 vanilla의 문제는 단순히 “특정 bank에 reader 7개가 몰린다”가 아니다. 과거에 사용한
`3/7/6`은 synthetic benchmark에서 고정한 reader 수이며, vanilla의 실제 interleaved K/V byte
분포가 아니다. 실제 병목 후보는 작은 tile packet, 주소 전환, chunk 반복 오버헤드,
reader/compute CB 동기화, head별 tail imbalance가 합쳐진 것이다.

### 3.2 Dual NoC와 endpoint 확장

host program factory에 opt-in BOS endpoint assignment를 추가했다. 각 active reader에 대해
physical core와 endpoint 사이의 route cost를 계산하고, endpoint load 및 NOC0/NOC1 load를
제약으로 두어 assignment를 찾는다. output writer와 reader가 같은 RISC/NoC 자원을 충돌시키지
않도록 output core에는 큰 swap cost도 둔다.

관련 토글은 다음과 같다.

```text
TT_METAL_SDPA_DECODE_DUAL_NOC=1
TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=3|5|6
```

3 endpoint 단계는 vanilla보다 나아졌지만, 6 endpoint로 확장하면서 K=128 기준 약
56.6 GB/s에 도달했다. 이는 endpoint service와 route 분산이 실제로 유효했음을 보여준다.
다만 saturation microbenchmark의 86.83 GB/s에는 도달하지 못했으므로 endpoint 수만으로
병목이 모두 해소되지는 않았다.

### 3.3 Tagged async와 double buffering

saturation kernel은 두 개의 L1 slot과 transaction ID를 이용해 이전 block이 끝나기 전에
다음 read를 발행한다. 이를 실제 SDPA에도 적용하기 위해 기존 double-buffered K/V CB를
cross-chunk pipeline으로 사용했다.

실험 경로에서는 `K_i` publish 후 compute가 QK를 시작하는 동안 transaction ID 2로 `V_i`를
읽고, transaction ID 1로 `K_(i+1)`을 prefetch한다. 지원하지 않는 shape는 vanilla full
barrier 경로로 fallback한다.

```text
TT_METAL_SDPA_DECODE_TAGGED_ASYNC=1
```

결과는 full barrier 56.500 GB/s, cross-chunk prefetch 56.374 GB/s였다. kernel duration 변화는
+0.032%, FW duration 변화는 +0.223%로 사실상 중립이다.

synthetic full-barrier A/B에서도 4 KiB packet은 차이가 0.1% 수준이었고, SDPA와 유사한
1,088 B × 16 packet에서는 tagged가 53.60 GB/s, full barrier가 52.87 GB/s로 차이가 약
1.4%였다. 반면 4 KiB packet에서 1,088 B packet으로 작아지는 비용은 약 19.4%였다.

따라서 full barrier 자체가 vanilla 격차의 주원인은 아니다. real SDPA에서는 compute와
CB 소비가 reader 공백의 상당 부분을 이미 가리거나, 병목이 K/V 경계가 아니라 tile별
interleaved/paged 접근에 존재한다.

### 3.4 K/V chunk 확대

K/V chunk를 128에서 256과 512로 키워 chunk당 반복되는 compute/softmax setup과
synchronization 횟수를 줄였다.

| K chunk | Mean max-core span | Effective K/V BW | 128 대비 latency |
|---:|---:|---:|---:|
| 128 | 2.51933 ms | 56.605 GB/s | 기준 |
| 256 | 2.03641 ms | 70.028 GB/s | -19.17% |
| 512 | 2.00242 ms | 71.217 GB/s | -20.52% |

256과 512의 full-layer 출력은 이 cur-pos-only 입력에서 128 출력과 bitwise identical였다.
128→256은 bandwidth를 약 23.7% 올렸지만 256→512는 약 1.7%만 올렸다. 이는 256에서 이미
작은 chunk의 반복 비용 대부분을 제거했고, 이후에는 reader/worker imbalance와 per-head
reduction tail이 지배하기 시작했음을 뜻한다.

### 3.5 Compute pipeline과 tail 분석

K=256에서 최대 span은 BRISC 2.0375 ms, NCRISC 1.9990 ms, TRISC 2.0357 ms로 서로 비슷했다.
즉 matrix engine만 계속 바쁘고 data movement만 늦는 단일 병목은 아니었다. 8개
reducer/worker pair의 종료 시점은 약 1.716–2.036 ms로 약 18.6% 차이가 났다.

일부 reducer는 local K/V 계산을 먼저 끝내도 paired worker와 cross-core softmax merge를
기다렸다. `cb_wait_front` 누적 probe는 core당 0.369–0.619 ms를 보였지만 compute pipeline의
여러 CB wait가 합쳐진 값이므로 순수 DRAM stall로 해석하지 않았다. 반복 scope를 넣은 더
세밀한 probe는 계측 간섭으로 kernel을 약 2.04 ms에서 3.27 ms로 늘려 절대 시간 분석에서
폐기했다.

### 3.6 Endpoint balancing 재시도

K=256에서 두 가지 정적 assignment를 추가했다.

- `TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1`: reducer/worker가 반대 NoC를 사용
- `TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1`: endpoint load를 `3/2/3/3/3/2`로 만들어
  nominal physical-bank reader split을 5/5/6으로 조정

pair-balanced는 head 종료 spread를 약 305 µs에서 136 µs로 줄였지만 빠른 head를 늦춘
결과였고 critical head는 빨라지지 않았다. bank-balanced는 route cost를 26에서 21로 줄였지만
critical path를 개선하지 못했다. 둘 다 bitwise correctness는 유지했지만 기본값으로
채택하지 않았다.

이 결과는 endpoint별 reader 개수나 pair-level NoC 대칭성이 paged/interleaved traffic의 실제
tail을 충분히 모델링하지 못한다는 뜻이다.

### 3.7 Paging과 DRAM sharding A/B

K=256에서 paging 자체의 비용과 memory layout을 분리했다.

| Active cores | KV layout | Max-core span | Effective K/V BW |
|---:|---|---:|---:|
| 16 | paged interleaved | 2.0295 ms | 70.266 GB/s |
| 16 | contiguous interleaved | 2.0213 ms | 70.551 GB/s |
| 16 | contiguous six-way DRAM sharded | 5.8520 ms | 24.369 GB/s |
| 8 | contiguous interleaved | 3.7264 ms | 38.269 GB/s |
| 8 | contiguous six-way DRAM sharded | 5.8546 ms | 24.358 GB/s |

identity page table 조건에서 paging 제거 이득은 약 0.4%뿐이었다. 반면 generic
DRAM-sharded `TensorAccessor`는 worker ownership과 shard ownership을 맞추지 못하고 tile
단위 접근을 유지하여 크게 느려졌다. Tech Report의 높은 bandwidth는 단순히 tensor를
DRAM-sharded로 선언해서 나오는 것이 아니라, bank-local reader와 긴 연속 read, 명시적
distribution이 함께 있어야 한다.

### 3.8 Six dedicated reader microbenchmark

Tech Report 구조를 SDPA에 가까운 형태로 옮기기 위해 6개의 endpoint-local producer와
16개 logical consumer stream을 구성했다.

첫 번째 full-block fanout은 각 owner가 읽은 전체 block을 group의 remote core에 복제했다.
최적 128 KiB block에서도 실제 DRAM bandwidth는 51.070 GB/s였다. remote copy와 ACK가
producer data-movement RISC와 NoC를 먼저 소모했기 때문이다.

두 번째 disjoint sequence-shard relay는 각 DRAM block을 정확히 한 consumer stream에만
할당했다. 6개 local stream은 owner L1에 남기고 10개 remote stream만 한 번 unicast했다.

| Block size | Effective DRAM BW |
|---:|---:|
| 32 KiB | 67.818 GB/s |
| 64 KiB | 74.772 GB/s |
| 128 KiB | 77.369 GB/s |
| 256 KiB | 71.873 GB/s |

128 KiB의 별도 5-run 평균은 76.915 GB/s, 범위는 75.255–78.664 GB/s였다. 이는 K=256 실제
SDPA의 70.266 GB/s보다 약 10% 높은 headroom이다. SDPA도 두 core가 서로 다른 sequence
chunk를 처리한 뒤 `(m, l, O)` partial state를 merge하므로 방향 자체는 구조적으로 맞는다.

하지만 이 수치는 reader/relay microbenchmark 결과다. 실제 SDPA에서는 6 owner가 기존 K/V
CB를 10 receiver에 공급하고 compute/reducer kernel과 함께 동작해야 하므로 아직 동등한
성능 향상으로 간주할 수 없다.

### 3.9 실제 six-reader SDPA 통합 상태

현재 program factory에는 다음 opt-in POC 스캐폴딩이 존재한다.

```text
TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=1
```

지원 조건은 Blackhole, B=1, S=64K, K-chunk=256, paged causal decode, KV heads 8,
16 active cores, 6 endpoint이며 mask/sink/sliding-window/MLA가 없는 경우로 제한되어 있다.
6 owner와 10 receiver를 선택하고 두 GlobalCircularBuffer를 K/V input CB에 alias하는 코드,
owner/remote reader kernel 선택 및 runtime argument plumbing이 들어 있다.

그러나 유효한 end-to-end 성능 결과는 없다. 통합 조사 중 처음에는 compute CB index만
`c30/c31`에서 `c14/c15`로 바뀌고 matching host/reader 경로가 없는 불일치가 발견되어
vanilla index로 복구했다. 이후 장치가 SDPA뿐 아니라 32×32 add에서도 worker-FW init에
진입하지 못했다. Watcher에서 target core는 kernel body에 들어가지 않았으므로 이 상태에서
발생한 timeout은 reader/compute barrier deadlock의 증거가 아니다.

마지막 host freeze는 직접 `RESET_PCIE_LINK`/`RESTORE_STATE` ioctl을 실행한 직후 발생했다.
따라서 six-reader 통합은 “구현 스캐폴딩 존재, 성능·정확성 미검증” 상태로 남겨야 한다.

### 3.10 MLP DRAM/NoC 후속 실험

기존 single-layer device profile의 kernel-duration 합은 6.362796 ms였다. 같은 profile에서 MLP를
구성하는 주요 row는 다음과 같다.

| MLP 구간 | Duration |
|---|---:|
| W1 projection | 0.584982 ms |
| W3 projection | 0.585295 ms |
| SiLU × gate | 0.048245 ms |
| W2 projection | 0.719463 ms |
| 합계 | 1.937985 ms |

MLP는 vanilla layer kernel 합의 30.46%다. SDPA를 K=256 결과로 대체한 추정 layer 합
4.917634 ms에서는 MLP 비중이 39.41%까지 올라간다. 따라서 다음 값은 측정 결과가 아니라
Amdahl식 조건부 상한 추정이다.

- MLP가 1.5배 빨라지면 현재 SDPA-optimized 전체 throughput은 약 15.1% 증가한다.
- MLP가 SDPA와 같은 1.703배 빨라지면 약 19.43% 증가한다.
- 28-layer kernel-duration-only 환산 7.26 tok/s는 각각 약 8.36 tok/s와 8.67 tok/s가 된다.

기존 MLP source에는 DRAM-sharded memory config와 matmul program config가 준비돼 있었지만 실제
weight allocation과 decode matmul argument는 주석 처리되어 있었다. 따라서 기본 경로는 interleaved
weight와 generic matmul program selection을 사용했다. 이를 비교하기 위해
`TT_METAL_MLP_DRAM_SHARDED=1` opt-in을 추가했다. opt-in에서는 W1/W3와 W2를 별도
DRAM-sharded layout으로 cache하고, W1/W3에는 기존 `DECODE_MLP_W1_W3_PRG_CONFIG`, W2에는
`DECODE_MLP_W2_PRG_CONFIG`를 전달한다. 기본값은 계속 off다.

non-profiled decode correctness 결과는 다음과 같다.

| 구성 | PCC | Completion |
|---|---:|---|
| baseline interleaved | 0.9996355767898077 | pass, 정상 close |
| DRAM-sharded opt-in | 0.9996585822234915 | pass, 정상 close |

opt-in run에서 Metal은 `Dram Interface Workers: 6`을 출력했다. 이는 선택된 data path의 interface
worker 수이며, 6 physical banks나 20 active compute cores를 뜻하지 않는다. 사용한 generic helper는
4×4, 즉 16 compute-core 구성이므로 이번 실험을 BOS 전용 20-core/6-endpoint mapping이라고 주장하지
않는다. custom BOS용 route와 shard ownership은 별도로 설계·검증해야 한다.

이후 baseline interleaved full-layer NoC profile이 timeout됐고, 다음 MLP-only profile도 warmup을
완료하지 못했다. 사후 minimal add도 완료되지 않아 사용자가 `failed to initialize FW`를 확인했다.
따라서 latency, NoC traffic, DRAM bandwidth 결과는 하나도 유효하게 얻지 못했으며 이 변경은
correctness만 확인된 연구용/default-off 상태다. 상세한 process tree와 상태 전이 평가는
`/home/iris_hb4/reports/incidents/2026-08-01-bos-mlp-noc-profile-fw-init-failure.md`에 기록했다.

## 4. 현재 채택 판단

| 변경 | 상태 | 이유 |
|---|---|---|
| Dual NoC | 유지 후보 | 명확한 성능 개선 |
| 6 endpoint assignment | 유지 후보 | K=128에서 vanilla 대비 약 37.7% 향상 |
| K-chunk 256 | 우선 채택 후보 | 큰 개선과 512 대비 작은 성능 차이 |
| K-chunk 512 | 실험 옵션 | 최고 수치지만 256 대비 1.7%, 검증 부담 증가 |
| Tagged cross-chunk prefetch | 기본 off | 효과 없음 |
| Pair-balanced endpoints | 기본 off | critical path 개선 없음 |
| Bank-balanced endpoints | 기본 off | route cost만 감소, latency 개선 없음 |
| Contiguous KV | production 부적합 | paging 제거 효과가 작고 실제 paged workload와 다름 |
| Generic DRAM-sharded KV | 폐기 | 24 GB/s대로 악화 |
| Six-reader full fanout | 폐기 | relay traffic 과다 |
| Six-reader disjoint shard | 연구 계속 | microbenchmark는 +10% headroom, SDPA 통합 미검증 |
| MLP generic DRAM-sharded opt-in | 연구 전용/default off | PCC 통과, latency/NoC 미측정; BOS 20-core mapping 아님 |

## 5. 다음 단계

장치가 out-of-band 방식으로 정상 복구되고 minimal Tensix smoke test가 통과한 뒤에만 다음을
진행한다.

1. vanilla isolated SDPA를 짧은 context에서 먼저 확인한다.
2. 6 endpoint + K=256을 현재 안정 기준으로 재확인한다.
3. per-core/per-bank NoC transaction accounting으로 slow head의 실제 bank phase를 측정한다.
4. static reader-count balancing 대신 chunk phase rotation을 A/B한다.
5. six-reader disjoint-shard 통합은 한 group, 한 K chunk부터 검증하고 K/V CB handoff와
   receiver ACK를 분리 계측한다.
6. random page table, non-zero KV, full-layer 및 모델 수준 정확도 검증을 추가한다.
7. MLP는 baseline/DRAM-sharded를 NoC trace 없이 짧은 latency A/B한 뒤에만 isolated single-op NoC
   capture로 진행한다. timeout의 직접 child는 Tracy process로 두고, timeout 또는 incomplete report가
   한 번이라도 나오면 그 session에서 장치 작업을 중단한다.
8. 성능 run마다 commit, build, KMD, kernel-cache hash, 환경변수와 artifact 경로를 함께 남긴다.

PCIe link reset, driver unbind/rebind 및 직접 reset ioctl은 이 성능 작업의 일부로 다시
시도하지 않는다.

## 6. 주요 구현 및 artifact

### 구현

- Endpoint/NoC assignment와 six-reader POC:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- Validation 및 program hash 관련 BOS 토글:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp`
- Paged K/V reader와 tagged prefetch:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- 실험 runner:
  `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py`
- DRAM saturation 및 dedicated-reader microbenchmark:
  `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/`

### 대표 profiler runs

- Vanilla:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33`
- Dual-NoC 3 endpoint:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_3ep_curpos_only_npe_2026_07_26_10_39_30`
- Dual-NoC 6 endpoint:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_curpos_only_2026_07_26_11_45_00`
- Tagged prefetch ON/OFF:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_cross_chunk_2026_07_27_04_45_00`
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_cross_chunk_off_2026_07_27_04_44_00`
- K-chunk 128/256/512:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_128_profile_retry_2026_07_31`
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_device_profile_2026_07_31`
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_512_device_profile_2026_07_31`
- Endpoint balancing A/B:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_off_2026_07_31`
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_on_2026_07_31`
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_bank_balance_on_2026_07_31`

## 7. 근거 문서와 한계

주 근거는 다음 문서와 raw profiler artifact다.

- `/home/iris_hb4/reports/benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md`
- `/home/iris_hb4/reports/benchmark-results/2026-07-26-llama32-3b-64k-sdpa-dram.md`
- `/home/iris_hb4/reports/incidents/2026-07-31-blackhole-worker-fw-host-freeze.md`
- `/home/iris_hb4/reports/incidents/2026-08-01-bos-mlp-noc-profile-fw-init-failure.md`

측정은 서로 다른 날짜의 profiler build와 일부 단일-row run을 포함한다. 가장 신뢰할 수 있는
결론은 동일 binary의 연속 A/B와 반복 run에서 나온 방향성이다. 특히 3-endpoint 파생값,
cur-pos-only bitwise equality, microbenchmark의 six-reader 수치를 production 성능 또는 전체
모델 정확도로 과도하게 일반화하지 않는다.


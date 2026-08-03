# BOS 64K vanilla SDPA 실행 구조와 DRAM saturation benchmark 격차

- 작성일: 2026-08-01
- 대상 장치: Blackhole 코드/runtime을 사용하는 custom 20-core BOS NPU
- 대상 workload: Llama 3.2 3B decode SDPA, B=1, context=65,536
- 목적: 발표를 위해 vanilla 실행 상황과 이상적 DRAM 접근 패턴의 차이를 분해
- 성격: 기존 profiler, microbenchmark와 소스를 종합한 조사 보고서
- 신규 device 측정: 없음

## 1. 발표용 결론

Vanilla SDPA는 41.12 GB/s의 effective K/V bandwidth를 보였고, 20-reader/dual-NoC/6-endpoint
read-only saturation benchmark는 86.83 GB/s를 기록했다. 두 값을 그대로 비교하면 vanilla는
이상적 benchmark의 약 47.4% 수준이다.

이 격차는 하나의 원인으로 설명되지 않는다. 주요 차이는 다음 네 층에 걸쳐 있다.

1. vanilla는 16 active cores와 제한된 NoC/endpoint 경로를 사용하지만 benchmark는 20 readers,
   NOC0/NOC1 각 10 readers와 6 endpoints를 모두 사용한다.
2. benchmark는 endpoint에 고정된 4 KiB 연속 request를 사용하지만 vanilla는 1,088 B BF8 tile
   request와 paged/interleaved 주소를 사용한다.
3. benchmark는 두 L1 slot과 transaction ID로 DRAM traffic만 지속하지만 vanilla는 K/V CB,
   QK, online softmax, V accumulation과 reducer 동기화를 함께 수행한다.
4. vanilla K=128은 약 17 KiB의 K 또는 V tile 묶음마다 barrier와 chunk-level compute/setup을
   반복한다.

검증된 개선은 dual NoC/6 endpoint와 K-chunk 256이다. 이 조합은 약 70.03 GB/s로 raw saturation
benchmark의 약 80.7%에 도달했고, vanilla와 benchmark 사이의 수치상 격차 중 약 63.2%를 회수했다.
그러나 이는 saturation benchmark와 동일한 메모리 패턴이 됐다는 뜻이 아니다. packet 크기,
endpoint locality, K transpose, compute backpressure와 head tail은 여전히 다르다.

발표의 핵심 메시지는 다음과 같이 정리한다.

> Vanilla의 낮은 대역폭은 단순한 full barrier나 단일 DRAM bank 집중으로 설명되지 않는다.
> dual NoC/6 endpoint와 K=256으로 큰 고정 비용을 제거했지만, 남은 격차는 작은 tile request,
> 물리 주소/endpoint 전환, K transpose와 compute/reduction tail이 결합된 결과다.

## 2. 비교 조건

### 2.1 실제 SDPA

| 항목 | 값 |
|---|---|
| Batch | 1 |
| Context | 65,536 |
| Q heads / KV heads | 24 / 8 |
| Head dimension | 128 |
| Q / K,V format | BF16 / BF8 |
| KV layout | paged interleaved DRAM |
| Program grid | 5×4 available worker grid |
| Active compute cores | 16 |
| Vanilla K-chunk | 128 tokens |

5×4 grid는 사용 가능한 20-worker topology다. 이 SDPA shape는 KV head 8개에 reducer/worker
2 cores씩을 사용하므로 active core는 16개다. 20-reader benchmark와 동일한 core 수가 아니다.

BOS DRAM topology는 3 physical banks와 bank당 2 worker NoC endpoints, 총 6 endpoints다.
physical bank 수와 endpoint 수를 같은 개념으로 해석하지 않는다.

### 2.2 Saturation benchmark

| 항목 | 값 |
|---|---|
| Readers | 20 |
| NoC | NOC0 10 / NOC1 10 |
| Endpoints | 6 |
| Readers per endpoint | 3 또는 4 |
| Readers per physical bank | 7 / 7 / 6 |
| Request | 4 KiB |
| Requests per block | 8 |
| Pipeline block | 32 KiB |
| L1 slots | 2 |
| Reader address range | endpoint-local contiguous 2 MiB |
| Traffic per measured run | 약 0.604 GB read |

3-reader endpoint는 reader당 16 iterations, 4-reader endpoint는 reader당 12 iterations를
수행한다. 따라서 reader 수가 달라도 endpoint, NoC와 physical bank의 byte 수는 동일하다.
4개의 virtual channel도 endpoint 또는 동일 worker-row route에서 재사용되지 않도록 배정된다.

각 reader는 자기 DRAM view의 겹치지 않는 2 MiB 구간을 4 KiB 단위로 순차 읽고, iteration마다
같은 구간을 다시 읽는다. page table, K/V 교대, transpose와 compute consumer는 없다.

## 3. Vanilla SDPA의 실제 실행 흐름

### 3.1 주소 생성과 DRAM read

Vanilla reader는 logical K/V tile마다 다음 작업을 수행한다.

```text
logical sequence row와 head-dimension tile
→ page table에서 physical KV block 조회
→ page 내부 tile offset 계산
→ TensorAccessor가 DRAM view와 NoC address 계산
→ noc_async_read_tile() 발행
→ L1 K/V circular buffer에 기록
```

K/V cache가 interleaved이므로 연속 logical tile이 하나의 endpoint-local 연속 stream이 된다고
보장할 수 없다. page 경계에서는 physical block이 바뀔 수 있고, interleaved mapping에 따라
DRAM view와 endpoint도 전환될 수 있다.

현재 보존된 주 A/B는 identity page table을 사용했다. 따라서 paging을 제거한 contiguous
interleaved 결과가 0.4%만 빨랐다는 사실은 이 조건에서 software page lookup과 page 경계가
주원인이 아니었음을 보여준다. random/non-identity page table의 locality는 아직 검증하지 않았다.

### 3.2 K=128 chunk 한 번의 작업량

tile은 32×32이고 head dimension은 128이므로 head-dimension 방향은 4 tiles다.
K-chunk 128은 sequence 방향도 4 tiles이므로 K 또는 V 한 chunk는 4×4, 총 16 BF8 tiles다.

```text
K: 16 × 1,088 B ≈ 17 KiB read → barrier → K CB publish
QK + local softmax 진행
V: 16 × 1,088 B ≈ 17 KiB read → barrier → V CB publish
V accumulation과 online state update
다음 chunk
```

이 약 34 KiB는 하나의 연속 request가 아니다. endpoint가 달라질 수 있는 1,088 B tile request
32개의 K+V 합이다. K는 QK matmul이 요구하는 순서 때문에 L1 destination도 transpose된 stride를
사용한다.

### 3.3 Compute와 reduction

각 core는 할당된 sequence 영역에서 QK, max/exp/sum, V accumulation을 수행하고 chunk마다
online-softmax state `(m, l, O)`를 갱신한다. KV head당 두 core는 서로 다른 sequence shard를
처리하며 reducer가 두 partial state를 merge한다.

따라서 reader가 먼저 끝나도 compute CB 소비나 paired worker의 partial state를 기다릴 수 있다.
Vanilla은 DRAM read-only kernel이 아니라 reader, compute와 reducer의 겹친 critical path다.

## 4. 이상적 benchmark와의 단계별 격차

### 4.1 측정 사다리

| 단계 | 실행 패턴 | Bandwidth | 86.83 대비 |
|---|---|---:|---:|
| A | 20 readers, dual NoC, 6 endpoint, 4 KiB direct saturation | 86.83 GB/s | 100% |
| B | 16 readers, NOC1, fixed 3 endpoint synthetic topology | 66.47 GB/s | 76.6% |
| C | B의 topology, 1,088 B × 16, tagged | 53.60 GB/s | 61.7% |
| D | B의 topology, 1,088 B × 16, full barrier | 52.87 GB/s | 60.9% |
| E | 실제 paged/interleaved vanilla SDPA, K=128 | 41.12 GB/s | 47.4% |

이 표는 원인별 완전한 직교 A/B가 아니라 기존 측정으로 만든 gap ladder다. 특히 B의 fixed
3-endpoint assignment는 vanilla의 실제 endpoint trace가 아니라 synthetic reference다.

### 4.2 A→B: core, NoC와 endpoint 활용

20-reader/6-endpoint에서 16-reader/NOC1/3-endpoint로 제한하면 86.83→66.47 GB/s, 약 23.5%가
감소한다. 이는 양쪽 NoC와 6 endpoint를 사용하는 것이 유효함을 보여준다.

실제 SDPA에서도 dual NoC/6 endpoint를 적용했을 때 K=128이 41.12→56.61 GB/s로 약 37.7%
개선됐다. 따라서 이 부분은 benchmark에서 실제 operation으로 성공적으로 이전된 최적화다.

### 4.3 B→C: request granularity

같은 synthetic topology에서 4 KiB request를 SDPA와 유사한 1,088 B BF8 tile request로 줄이면
약 19.4%가 감소했다. 이는 protocol overhead, endpoint/request-state 전환 및 DRAM burst 효율을
포함한 transaction granularity 비용의 직접 증거다.

다만 DRAM row-buffer hit rate를 측정하지 않았으므로 이 감소를 row miss로 단정하지 않는다.

### 4.4 C→D: full barrier

Tagged 53.60 GB/s와 full barrier 52.87 GB/s의 차이는 약 1.4%다. 4 KiB packet에서는 차이가
0.1% 수준이었다. 따라서 full barrier 자체는 vanilla와 saturation 사이의 주된 격차가 아니다.

실제 SDPA의 cross-chunk tagged-prefetch도 56.500→56.374 GB/s로 중립이었다. compute와 기존
CB scheduling이 reader gap의 상당 부분을 이미 가리거나, 병목이 chunk 경계가 아닌 tile별
주소 전환에 있음을 시사한다.

### 4.5 D→E: 실제 SDPA semantics

52.87 GB/s synthetic tile-packet reference와 vanilla 41.12 GB/s 사이에는 약 22.2% 차이가 남는다.
후보는 다음과 같다.

- paged/interleaved tile별 physical address 및 endpoint 전환
- K의 strided/transpose L1 destination
- TensorAccessor와 page-table address calculation
- K/V CB reserve, publish와 compute backpressure
- QK, online softmax와 V accumulation
- paired worker/reducer partial-state merge
- physical-x별 read completion과 head별 tail imbalance

이 항목들의 개별 기여도는 아직 분리 측정되지 않았다.

## 5. Chunk 확대가 닫은 격차

K-chunk를 키우면 전체 KV byte는 같지만 barrier 사이의 read 묶음과 한 번의 online-softmax
iteration이 커진다.

| K-chunk | Tile shape | K 또는 V phase | Bandwidth | K=128 대비 latency |
|---:|---:|---:|---:|---:|
| 128 | 4×4 | 약 17 KiB | 56.605 GB/s | 기준 |
| 256 | 8×4 | 약 34 KiB | 70.028 GB/s | -19.17% |
| 512 | 16×4 | 약 68 KiB | 71.217 GB/s | -20.52% |

K=256은 K 또는 V phase의 총 in-flight payload를 saturation benchmark의 32 KiB pipeline block과
비슷한 수준으로 만든다. 동시에 barrier, CB handoff, matmul/softmax setup과 online-state merge
횟수를 절반으로 줄인다. 이것이 큰 개선의 핵심이다.

그러나 개별 request는 여전히 1,088 B이고 주소도 endpoint-local 연속 stream이 아니다.
K=256→512가 1.7%만 개선된 것은 256에서 작은 chunk의 고정 비용 대부분이 이미 제거됐고,
남은 병목이 request/address granularity와 reader/compute/reducer tail로 이동했음을 보여준다.

## 6. 현재 최적점과 남은 거리

| 구성 | Bandwidth | Raw 86.83 대비 | 상태 |
|---|---:|---:|---|
| Vanilla K=128 | 41.12 GB/s | 47.4% | 기준 |
| Dual NoC, 6 endpoint, K=128 | 56.61 GB/s | 65.2% | 검증된 개선 |
| Dual NoC, 6 endpoint, K=256 | 70.03 GB/s | 80.7% | 실용 후보 |
| Dual NoC, 6 endpoint, K=512 | 71.22 GB/s | 82.0% | 최고 수치, +1.7% 한계효용 |
| Six-reader disjoint relay microbenchmark | 77.37 GB/s | 89.1% | SDPA 통합 미검증 |

Vanilla에서 K=256까지의 증가는 28.91 GB/s다. raw benchmark와 vanilla의 차이 45.71 GB/s 중
약 63.2%를 수치상 회수했다. 남은 16.80 GB/s를 모두 실제 SDPA에서 회수할 수 있다고 기대하면
안 된다. raw benchmark에는 compute와 algorithmic synchronization이 없기 때문이다.

K=256에서 BRISC 2.0375 ms, NCRISC 1.9990 ms, TRISC 2.0357 ms로 processor span이 비슷했다.
8개 reducer/worker pair 종료 시점도 약 1.716–2.036 ms로 18.6% 퍼져 있었다. 따라서 reader
bandwidth만 개선해도 compute 또는 느린 head tail이 critical path로 남을 수 있다.

## 7. 발표에서 구분해야 할 사실과 가설

### 측정으로 확인된 사실

- dual NoC/6 endpoint는 실제 SDPA K=128을 약 37.7% 개선했다.
- K=128→256은 bandwidth를 약 23.7% 올리고 latency를 약 19.2% 줄였다.
- K=256→512의 추가 개선은 약 1.7%다.
- tagged cross-chunk prefetch와 static endpoint balancing은 critical path를 개선하지 못했다.
- identity page table에서 paging 제거 이득은 약 0.4%다.
- generic six-way DRAM-sharded TensorAccessor 경로는 약 24 GB/s대로 악화됐다.
- six-reader disjoint-shard microbenchmark는 약 77.37 GB/s를 기록했다.

### 아직 검증되지 않은 가설

- 실제 paged address sequence의 DRAM row-buffer hit rate가 낮다.
- 같은 endpoint의 연속 tile 4개를 약 4.25 KiB request로 합치면 실제 SDPA가 빨라진다.
- locality-aware page allocation이 random page table에서도 유효하다.
- 6 endpoint-local owner reader와 disjoint handoff가 실제 compute/merge와 함께 이득을 유지한다.

### 발표에서 피할 표현

- `3/7/6`이 vanilla의 실제 bank traffic 분포라는 주장
- 86.83 GB/s가 실제 SDPA가 달성 가능한 보장값이라는 주장
- full barrier가 전체 격차의 주원인이라는 주장
- generic DRAM sharding 결과가 specialized bank-local reader를 반증한다는 주장
- identity page table 결과를 random production paging으로 일반화하는 주장

## 8. 다음 분석과 실험

장치가 정상 복구되고 안전 절차를 통과한 뒤에만 device 실험을 재개한다. 그 전에는 host-side
분석만 수행한다.

1. 실제 page table과 tile loop로 `(logical tile, physical page, DRAM view, bank, endpoint,
   bank-local offset)` sequence를 생성한다.
2. endpoint별 contiguous run length, page/bank 전환 횟수와 합칠 수 있는 burst 크기를 계산한다.
3. 실제 주소 sequence를 재생하는 read-only trace-replay microbenchmark를 만든다.
4. 기존 1,088 B tile reader와 동일 주소의 coalesced reader를 A/B한다.
5. V의 contiguous destination부터 약 4.25 KiB burst를 적용한다.
6. K는 contiguous staging CB와 local transpose 비용을 별도 측정한다.
7. 이후에만 6 endpoint-local owner와 disjoint sequence handoff를 실제 SDPA에 통합한다.
8. aggregate bandwidth뿐 아니라 slowest core, reducer/worker 종료 spread와 end-to-end latency를
   성공 기준으로 사용한다.

## 9. 근거

### 보고서

- `/home/iris_hb4/reports/benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md`
- `/home/iris_hb4/reports/benchmark-results/2026-07-26-llama32-3b-64k-sdpa-dram.md`
- `/home/iris_hb4/reports/investigations/2026-08-01-sdpa-dram-performance-optimization-history.md`
- `/home/iris_hb4/reports/investigations/2026-07-26-ttnn-visualizer-bos-blackhole-npe.md`

### 주요 소스

- Vanilla/optimized SDPA program factory:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- Paged K/V reader:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- Saturation benchmark host:
  `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc.cpp`
- Saturation reader kernel:
  `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/kernels/reader_dram.cpp`

### 주요 profiler runs

- Vanilla:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33`
- Dual-NoC 6 endpoint:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_curpos_only_2026_07_26_11_45_00`
- K=256:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_device_profile_2026_07_31`
- K=512:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_512_device_profile_2026_07_31`

## 10. 한계

이 보고서는 날짜가 다른 profiler run과 일부 reconstructed duration을 포함한다. bandwidth는
통일된 effective K/V payload 기준을 우선 사용했지만 raw DRAM throughput, NPE utilization과
완전히 같은 counter가 아니다.

주 정확도 비교는 cur-pos-only, identity page table과 초기화된 KV cache를 사용했다. random page
table, non-zero KV, full model, 다양한 context와 production allocator의 locality는 검증되지 않았다.
86.83 GB/s는 read-only silicon saturation reference이며 실제 Flash Decode의 달성 보장값이 아니다.

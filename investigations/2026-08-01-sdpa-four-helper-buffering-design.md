# BOS Flash Decode의 4 idle-core buffering 및 data-processing helper 설계 조사

- 작성일: 2026-08-01
- 대상: Llama 3.2 3B, B=1, 64K paged Flash Decode
- 장치: Blackhole 코드/runtime을 사용하는 custom 20-core BOS NPU
- 목적: 현재 idle인 4개 worker core를 DRAM read, buffering, layout processing과 relay에 활용할 수
  있는지 검토
- 상태: host-side 설계 조사, device 성능·정확성 미검증
- 신규 device 실행: 없음

## 1. 결론

현재 5×4 program grid에는 20 worker cores가 있지만, 8 KV heads에 reducer/worker 2 cores씩을
배정하는 64K SDPA의 실제 compute-active core는 16개다. 나머지 4개 core를 추가 reducer로
단순 배치하는 것보다 reader helper로 사용하는 편이 구조적으로 더 타당하다.

제안하는 topology는 다음과 같다.

```text
4 idle helper cores
+ 2 active compute-reader owners
= 6 endpoint-local owners

6 owners
→ 6 local compute streams
→ 10 remote compute streams에 disjoint sequence shard 전달
```

Helper의 역할은 큰 KV cache를 장기간 보관하는 것이 아니다. B=1 Flash Decode에서는 KV의 추가
재사용이 작기 때문에 L1은 짧은 streaming window로 사용한다. 성능 이득을 내려면 helper가
다음 세 작업 중 하나 이상을 실제 critical path에서 제거해야 한다.

1. 같은 endpoint의 physically contiguous request를 더 큰 NoC read로 coalesce
2. K의 DRAM order와 QK matmul CB order 사이의 reorder/transpose를 active core에서 offload
3. page-table 결과를 endpoint-local run descriptor로 변환하고 disjoint stream만 전달

단순히 DRAM에서 읽어 compute core로 한 번 더 복사하는 relay는 이득이 없다. 기존 full-block
fanout microbenchmark가 51.070 GB/s로 악화된 것이 직접적인 반례다. 반면 중복 없는 six-reader
disjoint-shard relay는 77.369 GB/s를 기록했으므로, specialized helper의 유효성은
`coalescing/reorder로 절약한 비용 > helper→consumer NoC hop과 동기화 비용`인지에 달려 있다.

## 2. 현재 core 역할

### 2.1 Compute-active core

현재 shape는 KV heads 8개에 동일한 two-core group을 할당한다.

```text
KV head 0: reducer 0 + worker 0
KV head 1: reducer 1 + worker 1
...
KV head 7: reducer 7 + worker 7
```

Reducer와 worker는 서로 다른 compute kernel binary가 아니다. 모든 active core는
`sdpa_flash_decode.cpp`를 실행하고 runtime argument `do_reduce`로 역할을 선택한다.
Reducer도 자신의 sequence chunks를 계산한 뒤 worker의 partial `(m,l,O)`를 추가로 merge한다.

각 Tensix core에서는 data-movement reader, writer와 TRISC compute가 함께 동작한다.

```text
active Tensix
├── reader_decode_all.cpp
├── writer_decode_all.cpp
└── compute/sdpa_flash_decode.cpp
```

따라서 helper의 목적은 단순 core-count 증가가 아니라 active core의 data-movement RISC와 L1
layout-processing 부담을 분리하는 것이다.

### 2.2 Idle core

5×4 grid 중 계산에 배정되지 않은 4개 위치는 현재 operation의 active compute set이 아니다.
Program 범위에 포함되더라도 idle sentinel runtime argument를 받으면 kernel이 early return한다.

이 core에 reader-only 또는 reader+local-reorder kernel을 배치하려면 compute-active core count와
별도의 helper core set, kernel handle과 runtime arguments가 필요하다. 단순히 기존 active-core
수를 20으로 바꾸는 것으로는 원하는 topology가 만들어지지 않는다.

## 3. 현재 vanilla reader의 비용

K-chunk 256, head dimension 128에서 K 또는 V 한 phase는 다음과 같다.

```text
8 sequence tiles × 4 head-dimension tiles
= 32 BF8 tiles
= 약 34 KiB logical payload
```

그러나 이는 하나의 34 KiB request가 아니다. Paged/interleaved KV에서 reader는 tile마다
`noc_async_read_tile()`을 호출한다.

- K: physical row order로 읽고 L1에는 transposed/strided order로 기록
- V: physical row order로 읽고 L1에도 row-major order로 기록
- page table은 logical sequence row를 physical KV block으로 변환
- interleaved mapping은 tile마다 DRAM view/endpoint를 바꿀 수 있음

따라서 aggregate payload가 saturation benchmark의 32 KiB block과 비슷해도 endpoint-local
contiguous run과 request 크기는 다르다.

중요한 제한은 연속 logical tile 4개가 현재 interleaved allocation에서 물리적으로 같은 endpoint의
연속 주소라는 보장이 없다는 점이다. 약 4.25 KiB burst는 다음 중 하나가 확인돼야만 가능하다.

- 실제 address trace에서 같은 DRAM view의 contiguous bank-local offset이 발견됨
- KV allocation/page placement를 specialized bank-local layout으로 변경함

Reader code만 보고 임의의 네 tile을 하나의 NoC read로 합치면 안 된다.

## 4. 제안 topology

### 4.1 Six-owner hybrid

BOS topology는 3 physical DRAM banks, bank당 2 worker NoC endpoints로 총 6 endpoints다.
Helper가 4개뿐이므로 다음 hybrid mapping을 사용한다.

```text
endpoint 0 ← idle helper A
endpoint 1 ← idle helper B
endpoint 2 ← idle helper C
endpoint 3 ← idle helper D
endpoint 4 ← endpoint에 가까운 active compute owner E
endpoint 5 ← endpoint에 가까운 active compute owner F
```

실제 endpoint 선택은 고정 번호가 아니라 physical worker 위치, native NoC ring과 route cost로
결정한다. 네 helper는 active compute core가 멀리 있거나 measured barrier tail이 큰 endpoint에
우선 배치한다.

6 owner가 담당할 16 logical compute streams는 기존 disjoint benchmark와 같은
`3/2/3/3/3/2` 유형의 group으로 나눌 수 있다.

- owner local stream: 6개
- remote stream: 10개
- NOC0/NOC1 stream count: 각각 8개
- physical-bank stream load: 5/5/6

정적 reader 수의 완전 대칭보다 slowest reducer/worker pair의 종료 시간을 균형화하는 것이
우선이다.

### 4.2 Streaming slots

Helper L1은 cache가 아니라 bounded ring buffer로 사용한다.

```text
slot 0: 현재 consumer가 계산 중
slot 1: 다음 physical run을 DRAM에서 읽는 중
optional slot 2: reorder 또는 remote handoff 대기
```

각 slot은 consumer ACK 또는 remote-CB pop이 확인된 뒤에만 재사용한다. Slot 수를 크게 늘리는
것은 목표가 아니다. Tagged cross-chunk prefetch가 실제 SDPA에서 중립이었으므로 buffer depth만
늘려서는 성능 개선을 기대하지 않는다.

## 5. Helper에서 가능한 데이터 가공

### 5.1 Address-run generation

먼저 logical read sequence를 다음 descriptor로 바꾼다.

```text
(KV kind, head, logical row, physical page, DRAM view,
 physical bank, endpoint, bank-local offset, run bytes)
```

동일 endpoint에서 bank-local offset과 destination offset이 모두 연속인 구간만 direct burst로
합친다. Source만 연속이고 destination이 strided인 K는 staging/reorder 대상으로 분류한다.

Descriptor는 host-side에서 분석할 수 있지만 production page table이 runtime에 바뀌면 device
helper가 page-table entry를 읽어 생성하거나 page allocator가 locality metadata를 제공해야 한다.

### 5.2 V direct burst

V는 source tile order와 compute CB의 row-major destination order가 일치하므로 첫 후보이다.
동일 endpoint의 물리 연속 run이 존재한다면:

```text
여러 noc_async_read_tile()
→ 하나의 noc_async_read(source, contiguous_L1, run_bytes)
```

로 바꿀 수 있다. 현재 allocation에서 run이 짧다면 V-only burst의 이득도 제한된다.

### 5.3 K staging 및 transpose

K는 source row의 head-dimension tiles가 논리적으로 연속이어도 final K CB destination이
sequence-strided다. 따라서 direct burst 조건을 만족하지 않는다.

```text
DRAM physical run
→ helper contiguous staging
→ local tile reorder/transpose
→ owner local K CB 또는 receiver remote K CB
```

Local reorder 구현 후보는 다음과 같다.

- data-movement RISC의 L1-to-L1 copy
- helper core의 idle TRISC를 사용하는 tile transpose/retilize kernel
- compute kernel이 staging order를 직접 소비하도록 QK matmul blocking 변경

세 번째 방법은 copy를 없앨 수 있지만 compute kernel 변경 범위가 가장 크다.

### 5.4 Page-table predecode

현재 reader가 chunk loop 안에서 반복하는 page-table mapping을 helper가 미리 수행할 수 있다.
장점은 산술 비용보다 address run과 endpoint별 queue를 미리 구성할 수 있다는 데 있다.

Identity page-table A/B에서 paging 제거 이득이 약 0.4%뿐이었으므로 page lookup 계산 자체를
주요 성능 목표로 삼지는 않는다. Random/non-identity page table에서 locality가 깨질 때의
address scheduling 효과는 별도로 검증해야 한다.

### 5.5 수행하지 않을 가공

- BF8→BF16 변환: L1/NoC payload 증가
- K에 RoPE 적용: cache update 시 이미 rotated K가 기록됨
- 전체 K/V block fanout: 중복 relay traffic 증가
- 큰 KV 영역의 장기 cache: B=1에서 추가 consumer/reuse 부족
- helper에서 임의의 softmax partial 계산: head별 compute core 수가 불균형해지고 역할이 reader
  offload에서 dynamic compute scheduling 문제로 바뀜

## 6. 재사용 가능한 기존 POC

현재 opt-in은 다음 환경변수를 사용한다.

```text
TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=1
```

지원 조건은 B=1, S=64K, K-chunk 256, paged causal decode, KV heads 8, active cores 16과
6 endpoints다.

이미 구현된 요소는 다음과 같다.

- endpoint group별 6 owners와 10 receivers 선택
- owner/receiver별 reader kernel compile define
- K/V owner staging CB
- K/V GlobalCircularBuffer sender/receiver mapping
- receiver local `c1/c2`와 remote `c30/c31` alias
- remote reserve/push/pop 및 slot reuse handshake
- owner가 remote stream의 disjoint sequence chunk를 별도 read해 전달하는 plumbing

그러나 현재 owner 6개는 active 16-core set 안에서 선택된다. 제안안과의 차이는 다음과 같다.

| 항목 | 현재 POC | 제안 helper |
|---|---|---|
| Owner 위치 | active compute cores 중 6 | idle helper 4 + active owner 2 |
| DRAM read | tile별 TensorAccessor | contiguous-run coalescing 후보 |
| K processing | staging에 strided write | contiguous staging 후 local reorder 후보 |
| Owner compute | 모든 owner가 compute도 수행 | helper 4개는 compute 없음 |
| 성능 상태 | end-to-end 미검증 | 설계 단계 |

기존 GCB handoff를 재사용할 수 있지만 owner set, active/idle kernel ranges, runtime chunk assignment와
helper용 compile/runtime arguments는 새로 설계해야 한다.

## 7. 성능 근거와 기대 범위

| 구성 | Bandwidth | 해석 |
|---|---:|---|
| 20-reader, 6-endpoint direct saturation | 86.83 GB/s | 이상적 read-only 상한 |
| Six-reader full-block fanout | 51.070 GB/s | 중복 relay/ACK 비용으로 실패 |
| Six-reader disjoint-shard relay | 77.369 GB/s | specialized ownership의 headroom |
| 실제 6-endpoint K=256 SDPA | 약 70.03–70.27 GB/s | 현재 실용 기준 |

현재 SDPA는 disjoint microbenchmark보다 약 10% 낮다. 이 차이 전체가 helper로 회수 가능한 것은
아니다. K=256에서 BRISC/NCRISC/TRISC span이 약 2.0 ms로 비슷하므로 reader만 빨라져도 compute와
reducer tail이 critical path로 남을 수 있다.

성공 기준은 다음과 같이 둔다.

- correctness와 random-page-table coverage 통과
- current K=256 기준 대비 latency 3% 이상 개선: 의미 있는 성공
- latency 5% 이상 개선: 강한 채택 후보
- bandwidth만 오르고 slowest-core latency가 그대로면 채택 보류
- helper/relay traffic 포함 aggregate NoC byte와 endpoint tail을 함께 보고

## 8. 주요 위험

1. Helper hop이 절약한 request overhead보다 비쌀 수 있다.
2. Remote consumer 하나가 늦으면 owner slot과 같은 group의 다른 stream까지 막을 수 있다.
3. K local reorder가 active core의 strided write보다 느릴 수 있다.
4. Current interleaved allocation에는 coalesce 가능한 physical run이 거의 없을 수 있다.
5. DRAM-sharded allocation을 도입하면 cache update/write path도 같은 layout을 지원해야 한다.
6. 4 helper와 2 compute-owner의 역할 차이로 per-endpoint service cadence가 불균형할 수 있다.
7. GlobalCircularBuffer/remote-CB lifecycle 오류는 deadlock으로 이어질 수 있다.
8. Profiler timeout 뒤 장치 작업을 계속하면 device/FW 상태를 오염시킬 수 있다.

Generic DRAM-sharded TensorAccessor가 약 24 GB/s로 악화된 결과는 specialized helper를 직접
반증하지 않는다. 당시에는 worker ownership, shard ownership과 burst granularity가 맞지 않았다.
반대로 allocation만 바꾸면 충분하다는 주장도 지지하지 않는다.

## 9. 단계별 검증 계획

### Phase 0 — host-side only

1. 실제 identity 및 random page table에서 physical tile sequence를 생성한다.
2. endpoint별 contiguous source/destination run-length histogram을 만든다.
3. V direct-burst 가능 byte 비율과 K staging 필요 byte 비율을 계산한다.
4. Helper 4개와 active owner 2개의 route-cost assignment를 계산한다.
5. 예상 DRAM, helper-local-copy와 relay byte accounting을 만든다.

### Phase 1 — isolated helper primitive

장치가 정상 복구되고 add, non-profiled SDPA 및 Watcher 안전 절차가 통과한 뒤에만 실행한다.

1. helper 1개, consumer 1개, V-only, 한 chunk로 시작한다.
2. tile reader와 동일 주소의 coalesced reader를 A/B한다.
3. local consumer와 remote consumer를 각각 측정해 relay hop 비용을 분리한다.
4. timeout 없이 correctness와 slot reuse를 반복 검증한다.

### Phase 2 — K processing

1. K contiguous staging까지만 측정한다.
2. data-movement copy와 helper TRISC reorder를 A/B한다.
3. final K CB layout을 기존 reader 결과와 bytewise 비교한다.
4. active compute 없이 helper pipeline의 최대 sustainable rate를 측정한다.

### Phase 3 — one-head SDPA

1. KV head 하나, reducer/worker pair 하나에 helper를 연결한다.
2. K/V chunk 하나만 remote CB로 전달한다.
3. `m,l,O`, final output과 vanilla를 비교한다.
4. reader span, compute CB wait, helper/consumer semaphore wait를 분리 계측한다.

### Phase 4 — full 6-owner topology

1. 4 idle helpers + 2 active owners를 활성화한다.
2. 6 local + 10 remote disjoint streams로 확장한다.
3. aggregate bandwidth가 아니라 slowest reducer/worker pair를 먼저 본다.
4. identity/random page table, non-zero KV, full-layer와 model-level correctness를 검증한다.

NoC capture는 isolated single operation에서만 시작하고, timeout 또는 incomplete report가 한
번이라도 발생하면 그 session에서는 add를 포함한 추가 device workload를 실행하지 않는다.

## 10. 구현 후보 위치

- Core/endpoint/owner assignment 및 GCB:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- Paged K/V read, owner staging 및 receiver remote CB:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- Partial-state 및 output data movement:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
- Flash Decode compute:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
- Six-reader microbenchmark:
  `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/`

## 11. 관련 보고서

- `/home/iris_hb4/reports/investigations/2026-08-01-vanilla-sdpa-vs-dram-saturation-gap.md`
- `/home/iris_hb4/reports/investigations/2026-08-01-sdpa-dram-performance-optimization-history.md`
- `/home/iris_hb4/reports/benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md`
- `/home/iris_hb4/reports/incidents/2026-08-01-bos-mlp-noc-profile-fw-init-failure.md`

## 12. 현재 판단

4 idle core를 활용하는 방향은 타당하지만, “buffer를 크게 만든다”가 아니라 “active reader의
작은 request와 K layout 변환을 endpoint-local helper로 옮긴다”가 정확한 목표다.

첫 의사결정 지점은 device code가 아니라 host-side address-run 분석이다. 현재 paged/interleaved
allocation에 충분한 contiguous run이 없다면 reader-only coalescing은 불가능하며, specialized
DRAM allocation과 KV update path 변경까지 범위가 커진다.

따라서 현재 채택 판단은 다음과 같다.

- 4-helper topology: 연구 계속
- 기존 six-reader GCB plumbing: 재사용 후보
- V direct burst: address-run 확인 후 첫 구현 후보
- K staging/reorder: V primitive가 이득을 보인 뒤 진행
- full fanout: 폐기
- generic DRAM sharding 단독 적용: 폐기
- end-to-end 기본 활성화: 성능·정확성 검증 전 금지

## 13. 2026-08-02 reduce-only helper POC 결과

Reader relay 대신 16개 active core의 direct K/V read를 유지하고, KV head 7의 두 partial
`(m,l,O)`만 idle core 하나로 보내 최종 correction/normalization/output write를 offload하는 opt-in
POC를 구현했다.

```text
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1
active producers: KV head 7의 기존 reducer/worker 2개
helper: logical (1,3), Watcher virtual (1,4)
other heads: vanilla reducer/worker 유지
output: GQA가 지원하는 interleaved DRAM output
```

Host factory object와 `ttnncpp` shared library build는 성공했다. 첫 sharded-output 사전 시도는 GQA
host validation에서 kernel launch 전에 거부됐고 정상 device close로 끝났다. 이를 interleaved output
writer로 수정한 뒤 실제 64K paged SDPA를 Watcher와 외부 timeout 아래 한 번 실행했다.

실제 run에서는 새 helper kernels가 compile/load됐지만 completion marker가 나오지 않았다. Watcher는
helper BRISC를 `NSW`(`noc_semaphore_wait`)에, helper TRISC를 compute wait 상태에 고정된 것으로
보여줬다. 동시에 active cores도 reader `CRBW`와 compute matmul 관련 waypoint에서 전진하지 않았다.
따라서 관측상 helper가 기다린 두 partial이 도착하지 않았지만, helper semaphore 자체만을 원인으로
확정할 수는 없다. active producer가 partial emission까지 도달하지 못한 global circular wait가 더
넓고 정확한 현재 진단이다.

150초 SIGINT timeout 뒤 15초 cleanup 상한에서 SIGKILL이 발동해 exit code 137이 됐고 Python PID는
PID 1 아래 zombie로 남았다. 이후 장치 workload는 실행하지 않았으며 장치는 다시 격리 상태다.

다음 재시작 뒤에는 현재 POC를 그대로 재실행하지 않는다.

1. 필수 32x32 add smoke 한 번
2. helper와 tagged async를 모두 끈 동일 isolated 64K baseline
3. producer partial-send 전후와 helper semaphore 값에 명시적 Watcher waypoint 추가
4. tagged async를 끈 짧은-context one-head reduction primitive
5. 위 단계가 통과한 뒤에만 64K 및 tagged async를 각각 독립적으로 추가

상세 장애 기록은
`/home/iris_hb4/reports/incidents/2026-08-02-bos-sdpa-reduce-helper-deadlock.md`에 있다.

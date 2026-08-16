# BOS workload-specific memory benchmark design

## 목적

Raw DRAM peak 하나를 SDPA와 MLP의 roof로 사용하지 않는다. 같은 custom 20-core BOS NPU에서도
주소 변환, tile 순서, destination layout, barrier cadence와 consumer가 다르면 달성 가능한 대역폭이
달라진다. 아래 계층을 분리한다.

1. raw transport ceiling
2. workload trace-replay transport ceiling
3. isolated operation performance
4. full-layer 및 end-to-end performance

장치는 Blackhole runtime/code path를 사용한다. 사용 가능한 worker grid는 5x4, 물리 DRAM은 3 banks,
worker NoC endpoint는 bank당 2개로 총 6개다. Runtime logical DRAM view와 물리 bank를 같은 개념으로
취급하지 않는다.

## 왜 workload별 benchmark가 필요한가

### Raw benchmark

현재 `12_dram_20_core_6_noc_read`는 지정된 DRAM view 안에서 연속 주소를 반복 읽는다. Compute, CB
backpressure, page-table translation이 없다. 20-reader 구성은 약 94.43 GB/s, 과거 최적점은 최대
96--98 GB/s 범위다. 이것은 read-only transport ceiling이다.

3-reader/3-bank 구성은 64/128/256 KiB block과 tagged depth 2/3에서 약 66.5--66.8 GB/s였다. 따라서
reader 하나가 bank 하나를 포화시킨다는 가정은 성립하지 않았다.

### Weight stream

Stable MLP weight는 DRAM-sharded다. 각 reader는 고정 shard의 연속 K/N block을 읽고 compute CB에
공급한다. Raw benchmark의 `mlp-strided` proxy도 86.560 GB/s에 도달했지만 실제 MLP consumer wait는
kernel의 약 67%였다. 주소 stride만으로 실제 MLP 병목을 재현하지 못한다.

Weight benchmark는 실제 matmul factory의 아래 계약을 재생해야 한다.

- shard-to-reader ownership
- W1/W3/W2별 `per_core_N`, K-block 순서와 block width
- BFP8/BF16 tile/page size
- fanout-2 destination 4:4:4
- tagged two-block issue/retire cadence
- producer CB reserve/push와 consumer pop cadence

### Paged KV stream

SDPA는 page table을 L1에 올린 뒤 logical sequence row를 physical page/tile ID로 변환한다. K는 DRAM에서
row-major로 읽고 L1 CB에는 transposed placement를 사용한다. V는 row-major destination을 사용한다.
Barrier threshold와 K/V tagged transaction도 실제 cadence 일부다.

기존 64K identity-page-table A/B는 다음 결과를 냈다.

| layout | 16-core effective K/V bandwidth |
|---|---:|
| paged interleaved | 70.266 GB/s |
| contiguous interleaved | 70.551 GB/s |
| generic six-way DRAM-sharded | 24.369 GB/s |

Identity page table에서 paging 제거 이득은 약 0.4%다. 따라서 page-table lookup 자체가 주 병목이라는
주장은 성립하지 않는다. Random/non-local page table과 page-boundary 전환 비용은 아직 미측정이다.

## Benchmark suite

| 층 | Weight | Attention |
|---|---|---|
| Raw ceiling | `12_dram_20_core_6_noc_read`, packed | 동일 raw ceiling |
| Trace replay | DRAM-sharded weight owner/reader + K-block cadence | page-table 기반 K/V tile sequence, K와 V 분리 |
| Isolated op | W1/W3/W2 matmul, compute/CB 포함 | cur-pos-only single-layer SDPA |
| Integration | one-layer MLP | one-layer attention 및 full layer |

Trace replay는 compute를 제거하되 실제 주소와 synchronization을 유지한다. Isolated op와의 차이가
consumer/compute/CB 비용이다.

## Weight trace-replay 사양

- 실제 `BufferDistributionSpec`과 `BufferShardingArgs`로 DRAM-sharded buffer 생성
- runtime의 BOS logical DRAM views와 선택된 6 interface workers 기록
- actual matmul factory의 W1/W3/W2 block geometry 재생
- one-packet tile과 최대 16 KiB coalesced burst 독립 A/B
- depth 1/2와 full barrier/tagged 독립 A/B
- destination-local과 fanout-2 remote-L1 write 분리 측정
- reader별 고유 data를 써서 L1 결과 검증

필수 sweep은 projection, format, block width, active reader 수, endpoint assignment다. Production
4:4:4와 의도적 불균형을 비교한다.

## Paged-KV trace-replay 사양

- production KV shape, tile size, 8 KV heads, head dim 128 사용
- page table은 identity, random permutation, locality-grouped permutation 비교
- K-chunk 128/256/512 비교
- production `virtual_seq_tile_id_to_physical_tile_id`와 `TensorAccessor` 재사용
- K-only/V-only run 분리
- K transposed L1 destination과 V contiguous destination 재현
- production barrier threshold와 tagged K/V transaction 재현
- coalescing은 연속 physical tile run 안에서만 허용

필수 출력은 endpoint별 bytes/request, contiguous-run 분포, page/endpoint transition, 16 KiB boundary
crossing, issue/barrier/tail cycles, slowest-core span, finish spread, aggregate/useful bandwidth다.

## 판정법

각 실험은 독립 변수 하나만 바꾸고 warmup 뒤 5회 측정한다.

1. raw 대비 trace replay: locality/addressing/synchronization 손실
2. trace replay 대비 isolated op: CB/consumer/compute 손실
3. isolated op 대비 layer: operator gap과 integration 손실
4. layer 대비 end-to-end: scheduler 및 나머지 model 손실

Bandwidth plateau만 포화 증거로 쓰지 않는다. Plateau와 함께 issue backpressure 또는 barrier latency가
증가하고 endpoint별 bytes와 slowest-core span이 안정적인지 확인한다.

## 기존 재현 경로

Raw transport:

```bash
build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-endpoint --pipeline-mode tagged \
  --page-size 2048 --pages-per-core 512 --pages-per-block 8 --num-tests 5
```

Actual paged SDPA:

```bash
TT_METAL_SDPA_DECODE_DUAL_NOC=1 TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 \
python_env/bin/python models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py \
  --precision-mode accuracy --context-len 65536 --sdpa-k-chunk-size 256 \
  --kv-layout paged
```

Contiguous control은 마지막 인자를 `--kv-layout contiguous`로 바꾼다. `contiguous-sharded`는 specialized
bank-local reader가 아닌 generic TensorAccessor 반례이므로 weight-sharded roof로 인용하지 않는다.

## 구현 순서

1. Weight: existing DRAM-sharded data-movement test를 correctness oracle로 사용하고 standalone timed
   reader를 추가한다.
2. Weight: production W1/W3/W2 args를 dump해 trace input으로 고정한다.
3. Attention: host에서 page table별 physical tile trace와 locality metrics를 생성한다.
4. Attention: trace를 그대로 재생하는 K-only/V-only kernel을 추가한다.
5. Profiler 없이 correctness와 짧은 latency run을 통과시킨 뒤 isolated single-op profile을 수행한다.

## 현재 결론

- Workload별 benchmark 필요하다.
- Raw benchmark를 복제해 이름만 weight/KV로 바꾸는 것은 무의미하다.
- Weight는 실제 DRAM shard ownership과 producer cadence가 핵심이다.
- Attention은 actual page table, K transpose destination, V contiguous destination이 핵심이다.
- Existing identity-page-table 결과상 paging lookup 자체는 현재 70 GB/s 상한의 주원인이 아니다.

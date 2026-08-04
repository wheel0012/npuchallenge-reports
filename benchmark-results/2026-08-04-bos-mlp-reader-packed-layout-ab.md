# BOS MLP reader-packed weight layout A/B

날짜: 2026-08-04 UTC

## 결론

Reader별 연속 multi-block packing은 채택하지 않는다. 실제 MLP fanout-2와 같은 12 readers,
6 endpoints, endpoint당 2 lanes 조건에서 효과가 재현되지 않았다. W2 exact payload는 +0.39%뿐이다.
W1 exact payload는 host jitter가 큰 15 MB 측정에서 +3.58%였지만, 4회 cadence 반복에서는 packed가
오히려 -2.46%였다.

현재 weight cadence 원인을 DRAM shard 내부의 lane-interleaved row layout으로 보기는 어렵다. 다음 분석은
compute consumer cadence, activation multicast phase, weight/compute overlap을 대상으로 한다.

## 장치와 topology

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- benchmark active readers: 12
- physical DRAM: 3 banks, bank당 2 worker endpoints, 총 6 endpoints
- endpoint readers: 2/2/2/2/2/2
- NoC readers: NOC0 6, NOC1 6
- runtime logical DRAM views: 6
- pipeline: tagged two-block

`p100` harvesting warning은 BOS identity가 아니다. Custom BOS topology를 위 기준으로 기록한다.

## 비교한 주소 패턴

기존 MLP-like strided:

```text
[K row 0: lane 0 | lane 1]
[K row 1: lane 0 | lane 1]
...
```

reader-packed:

```text
[lane 0: K row 0 | K row 1 | ...]
[lane 1: K row 0 | K row 1 | ...]
```

두 경로의 payload, reader 배치, endpoint, NoC, VC, tagged pipeline은 같다. 바뀐 것은 DRAM shard 내부
source block offset뿐이다.

## MLP shape 대응

| projection | read page | pages/block | blocks | pages/reader | total payload |
|---|---:|---:|---:|---:|---:|
| W1/W3 | 13,056 B | 6 | 16 | 96 | 15,040,512 B |
| W2 | 8,704 B | 16 | 16 | 256 | 26,738,688 B |

BFLOAT8_B tile은 1,088 B다. W1/W3 row burst는 12 tiles, W2 row burst는 8 tiles다.

## 결과

각 구성은 warmup 뒤 measured call 10회다. 모든 run이 validation, `Test Passed`, device close까지
완료됐다. Timeout, signal, exit 124/137은 없었다.

| shape | cadence repeat | layout | avg GB/s | min | max | packed 변화 |
|---|---:|---|---:|---:|---:|---:|
| W1/W3 | 1 | strided | 71.573 | 60.296 | 81.290 | - |
| W1/W3 | 1 | packed | 74.135 | 64.687 | 80.001 | +3.58% |
| W2 | 1 | strided | 78.031 | 74.690 | 83.396 | - |
| W2 | 1 | packed | 78.333 | 70.693 | 84.689 | +0.39% |
| W1/W3 | 4 | strided | 86.560 | 85.357 | 89.039 | - |
| W1/W3 | 4 | packed | 84.433 | 83.179 | 85.369 | -2.46% |

W1 single-payload 결과는 0.185--0.249 ms host timing 분산이 크다. 4x cadence 결과는 분산이 작고
packed 우위를 부정한다. W2도 packed 이득이 측정 잡음 수준이다.

## 판정

관측 사실:

1. Strided도 4x cadence에서 86.560 GB/s까지 도달했다.
2. Packed는 W1 sustained cadence를 높이지 않았다.
3. W2 packed 변화는 +0.39%뿐이다.
4. 두 layout 모두 correctness와 device close를 통과했다.

추론:

1. 두 concurrent lane의 row-stride는 BOS DRAM/NoC가 숨길 수 있다.
2. Full MLP의 약 24 us publish cadence는 단순 DRAM physical locality보다 producer-consumer phase와
   compute backpressure 영향을 더 받을 가능성이 높다.

미검증 가설:

1. Activation multicast와 weight arrival의 core별 phase alignment가 consumer wait를 만든다.
2. Math engine consumption cadence가 tagged reader의 추가 prefetch를 흡수하지 못한다.

## Fix

1. Model-load weight prepack과 cache format 변경을 구현하지 않는다.
2. 기존 fanout-2 weight layout을 유지한다.
3. Microbenchmark의 `mlp-fanout2`와 `--access-layout`은 향후 회귀 A/B용으로 유지한다.
4. 다음 계측은 compute block start/end와 `MLP_IN0_READY`/`MLP_IN1_READY`를 같은 block index로 결합한다.

## 재현

```bash
cmake --build build_home_release --target test_dram_20_core_6_noc -j

timeout --signal=INT --kill-after=15s 60s \
  build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config mlp-fanout2 --access-layout mlp-strided --pipeline-mode tagged \
  --page-size 13056 --pages-per-core 96 --pages-per-block 6 --iteration-quanta 1 --num-tests 10
```

`--access-layout packed`로 A/B한다. W2는 `--page-size 8704 --pages-per-core 256
--pages-per-block 16`을 쓴다. Sustained W1은 `--iteration-quanta 4`를 쓴다.

## 위치와 한계

- host: `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc.cpp`
- kernel: 같은 디렉터리의 `kernels/reader_dram.cpp`
- audit patch: `/home/iris_hb4/tmp/codex-patches/20260804-mlp-packed-access-ab.patch`
- profiler artifact: 없음. Profiler-free console measurement다.
- 이 microbenchmark에는 matmul compute와 activation multicast가 없다. Full-layer 성능 개선을 직접
  증명하지 않는다.
- 4x cadence는 동일 weight 주소를 반복한다. Exact one-layer traffic은 repeat 1 결과다.

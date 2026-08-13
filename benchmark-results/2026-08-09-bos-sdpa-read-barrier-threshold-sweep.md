# BOS SDPA K256 read-barrier threshold sweep

## 결론

64K paged SDPA의 K/V reader barrier cadence는 성능에 영향을 줬다. 기존 자동값 18 tiles 대비
8 tiles가 median latency를 `2.151875→2.133140 ms`, `0.871%` 줄였다. 그러나 재측정은 안전 정책
reviewer가 실행 전에 차단했다. threshold 8은 후보이며 stable 승격하지 않는다.

outer K256 + inner K128 publish는 이번에 구현하지 않았다. 현재 K tile layout에서 첫 128-token half는
CB에 연속하지 않는다. 이를 구현하려면 K/V 재배열, QK/PV 두 번, online-softmax merge 추가가 필요하다.
기존 K128은 K256보다 4.56% 느렸으므로 단순 half-chunk 전환은 기대 방향과 반대다.

## 대상

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- active SDPA cores/readers: 16 (`8 KV heads × 2 cores/head`)
- DRAM: 3 physical banks, 2 worker endpoints/bank, 6 endpoints
- reader mapping: endpoint loads `3/2/3/3/3/2`, NoC loads `8/8`
- sequence: 65,536, curpos 65,535
- Q/K chunk: 128/256 tokens
- page block: 128 tokens
- Q/KV dtype: BF16/BFLOAT8_B
- output: interleaved DRAM
- tagged async, six-reader relay, reduce-only helper: off

## 변경

새 opt-in:

```text
TT_METAL_SDPA_DECODE_READ_BARRIER_THRESHOLD=<1..64>
```

unset이면 기존 formula를 유지한다.

```text
((512 / num_readers) * (1024 + 128)) / q_tile_bytes
```

이번 16-reader/BF16-Q 구성의 자동값은 18이다. opt-in은 reader kernel compile define만 바꾼다.
CB 크기, tile layout, compute, softmax reduction order는 바꾸지 않는다.

변경 파일:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- `tests/bos_model/run_sdpa_kchunk_profile.py`

## 검증

`ttnncpp`와 monolithic `ttnn` 증분 빌드가 성공했다. runtime이 사용하는
`build_home_release/lib/_ttnncpp.so`에 새 binary를 배포하고 문자열과 SHA-256 일치를 확인했다.

profiler 없는 4K K256 threshold18 gate:

- override log: `BOS SDPA decode read barrier threshold override: 18 tiles`
- max absolute error: 0.0625
- completion/close: 정상
- exit: 0

검증된 64K PCC runner의 threshold18 gate:

- PCC: 0.9999178292703101
- max delta: 0.0273244083
- `SDPA_CORRECT`, `DEVICE_CLOSED`
- exit: 0

## 64K latency sweep

각 점은 profiler 없이 warmup 2, 10 calls/repeat, 5 repeats로 측정했다. 표의 bandwidth는 전체
16 cores가 읽는 K+V payload `142,606,336 bytes`를 host median latency로 나눈 effective 값이다.

| threshold | median ms | mean ms | effective K/V GB/s | vs 18 median |
|---:|---:|---:|---:|---:|
| 4 | 2.611240 | 2.612134 | 54.613 | -21.347% |
| 8 | 2.133140 | 2.132655 | 66.853 | +0.871% |
| 12 | 2.137461 | 2.140960 | 66.718 | +0.670% |
| 18 | 2.151875 | 2.150515 | 66.271 | 기준 |
| 24 | 2.164892 | 2.166466 | 65.872 | -0.605% |
| 32 | 2.235163 | 2.236384 | 63.801 | -3.870% |

모든 표 run은 completion, close, exit 0이다. 상수-V latency runner는 64K 누적에서 max absolute error
0.6875를 보였지만 모든 threshold에서 동일했다. 정확성 판정은 별도 PCC gate를 사용했다.

## 해석

### 관측 사실

- barrier를 너무 자주 실행한 threshold4는 크게 느렸다.
- 자동값보다 큰 24/32도 느렸다.
- 8/12가 18보다 소폭 빨랐다.
- 따라서 큰 outstanding queue가 항상 유리하지 않다.

### 강한 추론

현재 gap 일부는 의도적 barrier cadence에서 생긴다. 18/24/32는 한 barrier가 기다리는 service tail과
endpoint contention을 늘린다. 4는 barrier 호출 overhead가 지배한다. 8 근처가 두 비용의 균형점이다.
하지만 최대 이득이 0.9% 수준이라 SDPA가 전체 DRAM bandwidth를 못 쓰는 주원인은 아니다.

### 미검증 가설

- threshold8이 compute K/V CB wait의 critical-core tail도 줄인다.
- endpoint별 최적 threshold가 다를 수 있다.
- threshold8과 tagged cross-chunk prefetch 조합은 단독 tagged 결과와 다를 수 있다.

## 안전 상태와 다음 단계

threshold18→8 확인 run은 안전 정책 reviewer가 process 생성 전에 차단했다. device workload는 시작되지
않았다. 이 확인 없이 threshold8을 README best-stable이나 full-model profile에 넣지 않는다.

다음 승인 뒤 순서:

1. threshold18/8 각각 warmup 3, 20 calls × 10 repeats 재측정
2. threshold8 64K PCC gate
3. device-zone으로 K/V read barrier와 compute CB wait 분해
4. 효과가 재현되면 full-model 64K A/B

## Artifact

- run root: `/home/iris_hb4/benchmark_runs/sdpa_read_barrier_threshold_2026_08_09`
- source patch: `/home/iris_hb4/tmp/codex-patches/20260809-161955-sdpa-read-barrier-threshold.patch`
- runner patch: `/home/iris_hb4/tmp/codex-patches/20260809-162624-sdpa-kchunk-gate-runner.patch`

Profiler/Tracy artifact는 생성하지 않았다.

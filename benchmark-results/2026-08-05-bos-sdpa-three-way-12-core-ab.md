# BOS SDPA three-way 12-core A/B

## 결론

8 KV heads를 head당 3 sequence slice로 나누고, 4개 core group이 각각 2 heads를 순차 처리하는
12-active-core 경로는 정확했지만 64K에서 baseline보다 느렸다.

| 경로 | active cores | endpoint loads | SDPA wall mean | layer mean |
|---|---:|---|---:|---:|
| baseline, 2 slices/head | 16 | `3/3/2/3/3/2` | 2.221584 ms | 5.450323 ms |
| three-way, 3 slices/head | 12 | `2/2/2/2/2/2` | 2.532018 ms | 5.815643 ms |
| 변화 | -4 | 균등화 | **+13.97%** | **+6.70%** |

Endpoint 균등화만으로 active-core 감소와 추가 reduce 비용을 상쇄하지 못했다. 기본 활성화하지 않는다.

## Hardware와 실행 형태

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- baseline operation active cores: 16
- three-way operation active cores: 12
- physical DRAM: 3 banks
- worker NoC endpoints: bank당 2, 총 6
- runtime logical DRAM grid: 이 A/B에서 별도 계측하지 않음
- selected DRAM-interface workers: 이 로그만으로 추론하지 않음
- TurboQuant: off

UMD의 P100/P150 문자열은 BOS board identity로 사용하지 않았다.

## 구현

Opt-in:

```text
TT_METAL_SDPA_DECODE_THREE_WAY_12_CORE=1
```

Scheduler 구성:

```text
8 KV heads × 3 sequence slices/head = 24 head-slice tasks
4 persistent groups × 3 cores/group = 12 active cores
각 group: reducer 1 + worker 2
각 group: KV heads 2개를 순차 처리
```

변경 위치:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
- `tests/bos_model/run_sdpa_three_way_12_core.py`

Reader와 compute kernel의 기존 `num_heads_per_core` loop를 재사용했다. Writer worker도 두 head round를
순회한다. 각 head의 partial이 덮어써지지 않도록 c19 intermediate storage를 2배로 잡고
`head_round × workers + worker_id` slot을 사용한다. Reducer semaphore wait는 두 번째 round에서 첫 번째
count를 재사용하지 않도록 누적 threshold `(head_round + 1) × workers`를 사용한다.

첫 검증 경로는 DRAM interleaved output이다. Sharded output의 multi-round gather 계약은 미검증이다.

## Correctness

Standalone paged SDPA, BF16 Q, BF8 K/V, `QH=24`, `KVH=8`, `D=128`, block 128,
Q chunk 128, K chunk 256, `5×4` program grid를 사용했다.

| sequence | PCC | 결과 |
|---:|---:|---|
| 4K | 0.9991912495 | pass, exit 0, `DEVICE_CLOSED` |
| 64K | 0.9999206797 | pass, exit 0, `DEVICE_CLOSED` |

두 run 모두 새 경로의 `12 active cores`와 endpoint load `2/2/2/2/2/2`, NoC load `6/6`을 로그에서
확인했다.

## Performance 측정

Production Llama 3.2 3B single-layer curpos-only runner를 사용했다. 64K paged KV, block 32,
K chunk 256, performance precision, warmup 1회, iteration 2회 × repeat 3회다. Profiler와 Watcher는
사용하지 않았다. MLP DRAM-sharded, grouped concat, TurboQuant opt-in은 모두 껐다. 두 process 모두
exit 0과 정상 device close를 확인했다.

Samples:

```text
baseline layer ms: 5.328683, 5.493942, 5.528343
three-way layer ms: 5.821317, 5.833517, 5.7920945
baseline SDPA wall mean: 2.2215841667 ms
three-way SDPA wall mean: 2.5320181666 ms
```

Artifact:

```text
/home/iris_hb4/profiler_runs/sdpa_three_way_12_core_ab_2026_08_05_1448/
├── baseline.json
└── three_way_12_core.json
```

## 해석

관측 사실:

- 12-core 경로는 여섯 endpoint를 정확히 2 readers씩 사용했다.
- 그래도 SDPA wall은 13.97%, full layer는 6.70% 느려졌다.
- 결과 정확성과 device lifecycle은 정상이다.

추론:

- baseline core는 한 head의 절반을 처리한다. Three-way core는 한 head의 1/3을 두 번 처리한다.
  Core-group critical work는 대략 `1/2 → 2/3` head로 늘어난다.
- head당 worker partial 수가 1개에서 2개로 늘어 reducer traffic과 synchronization도 증가한다.
- 따라서 이 shape에서는 endpoint 균등화보다 16→12 active-core 감소와 serial second-head round 비용이 크다.

미검증:

- 남은 8 idle cores가 아닌 실제 4 idle-core 활용과 이 scheduler의 overlap 가능성
- sharded output multi-round gather
- NoC trace 기반 service-latency 분해

## Patch

```text
/home/iris_hb4/tmp/codex-patches/20260805-143800-sdpa-three-way-12-core-host.patch
SHA-256 396655dd6e2854adcf524361678b8fdf23bcf88b1b8efde0c3202889007b85cf

/home/iris_hb4/tmp/codex-patches/20260805-144000-sdpa-three-way-12-core-writer.patch
SHA-256 f18cf9bebf258283851cff3a83d730ff1ed1fa2d36971a1ac1f97087a6519754
```

## 권고

이 경로는 실험용 opt-in으로 유지한다. Production 후보는 16-core baseline을 유지한다. 다음 우선순위는
core 수를 줄이는 scheduler가 아니라 16 cores를 유지한 채 endpoint `3/3/2/3/3/2` imbalance를 줄이는
mapping 또는 reader cadence 변경이다.

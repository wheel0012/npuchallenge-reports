# BOS SDPA grouped concat exact Wo-input A/B

- Date: 2026-08-05
- Source revision: `142354481e02999186df74ef691bfb12f8d6d17e` plus local opt-in patches
- Device: custom 20-core BOS NPU
- Runtime/code architecture: Blackhole
- Available worker grid: `5×4 = 20 cores`
- Physical DRAM: 3 banks, 6 worker NoC endpoints
- Status: full-model bit-exact, single-layer latency improved; opt-in stable candidate

## Problem

12-core, 2-heads/core grouped concat는 single-layer actual concat row가 bit-exact이고 layer latency를 줄였다.
하지만 원래 grouped path를 28-layer Llama 3.2 3B decode에 적용하면 final logits가 baseline과 달랐다.

| 비교 | exact | PCC | max abs | changed |
|---|---:|---:|---:|---:|
| grouped OFF vs OFF repeat | true | 1.0 | 0 | 0 |
| grouped OFF vs original grouped ON | false | 0.999314 | 0.3125 | 99,701/128,256 |

## First-divergence capture

각 layer에서 attention의 Wo 이후 output과 TransformerBlock의 최종 residual을 저장했다. 실제 batch row 0과
padding rows 1–31을 분리했다.

관측 결과:

- Layer 0 Wo actual row에서 첫 차이가 발생했다.
- 차이는 150/3,072 values, max abs `9.5367432e-7`이었다.
- Layer 0 final residual은 BF16 반올림 뒤 다시 bit-exact였다.
- 최초 residual divergence는 layer 1이었다.
- 이후 차이가 누적되어 final logits max abs `0.3125`가 됐다.
- Padding rows에는 non-finite 값이 있었지만 actual row와 final logits는 finite였다.

## Root-cause inference

Baseline generic concat는 Wo에 DRAM interleaved TILE을 전달한다. Original grouped concat는 L1 sharded TILE을
전달한다. 두 경로의 logical actual row는 bit-exact지만, `ttnn.matmul`에 explicit program config가 없어서
input memory config에 따라 다른 program 또는 reduction order가 선택될 수 있다.

Wo 직전 memory config만 baseline과 같게 만들자 모든 actual rows와 final logits가 exact로 회복됐다. 따라서
padding leakage보다 Wo matmul의 layout-dependent accumulation order가 원인이라는 해석의 신뢰도가 높다.

## Exact Wo-input dataflow

Opt-in flags:

```text
TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1
TT_METAL_SDPA_DECODE_GROUPED_CONCAT_EXACT_WO_INPUT=1
```

Dataflow:

```text
SDPA DRAM output
  → height-sharded L1
  → 12-core grouped concat in L1
  → sharded_to_interleaved DRAM TILE
  → baseline-compatible Wo matmul
```

Generic baseline의 `L1 shard → DRAM → ROW_MAJOR → reshape → TILE`보다 변환 단계가 적다. 완전한 L1→Wo
fusion은 아니며 DRAM 왕복 하나는 남는다.

## Full-model correctness

Stable 6-endpoint SDPA와 stable 12-reader/12-compute MLP를 사용했다.

```text
SDPA endpoint loads: 3/2/3/3/3/2
SDPA NoC0/NoC1 loads: 8/8
MLP fanout: 2
MLP endpoint groups: 4:4:4
MLP read-page cap: 16 KiB
TurboQuant: off
```

| 항목 | 결과 |
|---|---:|
| Wo actual rows exact | 28/28 |
| Decoder residual actual rows exact | 28/28 |
| Combined layer captures exact | 56/56 |
| Final logits exact | true |
| Final logits changed values | 0/128,256 |
| Final logits PCC | 1.0 |
| Final logits max abs | 0.0 |

All capture runs exited 0 and printed `DEVICE_CLOSED`.

## Interleaved performance A/B

Multi-process reverse-order pairs showed process drift, so final result uses one process and one model instance. Mode order
was reversed every repeat.

```text
context: 64K, current_pos=65535
SDPA K chunk: 256 tokens
warmup: 3 per mode
iterations_per_sample: 10
repeats_per_mode: 14
order: baseline→exact, exact→baseline alternating
profiler: off
```

| 경로 | mean layer | median layer | mean SDPA wall |
|---|---:|---:|---:|
| Generic baseline | 4.469103 ms | 4.453900 ms | 2.237311 ms |
| Grouped exact Wo input | 4.352791 ms | 4.343528 ms | 2.236573 ms |
| 변화 | **-2.60%** | **-2.48%** | -0.03% |

SDPA wall은 동일하다. 이득은 post-SDPA concat/layout 경로에서 발생한다.

## L1 concat → DRAM-sharded Wo negative result

`TT_METAL_SDPA_DECODE_GROUPED_CONCAT_L1_WO_PROGRAM=1` opt-in으로 grouped concat output을 L1에 유지하고,
Wo weight만 DRAM width-sharded layout으로 추가 로드했다. 목적은 exact mode의
`sharded_to_interleaved(..., DRAM_MEMORY_CONFIG)` bridge를 제거하는 것이었다.

첫 실행은 기존 `ATTN_OUTPUT_PROGCFG`의 20-compute-core 계약과 global MLP fanout-2 factory의
12-reader/12-compute-worker 계약이 달라 output이 유효하지 않았다. Wo 전용 program config를 실제
`6 DRAM endpoints × fanout 2 = 12 workers`에 맞춘 뒤 실행과 device close는 정상 완료됐다.

동일 isolated runner의 actual batch row 0을 exact-control과 비교한 결과:

| 항목 | 결과 |
|---|---:|
| Exact | false |
| Changed values | 545/3,072 |
| Max abs | 0.00146484375 |
| Mean abs | 0.0000361204 |
| PCC | 0.999996662 |
| L1-sharded-Wo smoke latency | 4.345214 ms |
| Exact-control smoke latency | 4.402932 ms |

두 latency 값은 각 warmup 1회, measured 1회뿐이므로 성능 개선 근거로 사용하지 않는다. DRAM-sharded
Wo matmul은 generic interleaved Wo matmul과 reduction/partition 순서가 달라 bit-exact를 보존하지 못한다.
따라서 이 경로는 stable exact 구성으로 채택하지 않는다.

관련 artifact:

- `perf_grouped_l1_sharded_wo_contract_smoke.json`
- `output_grouped_l1_sharded_wo_contract.pt`
- `perf_grouped_exact_contract_control.json`
- `output_grouped_exact_contract_control.pt`

## L1 concat → gather-in0 Wo operator-fusion trial

`TT_METAL_SDPA_DECODE_GROUPED_CONCAT_GATHER_WO=1` opt-in을 추가했다. 이것은 단일 monolithic kernel로
concat과 Wo를 합친 것은 아니다. 그러나 grouped concat이 만든 distributed L1 shards를 Wo matmul의
`gather_in0` reader가 직접 ring-gather하므로 중간 DRAM materialization을 없앤 dataflow/operator fusion이다.

Llama 3.2 3B는 Q heads가 24개다. `heads_per_core=2` grouped concat의 실제 producer는 16개가 아니라
12 cores다. 검증된 계약은 다음과 같다.

```text
worker grid: 5×4
concat/gather cores: first 12 row-major cores
core ranges: (0,0)-(4,1), (0,2)-(1,2)
input/output shard: [32, 256], WIDTH_SHARDED L1
Wo per_core_N: 8 tiles
Wo weight: DRAM interleaved
```

초기 16-core 가정은 input/output grid mismatch로 host validation에서 종료됐다. 12-core 계약으로 교정한
run은 exit 0, `measured_single_layer_decode_complete`, driver close를 모두 통과했다. Isolated layer의 유효
row 3,072 values는 exact-control과 bit-exact였다. Padding rows 1–31은 미초기화 영역이라 correctness에서
제외했다.

그러나 fresh 28-layer gate는 실패했다.

| 항목 | 결과 |
|---|---:|
| Attention Wo exact layers | 0/28 |
| Attention Wo changed values | 77,869 |
| Attention Wo max abs | 0.0013427734375 |
| Residual exact layers | 1/28 |
| Residual changed values | 77,199 |
| Residual max abs | 0.375 |
| Final logits changed values | 114,649/128,256 |
| Final logits max abs | 0.34375 |
| Final logits PCC | 0.9993150234 |

Isolated row exact는 layer-0 residual BF16 rounding이 작은 Wo 차이를 가린 결과다. Full-model 결과는
`gather_in0`의 K-fragment accumulation order가 baseline interleaved Wo와 다름을 보여준다. 따라서 이
경로는 기능적으로 실행 가능한 fusion prototype이지만 stable exact 구성으로 채택하지 않는다. 단일
smoke latency `4.500277 ms`는 warmup 1회, measured 1회라 성능 근거로 사용하지 않는다.

## Decision

- Original grouped-only mode는 full-model exact gate 실패다.
- Exact Wo-input mode는 full-model bit-exact와 interleaved single-layer `2.60%` 개선을 통과했다.
- Gather-in0 Wo mode는 DRAM bridge를 제거했지만 full-model exact gate 실패다.
- 아직 opt-in으로 유지한다.
- 다음 단계는 28-layer full decode latency를 동일-process baseline/exact로 비교하는 것이다.

## Artifacts

```text
/home/iris_hb4/profiler_runs/llama32_3b_64k_grouped_concat_first_divergence_2026_08_05/
  grouped_off/
  grouped_on/
  grouped_on_exact_wo_input/
  grouped_on_exact_fresh_balanced_k256/
  grouped_on_gather_wo_k256/
  perf_grouped_gather_wo_smoke.json
  comparison.json
  comparison_exact_wo_input.json
  perf_baseline.json
  perf_baseline_2.json
  perf_grouped_exact_wo_input.json
  perf_grouped_exact_wo_input_2.json
  perf_pooled_summary.json
  perf_interleaved_grouped_exact_ab.json
```

Checksums:

- `comparison_exact_wo_input.json`:
  `36fa1f7fbf6d82d29dac5d9b234d66039aa6067debb5387c9e3d52811a59a38a`
- `perf_interleaved_grouped_exact_ab.json`:
  `2ba4e0cdd06d838b8a68007036ebf1811565dd2c9cc0eed5ec1326a79d67c702`

Audit patches:

- `/home/iris_hb4/tmp/codex-patches/20260805-125000-grouped-concat-exact-wo-input.patch`
  - SHA-256: `3c6f13b93ffc00911c38305c293563851206fbef0a6d4203634758f4294cf773`
- `/home/iris_hb4/tmp/codex-patches/20260805-130000-interleaved-grouped-exact-ab-runner.patch`
  - SHA-256: `fa36c90f290f20cda35a3456b07956f98fff8f0b73ad8b7a71b8af5f7c55af7b`
- `/home/iris_hb4/tmp/codex-patches/20260805-124635-grouped-l1-sharded-wo.patch`
  - SHA-256: `81a30d8ec89c17d9f547edc4fb71fa3b60f7e2b303a40cf07372e9c6f5510aec`
- `/home/iris_hb4/tmp/codex-patches/20260805-125125-grouped-l1-wo-worker-contract.patch`
  - SHA-256: `623e0d4b44ad107189054fa535d1681cec90b0817c6d19d5b09fedfbbc744c3b`
- `/home/iris_hb4/tmp/codex-patches/20260805-131000-grouped-concat-gather-wo.patch`
  - SHA-256: `de16f7571081de1a65150e62d8c13afee2ce30acd444ca336c887da0768070d7`
- `/home/iris_hb4/tmp/codex-patches/20260805-131430-grouped-gather-output-shard.patch`
  - SHA-256: `a426761a9c1bfdd61177ee6a44e693412165f5836171c9bce96baa6ccfd75027`
- `/home/iris_hb4/tmp/codex-patches/20260805-131700-grouped-gather-matching-grid.patch`
  - SHA-256: `dba121cf5c82a923f7e17d19a415aade7c4b3d0ae657c900cf495d15c3e02deb`
- `/home/iris_hb4/tmp/codex-patches/20260805-132300-grouped-gather-12-core-contract.patch`
  - SHA-256: `b46d3eb05b55ba94753920443a2109380d91214081b81149b32f792e485a405c`
- `/home/iris_hb4/tmp/codex-patches/20260805-132600-remove-grouped-gather-grid-marker.patch`
  - SHA-256: `26710d5db4f5c5599c71e31fa734481450fe137ddbe5339b59b014e24a0a85a6`

## Limitations

- Actual 65,535-token prefill은 실행하지 않았다. KV cache는 cur-pos-only test 구성이다.
- 28-layer full-model correctness는 측정했지만 full-model latency는 아직 측정하지 않았다.
- NoC profiler와 DRAM bandwidth capture는 사용하지 않았다.
- P100/P150 runtime log는 board identity가 아니다.

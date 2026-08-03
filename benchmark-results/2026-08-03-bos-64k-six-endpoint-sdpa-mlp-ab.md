# BOS Llama 3.2 3B 64K SDPA·MLP 6-endpoint A/B

측정일: 2026-08-03 UTC

## 요약

Llama 3.2 3B layer 0의 curpos-only 64K decode에서 검증된 기존 방식을 재현했다. SDPA는 active
reader 16개를 유지한 채 DRAM worker endpoint 6개와 두 NoC에 분산했고, MLP는 DRAM-sharded weight
data path의 interface worker 6개와 W2 `in0_block_w=16`을 사용했다.

vanilla K256 대비 SDPA 6-endpoint만 적용하면 layer 중앙 latency가 18.65% 감소했고, MLP까지 결합하면
26.66% 감소했다. 처리율의 역수 기준 증가는 각각 22.92%, 36.35%다. 모든 비교 output은 동일 shape,
finite이며 bit-exact 일치했다.

## 구성 구분

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- SDPA program의 operation active cores: 16개
- SDPA active DRAM readers: 16개
- physical DRAM topology: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- SDPA endpoint load: `3/2/3/3/3/2`
- SDPA NoC load: NoC0/NoC1 `8/8`
- MLP program grid 설정: 4×4
- MLP runtime log: `Dram Interface Workers: 6`

`Dram Interface Workers: 6`은 선택된 MLP matmul data path의 interface-worker 수다. 이를 physical bank
수, tensor shard 수 또는 operation active-core 수와 동일하게 해석하지 않는다. runtime의 P150 표시는
custom BOS hardware identity가 아니라 `model_config.py` heuristic 결과다.

## 측정 조건

- 모델: `meta-llama/Llama-3.2-3B-Instruct`
- layer: 0, attention부터 MLP까지 포함한 single decoder layer
- context/curpos: 65,536 / 65,535
- KV layout: paged, block size 32
- SDPA K chunk: 256 tokens
- cores per KV head: 2
- precision mode: performance
- prefill: 실행하지 않음; 64K KV cache를 준비하고 decode만 측정
- warmup: 3회
- sample: 3 iterations 평균 × 5 repeats
- profiler/Watcher: 사용하지 않음
- timeout: `SIGINT`, 120초 또는 MLP 결합 run 180초, `--kill-after=15s`

## 결과

| 구성 | Layer mean (ms) | Layer median (ms) | Layer min (ms) | SDPA wall mean (ms) |
|---|---:|---:|---:|---:|
| vanilla K256 | 6.690407 | 6.686590 | 6.640358 | 3.491338 |
| SDPA 16-reader, 6-endpoint | 5.440844 | 5.439694 | 5.415174 | 2.248903 |
| SDPA 6-endpoint + MLP 6-interface/block16 | 4.892870 | 4.904165 | 4.857650 | 2.260430 |

| 비교 | Layer latency 감소 | 역수 처리율 증가 |
|---|---:|---:|
| vanilla → SDPA 6-endpoint | 18.65% | 22.92% |
| SDPA 6-endpoint → MLP 결합 | 9.84% | 10.92% |
| vanilla → SDPA+MLP 결합 | 26.66% | 36.35% |

SDPA wall 자체는 vanilla 대비 35.59% 감소했다. MLP 결합 run의 SDPA wall은 SDPA-only보다 0.51%
높았으며, 이는 별도 process에서 얻은 작은 session 간 변동 범위로 본다. MLP 추가 효과는 whole-layer
중앙 latency 차이로 평가했다.

## Correctness

첫 single run에서 output tensor를 저장해 host에서 비교했다.

- SDPA 6-endpoint output 대 vanilla K256 output: `MAX_ABS=0`, `MEAN_ABS=0`, `PCC=1.0`
- MLP 결합 output 대 SDPA-only output: `MAX_ABS=0`, `MEAN_ABS=0`, `PCC=1.0`
- 공통 shape: `[1, 1, 32, 3072]`
- 모든 값 finite

## 핵심 환경변수

```bash
TT_METAL_SDPA_DECODE_DUAL_NOC=1
TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6
TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1
TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1
TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0
```

MLP 결합 run에는 다음을 추가했다.

```bash
TT_METAL_MLP_DRAM_SHARDED=1
TT_METAL_MLP_W2_IN0_BLOCK_W=16
```

runner:

```text
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py
```

공통 runner 인자:

```text
--context-len 65536 --chunk-size 2048 --sdpa-k-chunk-size 256
--cores-per-kv-head 2 --kv-layout paged --precision-mode performance
--warmup 3 --iterations 3 --repeats 5
```

## Artifact

- vanilla JSON: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_6endpoint_ab_2026_08_02_18_30_00/layer_k256_control.json`
- SDPA 6-endpoint JSON: `/home/iris_hb4/benchmark_runs/sdpa_16reader_6endpoint_2026_08_03/k256_measured_3x5.json`
- SDPA+MLP JSON: `/home/iris_hb4/benchmark_runs/sdpa_16reader_6endpoint_2026_08_03/k256_mlp6_block16_measured_3x5.json`
- correctness tensors: 같은 2026-08-03 run 디렉터리의 `k256_single.pt`와
  `k256_mlp6_block16_single.pt`

## 6-endpoint와 실패한 6-reader POC의 구분

성공 구성은 16개 active reader가 각자 DRAM에서 local CB로 읽고 endpoint만 6개로 분산한다. 실패한
`TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=1` POC는 reader를 6 owner로 줄이고 나머지 10 consumer에
GlobalCircularBuffer로 K/V를 전달하는 별도 구조다. 이 구조는 2026-08-03 첫 warmup 이전에 정지하고
timeout cleanup이 exit 137로 끝났으므로 성능 후보에서 제외한다. 6-endpoint 성공 결과를 6-reader
remote-CB 결과로 표기하지 않는다.

## 한계

- vanilla와 최적화 결과는 동일 runner/shape/반복 조건이지만 서로 다른 process/session에서 측정했다.
- single layer, batch 1 decode 결과이며 full model token/s를 직접 측정한 값이 아니다.
- MLP DRAM-sharded path는 curpos-only decode에서 검증했다. 기존 full-demo prefill에는 weight layout
  validation 문제가 있어 그대로 적용할 수 없다.
- 이번 run은 latency와 correctness 측정이며 NoC/DRAM saturation profile을 새로 수집하지 않았다.
- SDPA는 20-core grid를 사용할 수 있어도 실제 active compute/readers가 16개인 구성이다.

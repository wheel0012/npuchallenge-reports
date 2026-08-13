# BOS attention QKV·Wo DRAM-sharded A/B

## 결론

Llama 3.2 3B layer-0의 64K decode에서 QKV와 Wo weight reader를 MLP의
balanced fanout-2 DRAM-sharded data path로 옮겼다. 두 matmul kernel은 각각
`35.64%`, `29.74%` 빨라졌다. 전체 layer device makespan 개선은 `1.41%`다.
동일-session 28-layer 50-token throughput는 `7.772611→8.213245 tok/s`, `+5.67%`다.
fixed-input full-model logits는 top-1과 top-5가 유지됐지만 bitwise exact가
아니다. 사용자 결정에 따라 PCC `0.999` 수준을 허용하는 best-stable profile로 승격했다.

## 장치와 topology

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- QKV/Wo DRAM-sharded path: 6 DRAM interface workers, 12 reader/compute workers
- physical DRAM: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- endpoint grouping: `4:4:4`

`Dram Interface Workers: 6`은 이 matmul data path의 interface-worker 수다.
physical bank 수나 전체 program active-core 수와 같은 값으로 해석하지 않는다.

## 구현

파일: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tt/attention.py`

### QKV

`TT_METAL_ATTN_DECODE_DRAM_SHARDED_QKV=1`에서 decode 전용 fused QKV weight를
DRAM width-sharded 형태로 적재한다. prefill은 기존 interleaved weight를 유지한다.
QKV logical shape는 `[3072, 5120]`이다. 6 endpoints × fanout-2 padding 때문에
physical shard width는 896이다. activation K partition은 기존 16-way contract를
유지한다. DRAM factory의 실제 reader/compute worker는 12개다.

### Wo

기존 `TT_METAL_SDPA_DECODE_GROUPED_CONCAT_L1_WO_PROGRAM=1` 경로를 사용한다.
Wo logical shape는 `[3072, 3072]`, physical DRAM shard width는 512다.
grouped concat 출력에서 DRAM-sharded Wo matmul로 직접 진입한다.

공통 reader path는 6 interface workers, fanout-2, 12 readers, 16 KiB capped
row-burst, balanced endpoint group `4:4:4`다.

## 측정법

- model: `meta-llama/Llama-3.2-3B-Instruct`
- layer: 0 한 개
- batch: 1, tile padded rows 32
- paged KV: synthetic 64K, current position 65,535
- warmup: 1회
- measured: signpost 내부 1회
- baseline: 2026-08-09 best-stable exact-Wo-input trace
- optimized: QKV DRAM-sharded + grouped L1 Wo DRAM-sharded
- timeout: `300s`, exit 0, 정상 device close

full-model throughput는 같은 source/session에서 각 profile warmup 3 tokens,
measured 50 tokens로 1회씩 측정했다. synthetic zero-initialized 64K KV를 사용했다.

단일 op 비교는 device kernel duration을 사용했다. layer latency는 op별 FW
duration 합이 아니라 signpost 내부 첫 device FW start부터 마지막 FW end까지의
cycle span을 650 MHz로 변환했다. producer-consumer overlap 때문에 FW duration을
단순 합하면 QKV bridge와 Wo reshard 시간이 중복된다.

## 결과

| 항목 | Baseline | Optimized | 변화 |
|---|---:|---:|---:|
| QKV matmul kernel | 433.448 us | 278.943 us | -35.64% |
| QKV packed effective weight BW | 38.55 GB/s | 59.91 GB/s | +55.41% |
| Wo matmul kernel | 236.042 us | 165.835 us | -29.74% |
| Wo packed effective weight BW | 42.48 GB/s | 60.46 GB/s | +42.33% |
| layer device makespan | 3904.725 us | 3849.635 us | -55.090 us, -1.41% |
| full-model latency/token | 128.656900 ms | 121.754557 ms | -5.36% |
| full-model throughput | 7.772611 tok/s | 8.213245 tok/s | +5.67% |

QKV의 `ShardedToInterleaved` kernel 자체는 약 4 us다. optimized FW span은
151.943 us지만 QKV matmul과 겹친다. Wo 뒤 reshard도 kernel은 3.645 us이며
167.448 us FW span 대부분이 Wo producer와 겹친다. `op_perf_results.py`의 FW
duration 합은 baseline `4661.931 us`, optimized `4714.258 us`로 역전되지만
critical-path makespan이 아니다.

## 정확도

동일 fixed input의 full-model logits A/B:

- exact: false
- changed: `98,097 / 128,256`
- PCC: `0.9992725253`
- max absolute difference: `0.40625`
- mean absolute difference: `0.04416167736`
- top-1 token: baseline/optimized 모두 `320`
- top-5 overlap: `5/5`

QKV와 Wo를 동시에 켠 결과다. QKV activation K partition은 16-way로 유지된다.
DRAM-sharded factory의 reader/compute workers는 12개다. 따라서 단순한 전체
compute core 16→12 축소가 아니다. weight tile 소유권, multicast, core별 partial
accumulation과 reduction 순서가 바뀌어 BF8/BF16 부동소수점 비결합성에 따른
rounding drift가 생긴 것이 가장 강한 원인 추론이다. 두 op별 기여는 분리하지 않았다.
bit-exact 요구 profile은 README의 exact fallback을 사용한다.

## 재현 환경

현재 Best-stable 환경은 아래 값을 포함한다.

```bash
export TT_METAL_ATTN_DECODE_DRAM_SHARDED_QKV=1
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT_EXACT_WO_INPUT=0
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT_L1_WO_PROGRAM=1
```

runner:

```text
models/bos_model/llama32/run_llama32_3b_64k_single_layer_profile.py
```

## Artifact

- run root: `/home/iris_hb4/profiler_runs/attn_qkv_wo_dram_sharded_ab_2026_08_09_15_05_00`
- optimized Tracy/device report: `perf_optimized/reports/attn_qkv_wo_dram_sharded/2026_08_09_15_30_31`
- baseline profile: `/home/iris_hb4/profiler_runs/llama32_3b_64k_stable_layer0_visualizer_2026_08_09_14_27_41`
- correctness: `correctness_baseline/final_logits.pt`, `correctness_optimized/final_logits.pt`
- source patches: `/home/iris_hb4/tmp/codex-patches/20260809-150000-attn-qkv-dram-sharded-optin.patch`, `/home/iris_hb4/tmp/codex-patches/20260809-152500-attn-qkv-program-16core-fix.patch`

## 한계와 다음 단계

- QKV-only, Wo-only logits A/B는 아직 분리하지 않았다.
- layer 개선폭이 작다. 64K에서는 SDPA 약 2.04 ms가 여전히 가장 크다.
- 승격 뒤에도 full-model 반복 성능과 generation 품질을 계속 감시한다.

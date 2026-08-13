# BOS Llama 3.2 3B 56K all-BF16 layer-0 performance profile

- Date: 2026-08-12
- Commit/build: `62f2e0309526a314447bb07c9bd96cb5d83a4f9e` + uncommitted profiler-runner BF16 option
- Device and topology: custom 20-core BOS NPU; Blackhole runtime; 5×4 workers; 3 physical DRAM banks; 2 worker NoC endpoints per bank; 6 endpoints total
- Status: verified

## Question

All-BF16의 allocated footprint가 quantized 64K와 비슷해도 decode throughput가 크게 낮은 이유를 layer-0 device profile과 useful-payload traffic으로 확인한다.

## Configuration

- Model: `meta-llama/Llama-3.2-3B-Instruct`, layer 0, batch 1
- Context: 57,344 tokens; decode position 57,343
- KV: paged layout, block 32 tokens, 1,792 blocks, synthetic zero KV; 실제 prefill 생략
- Precision: 모든 `TensorGroup` BF16; math fidelity는 기존 HiFi2/HiFi4 설정 유지
- SDPA: K-chunk 256, dual NoC, endpoint count 6, pair/bank balancing, grouped concat
- Attention projection: QKV/Wo DRAM-sharded stable path
- MLP: DRAM-sharded, W2 input block width 16, 16 KiB read page, fanout-2 tagged, balanced endpoints
- 금지 경로: reduce-only helper, fanout-3, endpoint-local, TurboQuant 모두 0
- 5×4는 available grid다. CSV의 active core count는 op별로 다르다. 6 endpoint는 3 physical banks와 같은 수가 아니다.

## Method

1. 동일 runner와 kernel cache로 profiler-free warmup 1회, measured 1회 gate를 통과했다.
2. Tracy/device profiler에서 warmup 1회를 제외하고 signpost 내부 measured call 1회만 `ops_perf_result.csv`로 잘랐다.
3. NoC trace는 사용하지 않았다.
4. TTNN detailed-buffer capture를 별도로 1회 실행해 Visualizer memory DB를 만들었다. 이 capture latency는 사용하지 않는다.
5. 모든 실행은 exit 0, `measured_single_layer_decode_complete`, `DEVICE_CLOSED`를 확인했다.

## Results

| Metric | Result |
|---|---:|
| Profiler-free layer wall | 7.046747 ms |
| Tracy run layer wall | 7.144251 ms |
| Tracy run SDPA wall | 3.086033 ms |
| Measured operations | 22 |
| Device FW duration sum | 8.821102 ms |
| Attention-side FW sum | 4.363399 ms |
| MLP-side FW sum | 4.457703 ms |

Device FW duration sum은 병렬·중첩 op를 더한 값이다. layer wall과 직접 같지 않다.

주요 op:

| Op | Device FW duration |
|---|---:|
| QKV matmul | 505.892 µs |
| SDPA | 2,845.089 µs |
| Wo matmul | 291.657 µs |
| W1 matmul | 802.935 µs |
| W3 matmul | 1,481.958 µs |
| SwiGLU | 698.403 µs |
| W2 matmul | 764.172 µs |

BF16 useful payload estimate:

| Source | Bytes/layer/token |
|---|---:|
| QKV weight | 31,457,280 |
| Wo weight | 18,874,368 |
| W1/W3/W2 weights | 150,994,944 |
| K+V cache at 57,344 | 234,881,024 |
| Total | 436,207,616 |

`436,207,616 B / 7.144251 ms = 61.06 GB/s`다. 이는 weight를 layer당 1회, K와 V를 각각 1회 읽는다고 둔 useful-payload estimate다. 실제 DRAM counter가 아니다.

## Interpretation

### Observed

- SDPA와 MLP가 layer FW 합의 대부분이다.
- BF16 layer는 약 436.2 MB의 핵심 weight/KV payload를 token당 읽는다.
- LoFi가 아니라 기본 fidelity에서도 effective useful-payload가 약 61.1 GB/s다.
- Visualizer CSV의 `DRAM BW UTIL (%)`와 `NOC UTIL (%)`는 비어 있다. NoC/NPE capture를 하지 않았기 때문이다.

### Inference

Allocated footprint가 비슷해도 traffic/token은 같지 않다. 56K로 context를 줄여 BF16 KV와 weight를 capacity 안에 넣었지만, 실행 중 읽는 BF16 element는 2 bytes다. BF8/BFP8 계열은 tile metadata가 있어 정확히 절반은 아니어도 핵심 payload가 더 작다. 따라서 pack/unpack 제거 이득보다 DRAM service bytes 증가가 크며 throughput가 낮아진다.

61.06 GB/s는 microbenchmark peak가 아니라 model-useful payload 기준이다. small tensor, page table, norm parameter, activation movement, reread와 write traffic을 제외하므로 실제 bus traffic과 utilization을 판정할 수 없다.

## Reproduction command

Stable opt-in은 repository `README.md`의 `Best-stable opt-in 실행` 블록을 적용한다. 핵심 measured runner:

```bash
export TT_METAL_LLAMA32_SDPA_DECODE_K_CHUNK_SIZE=256
export TT_METAL_SDPA_DECODE_DUAL_NOC=1
export TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6
export TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1
export TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1
export TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0
export TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1
export TT_METAL_ATTN_DECODE_DRAM_SHARDED_QKV=1
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT_EXACT_WO_INPUT=0
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT_L1_WO_PROGRAM=1
export TT_METAL_SDPA_DECODE_GROUPED_CONCAT_GATHER_WO=0
export TT_METAL_MLP_DRAM_SHARDED=1
export TT_METAL_MLP_W2_IN0_BLOCK_W=16
export TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1
export TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0
export TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1
export TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1
export TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1
export TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0
export TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0
export TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=0
export TT_METAL_MLP_FUSED_GATE_UP=0
export TT_METAL_TURBOQUANT=0

timeout --signal=INT --kill-after=15s 180s /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy -p -r -v --check-exit-code --sync-host-device models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py --context-len 57344 --chunk-size 2048 --sdpa-k-chunk-size 256 --kv-block-size 32 --cores-per-kv-head 2 --kv-layout paged --precision-mode bf16 --warmup 1 --iterations 1 --repeats 1
```

실제 실행은 profiler safety contract에 따라 `timeout`의 direct child를 Tracy Python process로 두고 180초 상한을 사용했다.

## Artifact paths

Run root:

`/home/iris_hb4/profiler_runs/llama32_3b_56k_bf16_layer0_perf_2026_08_12_10_40_00`

```text
memory_report_visualizer/
├── config.json
└── db.sqlite
perf_report_visualize/
├── ops_perf_result.csv
├── profile_log_device.csv
└── tracy_profile_log_host.tracy
```

- Gate: `gate-result.json`
- Tracy result: `profile-result.json`
- Memory capture result/log: `memory-result.json`, `memory.log`
- Memory DB: integrity `ok`, 126 recorded operations

## Limitations and next steps

- synthetic zero paged KV다. 실제 57,343-token prefill data dependency와 output PCC를 검증하지 않았다.
- single layer, single measured sample이다. 분산값이 없다.
- memory capture는 logging 때문에 3.600 s였다. 성능값으로 쓰면 안 된다.
- hardware DRAM BW/utilization 확정에는 profiler-free correctness 뒤 isolated NoC/NPE capture가 별도로 필요하다.
- BF8/BFP8와 동일 56K, 동일 runner, 동일 signpost의 direct A/B가 다음 실험이다.

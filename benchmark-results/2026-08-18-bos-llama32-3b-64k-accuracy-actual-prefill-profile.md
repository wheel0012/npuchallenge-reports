# BOS Llama 3.2 3B 64K accuracy actual-prefill stable layer-0 profile

- Date: 2026-08-18
- Commit/build: `62f2e0309526a314447bb07c9bd96cb5d83a4f9e`; runtime `_ttnn.so` SHA-256 `dc74b2b027fe28d01d459b97a9af9097c9629b6a6ae295e2fad7e37f61fdeefb`
- Device and topology: custom 20-core BOS NPU; Blackhole runtime/code architecture; 5×4 available workers; 3 physical DRAM banks and 6 worker NoC endpoints
- Status: verified

## Question

Run the final stable optimization-waterfall configuration with the accuracy preset and an actual
65,535-token prefill, then regenerate the layer-0 decode performance and TTNN Visualizer memory
artifacts.

## Configuration

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Layer / batch | layer 0 / batch 1 |
| Precision preset | `accuracy` |
| Context | 65,536 positions |
| Actual prefill | 65,535 tokens, 32 chunks × 2,048 |
| Paged KV | 2,048 page blocks × 32 tokens |
| Warmup | one decode after prefill |
| Measured interval | exactly one layer-0 decode between signposts |
| Signpost | `llama32_3b_64k_accuracy_mode_actual_prefill_single_layer_decode` ... `_end` |

Stable opt-ins were K-chunk 256, SDPA dual-NoC and six endpoints with pair/bank-balanced
assignment, grouped concat with L1-sharded Wo input, DRAM-sharded QKV/Wo, and MLP DRAM-sharded
12-reader fanout-2 with balanced endpoints, tagged two-block issue, 16 KiB read pages, and W2
input block width 16. Experimental reduce-helper, six-reader-sharded SDPA, fanout-3,
endpoint-local MLP, fused gate/up, TurboQuant, and SnapKV paths were disabled.

The accuracy preset is not storage-equivalent to the earlier performance preset; in particular,
the relevant MLP weight path uses BFP8 rather than the performance BFP4 path. This report
therefore does not treat their latency difference as the isolated cost of actual prefill.

## Method

Three independent processes used the same source, build, model configuration, and opt-ins.

1. A profiler-free gate completed all 32 prefill chunks, performed warmup, executed the measured
   layer once, and closed the device normally.
2. A direct-child Tracy invocation captured host and device profiling around the one measured
   layer. The delivered operation CSV contains the 22 rows strictly between the signposts and
   retains all 124 source columns.
3. A separate TTNN detailed-buffer run generated the memory database. Its synchronized execution
   time is intentionally excluded from performance results.

| Source | SHA-256 |
|---|---|
| profile runner | `20c7f0501b6c1337566d4d676ba81eeb1d402270dab462b8d3c860ef6fcf99e9` |
| `attention.py` | `ef963fbfad86f6d52090faa5aa546cc9a7a28614db5a01a9e8d548f6367b8a8d` |
| `mlp.py` | `b956d3107228d79528ac2426dd64cd6a8c182954508b3f65b8708a852757ccf8` |

## Results

### Layer-0 device-operation waterfall

| Region | FW duration | Share of summed FW duration |
|---|---:|---:|
| Full measured layer, 22 ops | 5,425.120 µs | 100.00% |
| Attention sublayer, first 14 ops | 2,901.671 µs | 53.49% |
| SDPA operation | 2,040.840 µs | 37.62% |
| MLP sublayer, last 8 ops | 2,523.449 µs | 46.51% |
| W1 + W3 + SwiGLU + W2 | 2,096.316 µs | 38.64% |

The largest individual operations were SDPA 2,040.840 µs, W3 812.235 µs, W1 450.802 µs,
W2 429.794 µs, SwiGLU 403.485 µs, QKV matmul 277.497 µs, and Wo matmul 167.766 µs.
The reciprocal of the single-layer summed FW duration is 184.33 layer-equivalents/s; it is not
full-model token throughput.

### Historical comparison boundary

| Run | Precision / KV state | Layer FW sum |
|---|---|---:|
| Current | accuracy / actual 65,535-token prefill | 5.425120 ms |
| 2026-08-09 stable profile | performance / synthetic 64K paged KV | 4.661931 ms |
| 2026-08-09 final waterfall stage | performance / synthetic 64K paged KV | 4.663924 ms |

The current sum is 16.37% above the prior stable-profile sum. Because precision mode and KV
construction both changed, this is a deployment-condition comparison, not a causal A/B for
actual prefill or accuracy mode alone.

### Memory-report validation

The delivered SQLite database passed `PRAGMA integrity_check` with `ok` and has zero rows in
`errors`.

| Table | Rows |
|---|---:|
| operations | 1,823 |
| tensors | 1,301 |
| buffers | 114,856 |
| buffer_pages | 29,928 |
| device_tensors | 1,216 |
| input_tensors | 2,646 |
| output_tensors | 1,216 |
| stack_traces | 1,823 |
| devices | 1 |

## Interpretation

The stable optimized path survives a real 64K prefill in accuracy mode and produces a complete
Visualizer delivery. SDPA remains the largest single operation, while the complete MLP sublayer
is nearly half of the summed layer time. These timings establish an accuracy/actual-prefill
baseline for future like-for-like optimization tests; they do not replace the earlier
performance-mode throughput waterfall.

All three device processes completed normally with exit code 0 and a normal device close. No
timeout, signal termination, or quarantine-triggering event occurred.

## Reproduction command

From `/home/iris_hb4/tt-metal-hb4`, activate `python_env`, export the stable opt-ins listed
above, and use the checked runner:

```bash
python models/bos_model/llama32/tt/run_llama32_3b_actual_prefill_single_layer_profile.py \
  --context-len 65536 --chunk-size 2048 --precision-mode accuracy
```

For the performance capture, the timeout direct child was the Tracy Python process:

```bash
python -m tracy -p -r -v --check-exit-code \
  -o <run>/perf_capture \
  -n llama32_3b_64k_accuracy_actual_prefill_stable_layer0 \
  models/bos_model/llama32/tt/run_llama32_3b_actual_prefill_single_layer_profile.py \
  --context-len 65536 --chunk-size 2048 --precision-mode accuracy
```

The memory process additionally set `TTNN_CONFIG_OVERRIDES` with logging and
`enable_detailed_buffer_report=true`, and with fast runtime mode disabled. Performance and
memory captures must remain separate.

## Artifact paths

Run root:

`/home/iris_hb4/profiler_runs/llama32_3b_64k_accuracy_actual_prefill_stable_visualizer_2026_08_18_05_50_00`

Visualizer contract:

```text
memory_report_visualizer/
├── config.json
└── db.sqlite
perf_report_visualize/
├── ops_perf_result.csv
├── profile_log_device.csv
└── tracy_profile_log_host.tracy
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `memory_report_visualizer/config.json` | 969 | `46bb08a7509b2fd8260bcf654609a9d04f8be23301d84d7cbdb27c226a33695e` |
| `memory_report_visualizer/db.sqlite` | 17,670,144 | `bf7949644623d8ba3fb22631a9ef3ac666bf3323d5de6ec739b960a38787496f` |
| `perf_report_visualize/ops_perf_result.csv` | 39,137 | `c0b9fb79ed0ca34725b753218beb8e8d77f70153434316c2337d9c53c25c5461` |
| `perf_report_visualize/profile_log_device.csv` | 126,675,561 | `943b6b8ab2483d9f92fb9e15f8dad9fbb701f43939e32a17150929260101240f` |
| `perf_report_visualize/tracy_profile_log_host.tracy` | 7,281,590 | `1a007cf8a24e4e487be7ae92b05940dcc2d0907fe0c56219cadfc4c326931740` |
| `perf_summary.txt` | 1,027 | `03dc4b242fee6778f258864f419aaee3d16ee2bc3c6e034cc7e4d063c3311211` |

## Limitations and next steps

- This is one measured layer call after one warmup, not a repeated latency distribution.
- FW duration is the sum of device-operation durations, not end-to-end wall-clock token time.
- The memory run intentionally enables intrusive detailed reporting and cannot provide latency.
- Actual prefill uses deterministic generated token IDs, not a natural-language 64K corpus.
- This run did not perform a new output PCC or bit-exact comparison.
- No NoC/NPE trace was requested or generated.
- An accuracy-mode synthetic-KV control is required to separate actual-prefill state effects from
  precision-mode effects.

# Profiler runs

Profiler artifact root managed according to `/home/iris_hb4/QWEN_AND_PROFILER_README.md`.
Performance and synchronization experiments are stored as separate runs; failed runs are retained with their failure reason.

| Run | Status | Purpose | Key result |
|---|---|---|---|
| `llama31_8b_speculative_4lane_sdpa_tracy_2026_07_22_07_39_07` | success | 1B draft + 8B target, 4-lane, 47-token prompt, one round | decode SDPA kernel 2.229 ms / all decode kernels 321.103 ms = 0.694% |
| `llama31_8b_speculative_4lane_sdpa_wall_2026_07_22_07_43_54` | success, intrusive | per-layer synchronize upper-bound measurement | 1B 15.496 ms + 8B 12.173 ms; sync overhead included |
| `llama31_8b_vanilla_sdpa_tracy_2026_07_22_07_52_00` | success | vanilla Llama 3.1 8B, batch 1, 47-token prompt | SDPA 6.232 ms / 10 steps; 0.310% of summed decode kernels |
| `llama31_8b_32k_decode_perf_2026_07_22_08_02_00` | success | vanilla Llama 3.1 8B, actual 32K context, decode-only delivery CSV | SDPA 54.721 ms/token; 21.409% of summed decode kernels |
| `llama31_8b_32k_decode_memory_2026_07_22_08_12_00` | incomplete | full-model 32K detailed-buffer capture | interrupted; replaced by planned single-layer capture |
| `llama31_8b_single_layer_long_context_notracy_2026_07_22_08_26_00` | partial | Llama 8B layer 0 synthetic long-context, no Tracy | 32K 9.097 ms; 128K kernel hang, reset required |
| `llama31_8b_48k_single_layer_decode_perf_2026_07_22_13_52_00` | success | Llama 8B layer 0, synthetic 48K paged KV, decode-only Tracy | layer 8.835 ms; SDPA 2.637 ms (29.844%) |
| `llama31_8b_vanilla_sdpa_wall_failed_2026_07_22_07_46_22` | failed | vanilla 8B synchronized SDPA attempt | synchronize invoked during trace capture; values are invalid |
| `llama32_3b_64k_accuracy_mode_sdpa_dual_noc_5ep_2026_07_26_02_49_30` | success | Llama 3.2 3B layer 0, actual 65,535-token prefill, accuracy mode | Visualizer memory/perf delivery artifacts; SDPA NPE columns unavailable |
| `sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33` | success, microbenchmark | isolated vanilla paged SDPA decode at 64K with NoC trace/NPE | 3.467 ms; logical 38.72 GB/s; NPE DRAM BW util 4.3% |
| `mlp_decode_dram_sharded_w2_block16_noc_2026_08_02_09_15_00` | failed before device workload | isolated MLP, DRAM-sharded + W2 block 16, warmup 1 + measured 1 NoC capture | child used `/usr/bin/python3`; no device open, raw trace or ops CSV |
| `mlp_decode_dram_sharded_w2_block16_noc_2026_08_02_09_10_00` | success | isolated MLP, DRAM-sharded + W2 block 16, correctness/JIT 1 + measured 1 NoC capture | W1/W3/W2 45.0–45.9 GB/s; aggregate 45.40 GB/s = microbenchmark peak의 52.28%; six readers balanced but destinations 3:2:1 |
| `mlp_decode_dram_sharded_dual_noc_2026_08_03_05_07_00` | operationally complete, experiment invalid | dual-NoC opt-in capture 의도, warmup 1 + measured 1 | loaded stale `build_Release/lib/_ttnncpp.so`; raw W1은 전부 NOC1이므로 dual-NoC 결과로 사용 금지 |
| `mlp_decode_dram_sharded_w2_block16_page16k_engine_2026_08_03_07_03_30` | success | isolated MLP 16 KiB page, correctness/JIT 1 + measured 1 device-span capture | BRISC와 TRISC가 같은 critical path; W1/W3/W2 kernel 579.502/583.205/563.248 us |
| `mlp_decode_dram_sharded_w2_block16_page16k_counters_2026_08_03_07_05_00` | success | 같은 구성의 all-group performance-counter capture | FPU-active core는 20 program cores 중 6개뿐; 활성 core FPU util W1/W3 39--41%, W2 36--37% |
| `mlp_fanout2_rowburst_counters_2026_08_03_08_30_00` | success | fanout-2 row-burst isolated MLP performance-counter capture | 12 FPU-active cores, 각 약 18.3--19.5%; pack/unpack/L1/instrn raw counter는 미생성 |
| `mlp_fanout2_rowburst_noc_2026_08_03_08_35_00` | success, raw trace only | 같은 구성의 isolated NoC capture | reconstructed aggregate 48.60 GB/s; destinations 6:5:1; tt-npe import 실패로 timeline 없음 |

Visualizer-compatible performance artifacts are under each successful Tracy run's `perf_report_visualize/` directory.

The Llama 3.2 3B 64K SDPA DRAM/NoC methodology and result interpretation are documented in
[`DRAM_BENCHMARK_README.md`](DRAM_BENCHMARK_README.md).

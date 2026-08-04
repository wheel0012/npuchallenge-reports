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
| `mlp_fanout2_rowburst_balanced_noc_2026_08_03_09_15_00` | success, raw trace only | 5×4 partner placement로 NOC1 destination을 4:4:4로 교정한 isolated MLP | aggregate 62.93 GB/s; profiler-free mean 1.472280 ms; PCC 0.999641; tt-npe timeline 없음 |
| `mlp_prefetch_helpers_correctness_2026_08_03_13_30_00` | failed, Watcher abort, exit 137 | 6 compute owner + 6 prefetch helper isolated MLP correctness | 12 readers/6 compute and 4:4:4 confirmed; helper BRISC write barrier 누락; no PCC/close/perf result; device quarantined |
| `mlp_prefetch_helpers_barrier_noc_2026_08_03_15_55_00` | success, raw trace only | barrier-corrected 6-compute + 6-helper isolated NoC capture | PCC 0.999641; 4:4:4 maintained; aggregate 59.58 GB/s; tt-npe timeline 없음 |
| `mlp_prefetch_helpers_barrier_counters_2026_08_03_16_05_00` | success | 같은 구성의 performance-counter capture | 6 FPU-active cores; W1/W3 46.91--49.29%, W2 44.31--46.63%; other requested counter groups 미생성 |
| `mlp_fanout3_18_compute_correctness_2026_08_03_17_35_00` | success, profiler-free | capacity-aware fanout-3 isolated MLP correctness | 18 readers/18 compute, NOC1 groups 7:7:4; PCC 0.999641; normal close/exit 0 |
| `mlp_fanout3_18_compute_latency_2026_08_03_17_40_00` | success, profiler-free | 같은 구성의 5-sample latency | mean 1.703471 ms; balanced fanout-2보다 15.70% 느림; normal close/exit 0 |
| `mlp_fanout3_dual_noc_3x6_correctness_2026_08_03_19_10_00` | failed, timeout/SIGKILL, exit 137 | physical 6 endpoints × 3 readers, NOC0/NOC1 9:9 isolated MLP correctness | mapping confirmed; no PCC/completion/close; fixed-per-endpoint VC violated microbenchmark coloring contract; device quarantined |
| `mlp_fanout3_dual_noc_3x6_vc_correctness_2026_08_03_20_35_00` | failed, timeout/SIGKILL, exit 137 | conflict-free VC corrected physical 6-endpoint MLP correctness | same no-completion signature; VC not sole root cause; no PCC/close; device quarantined |
| `mlp_fanout3_dual_noc_3x6_separate_writer_correctness_2026_08_03_13_31_24` | failed, Watcher + timeout/SIGKILL, exit 137 | reader opposite-NoC output write correctness | 3:3:3:3:3:3 and reader NoCs 9:9 confirmed; no PCC/completion/close; writer NoC coupling not sole root cause |
| `mlp_existing_fanout2_balanced_correctness_2026_08_03_13_41_50` | success, profiler-free | same-build normal-path control after reboot/add gate | NOC1 groups 4:4:4; 12 readers/compute; PCC 0.999641; 1.487526 ms; normal close/exit 0 |
| `mlp_fanout3_split_kernel_only_correctness_2026_08_03_14_02_23` | failed, Watcher + timeout/SIGKILL, exit 137 | standard fanout-3 mapping/addressing with only RISCV0 kernel handles split NOC0/NOC1 | 18 readers/compute and NoCs 9:9 confirmed; no PCC/completion/close; split-handle/kernel-group contract isolated as leading cause; device quarantined |
| `llama32_3b_64k_full_decode_ab_2026_08_03_14_30_00` | success, profiler-free A/B | full 28-layer Llama 3.2 3B decode, synthetic paged 64K KV, 50 measured tokens | vanilla K128 5.123448 tok/s; SDPA 6-endpoint K256 + MLP balanced fanout-2 7.645991 tok/s; +49.24%; both normal close |
| `llama32_3b_64k_full_logits_ab_2026_08_03_15_20_00` | success, profiler-free correctness A/B | fixed seed/token/position, one full-model decode logits comparison | not bitwise exact; PCC 0.999326, max abs 0.3125; same top-1 and 5/5 top-5 overlap; both normal close |
| `mlp_triple_buffer_stall_2026_08_03_16_16_25` | success | balanced fanout-2 isolated MLP, profiler-free gate + device-zone capture | weight CB already triple-buffered; critical input wait 70.4--70.9% of projection kernel, reserve median 0.77--1.22 us; PCC 0.999641; normal close |
| `mlp_input_readiness_2026_08_03_17_10_00` | success | balanced fanout-2 activation/weight block publish timestamps, profiler-free gate + device-zone capture | weight late on W1/W3/W2 76.6/68.8/75.0% of core-block pairs and all slowest input-wait cores; activation late on cores 2:4 and 3:4; PCC 0.999641; normal close |
| `mlp_tagged_wait_decomposition_2026_08_04_00_00_00` | success | tagged direct fanout-2 DRAM/CB/consumer wait profile | BRISC barrier reduced 18.5--36.8%; consumer critical wait stayed about 70%; PCC 0.999641; normal close |
| `mlp_helper_wait_decomposition_2026_08_04_04_05_00` | success | helper/owner DRAM, semaphore, remote-L1 and consumer wait profile | remote write 5.1--5.3 us; owner delivery 25.9--27.2 us vs direct 24.0--24.9 us; PCC 0.999641; normal close |
| `mlp_compute_block_cadence_2026_08_04_04_38_00` | success | actual MLP compute block release/next-input cadence | W1/W3/W2 consumer CB wait 66.75/66.94/67.35% of kernel; PCC 0.999641; normal close |
| `mlp_compute_wait_in1_first_2026_08_04_04_52_00` | success | in1-first wait-order A/B | W1 weight-late 72.9%; W3 balanced; W2 activation-late 58.3%; total wait unchanged; PCC 0.999641; normal close |
| `mlp_merge_k_blocks2_2026_08_04_05_09_20` | success | consumer/weight K-block 경계 16→8 A/B | median 1.282% slower; wait +1.74--7.28%; CB call count가 주 병목 아님; PCC 0.999555; normal close |
| `mlp_tagged_depth3_2026_08_04_05_21_00` | success, block marker invalid | tagged pending depth-3 first profile | aggregate waits valid; publish block ID was N-1 instead of N-2, so block phase evidence 금지; normal close |
| `mlp_tagged_depth3_phase_2026_08_04_05_23_00` | success | corrected tagged depth-3 cadence/phase profile | BRISC barrier 감소가 activation wait로 이동; total wait +1.44--5.04%; PCC 0.999641; normal close |
| `mlp_dependency_chain_2026_08_04_05_43_00` | success | activation multicast, weight request, compute dependency-chain profile | target `(0,2)/(1,1)/(0,0)`은 W1/W3 전 block과 W2 45/48 block에서 weight-late; 전체 W3/W2는 혼합 병목; PCC 0.999641; normal close |
| `mlp_activation_credit_wait_2026_08_04_06_57_30` | success | activation sender credit와 multicast delivery 분해 | multicast last-arrival 0.48/0.48/0.69 us; 큰 sender wait는 future-block credit 대기; PCC 0.999641; normal close |
| `mlp_activation_credit_depth3_2026_08_04_07_02_00` | success | activation CB/credit depth 2→3 A/B | 20회 mean +1.921%; in0 감소가 in1로 이동; PCC 0.999641; normal close |
| `mlp_local_swiglu_2026_08_04_09_53_00` | success | fused W3/W1 + core-local L1 SwiGLU isolated profile | profiler-free 20회 mean -2.02%; profile PCC 0.999695; local SwiGLU 48.145 us; irregular 16-core→rectangular 16-core L1 remap 4.042 us; normal close |
| `llama32_3b_64k_local_swiglu_ab_2026_08_04_10_00_00` | operationally complete, experiment invalid | intended optimized full decode A/B | SDPA 6-endpoint opt-ins omitted; 6.061334/6.071251 tok/s excluded; both normal close |
| `llama32_3b_64k_local_swiglu_ab_corrected_2026_08_04_11_00_00` | success, profiler-free A/B | exact SDPA 6-endpoint K256 + MLP fanout-2 baseline/fused | baseline 7.643404 vs fused 7.666719 tok/s, +0.3050%; dual-NoC marker confirmed; both normal close |
| `llama32_3b_64k_local_swiglu_logits_ab_2026_08_04_10_30_00` | success, profiler-free correctness A/B | fixed token/position/zero KV full-model logits | PCC 0.999288; same top-1 and 5/5 top-5 overlap; not bitwise exact; both normal close |

Visualizer-compatible performance artifacts are under each successful Tracy run's `perf_report_visualize/` directory.

The Llama 3.2 3B 64K SDPA DRAM/NoC methodology and result interpretation are documented in
[`DRAM_BENCHMARK_README.md`](DRAM_BENCHMARK_README.md).

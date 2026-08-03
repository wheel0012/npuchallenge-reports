# Report manifest

기준일: 2026-08-03

중앙 파일은 아래 원본의 byte-for-byte 스냅샷이다. SHA-256은 중앙 파일을 기준으로 한다.

| 유형 | 중앙 파일 | 원본 | SHA-256 |
|---|---|---|---|
| benchmark | `benchmark-results/2026-07-26-llama32-3b-64k-sdpa-dram.md` | `/home/iris_hb4/profiler_runs/DRAM_BENCHMARK_README.md` | `538e7ebe746d5ae297cbf2cad29f1e1b66e4338e06da1e1402bcc72cee0a2148` |
| benchmark | `benchmark-results/2026-07-26-llama32-3b-gemm-20core.md` | `/home/iris_hb4/gemm_benchmark_llama32_3b_20core_2026_07_26/README.md` | `9ed435b348c46cb3300e553bc0ce81a53ba514982fbfaab329f8ec9b4836ba67` |
| benchmark | `benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md` | `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/README.md` | `37bf584967df7d2186b6805283727aa4880eee5edd81d6777830ca62367c2975` |
| benchmark | `benchmark-results/2026-08-02-bos-mlp-w2-block-width-ab.md` | canonical report (중앙 저장소에서 직접 작성) | `c83b94e4c27299d340eb6a3bc36cdd3c441f26f4872a8d220e74202b7e90d060` |
| benchmark | `benchmark-results/2026-08-03-bos-64k-six-endpoint-sdpa-mlp-ab.md` | canonical report (중앙 저장소에서 직접 작성) | `9c415d0068d6ac97e3873f654df58bdf3c6f7f0c8d8f5a77acea6a6958891595` |
| benchmark | `benchmark-results/2026-08-03-bos-dram-read-write-pipeline.md` | canonical report (중앙 저장소에서 직접 작성) | `1f1174bd281279a76ad6110ba12426c8d263b7d6f0e3a00cfeedcdd11db9c407` |
| benchmark | `benchmark-results/2026-08-03-bos-mlp-six-endpoint-fanout2-row-burst.md` | canonical report (중앙 저장소에서 직접 작성) | `835df96a8bc64d4b47a8616b569355c36d149c00e67885e464fda5d05a442328` |
| investigation | `investigations/2026-07-24-bos-llama32-kv-cache-test-demo.md` | `/home/iris_hb4/2026npu/readme_llama32_test_demo_analysis.md` | `9aa3dfa5f61cdf13847e8488c867068fc96808a8faf4669ffaf3687263825872` |
| investigation | `investigations/2026-07-25-llama31-8b-kv-cache.md` | `/home/iris_hb4/2026npu/readme_llama32_analysis.md` | `b9f4ab7f6c3bbb77ef323c7bd71bbbb242df0b72c2fcb0691d84fb2ae7991c3c` |
| investigation | `investigations/2026-07-26-ttnn-visualizer-bos-blackhole-npe.md` | `/home/iris_hb4/TTNN_VISUALIZER_BOS_BLACKHOLE_NPE_REPORT_KO.md` | `afc2148ba76fa5bda226d8f81de20032bd291cf203d3b9bd16d2e56587ed8d8c` |
| investigation | `investigations/2026-08-01-sdpa-dram-performance-optimization-history.md` | canonical report (중앙 저장소에서 직접 작성) | `08c06cf5de0c8b071b6b5a82fabe24bf1fa90f1c2a4c5bf7853a4ff9a52fe8a9` |
| investigation | `investigations/2026-08-01-vanilla-sdpa-vs-dram-saturation-gap.md` | canonical report (중앙 저장소에서 직접 작성) | `9ee9e23506d50336008764076744968cfa101e03247faf1f5ffd84a55b0aaf9d` |
| investigation | `investigations/2026-08-01-sdpa-four-helper-buffering-design.md` | canonical report (중앙 저장소에서 직접 작성) | `a6d4b9000fc4adab5bec1c972c590ecd2f580b06593e762fcad54db816a7491a` |
| incident | `incidents/2026-07-31-blackhole-worker-fw-host-freeze.md` | `/home/iris_hb4/tt-metal-hb4/tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/INCIDENT_REPORT_2026_07_31_DEVICE_FREEZE.md` | `c835200868f68b245f4bd52e77a54e274996bdca36730a6c7f3cd6ed73008611` |
| incident | `incidents/2026-08-01-bos-mlp-noc-profile-fw-init-failure.md` | canonical report (중앙 저장소에서 직접 작성) | `f8f97df2e72b9d0672a0271f107e993f4be6e369f0ff4bd9463c255885a7366b` |
| incident | `incidents/2026-08-02-bos-sdpa-reduce-helper-deadlock.md` | canonical report (중앙 저장소에서 직접 작성) | `3a14bdd61b1a57d18621d0bd40fc74a597440049451f272cb4ff449110d7482c` |
| incident | `incidents/2026-08-02-bos-llama32-dram-sharded-prefill-validation-failure.md` | canonical report (중앙 저장소에서 직접 작성) | `176564a08e0011cf8206ca6d8bf66fb3b6b72c2d6a15c2b7df395c64b1aaa5fa` |
| incident | `incidents/2026-08-03-bos-mlp-dual-noc-reader-timeout.md` | canonical report (중앙 저장소에서 직접 작성) | `a2504adad42c44dcb0cc16556678cbfb47a3c957b3450dc9a9cda23870e82795` |
| handoff | `handoffs/2026-07-21-qwen25-3b-32k-profiling.md` | `/home/iris_hb4/QWEN_AND_PROFILER_README.md` | `b7ccdfc275c11a2d554ae55bb8afdbf725316d999f404adf49a5b0adb36a2a4a` |
| guide | `guides/gemm-benchmark-measurement.md` | `/home/iris_hb4/GEMM_BENCHMARK_GUIDE_KO.md` | `2c1c7858deb47870bd8b90b97484095fd4cbade1c2d09d37f043e0eeb1dd2643` |
| guide | `guides/tt-metal-venv-path.md` | `/home/iris_hb4/TT_METAL_VENV_PATH_GUIDE_KO.md` | `da9bd20162b7d626ff2b83a5659eba1b6c65c1f11861b7cddc2b2d96541d06c9` |
| index | `indexes/profiler-runs.md` | canonical index (중앙 저장소에서 직접 갱신) | `5df24d697058168efb6157839f8e8b43945fb242f07afbc164575527c312be4c` |

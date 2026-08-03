# Llama 3.2 3B reference GEMM benchmark — 20 cores

`/home/iris_hb4/tt-metal-hb4/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`와
`test_matmul_2d_host_perf`의 측정 메커니즘을 현재 5×4 grid에 적용한 결과다.

## 측정 조건

- Grid: `5×4` (20 Tensix cores)
- Mode: host non-trace steady state
- Warmup / measurement: 5 / 100 iterations
- Precision: mixed BF16/BFP8 HiFi2, original-report BFP8 LoFi, BFP4 LoFi
- Math: HiFi2, `fp32_dest_acc_en=False`, `packer_l1_acc=True`
- Clock reference: 650 MHz (utilization 계산에만 사용)
- Completed: 80 cases (20 shapes × 4 precision/fidelity paths)

## 핵심 결과

| Precision | Peak shape M×K×N | Peak TFLOPS | Average time |
| --- | ---: | ---: | ---: |
| BF16 × BFP8 → BF16 HiFi2 | 16384×20480×20480 | 23.9963 | 572.751 ms |
| BFP8 × BFP8 → BF16 HiFi2 | 16384×20480×20480 | 24.0990 | 570.310 ms |
| BFP8 × BFP8 → BFP8 LoFi | 14336×17920×17920 | 43.6569 | 210.902 ms |
| BFP4 × BFP4 → BFP4 LoFi | 14336×17920×17920 | 46.6616 | 197.322 ms |

## Large-matrix extension

원문의 1-core base-shape scaling을 5×4 grid에 적용했다.
정방 base `S`마다 실제 shape는 `M=4S`, `K=N=5S`다.

| Base S | Actual M×K×N | GFLOP | BF16×BFP8 H2 | BFP8×BFP8 H2 | BFP8 LoFi | BFP4 LoFi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2560 | 10240×12800×12800 | 3355.443 | 23.7755 | 23.9196 | 42.7667 | 46.1242 |
| 3072 | 12288×15360×15360 | 5798.206 | 23.8822 | 23.9933 | 42.7145 | 46.3399 |
| 3584 | 14336×17920×17920 | 9207.336 | 23.9592 | 24.0668 | 43.6569 | 46.6616 |
| 4096 | 16384×20480×20480 | 13743.895 | 23.9963 | 24.0990 | 43.1945 | 46.4661 |

원문 scatter plot과 같이 `M×K×N` log 축을 사용한 별도 plot을 생성했다.
추가 shape는 기존 최대 case와 동일하게 core-local K block 16 tile, output block 8×8 tile을 유지한다.


650 MHz, 20 core의 fidelity별 arithmetic ceiling과 실측 peak는 다음과 같다.

| Fidelity | Theoretical ceiling | Best measured | Utilization |
| --- | ---: | ---: | ---: |
| HiFi2 | 26.624 TFLOPS | 24.0990 | 90.52% |
| LoFi | 53.248 TFLOPS | 46.6616 | 87.63% |

## Llama profile marker

`Source: /home/iris_hb4/profiler_runs/llama32_3b_64k_actual_prefill_single_layer_decode_accuracy_mode_perf_2026_07_25_09_14_33/perf_report_visualize/ops_perf_result.csv`

| M×K×N | Calls | Precision | Acc | Cores | Workload | FW TFLOPS |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 32×3072×3072 | 1 | BF16 × BFP8 → BF16 | FP32 | 20 | 0.604 GFLOP | 2.3352 |
| 32×3072×5120 | 1 | BF16 × BFP8 → BF16 | FP32 | 16 | 1.007 GFLOP | 2.3304 |
| 32×3072×8192 | 2 | BF16 × BFP8 → BF16 | FP16 | 16 | 1.611 GFLOP | 2.3407 / 2.2515 |
| 32×8192×3072 | 1 | BFP8 × BFP8 → BF16 | FP16 | 16 | 1.611 GFLOP | 2.3532 |

그래프 x축은 log scale의 `2MKN` workload이며 위 shape를 세로 marker로 표시한다.
동일 workload인 마지막 두 shape는 같은 세로선에 전체 shape label을 함께 표시한다.

세로선은 workload 크기 참고선이지 benchmark curve에서 실제 decode 성능을 보간하는 선이 아니다.
실제 decode는 `M=32`, width-sharded/model-specific config, 16 또는 20 core를 사용한다.
또한 첫 두 attention projection은 FP32 destination accumulation이므로 FP16 누적 benchmark line과 math가 다르다.
benchmark는 5×4 2D multicast shape이고 host 100회 평균이므로 profile FW time과 측정 범위도 다르다.

## 산출물

- `gemm_20core_results.csv`: 80개 raw benchmark 결과
- `gemm_20core_llama32_markers.png`: raster plot
- `gemm_20core_llama32_markers.svg`: vector plot
- `gemm_20core_tech_report_matrix_elements.png`: 원문식 M×K×N scatter/line plot
- `gemm_20core_tech_report_matrix_elements.svg`: 원문식 vector plot
- `llama32_3b_matmul_markers.csv`: profile에서 추출한 unique shape
- `benchmark.log`: 전체 장치 실행 log
- `run_gemm_benchmark.py`, `plot_gemm_benchmark.py`: 재현 script

## 재현

```bash
cd /home/iris_hb4/gemm_benchmark_llama32_3b_20core_2026_07_26
source /home/iris_hb4/.venv/bin/activate

export TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4
export PYTHONPATH=/home/iris_hb4/tt-metal-hb4:/home/iris_hb4/tt-metal-hb4/ttnn
export LD_LIBRARY_PATH=/home/iris_hb4/tt-metal-hb4/build_home_release/lib:/home/iris_hb4/tt-metal-hb4/build_home_release/tt_metal

python -u run_gemm_benchmark.py --output gemm_20core_results.csv
```

기존 CSV row는 resume 시 skip한다. 완전 재실행은 새 output filename을 사용한다.

```bash
python plot_gemm_benchmark.py \
  --benchmark-csv gemm_20core_results.csv \
  --profile-csv /home/iris_hb4/profiler_runs/llama32_3b_64k_actual_prefill_single_layer_decode_accuracy_mode_perf_2026_07_25_09_14_33/perf_report_visualize/ops_perf_result.csv \
  --markers-csv llama32_3b_matmul_markers.csv \
  --output-png gemm_20core_llama32_markers.png \
  --output-svg gemm_20core_llama32_markers.svg \
  --output-original-png gemm_20core_tech_report_matrix_elements.png \
  --output-original-svg gemm_20core_tech_report_matrix_elements.svg
```

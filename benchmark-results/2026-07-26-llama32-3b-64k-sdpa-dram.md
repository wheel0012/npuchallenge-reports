# Llama 3.2 3B 64K SDPA DRAM benchmark

## 요약

이 문서는 Llama 3.2 3B의 64K decode SDPA에 대해 수행한 DRAM/NoC 계측을 정리한다.
기준 결과는 다음 독립 SDPA run이다.

```text
/home/iris_hb4/profiler_runs/
└── sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33/
```

이 run은 64K 위치의 `SdpaDecodeDeviceOperation` 한 번만 캡처한
`cur_pos-only` microbenchmark다. 실제 prefill을 수행한 end-to-end benchmark는 아니며,
warmup 이후 SDPA 자체의 DRAM/NoC 동작을 분리해 보기 위한 실행이다.

| 항목 | 결과 |
|---|---:|
| Device kernel duration | 3,466,786 ns |
| Algorithmic matmul FLOPs | 805,306,368 FLOPs |
| Algorithmic throughput | 0.2323 TFLOP/s |
| Logical matmul DRAM traffic | 134,225,920 bytes |
| Logical matmul bandwidth | 38.72 GB/s |
| NPE NoC utilization | 2.2% |
| NPE DRAM bandwidth utilization | 4.3% |
| NPE congestion impact | 0.0% |

`38.72 GB/s`와 `4.3%`는 서로 다른 정의다. 전자는 tensor shape로 계산한 논리적
QK/AV matmul traffic이고, 후자는 수집한 NoC event trace를 tt-npe가 해석한
DRAM payload를 golden device cycles 동안의 모델 peak와 비교한 값이다.

## Benchmark configuration

| 항목 | 설정 |
|---|---|
| Model geometry | Llama 3.2 3B |
| Operation | `SdpaDecodeDeviceOperation` |
| Batch | 1 |
| Query heads | 24 logical heads, 32 padded heads |
| KV heads | 8 |
| Head dimension | 128 |
| Context length | 65,536 |
| Page block size | 32 |
| Page-table entries | 2,048 |
| SDPA grid | 5 x 4, 20 cores |
| Q/K chunk size | 128 / 128 |
| Math fidelity | HiFi2 |
| Q | BF16, height-sharded L1 |
| K/V cache | BFP8_B, interleaved DRAM |
| Output | BF16, interleaved DRAM |
| NPE device model | Wormhole B0 |

CSV에 기록된 주요 shape는 다음과 같다.

```text
Q:          [1, 1, 32[24], 128] BF16 L1
K cache:    [2048, 8, 32, 128]  BFP8_B DRAM
V cache:    [2048, 8, 32, 128]  BFP8_B DRAM
cur_pos:    [1, 1, 1, 1]        INT32 DRAM
page table: [1, 1, 1, 2048]     INT32 DRAM
output:     [1, 1, 32[24], 128] BF16 DRAM
```

## 산식

### Matmul FLOPs

softmax와 scale 등은 제외하고 QK와 AV matmul만 센다.

```text
QK FLOPs = 2 * B * Hq * S * D
          = 2 * 1 * 24 * 65,536 * 128
          = 402,653,184

AV FLOPs = 2 * B * Hq * S * Dv
          = 402,653,184

total FLOPs = 805,306,368
TFLOP/s = total FLOPs / 3,466,786 ns
        = 0.2323 TFLOP/s
```

### Logical matmul DRAM traffic

기존 matmul report와 같은 방식으로 DRAM에 있는 matmul operand와 output만 합산한다.
Q는 L1에 있으므로 제외하고, `cur_pos`, page table, softmax intermediate도 제외한다.

```text
K read       = 2048 * 8 * 32 * 128 * 1 byte = 67,108,864 bytes
V read       = 2048 * 8 * 32 * 128 * 1 byte = 67,108,864 bytes
output write = 1 * 1 * 32 * 128 * 2 bytes    =      8,192 bytes

total        = 134,225,920 bytes
logical BW   = total / 3,466,786 ns
             = 38.72 GB/s
```

이 값은 K/V 전체를 논리적으로 한 번 읽는다고 가정한 algorithmic bandwidth다.
NoC multicast, cache reuse, packetization, page-table access 또는 실제 전송 재사용을 반영한
hardware counter 값은 아니다.

### NPE DRAM bandwidth utilization

tt-npe는 raw NoC trace의 DRAM-origin read와 DRAM-destination write payload를 합산한다.

```text
DRAM BW util = traced DRAM read/write bytes
             / (golden cycles * modeled DRAM bytes/cycle)
             * 100
```

현재 checkout의 Wormhole B0 모델은 controller 6개와 controller당
`2 * average(23.2, 24.0) = 47.2 bytes/cycle`을 사용하므로 chip peak는
283.2 bytes/cycle이다. 1 GHz를 가정하면 CSV의 반올림된 4.3%는 약 12.2 GB/s에
해당한다. 이 환산값은 실측 DRAM counter가 아니라 tt-npe 모델 기준 근사치다.

## 산출물

```text
sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33/
└── perf_capture/
    ├── .logs/
    │   ├── noc_trace_dev0_ID1024.json
    │   ├── noc_trace_ID1_merged.json
    │   ├── profile_log_device.csv
    │   └── tracy_profile_log_host.tracy
    ├── npe_viz/
    │   ├── manifest.json
    │   └── UnknownOP_ID1*.npeviz.zst
    └── reports/sdpa_decode_64k_vanilla_curpos_only_npe/
        └── 2026_07_26_03_32_30/
            ├── ops_perf_results_*.csv
            ├── profile_log_device.csv
            ├── tracy_profile_log_host.tracy
            └── npe_viz/
```

가장 먼저 볼 파일은 다음 CSV다.

```text
/home/iris_hb4/profiler_runs/sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33/
  perf_capture/reports/sdpa_decode_64k_vanilla_curpos_only_npe/2026_07_26_03_32_30/
  ops_perf_results_sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_32_30.csv
```

`npe_viz/manifest.json`의 `global_call_count=1024`가 이 CSV의 SDPA row와 연결된다.

## 재현 절차

NoC trace 수집은 profiler의 `--collect-noc-traces` 옵션을 사용한다.

```bash
cd /home/iris_hb4/tt-metal-hb4
source python_env/bin/activate
source /home/iris_hb4/tt-npe/ENV_SETUP

python tools/tracy/profile_this.py \
    --collect-noc-traces \
    -c '<64K paged SdpaDecodeDeviceOperation을 한 번 실행하는 명령>' \
    -o /home/iris_hb4/profiler_runs/<new-run>/perf_capture
```

저장된 raw trace만 다시 NPE timeline으로 변환할 때는 장치를 재실행할 필요가 없다.

```bash
npe_analyze_noc_trace_dir.py \
    /home/iris_hb4/profiler_runs/<run>/perf_capture/.logs \
    -e
```

이번 독립 SDPA 실행에 사용한 일회성 launcher는 저장소에 남아 있지 않으므로,
위 `<...>` 부분은 현재 그대로 복사 가능한 완전한 재현 명령이 아니다.
full-layer `cur_pos-only` 입력 구성은 다음 스크립트에서 확인할 수 있다.

```text
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/
run_llama32_3b_curpos_only_single_layer_npe.py
```

## 실제 prefill run과의 관계

실제 65,535 token prefill 후 layer 0 decode를 수행한 accuracy-mode 결과는 아래에 있다.

```text
/home/iris_hb4/profiler_runs/
llama32_3b_64k_accuracy_mode_sdpa_dual_noc_5ep_2026_07_26_02_49_30/
```

이 run은 전달 디렉터리 계약에 맞는 산출물을 가진다.

```text
<run>/
├── memory_report_visualizer/
│   ├── config.json
│   └── db.sqlite
└── perf_report_visualize/
    ├── ops_perf_result.csv
    ├── profile_log_device.csv
    └── tracy_profile_log_host.tracy
```

다만 해당 full-layer report의 SDPA row에는 `NOC UTIL (%)`와
`DRAM BW UTIL (%)`이 비어 있다. 따라서 이 문서의 NPE DRAM 결과는 실제 prefill run에서
직접 얻은 값이 아니라, 동일한 64K SDPA geometry를 독립 실행해 얻은 값이다.

## 해석 시 주의사항

- 이 benchmark는 `cur_pos-only`이므로 실제 prompt의 K/V 내용이나 prefill 비용을 측정하지 않는다.
- `vanilla`는 speculative decode가 아닌 표준 paged SDPA decode라는 뜻이다.
- NPE utilization은 captured NoC events와 device model을 이용한 trace-driven estimate다.
  PMU 기반 DRAM hardware counter로 읽은 수치가 아니다.
- logical bandwidth와 NPE bandwidth는 byte accounting 범위가 다르므로 직접 일치할 필요가 없다.
- `DRAM BW UTIL (%)=4.3`은 CSV에서 한 자리 소수로 반올림된 값이다.
- `NPE CONG IMPACT (%)=0.0`은 이 trace와 모델에서 예측된 값이며, 모든 64K workload에
  congestion이 없다는 일반적인 결론은 아니다.
- 독립 NPE run은 원본 분석용으로 보존한다. 원격 Visualizer 전달에는 위 actual-prefill
  run의 계약 디렉터리를 사용한다.

## 사용한 checkout

```text
tt-metal-hb4: 1423544
tt-npe:       3ce4534
```

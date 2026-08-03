# TTNN GEMM FLOPS Benchmark 측정 메커니즘

이 문서는 Tenstorrent `GEMM_FLOPS` Tech Report의 수치를 생성하는
`test_matmul_2d_host_perf` benchmark가 실제로 무엇을 구성하고 어떻게 시간을 재는지
설명한다. 현재 BOS용 5×4 grid 및 Qwen mixed-precision 확장도 함께 구분한다.

기준 소스:

```text
/home/iris_hb4/tt-metal-prof-src/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md
/home/iris_hb4/tt-metal-prof-src/tests/ttnn/unit_tests/benchmarks/test_benchmark.py
```

## 1. 측정 대상

benchmark가 실행하는 연산은 다음 GEMM이다.

```text
A[M, K] × B[K, N] → C[M, N]
```

연산량은 multiply와 add를 각각 한 operation으로 세어 다음과 같이 계산한다.

```text
operations = 2 × M × K × N
TFLOPS = operations / elapsed_seconds / 10^12
```

이 benchmark는 한 개의 고정 square GEMM만 측정하지 않는다. dtype마다 준비된 여러 base
shape, math fidelity, L1/DRAM placement, sharding, blocking 및 trace on/off 조합을 sweep한다.
작은 shape에서는 dispatch와 data movement의 영향이 보이고, 큰 shape에서는 Tensix compute
ceiling에 접근하는 과정을 관찰한다.

## 2. 전체 실행 흐름

case 하나의 실행 순서는 다음과 같다.

```text
base shape 선택
  ↓
grid 크기에 맞춰 M/K/N 확장
  ↓
Torch A/B 생성
  ↓
TTNN tile tensor로 변환하여 L1 또는 DRAM에 배치
  ↓
2D multicast MatMul program config 생성
  ↓
최초 MatMul 1회: program 생성/JIT 및 cache 준비
  ↓
warmup MatMul 5회
  ↓
device synchronize
  ↓
MatMul 100회 측정(non-trace 또는 trace)
  ↓
elapsed/100, TFLOPS 및 utilization 계산
  ↓
최종 output readback 및 tensor deallocation
```

입력 tensor 생성과 host→device 업로드, 최초 program 생성 및 warmup은 측정 구간 밖이다.
측정 중에는 같은 `A`와 `B` tensor 및 동일 program configuration을 반복 사용하므로 program
cache가 warm 상태다.

## 3. Shape가 grid에 맞춰 확장되는 방법

shape table은 1×1 core 기준의 base shape를 저장한다.

```text
(m_base, k_base, n_base,
 in0_sharded, out_sharded,
 in0_block_w_div, num_out_blocks_h, num_out_blocks_w)
```

실제 shape는 선택한 grid `(X, Y)`에 따라 다음처럼 확장된다.

```text
M = m_base × Y
K = k_base × X
N = n_base × X
```

예를 들어 base `(64, 128, 256)`을 BOS 5×4 grid에서 실행하면:

```text
M = 64  × 4 = 256
K = 128 × 5 = 640
N = 256 × 5 = 1280

A: [256, 640]
B: [640, 1280]
C: [256, 1280]
```

이 확장은 core 수만큼 행렬 전체를 단순 복제하는 것이 아니다. core당 M/N tile 분할이
정수가 되도록 전체 행렬을 grid에 맞춰 확대한다.

```text
per_core_M = M / Y / tile_height
per_core_N = N / X / tile_width
```

원 Tech Report의 Wormhole 8×8 실행에서는 base shape가 M×8, K×8, N×8로 확장된다.
BOS에서는 `--grid-size 5x4`를 사용해 M×4, K×5, N×5로 확장한다. 따라서 BOS 결과는
동일 base configuration을 20-core topology에 이식한 것이며, Wormhole의 최종 행렬 크기를
그대로 실행한 결과는 아니다.

## 4. Tile 표현과 dtype

기본 tile은 32×32다.

```text
A tile:  tile_height × 32
B tile:  32 × tile_width
C tile:  tile_height × tile_width
```

현재 sweep에서 `tile_height=32`, `tile_width=32`를 사용한다. host의 Torch tensor는 BF16으로
생성되지만 `ttnn.from_torch`에서 case별 input dtype으로 pack된다.

원 benchmark의 대표 조합은 다음과 같다.

| Input/Output | Fidelity |
| --- | --- |
| BF16 × BF16 → BF16 | HiFi2, HiFi4 |
| BFP8 × BFP8 → BFP8 | HiFi2, LoFi |
| BFP4 × BFP4 → BFP4 | LoFi |

BOS Qwen 확장에서 추가한 실제 모델형 조합은 다음과 같다.

| Activation × Weight → Output | Fidelity |
| --- | --- |
| BF16 × BFP8 → BF16 | HiFi2 |
| BF16 × BFP4 → BF16 | LoFi |
| BFP8 × BFP8 → BF16 | HiFi2 |

BFP8/BFP4는 단순히 원소당 정확히 1 byte/0.5 byte인 raw integer 형식이 아니다. 32×32
tile 단위 mantissa와 exponent block을 포함하는 packed block-floating-point 형식이다.

## 5. 입력과 출력의 memory placement

benchmark의 기본 data path는 다음과 같다.

```text
A(in0): case에 따라 block-sharded L1 또는 interleaved DRAM
B(in1): 항상 interleaved DRAM
C(out): case에 따라 block-sharded L1 또는 interleaved DRAM
```

작은 shape는 A와 C를 L1에 sharding할 수 있다.

```text
L1 block-sharded A
        │
        ├──▶ per-core CB ──▶ Tensix matrix engine ──▶ L1 block-sharded C
        │                         ▲
DRAM interleaved B ──NoC/CB───────┘
```

큰 shape에서는 A와 C도 DRAM에 둔다.

```text
DRAM A ──reader/NoC──▶ CB ─┐
                            ├──▶ Tensix matrix engine ──writer/NoC──▶ DRAM C
DRAM B ──reader/NoC──▶ CB ─┘
```

여기서 “A 전체를 L1에 넣고 B를 tile 하나씩 읽는다”고 단순화하면 정확하지 않다. A가
L1인 case도 core별 shard로 나뉘며, MatMul kernel은 A/B를 block 단위 CB window로 공급한다.
B는 항상 DRAM tensor지만 같은 B block을 여러 M output block 계산에 재사용할 수 있다.
마찬가지로 A block은 여러 N output block에 재사용될 수 있다.

행렬이 커져도 GEMM이 compute-bound가 될 수 있는 이유는 전체 입력을 한 번에 L1에 넣기
때문이 아니라, CB에 들어온 block을 여러 output tile 계산에 재사용해 byte당 MAC 수를
높이기 때문이다.

## 6. 2D multicast와 core 작업 분할

benchmark는 다음 explicit program config를 사용한다.

```python
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=grid_size,
    in0_block_w=in0_block_w,
    out_subblock_h=out_subblock_h,
    out_subblock_w=out_subblock_w,
    out_block_h=out_block_h,
    out_block_w=out_block_w,
    per_core_M=per_core_M,
    per_core_N=per_core_N,
    transpose_mcast=False,
    fused_activation=None,
)
```

각 core는 전체 C의 `per_core_M × per_core_N` tile 영역을 담당한다. 동일한 A 또는 B block이
필요한 core들이 존재하므로 2D multicast/reuse dataflow를 통해 각 core가 DRAM에서 완전히
독립적으로 중복 read하는 비용을 줄인다.

개념적으로는 다음과 같다.

```text
                 N tile blocks
             core  core  core  core  core
M blocks     core  core  core  core  core
             core  core  core  core  core
             core  core  core  core  core

각 core: C의 한 2D block 담당
A/B block: 필요한 core 집합에 multicast 또는 재사용
K 방향: 여러 block을 순서대로 누적
```

실제 NoC 방향과 reader core 선택은 architecture 및 program factory가 결정한다. 따라서
이 그림을 “특정 X/Y 방향으로 반드시 A/B가 전송된다”는 물리 routing diagram으로
해석하면 안 된다.

## 7. L1과 Circular Buffer 크기를 제어하는 blocking

GEMM은 A/B 전체를 L1 CB에 동시에 담지 않는다. 다음 parameter로 작업을 더 작은 block으로
나눠 L1 사용량을 제한한다.

```text
in0_block_w_div
num_out_blocks_h
num_out_blocks_w
```

K 방향 A block 폭은 다음과 같다.

```text
in0_block_w = K / X / 32 / in0_block_w_div
```

`in0_block_w_div`가 커질수록 한 번에 CB에 넣는 K tile 수가 줄고 K accumulation iteration은
늘어난다. output 영역은 다음처럼 나뉜다.

```text
out_block_h = per_core_M / num_out_blocks_h
out_block_w = per_core_N / num_out_blocks_w
```

output subblock은 후보 목록에서 block을 정확히 나누면서 matrix-engine/packer 제약을
만족하는 가장 큰 조합을 고른다. 일반적으로 subblock tile area는 최대 8이며,
`out_sharded=True`에서는 `out_subblock_h=1`이고 N 방향이 정확히 나뉘어야 한다.

따라서 큰 행렬을 실행할 수 있는 것은 행렬 자체가 L1보다 작아서가 아니다. core별 shard,
K block, output block 및 subblock으로 계층적으로 나눠 CB window를 재사용하기 때문이다.

## 8. Compute kernel 설정과 math fidelity

Blackhole/Wormhole 계열 경로에서는 다음 설정을 사용한다.

```text
math_approx_mode = True
fp32_dest_acc_en = False
packer_l1_acc    = True
throttle_level   = NO_THROTTLE
```

`packer_l1_acc=True`는 긴 K accumulation에서 intermediate partial output을 L1 packer 경로로
관리하는 최적화다. `fp32_dest_acc_en=False`이므로 FP32 destination accumulator를 사용한
정확도/용량 조건은 측정하지 않는다.

Tech Report의 ideal tile cycle 모델은 다음 값을 사용한다.

| Fidelity | Ideal cycles per tile multiply |
| --- | ---: |
| LoFi | 16 |
| HiFi2 | 32 |
| HiFi3 | 48 |
| HiFi4 | 64 |

높은 fidelity는 input mantissa의 더 많은 bit/pass를 처리하므로 일반적으로 peak throughput이
낮아진다. dtype과 fidelity는 별개다. BFP8이라고 항상 LoFi인 것도 아니며, 모델 정확도
요구에 따라 BFP8 HiFi2를 사용할 수 있다.

## 9. Warmup과 program cache

측정 전에 MatMul을 최초 1회 실행하고 추가 warmup을 5회 enqueue한다.

```text
first call: program 생성/JIT, allocation 및 cache 준비 가능
warmup 5:  동일 shape/config program cache 재사용
synchronize: warmup 전부 완료
measure: warm cache 상태
```

따라서 보고된 평균은 cold-start compile latency가 아니라 steady-state 반복 throughput이다.
case마다 shape, dtype 또는 config가 바뀌면 새로운 program이 준비될 수 있지만 그 비용은
해당 case의 측정 전에 지불한다.

## 10. Non-trace 측정

non-trace는 MatMul마다 synchronize하지 않는다.

```python
start_timer()
for _ in range(100):
    output = ttnn.matmul(...)
ttnn.synchronize_device(device)
stop_timer()

average = total_elapsed / 100
```

즉 측정값에는 다음이 포함된다.

- 100개 TTNN operation의 host enqueue/dispatch
- device command processing
- 100개 GEMM kernel 실행
- 마지막 synchronize 완료 대기

반면 input 생성, host→device upload, 최초 compile/warmup 및 최종 `to_torch` readback은
포함되지 않는다. 100회 batching 덕분에 timer 호출과 마지막 synchronize의 고정 비용은
한 operation당 1/100로 줄지만, 각 operation의 host dispatch 비용은 남는다.

## 11. Trace 측정

trace mode는 100개의 MatMul command sequence를 먼저 DRAM trace buffer에 capture한다.

```text
begin_trace_capture
  MatMul command × 100 기록
end_trace_capture

start_timer
execute_trace 한 번
synchronize 한 번
stop_timer

average = replay elapsed / 100
```

replay 시에는 Python이 MatMul 100개를 다시 dispatch하지 않으므로 반복 loop의 host overhead가
크게 줄어든다. 작은/빠른 GEMM에서는 dispatch 시간이 kernel 시간과 비슷해 trace 효과가
크고, 큰 GEMM에서는 device kernel 시간이 지배적이라 trace/non-trace 차이가 작다.

trace가 compute kernel을 더 빠른 kernel로 바꾸는 것은 아니다. 동일 command sequence의
host dispatch를 제거하는 runtime 최적화다. 또한 100개 operation을 capture하므로 trace
region이 충분히 커야 한다. BOS full sweep의 BFP8 case는 16 MiB를 초과해 32 MiB trace
region을 사용했다.

## 12. Host TFLOPS 계산

측정된 총 elapsed를 100으로 나눈 `inference_time_avg`로 계산한다.

```text
inference_time_avg = elapsed(100 GEMMs) / 100
TFLOPS = 2 × M × K × N / inference_time_avg / 10^12
```

예를 들어 `M=K=N=2560`, device time이 1.549 ms라면:

```text
operations = 2 × 2560^3 = 33,554,432,000
throughput = 33,554,432,000 / 0.001549 / 10^12
           ≈ 21.66 TFLOPS
```

정수 또는 block-floating-point 형식에도 benchmark CSV는 역사적으로 `TFLOPS`라는 이름을
사용한다. 발표에서 dtype 전체를 함께 비교할 때는 `TOPS` 또는 “2 ops/MAC 기준 throughput”을
사용하면 오해가 적다.

## 13. Utilization 계산

benchmark의 ideal cycle 모델은 다음과 같다.

```text
ideal_cycles =
    M × K × N
    / (tile_height × tile_width × 32)
    × fidelity_cycles_per_tile
    / number_of_cores
```

Host utilization은 host 평균 시간을 device cycle로 변환한다.

```text
inference_cycles = inference_time_avg × device_frequency_hz
host utilization = ideal_cycles / inference_cycles
```

기본 Blackhole clock 상수는 1350 MHz지만 BOS 실측 clock은 650 MHz이므로
`TT_METAL_BENCHMARK_DEVICE_FREQ_MHZ=650` override가 필요하다. 잘못된 clock을 쓰면 TFLOPS
자체는 변하지 않지만 utilization percentage가 틀어진다.

device profiler가 활성화되면 profiler log의 평균 TRISC1 math-kernel cycle을 사용한다.

```text
device utilization = ideal_cycles / mean(TRISC1 kernel cycles)
```

이 device utilization은 host dispatch를 제외하지만 TRISC1 zone 정의와 profiler marker가
정상이어야 한다. 전체 sweep에서 marker buffer가 포화되면 누락된 row를 그대로 사용하지
말고 case별로 분리해 profile해야 한다.

## 14. Device kernel time과 host time의 차이

| 시간 | 포함 범위 | 적합한 용도 |
| --- | --- | --- |
| Host non-trace average | enqueue + dispatch + device execution | 일반 runtime throughput |
| Host trace average | trace replay + device execution | host dispatch 최소화 throughput |
| Tracy `DEVICE KERNEL DURATION` | device operation/kernel 구간 | empirical compute roof |
| TRISC1 kernel cycles | math RISC kernel zone | ideal-cycle utilization |

최종 Roofline의 compute ceiling에는 Tracy device-kernel median을 사용하는 것이 가장
명확하다. 원 benchmark의 host average는 100회 반복 throughput을 비교하는 데 유용하지만,
device kernel만의 peak와 동일한 값은 아니다.

## 15. GEMM에서 DRAM traffic이 있어도 compute-bound가 되는 이유

B는 모든 case에서 DRAM에 있으므로 이 benchmark는 순수 register-only matrix-engine
microbenchmark가 아니다. 큰 case에서는 A와 C도 DRAM이다. 그럼에도 큰 정사각 GEMM이
compute roof에 접근할 수 있는 이유는 높은 재사용률이다.

대략적인 algorithmic arithmetic intensity는 다음과 같다.

```text
AI = 2MKN / (bytes(A) + bytes(B) + bytes(C))
```

정사각 행렬의 차원이 커지면 연산량은 세제곱으로 증가하지만 최소 tensor byte는 제곱으로
증가한다. 또한 tile/block dataflow에서:

- B block은 여러 M tile row 계산에 재사용된다.
- A block은 여러 N tile column 계산에 재사용된다.
- K block의 partial sum은 output block에 누적된다.

따라서 DRAM access가 사라지는 것이 아니라, DRAM에서 가져온 byte당 수행하는 MAC이 많아져
matrix engine이 병목이 된다. 반대로 Qwen batch-1 decode처럼 `M≈1`인 skinny GEMM은 B
weight를 M 방향으로 거의 재사용하지 못하므로 memory-bound가 되기 쉽다.

## 16. 이 benchmark가 측정하지 않는 것

- 순수 matrix engine의 register-only theoretical peak
- DRAM interface의 독립적인 최대 bandwidth
- 실제 모델의 layer fusion, KV cache, SDPA 또는 inter-op gap
- 각 tensor의 물리 DRAM transaction byte와 NoC packet overhead
- numerical accuracy/PCC; 이 테스트는 최종 output을 host로 가져오지만 golden과 비교하지 않는다.

따라서 이 결과는 “해당 TTNN GEMM config가 달성한 empirical throughput”이다. DRAM roof는
별도 streaming benchmark로 측정하고, 실제 Qwen operator는 동일 shape/dtype의 GEMM 결과와
별도로 비교해야 한다.

## 17. BOS 재실행 명령

```bash
cd /home/iris_hb4/tt-metal-prof-src
source /home/iris_hb4/.venv/bin/activate

export TT_METAL_HOME=/home/iris_hb4/tt-metal-prof-src
export PYTHONPATH=/home/iris_hb4/tt-metal-prof-src:/home/iris_hb4/tt-metal-prof-src/ttnn:$PYTHONPATH
export TT_METAL_RUN_BENCHMARKS=1
export TT_METAL_BENCHMARK_DEVICE_FREQ_MHZ=650
export TT_METAL_BENCHMARK_REPORT_PATH="$HOME/roofline_results/gemm_report.csv"

pytest -s tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf \
  --grid-size 5x4
```

Qwen mixed precision만 사용하려면:

```bash
export TT_METAL_BENCHMARK_QWEN_PRECISIONS=1
```

trace case만 실행하려면:

```bash
export TT_METAL_BENCHMARK_TRACE_ONLY=1
export TT_METAL_BENCHMARK_TRACE_REGION_SIZE=33554432
```

외부 Tracy wrapper로 profile할 때는 benchmark 내부 profiler postprocessing과 충돌하지 않도록:

```bash
export TT_METAL_BENCHMARK_EXTERNAL_PROFILER=1
```

Python device fixture가 정상 teardown에 도달하면 device를 close한다. 다만 native device call이
영구 정지하면 같은 process의 teardown도 실행될 수 없으므로 긴 sweep은 dtype/shape별
subprocess와 timeout으로 분리하는 것이 안전하다.

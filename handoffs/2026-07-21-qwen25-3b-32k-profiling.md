# TT-Metal Qwen2.5-3B 32K profiling handoff

이 문서는 현재 서버에서 진행한 TT-Metal/TTNN Qwen2.5-3B long-context profiling 작업을
다른 서버의 Codex가 이어받기 위한 인수인계 문서다. 기준일은 2026-07-21이다.

## 1. 목표와 현재 결론

- Blackhole 계열 장치에서 `Qwen/Qwen2.5-3B-Instruct`의 32K context decode를 분석했다.
- 주 측정 대상은 전체 36-layer 모델이 아니라 **layer 0 하나**다.
- 기본 테스트는 실제 30K prompt를 prefill하지 않는다. 32K paged KV cache를 할당하고
  `current_pos=30314`를 전달해 긴 context에서 토큰 하나를 decode하는 메모리 접근 형태를 재현한다.
- 별도 옵션 `QWEN_32K_PREFILL_CACHE=1`을 켜면 2,048-token chunk 단위의 synthetic prefill로
  KV cache를 먼저 채운 뒤 decode할 수 있다.
- performance와 memory capture는 계측 간섭 때문에 별도 실행/디렉터리로 관리한다.
- SDPA의 자동 TFLOPS/GB/s performance model은 현재 로컬 소스에 없다. 향후 QK/AV matmul
  FLOPs를 shape으로 계산하고, NPE/NoC trace payload로 memory bandwidth를 계산하는 전용
  postprocessor 또는 op performance model이 필요하다.

## 2. 핵심 디렉터리 트리

다른 서버에서는 `/home/iris_hb4`를 새 홈 경로로 치환한다. TT-Metal checkout 이름과
내부 구조도 다를 수 있으므로 이 트리는 **현재 서버의 위치를 설명하는 참고 구조**이지,
새 서버에서 그대로 존재해야 하는 고정 경로가 아니다. 실제 경로는 3절의 탐색 절차로 찾는다.

```text
/home/iris_hb4/
├── .venv/                              # 모든 Python 실행에 사용한 virtualenv
├── TT_METAL_QWEN32K_HANDOFF.md         # 이 문서
├── 2026npu/
│   ├── DEVICE_PROGRAM_PROFILER_GUIDE_KO.md
│   ├── LONG_CONTEXT_ATTENTION_SWEEP_KO.md
│   ├── README.md
│   └── trisc1_MMfusedbiasActivation.txt # TRISC1 ELF/disassembly 조사 결과
├── tt-metal-prof-src/                  # 실제 Metal/TTNN 소스와 build 사용 위치
│   ├── env_set.sh
│   ├── build/
│   ├── generated/
│   ├── ttnn/
│   │   └── cpp/ttnn/operations/
│   │       ├── matmul/device/kernels/compute/
│   │       │   └── bmm_large_block_zm_fused_bias_activation.cpp
│   │       └── transformer/sdpa_decode/device/kernels/compute/
│   │           └── sdpa_flash_decode.cpp
│   └── models/                         # 별도 nested Git repository
│       └── tt_transformers/
│           ├── tests/
│           │   ├── test_qwen_32k_decode_profile.py
│           │   ├── QWEN_32K_PROFILING_GUIDE.md
│           │   ├── qwen_profiling_prompt.md
│           │   └── roofline_tests/
│           └── scripts/op_perf_results.py
└── profiler_runs/                      # 모든 profiler artifact 저장 위치
    ├── qwen25_3b_32k_kernel_zones_2026_07_20_final/ # 권장 performance 성공본
    │   ├── .logs/
    │   │   ├── profile_log_device.csv
    │   │   ├── tracy_ops_data.csv
    │   │   ├── tracy_ops_times.csv
    │   │   └── tracy_profile_log_host.tracy
    │   ├── reports/qwen25_3b_32k_kernel_zones/<timestamp>/
    │   │   ├── ops_perf_results_*.csv
    │   │   ├── profile_log_device.csv
    │   │   └── tracy_profile_log_host.tracy
    │   └── perf_report_visualize/
    │       ├── ops_perf_result.csv
    │       ├── profile_log_device.csv
    │       └── tracy_profile_log_host.tracy
    ├── qwen25_3b_32k_actual_prefill_decode_2026_07_19/
    │   ├── memory_report_visualizer/
    │   │   ├── config.json
    │   │   └── db.sqlite
    │   └── perf_report_visualize/
    │       ├── ops_perf_result.csv
    │       ├── profile_log_device.csv
    │       └── tracy_profile_log_host.tracy
    └── qwen25_3b_32k_single_layer_memory_2026_07_18_16_09_54/
        └── memory_report_visualizer/
            ├── config.json
            └── db.sqlite
```

Visualizer 전달 디렉터리의 계약은 아래처럼 유지한다.

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


### 새 서버에서 경로와 버전을 찾는 방법

TT-Metal 버전에 따라 checkout 이름, Git 경계, Python package 설치 위치, build/generated
디렉터리 및 SDPA/model test의 내부 경로가 달라질 수 있다.

먼저 홈 아래에서 후보 checkout을 찾는다. 시스템 루트 전체를 검색하지 않는다.

```bash
find "$HOME" -maxdepth 4 -type f -name env_set.sh -print
find "$HOME" -maxdepth 5 -type d -path '*/ttnn/cpp/ttnn/operations' -print
find "$HOME" -maxdepth 6 -type d -name tt_transformers -print
```

후보를 찾으면 실제 경로를 변수로 고정한다.

```bash
export TT_METAL_ROOT=/path/to/actual/tt-metal-checkout
export PROFILE_ROOT="$HOME/profiler_runs"
export TT_PYTHON="$HOME/.venv/bin/python"

cd "$TT_METAL_ROOT"
test -f env_set.sh && source ./env_set.sh
mkdir -p "$PROFILE_ROOT"
```

소스와 실제 import 대상이 일치하는지 확인한다.

```bash
"$TT_PYTHON" - <<'PY'
import pathlib
import ttnn
print("ttnn module:", pathlib.Path(ttnn.__file__).resolve())
PY
```

출력은 새 checkout 또는 그 checkout으로 빌드한 wheel을 가리켜야 한다. 이전 서버의 wheel이나
다른 checkout을 가리키면 profile을 실행하지 않는다. source-tree import가 필요한 버전에서는
다음처럼 설정하되, 새 버전이 wheel 설치를 요구하면 해당 README/build 지침을 우선한다.

```bash
export PYTHONPATH="$TT_METAL_ROOT/ttnn:$TT_METAL_ROOT:${PYTHONPATH}"
```

Git 경계와 revision도 각각 확인한다.

```bash
git -C "$TT_METAL_ROOT" rev-parse --show-toplevel 2>/dev/null || true
git -C "$TT_METAL_ROOT" status --short --branch 2>/dev/null || true
git -C "$TT_METAL_ROOT/models" rev-parse --show-toplevel 2>/dev/null || true
git -C "$TT_METAL_ROOT/models" status --short --branch 2>/dev/null || true
```

`models`가 별도 repository인지 여부는 `rev-parse --show-toplevel` 결과로 판단한다. 파일 위치는
고정 경로보다 symbol 검색으로 대응시킨다.

```bash
rg -n 'paged_scaled_dot_product_attention_decode|sdpa_flash_decode' \
  "$TT_METAL_ROOT/ttnn" "$TT_METAL_ROOT/models" 2>/dev/null
rg -n 'cb_matmul_blocks|matmul_block' "$TT_METAL_ROOT/ttnn/cpp" 2>/dev/null
rg -n 'class TransformerBlock|def forward_decode' "$TT_METAL_ROOT/models" 2>/dev/null
```

기존 파일을 경로째 덮어쓰지 않는다. 새 버전의 signature와 control flow를 비교한 뒤 세
`DeviceZoneScopedN` scope만 동등한 QK, softmax, AV 영역에 이식한다. 테스트도 새 checkout의
model API와 기존 테스트를 기준으로 port한다.

profiler CLI와 실행 파일 위치도 로컬 버전에서 다시 확인한다.

```bash
"$TT_PYTHON" -m tracy --help
find "$TT_METAL_ROOT/build" -type f \
  \( -name 'capture-release' -o -name 'csvexport-release' \) -print 2>/dev/null
```

## 3. 저장소 및 브랜치 상태

- `/home/iris_hb4/tt-metal-prof-src` 최상위는 Git repository가 아니다.
- `/home/iris_hb4/tt-metal-prof-src/models`가 Git repository이며 현재 브랜치는
  `iris_hb2-roofline`이다.
- `models`에는 추적되지 않은 파일과 수정 파일이 남아 있다. 다른 서버로 옮길 때 단순히
  branch 이름만 checkout하면 현재 전체 상태가 복원되지 않는다.
- 확인 당시 주요 상태:

```text
M  tt_transformers/tests/roofline_tests/README.md
?? tt_transformers/tests/test_qwen_32k_decode_profile.py
?? tt_transformers/tests/QWEN_32K_PROFILING_GUIDE.md
?? tt_transformers/tests/qwen_profiling_prompt.md
?? tt_transformers/tests/roofline_tests/qwen_like_bandwidth.py
```

최상위 TTNN C++ kernel 수정은 nested `models` Git 상태에 포함되지 않으므로 소스 파일을
별도로 복사하거나 patch로 보존해야 한다.

## 4. 구현한 프로파일링 코드

### Qwen single-layer decode test

파일:
`tt-metal-prof-src/models/tt_transformers/tests/test_qwen_32k_decode_profile.py`

주요 설정:

```text
model                 Qwen/Qwen2.5-3B-Instruct
max_seq_len           32768
decode current_pos    30314
batch                 1
layer                 0 한 개
mode                   decode
paged KV block size    32
max blocks             1024
prefill chunk          2048
```

실행 순서:

```text
model/setup
→ warmup layer
→ synchronize + ReadDeviceProfiler
→ qwen_32k_single_layer_decode signpost
→ measured layer
→ qwen_32k_single_layer_decode_end signpost
→ ReadDeviceProfiler
```

측정 구간 안에서 synchronize, `ReadDeviceProfiler`, host copy를 하면 안 된다.

### Device kernel custom zones

`ttnn/.../sdpa_flash_decode.cpp`에 다음 zone을 추가했다.

```cpp
DeviceZoneScopedN("sdpa_decode_qk_matmul");
DeviceZoneScopedN("sdpa_decode_online_softmax");
DeviceZoneScopedN("sdpa_decode_av_matmul");
```

위치는 각각 QK matmul, online softmax, attention-value matmul 호출을 감싼다. 최종 성공본에서
관찰한 device-zone 통계는 다음과 같다. 여러 core/chunk의 zone이므로 합계를 SDPA wall time으로
해석하면 안 된다.

```text
zone                         count    mean
sdpa_decode_qk_matmul        1422     7.070 us
sdpa_decode_online_softmax   1422     3.405 us
sdpa_decode_av_matmul        1422     8.563 us
```

일반 matmul kernel의
`bmm_large_block_zm_fused_bias_activation.cpp`에는 `DeviceZoneScopedN("matmul_block")`도
추가되어 있다.

## 5. 환경 설정

반드시 `tt-metal-prof-src`의 Metal 코드와 홈의 `.venv`를 함께 사용한다.

```bash
cd /home/iris_hb4/tt-metal-prof-src
source ./env_set.sh
source /home/iris_hb4/.venv/bin/activate

export PYTHONPATH=/home/iris_hb4/tt-metal-prof-src/ttnn:/home/iris_hb4/tt-metal-prof-src:${PYTHONPATH}
export HF_MODEL=Qwen/Qwen2.5-3B-Instruct
export TT_CACHE_PATH=/home/iris_hb4/tt-metal-prof-src/generated/tt_cache_qwen3b
export 'Format=$Format'
```

`PYTHONPATH`의 첫 항목이 local `ttnn`이어야 한다. `Format` export는 현재 로컬 `_ttnn.so`가
잘못된 `GIT_COMMIT_HASH="$Format:%h$"` 문자열로 빌드된 문제를 피하기 위한 임시 workaround다.
새 서버에서 정상적으로 rebuild했다면 제거 가능한지 먼저 확인한다.

## 6. 실행 명령

### Synthetic 32K-context single-layer decode performance

```bash
RUN=/home/iris_hb4/profiler_runs/qwen25_3b_32k_kernel_zones_$(date +%Y_%m_%d_%H_%M_%S)

python -m tracy \
  -r -p -v --check-exit-code \
  -o "$RUN" \
  -n qwen25_3b_32k_kernel_zones \
  -m pytest -s \
  models/tt_transformers/tests/test_qwen_32k_decode_profile.py
```

setup/JIT/warmup 전체를 합산하지 말고 두 signpost 사이만 분석한다.

```bash
python models/tt_transformers/scripts/op_perf_results.py \
  "$RUN"/reports/qwen25_3b_32k_kernel_zones/*/ops_perf_results_*.csv \
  --signpost qwen_32k_single_layer_decode
```

과거 검증값은 measured pass 25 TTNN device ops, single-layer 6,668.4 us,
149.96 tokens/s/user였다. 36 layers를 단순 곱하면 근사치일 뿐이며 embedding, final norm,
LM head, dispatch gap은 포함되지 않는다.

### 실제 synthetic prefill 후 decode

```bash
export QWEN_32K_PREFILL_CACHE=1
pytest -s models/tt_transformers/tests/test_qwen_32k_decode_profile.py
```

현재 구현은 각 prefill chunk 뒤 `ReadDeviceProfiler()`를 호출하여 profiler buffer를 비운다.
따라서 prefill 전체 성능을 정확히 재는 코드라기보다 KV cache를 실제로 채운 뒤 decode를 확인하는
실험용이다. decode 측정 구간에는 이 호출이 들어가지 않는다.

### TTNN Visualizer memory report

```bash
RUN=/home/iris_hb4/profiler_runs/qwen25_3b_32k_memory_$(date +%Y_%m_%d_%H_%M_%S)
export BOS_DISABLE_TRACE=1
export TTNN_CONFIG_OVERRIDES="{\
  \"enable_fast_runtime_mode\": false,\
  \"enable_logging\": true,\
  \"enable_graph_report\": false,\
  \"enable_detailed_buffer_report\": true,\
  \"enable_detailed_tensor_report\": false,\
  \"enable_comparison_mode\": false,\
  \"root_report_path\": \"$RUN/visualizer_raw\",\
  \"report_name\": \"qwen25_3b_32k_single_layer_memory\"\
}"

pytest -s models/tt_transformers/tests/test_qwen_32k_decode_profile.py
```

생성된 hash 디렉터리에서 `config.json`과 `db.sqlite`를
`$RUN/memory_report_visualizer/`로 복사한다. Visualizer 계측은 op별 동기화/DB 기록을
추가하므로 이 실행의 latency는 성능값으로 사용하지 않는다.

## 7. 측정 의미와 주의사항

- `current_pos=30314`만 주고 decode하면 cache 주소 범위는 long context와 같지만, cache 내용은
  실제 prompt에서 생성된 K/V가 아니다.
- `QWEN_32K_PREFILL_CACHE=1`을 사용해야 synthetic 입력으로 K/V cache가 실제로 채워진다.
- prefill은 attention을 포함한 transformer layer 전체를 실행하면서 K/V cache를 채운다.
  단순한 메모리 초기화 전용 단계가 아니다.
- profile CSV의 약 25개 op는 한 layer의 TTNN device operation 수이지 kernel 수가 아니다.
- custom device-zone 시간은 core별/작업 chunk별 timestamp다. zone 평균이나 합을 곧바로 operator
  end-to-end latency로 쓰지 않는다.
- `ReadDeviceProfiler()`는 command queue를 `Finish()`하므로 측정 도중 호출하면 실행을 방해한다.
- profiler buffer가 작아 36 layers 전체 kernel zone을 한 번에 기록하면 overflow 위험이 있다.
  그래서 현재는 한 layer를 측정하고 전체 모델 값은 별도로 검증하는 접근을 사용했다.
- TTNN Python demo가 `ttnn.open_device()`를 직접 호출한다면 예외/freeze 상황에도 닫히도록
  `try/finally`에서 `ttnn.close_device()`를 반드시 호출한다. 현재 테스트는 pytest
  `mesh_device` fixture를 사용한다.

## 8. SDPA TFLOPS와 memory-bandwidth 후속 작업

현재 device profiler의 자동 TFLOPS/BW 열은 주로 matmul performance model에 의존하며,
로컬 `sdpa_decode`에는 `create_op_performance_model`이 없다.

권장 출력은 다음 세 값을 분리하는 것이다.

1. **Algorithmic achieved TFLOPS**: SDPA input shape로 QK와 AV의 FLOPs를 계산하고 SDPA
   device wall duration으로 나눈다.
2. **Logical KV bandwidth**: KV shape/dtype에서 논리적 read bytes를 계산해 duration으로 나눈다.
3. **Measured DRAM bandwidth**: NPE/NoC event trace에서 SDPA 구간의 실제 DRAM read/write
   payload를 합산해 duration으로 나눈다.

Decode attention matmul FLOPs의 기본식:

```text
QK FLOPs = 2 * B * Hq * context_length * head_dim
AV FLOPs = 2 * B * Hq * context_length * value_dim
```

softmax FLOPs는 관례와 구현별 차이가 크므로 matmul FLOPs와 분리해 표기한다. hardware counter의
FPU utilization도 algorithmic TFLOPS와 별도 열로 두는 것이 안전하다. NPE trace가 실제 payload를
제공하기 전에는 logical BW를 measured BW라고 부르면 안 된다.

참고할 최신 공식 문서:

- TT-NN profiling: <https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html>
- Device program profiler: <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html>
- Performance/NoC tools: <https://docs.tenstorrent.com/tt-lang/reference/performance-tools.html>

새 서버의 checkout/build가 동일 기능을 지원하는지는 `python -m tracy --help`, profiler build flags,
NoC event trace 환경변수를 직접 확인해야 한다. 최신 문서는 기능 기반을 설명하지만 SDPA 전용
TFLOPS/BW 자동 모델이 이미 제공된다는 증거는 확인하지 못했다.

## 9. TRISC1/matmul 조사 메모

- compute ELF/disassembly는 TRISC1을 중심으로 조사했다.
- `2026npu/trisc1_MMfusedbiasActivation.txt`에서 `ttmvmul` 명령이 반복되는 것을 확인했다.
- TTNN `matmul_block`은 상위 tile/block API이며 하위 구현에서 여러 `TTI_MVMUL`/`ttmvmul`을
  조합해 32x32 tile matmul을 수행한다.
- `8x16 = 8x16 @ 16x16` 같은 micro-operation 하나만으로 전체 32x32가 끝나는 것은 아니다.
  row/column half와 K accumulation을 위한 여러 호출이 필요하다.

## 10. 다른 서버로 옮길 항목 및 시작 체크리스트

최소 복사 대상:

```text
tt-metal-prof-src/models/tt_transformers/tests/test_qwen_32k_decode_profile.py
tt-metal-prof-src/models/tt_transformers/tests/QWEN_32K_PROFILING_GUIDE.md
tt-metal-prof-src/models/tt_transformers/tests/qwen_profiling_prompt.md
tt-metal-prof-src/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp
tt-metal-prof-src/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp
2026npu/
profiler_runs/qwen25_3b_32k_kernel_zones_2026_07_20_final/
profiler_runs/qwen25_3b_32k_actual_prefill_decode_2026_07_19/
```

새 서버에서 먼저 할 일:

1. 장치 종류, firmware, TT-Metal checkout/commit, profiler-enabled build 여부를 기록한다.
2. Qwen 지원 branch를 checkout하고 위의 untracked/C++ 변경을 patch로 적용한다.
3. 홈 `.venv`를 만들고 local TTNN이 import되는지 `ttnn.__file__`로 확인한다.
4. 모델 weight/cache 접근 권한과 `HF_MODEL`, `TT_CACHE_PATH`를 확인한다.
5. profiler 없이 single-layer test를 먼저 실행한다.
6. Tracy 실행 후 두 signpost와 세 SDPA custom zone이 raw device CSV에 있는지 확인한다.
7. memory report와 performance report를 반드시 별도 실행으로 생성한다.
8. 새 측정값은 장치/commit/build가 같을 때만 기존 수치와 직접 비교한다.

더 자세한 기존 설명은
`tt-metal-prof-src/models/tt_transformers/tests/QWEN_32K_PROFILING_GUIDE.md`와
`2026npu/DEVICE_PROGRAM_PROFILER_GUIDE_KO.md`를 먼저 읽는다.

GEMM benchmark는 `$HOME/GEMM_BENCHMARK_GUIDE_KO.md`를 참고한다.

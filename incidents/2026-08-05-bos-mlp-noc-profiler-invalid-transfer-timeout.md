# BOS MLP NoC profiler invalid-transfer abort 및 timeout

- 날짜: 2026-08-05 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- 영향: NoC capture 불완전, profiler child D-state 잔존, 장치 재격리

## 요약

재부팅 뒤 32×32 BF16 add와 동일 12-compute balanced MLP의 profiler-free correctness는 정상
통과했다. 이어 `--collect-noc-traces` isolated capture에서 MLP 자체는 PCC 0.9996410623과
`MLP_COMPLETED`까지 도달했다. Device profiler가 결과를 파일로 변환하는 중
`Invalid NoC transfer type on device: 0`으로 abort했다. `DEVICE_CLOSED`는 출력되지 않았고 외부
timeout이 exit 124로 끝났다. PID 4339는 PID 1 아래 `D` 상태로 남았다.

## 타임라인

- 03:50 UTC: 재부팅 뒤 첫 32×32 add. `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0.
- 03:56 UTC: profiler-free isolated MLP. PCC 0.9996410623, 1.493424 ms,
  `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0.
- 03:57:22 UTC: isolated NoC capture 시작.
- 03:57:40 UTC: PCC 0.9996410623, 1.507990 ms, `MLP_COMPLETED` 출력 뒤 profiler fatal.
- 약 03:59 UTC: 120초 SIGINT timeout 종료. 최종 exit 124. `DEVICE_CLOSED` 없음.
- 종료 뒤: PID 4339 `D`, PPID 1. 추가 device workload 없음.

## 실행 구성

- MLP: Llama 3.2 3B layer 0, batch 1
- weight: BFP8, DRAM width-sharded
- readers/compute: 12/12 fanout-2
- endpoint destination: NOC1 4:4:4
- W2 `in0_block_w=16`
- 16 KiB read-page cap, tagged two-block enabled
- correctness/JIT 1회와 measured 1회
- Watcher, counter, fused gate/up epilogue, TurboQuant: off

```bash
env -u TT_METAL_MLP_FUSED_GATE_UP -u TT_METAL_MLP_FUSED_GATE_UP_EPILOGUE \
  -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER \
  PATH=/home/iris_hb4/tt-metal-hb4/python_env/bin:/usr/local/bin:/usr/bin:/bin \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 \
  TT_METAL_TURBOQUANT=0 timeout --signal=INT --kill-after=15s 120s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy -p -r --sync-host-device \
  --collect-noc-traces --check-exit-code \
  -o /home/iris_hb4/profiler_runs/mlp_fanout2_tagged_balanced_noc_idle_2026_08_05_04_00_00/perf_capture \
  -n mlp_fanout2_tagged_balanced_noc_idle \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## 오류

```text
TT_FATAL: Invalid NoC transfer type on device: 0.
TT_FATAL @ tt_metal/impl/profiler/profiler.cpp:448:
EMD::isValidEventType(EMD(markers[i].data).data.raw_event.noc_xfer_type)
```

Backtrace는 `coalesceFabricEvents`, `convertNocTracePacketsToJson`,
`DeviceProfiler::writeDeviceResultsToFiles`를 가리킨다. MLP kernel completion 뒤 profiler 결과 변환에서
발생했다. 따라서 이 오류 자체는 MLP kernel deadlock 증거가 아니다. 그러나 close가 없고 child가 D
상태이므로 장치 상태는 신뢰하지 않는다.

## Artifact

Root: `/home/iris_hb4/profiler_runs/mlp_fanout2_tagged_balanced_noc_idle_2026_08_05_04_00_00`

- `profile_log_device.csv`: 4,133,426 B,
  SHA-256 `26a3143932d60cad39255331e6839d33c2e8d3729e3a2115e3301605ce7aa286`
- `tracy_profile_log_host.tracy`: 589,312 B,
  SHA-256 `d1f224b9477eda7b9a41b4719bbf63f6732c075929a20d32eba40bb7d6555c7d`
- raw `noc_trace_dev0_ID*.json`: 없음
- ops CSV: 없음

따라서 성공한 NoC capture로 분류하지 않는다.

## 제한적 marker 분석

raw NoC trace 대신 `MLP_IN1_ISSUED`부터 `MLP_IN1_DRAM_DONE`까지를 reader별 outstanding DRAM read
구간으로 해석했다. 이는 software marker 기반 pending-request 근사치다. 실제 NoC link busy 또는 NPE
utilization이 아니다.

Measured projection host ID는 W1 7168, W3 8192, W2 11264다.

| projection | 평균 pending | 최대 pending | pending ≥12 시간 | 내부 pending-empty 시간 |
|---|---:|---:|---:|---:|
| W1 | 18.43 | 24 | 94.56% | 0% |
| W3 | 18.52 | 24 | 95.07% | 0% |
| W2 | 16.37 | 22 | 94.27% | 0% |

Phase 경계의 pending-empty gap은 W1→W3 15.185 µs, W3→W2 70.122 µs다. 합계 85.307 µs는
measured 1.507990 ms의 5.66%다.

따라서 기존 약 67% consumer input wait를 global NoC-empty 시간으로 해석할 수 없다. 각 projection
내부에는 항상 하나 이상의 DRAM read가 pending이고 대부분 12개 이상이다. 새 inter-core pipeline은
phase gap 일부를 숨길 수 있으나, request service rate를 개선하지 않으면 상한이 작다.

## 복구와 예방

- exit 124와 D-state child 때문에 장치를 즉시 격리했다.
- incident 뒤 add, open/close, MLP 등 추가 device workload를 실행하지 않았다.
- 다음 device workload 전 사용자의 host/server 재시작 확인이 필요하다.
- 같은 `--collect-noc-traces` 구성을 재실행하지 않는다.
- profiler의 event type 0 처리와 raw packet 생성 경로를 host-side에서 먼저 조사한다.
- pipeline 구현 전 기존 12-reader 경로의 DRAM service latency와 phase boundary만 별도 안전 marker로
  계측한다.

## 2026-08-18 재발: actual vanilla endpoint trace

Actual opt-in-off isolated MLP의 미계측 reader/endpoint mapping을 확인하기 위해 같은 NoC profiler를
사용했고 동일한 cleanup signature가 재발했다. Profiler-free gate는 PCC `0.9996410623`, measured
`2.291724 ms`, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0으로 통과했다. 이어진 capture의 MLP도 PCC
`0.9996410623`, `2.372136 ms`, `MLP_COMPLETED`까지 도달했지만 NoC JSON 변환에서 같은 fatal이 발생했다.
`DEVICE_CLOSED`는 없었고 180초 외부 timeout이 exit 124로 종료됐다. 종료 직후 profiler Python child는 PID 703267, PPID 1, `D` state였고, 최종 host-side 확인에서는
`Z`/<defunct>로 전이됐다. Zombie는 반복 signal 대상이 아니며 PID 1의 reap을 기다린다. 추가 device workload는 실행하지 않았고 장치를 재격리했다.

실행된 profiler child command는 다음과 같다.

```text
python3 -m tracy -v -p -o /home/iris_hb4/profiler_runs/mlp_actual_vanilla_endpoint_trace_2026_08_18/noc_capture_valid -n actual_vanilla_mlp --collect-noc-traces -m models.bos_model.llama32.tests.run_mlp_block_width_ab
```

Capture는 MLP opt-in을 모두 끈 generic DRAM-interleaved path, `MLP_AB_ITERATIONS=1`, default accuracy
precision을 사용했다. Raw trace root는
`/home/iris_hb4/profiler_runs/mlp_actual_vanilla_endpoint_trace_2026_08_18/noc_capture_valid/.logs`다.

- `profile_log_device.csv`: 12,239,276 B,
  SHA-256 `3be904f39143246c2704e53b966e3380d7033a2b8c0603bb5063368e162b6946`
- `tracy_profile_log_host.tracy`: 1,309,165 B,
  SHA-256 `8142ddef9c456164b8f668f58db094a0e9d36a6369154f2d358039f8c1eff68`
- converted NoC JSON/complete ops CSV: 없음

이번 raw CSV는 converter fatal의 범위를 더 좁혔다. NoC event timer `12345`의 여섯 MLP host IDs에는
invalid event가 0개였다. Type 0은 별도 timestamp timer `15282`와 `58581`의 data-zero marker에서 각
op/core마다 발생했다. Converter가 timestamped datapoint 전체를 NoC metadata로 넘기면서 non-NoC marker를
오인한 것이 직접 fatal 조건이다. Raw timer `12345`는 별도 해독 가능했고, measured W1/W3/W2 각각에서
16 BRISC readers와 physical endpoints `(2,3)/(3,3)`의 8:8 분포를 복구했다. 이 복구는 endpoint evidence로
사용하되 capture 전체를 성공으로 승격하지 않는다.


## 2026-08-18 재발: performance datatype-matched endpoint trace

사용자 승인과 재부팅 뒤 32×32 add gate, profiler-free performance vanilla 및 layout A/B가 모두 정상
close/exit 0으로 통과한 뒤 actual vanilla NoC capture를 isolated 1회 실행했다. MLP opt-in은 모두 껐고
`MLP_PRECISION_MODE=performance`, `MLP_PCC_THRESHOLD=0.98`, `MLP_AB_ITERATIONS=1`을 사용했다.

MLP는 PCC `0.9869040195`, latency `2.101122 ms`, `MLP_COMPLETED`까지 도달했다. 이후
`DeviceProfiler::writeDeviceResultsToFiles()`의 NoC JSON 변환에서 다시 `Invalid NoC transfer type on device: 0`으로
abort했다. `DEVICE_CLOSED`는 없었고 120초 SIGINT와 15초 cleanup 상한 뒤 SIGKILL/exit 137이었다. Python PID
18802는 D-state에서 PID 1 아래 Z/<defunct>로 전이했다. 추가 device workload는 실행하지 않았고 장치를 즉시
격리했다.

Artifact root: `/home/iris_hb4/profiler_runs/mlp_actual_vanilla_performance_endpoint_trace_2026_08_18`

- `noc_capture/.logs/profile_log_device.csv`: 12,239,154 B, 164,578 lines, SHA-256
  `b1b4fd2532f1e6ba7f8924e40276360fc69e67a0ba6a1de23e23b8c349cd4a94`
- `console.log`: 15,778 B, SHA-256
  `550ddcddc9e5912624e0d0c94f3d3bc0b5ef16627351cb909861d89159e908a6`
- converted NoC JSON, complete ops CSV, `DEVICE_CLOSED`: 없음

Raw timer `12345`의 measured host IDs W1/W3/W2 `5120/6144/8192`는 각각 16 BRISC readers를 포함했다.
Physical endpoints `(2,3)/(3,3)`에 reader가 8:8로 배치됐고 모두 NoC1이었다. Endpoint당 request/byte는
W1/W3 BFP4 `12,288 / 7,077,888 B`, W2 BFP8 `12,288 / 13,369,344 B`였다. Accuracy capture와
reader/endpoint mapping 및 request count는 같고 BFP4 W1/W3 payload만 줄었다. 이 raw endpoint evidence는
사용할 수 있지만 capture 전체는 성공으로 승격하지 않는다.

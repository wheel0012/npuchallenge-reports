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

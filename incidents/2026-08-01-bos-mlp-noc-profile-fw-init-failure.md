# Incident report: BOS MLP NoC profiling 후 worker-FW initialization failure

- 발생일: 2026-08-01 UTC
- 장치: Blackhole 코드/runtime을 사용하는 custom 20-core BOS NPU
- 대상 작업: Llama 3.2 3B decode MLP DRAM-sharded weight 및 NoC trace A/B
- 상태: 장치 실행 중단, out-of-band 복구 필요
- 이전 incident와의 관계: 2026-07-31 PCIe-link reset/host-freeze 사건과 별개

## 1. 요약

MLP weight를 DRAM-sharded로 배치하고 기존 DRAM-sharded matmul program config를 opt-in으로
활성화하는 실험을 진행했다. 실험 전 32×32 `ttnn.add` smoke test는 정상 완료됐고, profiler를
사용하지 않은 baseline 및 DRAM-sharded MLP correctness test도 모두 완료됐다.

이후 `--collect-noc-traces`를 사용한 profiler run이 warmup 또는 report 생성 단계에서 60초
timeout됐다. profiler 종료 뒤 다시 실행한 32×32 add는 장치를 열고 worker program을 시작하는
단계에서 결과를 반환하지 못한 채 45초 timeout됐다. 사용자는 이어서 보드가
`failed to initialize FW` 상태임을 확인했다.

현재 증거는 worker firmware가 Tensix workload를 시작하지 못하는 상태로 전이했음을 보여주지만,
NoC profiler, instrumented kernel compile/launch, 기존 device state 중 무엇이 근본 원인인지는
확정하지 못했다. DRAM-sharded MLP 자체는 profiler 없이 정상 완료됐고, interleaved baseline을
대상으로 한 NoC 수집도 timeout됐으므로 이 사건을 DRAM-sharded reader의 kernel deadlock으로
단정할 수 없다.

이번 조사에서는 device reset, PCIe link reset, driver unbind/rebind를 수행하지 않았다.
재시도도 중단했다.

## 2. 영향

- 장치가 minimal 32×32 add를 실행하지 못하는 상태가 됐다.
- MLP latency 및 NoC/DRAM bandwidth A/B 결과를 얻지 못했다.
- 네 개의 profiler run 디렉터리가 불완전한 상태로 남았다.
- 두 개의 Tracy `capture-release` process가 PID 1 아래 zombie로 남았다. 둘 다 `Z` 상태이므로
  실행 중인 profiler나 장치 process는 아니지만, host reboot 전까지 PID entry가 남을 수 있다.
- host freeze나 PCI function revision `ff`는 이번 incident에서 관측되지 않았다.

## 3. 실험 변경

변경 파일:

```text
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tt/mlp.py
```

새 opt-in 환경변수:

```text
TT_METAL_MLP_DRAM_SHARDED=1
```

옵션이 활성화되면 다음이 적용된다.

- W1/W3: 기존 `create_dram_sharded_mem_config()` 결과 사용
- W2: 별도의 DRAM width-sharded memory config 사용
- W1/W3: 기존 `DECODE_MLP_W1_W3_PRG_CONFIG` 전달
- W2: 기존 `DECODE_MLP_W2_PRG_CONFIG` 전달
- interleaved weight cache와 충돌하지 않도록 `.dram_sharded` cache suffix 사용

환경변수가 없으면 기존 interleaved memory config와 자동 matmul program 선택을 유지한다.
Attention K/V chunk 또는 SDPA kernel은 이번 변경에서 수정하지 않았다.

## 4. 하드웨어 해석 주의사항

이 장치는 표준 P100/P150이 아니라 Blackhole 코드와 runtime architecture를 사용하는 custom
20-core BOS NPU다. `model_config.py`는 runtime DRAM grid width가 7이면 P100, 그 외이면 P150으로
이름을 추정하기 때문에 이번 실행에서는 P150으로 출력됐다. 이 문자열은 authoritative board
identity가 아니다.

UMD는 다음 topology inconsistency warning도 출력했다.

```text
Board p100 expects 2 Tensix units harvested, but harvest mask indicates 0
Board p100 expects 1 DRAM unit harvested, but harvest mask indicates 0
Board p100 expects 14 ETH units harvested, but harvest mask indicates 0
```

이 warning이 custom BOS descriptor와 표준 product heuristic의 불일치인지, 실제 firmware
initialization failure와 관련된 상태 이상인지는 확인되지 않았다.

DRAM-sharded MLP 실행 시 Metal은 다음을 출력했다.

```text
Dram Interface Workers: 6
```

이는 해당 matmul data path가 6 interface workers를 선택했다는 뜻이다. physical bank 수,
logical DRAM shard 수 또는 active compute-core 수와 같은 개념으로 해석하지 않는다.

## 5. 타임라인

### 11:00 UTC — 사전 smoke test 성공

장치 0에서 32×32 BF16 tensor 두 개를 더했다.

```text
SMOKE_VALUE 2.0
```

worker program이 정상 실행되고 device close까지 완료됐다.

### 11:02 UTC — baseline MLP correctness 성공

기존 interleaved weight 경로로 Llama 3.2 3B decode MLP, sequence length 32를 실행했다.

```text
PCC: 0.9996355767898077
MLP Passed!
```

### 11:03 UTC — DRAM-sharded MLP correctness 성공

`TT_METAL_MLP_DRAM_SHARDED=1`로 동일 test를 실행했다.

```text
Dram Interface Workers: 6
PCC: 0.9996585822234915
MLP Passed!
```

따라서 opt-in weight layout과 matmul program config는 적어도 이 단일 non-profiled decode
correctness case에서 kernel completion과 PCC 기준을 만족했다.

### 11:04 UTC — profiler command 호환성 오류 두 건

첫 시도는 virtualenv가 활성화되지 않아 `/usr/bin/python3: No module named tracy`로 종료됐다.
두 번째는 wrapper가 전체 command를 Python module name으로 해석해
`ImportError: No module named python`으로 종료됐다. 두 run은 device workload에 진입하지
않았으므로 firmware failure의 직접 trigger로 보지 않는다.

### 11:05 UTC — full-layer interleaved NoC profile timeout

직접 `python -m tracy -p -r --collect-noc-traces` 형식으로 8K cur-pos-only single layer를
실행했다. device profiler와 model load는 시작됐지만 60초 내 report가 완성되지 않았다.
timeout 후 test process는 사라졌고 `capture-release`가 zombie로 남았다. 유효한 ops CSV나
NoC result는 생성되지 않았다.

실제 실행 process 구조는 다음과 같았다.

```text
timeout 60s
└── bash -lc
    └── python -m tracy ...
        ├── profiled test process
        └── capture-release
```

따라서 timeout의 직접 child는 Tracy가 아니라 바깥 `bash`였다. 60초 상한에서 SIGTERM이 먼저
shell에 전달됐으므로 Tracy의 signal handler와 device-profiler cleanup이 정상 완료됐다고 보장할 수
없다. test process가 사라진 뒤 `capture-release`만 orphan/zombie로 남고 report가 불완전했던 점은
teardown 미완료 가설과 일치하지만, 이것만으로 firmware 장애의 단일 원인이라고 확정하지는 않는다.

이 run은 `TT_METAL_MLP_DRAM_SHARDED`가 꺼진 baseline interleaved full-layer 구성이다. 따라서
상태 전이의 최초 후보를 DRAM-sharded MLP reader 자체로 좁힐 수 없다.

### 11:08 UTC — MLP-only runner compile typo

첫 MLP-only stdin runner는 Python import typo로 compile 단계에서 종료됐다. 장치는 열리지 않았다.

### 11:09 UTC — MLP-only interleaved NoC profile timeout

오타를 수정한 baseline interleaved MLP-only profiler는 device를 열고 model/weight cache를
load했지만, warmup completion marker인 `WEIGHT_MEMORY_CONFIG`를 출력하기 전에 60초 timeout됐다.
DRAM-sharded 환경변수는 이 run에서 사용하지 않았다. 유효한 device ops CSV나 NoC result는
생성되지 않았으며 두 번째 `capture-release` zombie가 남았다.

이 run은 첫 full-layer NoC timeout 뒤 실제 MLP worker completion을 다시 시도한 첫 실행이었다.
device open과 weight load까지 진행됐지만 warmup 완료 marker에는 도달하지 못했다. 두 NoC run 사이에
minimal add smoke를 실행하지 않았으므로 정확한 상태 전이 시점을 독립적으로 측정한 것은 아니다.
다만 두 번째 baseline run도 warmup을 끝내지 못했다는 사실은 첫 full-layer timeout 시점에 이미
device/profiler lifecycle 상태가 손상됐을 가능성을 강화한다.

### 11:10 UTC — 사후 add smoke 실패

reset 없이 32×32 add를 다시 실행했다. device open과 topology discovery는 진행됐지만
`POST_PROFILE_SMOKE` 결과가 출력되지 않았고 외부 45초 timeout이 exit code 124로 종료했다.
사용자는 이후 보드에서 `failed to initialize FW`가 발생했다고 보고했다.

이 시점부터 모든 device workload를 중단했다.

### 강제종료 사용 내역

- correctness command에도 외부 timeout 상한은 있었지만 baseline과 DRAM-sharded run 모두 정상
  종료했고 signal은 발동하지 않았다.
- full-layer interleaved NoC run과 MLP-only interleaved NoC run은 각각 실제 60초 timeout과 SIGTERM을
  겪었다. 두 경우 모두 `timeout -> bash -lc -> Tracy` 구조였다.
- 이후 host-side 정리를 위해 `capture-release`에 SIGTERM을 보내고 SIGKILL도 시도했으나, 확인 당시
  PID 10435와 12574는 이미 `Z`/`<defunct>` 상태였다. 즉 종료된 zombie에 signal을 보낸 것이며
  실행 중인 device workload를 강제 종료한 것은 아니다.
- 이번 조사에서는 device reset, PCIe reset, driver rebind 또는 reset ioctl을 실행하지 않았다.

## 6. 불완전한 artifact

다음 디렉터리는 성공 결과가 아니며 성능 비교나 Visualizer 전달에 사용하지 않는다.

```text
/home/iris_hb4/profiler_runs/mlp_decode_8k_interleaved_noc_2026_08_01_11_04_00
/home/iris_hb4/profiler_runs/mlp_decode_8k_interleaved_noc_retry_2026_08_01_11_05_00
/home/iris_hb4/profiler_runs/mlp_decode_only_interleaved_noc_2026_08_01_11_07_00
/home/iris_hb4/profiler_runs/mlp_decode_only_interleaved_noc_retry_2026_08_01_11_09_00
```

관측된 파일 상태:

- 첫 wrapper run의 Tracy file은 684 bytes이고 ops CSV는 header 수준이다.
- full-layer retry에는 topology와 zone-location metadata만 있다.
- compile-typo MLP-only run의 host Tracy file은 217,452 bytes지만 device workload가 없다.
- MLP-only retry에는 topology와 zone-location metadata만 있고 device ops report가 없다.

Zombie host processes:

```text
PID 10435 [capture-release] <defunct>
PID 12574 [capture-release] <defunct>
```

## 7. 원인 평가

### 확인된 사실

1. incident 전 minimal Tensix add가 성공했다.
2. baseline과 DRAM-sharded MLP는 profiler 없이 모두 kernel completion 및 PCC 검증에 성공했다.
3. baseline interleaved full-layer NoC run이 최초로 실제 timeout/SIGTERM을 겪었다.
4. 그 직후의 baseline interleaved MLP-only run은 warmup marker 전에 timeout됐고, 두 run 사이에
   minimal smoke는 없었다.
5. 두 번째 timeout 뒤 minimal add도 완료되지 않았다.
6. 사용자가 `failed to initialize FW`를 확인했다.
7. 이번 incident에서는 reset 또는 PCIe state 변경을 수행하지 않았다.

### 가장 이른 상태 전이 지점

현재 확보된 순서는 다음과 같다.

```text
DRAM-sharded MLP correctness 정상 종료
→ baseline full-layer interleaved NoC profile 강제종료
→ baseline MLP-only warmup 미완료
→ minimal add 미완료 / failed to initialize FW
```

따라서 가장 이른 관측 가능한 상태 전이 후보는 full-layer interleaved NoC profile의 timeout이다.
이 위치에 대한 신뢰도는 중간 이상, 바깥 `bash` 때문에 profiler cleanup이 우회됐다는 가설은 중간,
정확한 firmware 내부 root cause는 낮음으로 평가한다.

### 가능한 원인

- baseline full-layer의 NoC-instrumented kernel compile/launch 또는 profiler lifecycle이 worker-FW
  상태를 비정상으로 남김
- `timeout -> bash -lc -> Tracy` process 구조에서 SIGTERM이 shell에 먼저 전달되어 device profiler
  teardown이 완전하지 않았음
- custom BOS topology와 표준 Blackhole/P100 harvesting metadata 불일치
- 이전 incident 이후 잠재한 firmware/KMD/device state가 profiler 부하에서 다시 드러남

### 현재 증거로 지지되지 않는 결론

- DRAM-sharded MLP reader가 deadlock됐다는 결론
- 6 DRAM interface workers 자체가 문제라는 결론
- W1/W3/W2의 correctness failure
- NoC 또는 DRAM bandwidth saturation 때문에 firmware initialization이 실패했다는 결론

traffic pattern과 profiler lifecycle을 구분해야 한다. 현재 증거는 특정 DRAM bank의 포화나 6 interface
worker traffic보다 full-layer NoC capture의 시작·timeout·불완전 teardown 경로를 더 강하게 가리킨다.
그러나 Watcher, kernel entry trace 또는 이전 boot의 firmware/KMD log가 없어 정확히 어느 단계에서
worker FW가 멈췄는지는 확인하지 못했다. 근본 원인 신뢰도는 낮다.

## 8. 금지 조치

- 이 상태에서 MLP, SDPA 또는 add를 반복 실행하지 않는다.
- device open/close를 복구 수단으로 반복하지 않는다.
- 직접 `RESET_PCIE_LINK`, `RESTORE_STATE`, FLR, PCI remove/rescan을 실행하지 않는다.
- driver unbind/rebind 또는 UMD/KMD reset을 interactive session에서 실행하지 않는다.
- 불완전 profiler 디렉터리를 성공 run으로 취급하지 않는다.

## 9. 안전한 복구 및 다음 단계

사용자는 2026-08-02 아침에 server/host를 재시작할 예정이다. 사용자가 재시작 완료를 명시적으로
확인하기 전에는 device open을 포함한 어떤 장치 작업도 수행하지 않는다.

1. 재시작 뒤 workload보다 먼저 이전 boot의 `journalctl -k -b -1`, PCIe AER 및 BOS KMD/firmware
   log를 보존한다. 이전 boot log가 persistent journal에 없으면 그 사실도 기록한다.
2. custom 20-core BOS NPU의 harvesting/topology metadata가 KMD/UMD에 올바르게 전달되는지 host-side
   log로 확인한다. 표준 P100/P150 heuristic 출력은 board identity로 사용하지 않는다.
3. 외부 timeout이 있는 32×32 add를 딱 한 번 실행한다. 실패, timeout 또는 FW init 오류가 나면
   추가 open/close와 workload 없이 즉시 장치를 다시 격리한다.
4. add가 통과한 경우 baseline MLP와 DRAM-sharded MLP correctness를 profiler 없이 각각 한 번 실행한다.
5. 두 non-profiled MLP가 통과한 뒤 isolated baseline MLP를 Watcher로 한 번 실행한다. host-side에는
   compile begin/end, kernel load/enqueue, launch, synchronize begin/end marker를 남기고, Watcher에서는
   core별 kernel entry, 마지막 실행 RISC, CB/semaphore/barrier 및 NoC stall을 확인한다.
6. Watcher 결과는 다음 기준으로 실패 구간을 분류한다.
   - compile-end가 없으면 compile 단계
   - compile-end는 있지만 kernel entry가 없으면 binary load, enqueue 또는 worker-FW launch 경로
   - 일부 core만 entry하면 core dispatch 또는 worker-FW 경로
   - 모든 core가 entry한 뒤 CB/semaphore/barrier에서 멈추면 kernel synchronization 경로
   - 모든 kernel이 완료되고 report만 없으면 profiler capture/teardown 경로
7. Watcher 실행까지 정상 완료된 뒤, NoC trace 전에 동일 binary/kernel cache로 짧은 latency A/B를
   완료한다.
8. 첫 NoC capture는 full layer가 아니라 isolated MLP operation, warmup 1회와 measured call 1회로
   제한한다.
9. virtualenv와 환경변수를 먼저 구성하고 timeout의 직접 child가 Tracy process가 되게 한다.

   ```bash
   source /home/iris_hb4/tt-metal-hb4/python_env/bin/activate
   timeout --signal=INT --kill-after=15s 90s /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy <options> <script> <args>
   ```

10. exit code 124/137, kill-after 발동, child 잔류 또는 report 불완전 중 하나라도 발생하면 그 session의
   device state는 오염된 것으로 간주한다. add smoke를 포함해 어떤 장치 workload도 더 실행하지 않는다.
11. timeout 뒤에는 host-side process와 artifact만 확인한다. zombie는 반복 signal하지 않고 reboot/reap을
   기다리며, reset ioctl·PCIe reset·driver rebind로 복구하지 않는다.
12. 각 run에 exact command, signal/exit code, 마지막 완료 marker, child 상태와 artifact 완성 여부를
    함께 기록한다.

## 10. 현재 소스 상태

MLP 변경은 opt-in이며 기본값은 off다. Python syntax 및 formatter 검증은 통과했다.
성능과 NoC 결과는 미검증 상태다. 장치 복구 전에는 추가 build 또는 device validation을 수행하지
않는다.


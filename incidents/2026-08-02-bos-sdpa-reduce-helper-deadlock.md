# BOS SDPA reduce-only helper deadlock 및 timeout SIGKILL

- 일시: 2026-08-02 07:07--07:10 UTC
- 장치: Blackhole 코드/runtime을 사용하는 custom 20-core BOS NPU
- 상태: 장치 격리, host/server 재시작 전 device workload 금지
- 범위: isolated paged SDPA decode 한 번, Tracy/NoC profiler 없음, Watcher 사용
- 결과: completion 없음, timeout cleanup 실패, exit code 137

## 1. 영향

KV head 하나의 최종 attention partial reduction을 idle worker에 offload하는 첫 opt-in POC가 device
kernel launch 뒤 진행하지 않았다. 외부 timeout은 Python에 SIGINT를 전달했지만 C++ device call이
반환되지 않아 cleanup이 15초 안에 완료되지 않았고 SIGKILL이 발동했다.

`REDUCE_ONLY_HELPER_CORRECTNESS_PASS`와 `DEVICE_CLOSED`는 출력되지 않았다. 종료 뒤 Python PID
6251은 PID 1 아래 `Z`/`<defunct>` 상태였다. zombie에는 추가 signal을 보내지 않았다. exit 137 이후
add smoke를 포함한 추가 device workload, device reopen, reset, driver rebind는 수행하지 않았다.

현재 firmware 상태는 확인하지 않았다. 확인 자체가 새 device open을 요구하므로, 사용자가 다음
host/server 재시작 완료를 명시적으로 알리기 전까지 장치를 격리한다.

## 2. 실험 목적과 변경 구조

16개 active SDPA core가 K/V를 계속 직접 읽고, KV head 7의 두 core만 final partial producer로
동작하게 했다. 기존 idle core 하나가 두 `(m,l,O)` partial을 받아 stable softmax correction,
normalization과 interleaved output write를 수행하도록 했다.

```text
head 7 producer A -- (m0,l0,O0) --+
                                      +--> helper logical (1,3) --> output DRAM
head 7 producer B -- (m1,l1,O1) --+
```

기본 경로는 변경하지 않고 다음 환경변수로만 활성화했다.

```text
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1
```

새 kernel은 다음 두 파일이다.

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode_reduce_only.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_reduce_only.cpp`

## 3. 실행 전 검증

재시작 직후 필수 32x32 BF16 `ttnn.add`는 `SMOKE_VALUE 2.0`, exit 0과 명시적
`DEVICE_CLOSED`로 완료됐다. 이어 기존 direct DRAM saturation benchmark도 84.482 GB/s,
`Test Passed`, exit 0과 정상 device close로 끝났다.

POC의 host-side 검증은 다음을 통과했다.

```text
sdpa_decode_program_factory.cpp.o build: success
ttnncpp shared library build: success
```

첫 장치 사전 시도는 sharded GQA output을 사용해 host validation에서
`Sharded output not supported for GQA`로 거부됐다. device kernel은 launch되지 않았고 Watcher detach,
device close와 exit 1이 정상적으로 완료됐다. 이후 실제 지원 경로인 interleaved output writer로
수정하고 다시 host build를 완료했다.

## 4. 마지막 실행 구성

```text
B=1
Q heads=24, padded Q heads=32
KV heads=8
head dimension=128
context=65,536
cur_pos=65,535
page block=32
K chunk=256 tokens
program grid=5x4
active reader/compute cores=16
idle reduce-only helper cores=1
DRAM=3 physical banks, 6 worker NoC endpoints
NoC0/NoC1 reader loads=8/8
endpoint loads=3/2/3/3/3/2
K/V=BFP8_B interleaved DRAM
Q=BF16 height-sharded L1
output=BF16 interleaved DRAM
Watcher interval=1 second
Tracy/NoC capture=disabled
```

실행 명령의 핵심 wrapper는 다음과 같다. `timeout`의 직접 child는 Python process였다.

```bash
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  TT_METAL_WATCHER=1 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_FORCE_JIT_COMPILE=1 \
  TT_METAL_SDPA_DECODE_DUAL_NOC=1 \
  TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 \
  TT_METAL_SDPA_DECODE_TAGGED_ASYNC=1 \
  TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1 \
  TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1 \
  TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1 \
  timeout --signal=INT --kill-after=15s 150s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -c '<isolated paged SDPA call>'
```

## 5. 타임라인

- 07:07:35: Python 시작
- 07:07:37: Watcher attach
- 07:07:43: dual-NoC 6-endpoint mapping, reduce-only helper와 tagged prefetch 활성화 로그
- 이후: 새 helper compute/writer kernel ID 5/11 확인, completion 없음
- 약 13초부터 163초까지: Watcher waypoint가 같은 상태로 반복
- 150초: timeout이 SIGINT 전달
- SIGINT 뒤 15초: cleanup 미완료로 SIGKILL
- 최종: exit code 137, PID 6251 zombie, 추가 device 작업 중단

## 6. Watcher 관측 사실

Watcher artifact는 복제하지 않고 원래 위치에 유지한다.

```text
/home/iris_hb4/tt-metal-hb4/generated/watcher/watcher.log
SHA-256: 01d7ef127e047b1458926dd3f64340850edde45c637341e28036004095527090
```

새 kernel mapping:

```text
k_id 5  = sdpa_flash_decode_reduce_only.cpp
k_id 11 = writer_decode_reduce_only.cpp
helper logical=(1,3), Watcher virtual=(1,4)
```

고정된 helper 상태:

```text
NSW, W, K, MWDD, K
```

소스상 `NSW`는 `noc_semaphore_wait()` 내부 waypoint다. helper writer는 두 producer completion을
기다리도록 target value 2를 사용했다. `NSD`로 전이하지 않았으므로 Watcher가 관측한 동안 local
semaphore는 정확히 2가 되지 않았다.

동시에 active cores는 writer/read CB wait와 matmul 관련 waypoint에 고정돼 있었다. 대표 상태에는
`CRBW`(`cb_reserve_back` wait), `UPMD`와 `MWDD`가 포함된다. Watcher sanitizer error, ASSERT 또는
illegal NoC access 메시지는 발견되지 않았다.

## 7. 원인 분석

### 관측으로 확인된 사실

1. Host factory와 새 device kernels는 compile/load됐다.
2. Device program은 launch됐다.
3. Helper writer는 두 partial semaphore를 기다리다 완료하지 못했다.
4. Active producer를 포함한 전체 SDPA pipeline도 전진하지 않았다.
5. 정상 output, correctness 결과와 device close는 없었다.

### 현재 추론

helper가 기다린 partial이 오지 않은 것은 확실하지만 helper semaphore 자체가 최초 원인이라는 증거는
없다. active cores가 final partial emission 전의 CB/matmul pipeline에서 함께 정지했기 때문이다.
현재 가장 보수적인 분류는 producer/helper 변경이 만든 global circular wait다.

### 미검증 가설

1. 기존 reducer를 worker producer로 바꾸면서 causal-mask CB의 producer/consumer cadence가 달라졌다.
2. 두 partial용 `c19` 용량 및 producer offset 변경이 기존 CB accounting과 상호작용했다.
3. tagged cross-chunk K/V prefetch와 새 final producer lifecycle이 독립적으로는 안전하지만 함께
   circular wait를 만들었다.
4. helper가 아니라 기존 tagged 64K random-page-table 경로가 먼저 정지했고 helper wait는 downstream
   symptom일 수 있다.

각 가설은 baseline 및 단일 변수 A/B 없이 확정하지 않는다.

## 8. 복구와 다음 실행 규칙

현재 세션에서는 복구를 시도하지 않는다. UMD reset, PCIe reset, driver unbind/rebind 또는 device
reopen을 수행하지 않는다. PID 6251은 zombie이므로 반복 signal 대상이 아니다.

사용자가 host/server 재시작 완료를 확인한 뒤에도 현재 POC를 그대로 재실행하지 않는다.

1. 외부 timeout이 있는 32x32 add 한 번만 실행
2. 성공하고 정상 close된 경우 helper/tagged async를 모두 끈 동일 isolated 64K baseline 실행
3. partial send 직전/직후와 helper wait 직전/직후에 구분 가능한 Watcher waypoint 추가
4. 64K가 아닌 짧은 context의 one-head reduction primitive를 tagged async 없이 검증
5. producer 두 개의 destination coordinate, c19 offset/capacity와 semaphore address를 host log로 확인
6. 위 단계가 모두 통과한 뒤에만 64K helper를 실행
7. tagged async는 helper correctness 이후 별도 변수로 추가

현재 opt-in 코드는 기본 비활성 상태로 남아 있으나 `TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1`은
미검증·위험 경로로 분류한다.

## 9. 07:38 UTC 격리 해제 뒤 add smoke 실패

사용자는 helper 사건이 host hang을 유발하지 않았고 사건 뒤 별도로 수행한 device open/close가
정상이라고 확인했다. 이에 영구 안전 규칙의 사용자 확인 기반 격리 해제 조건을 적용했다. 실패한
helper flag는 사용하지 않았고, 에이전트의 첫 device workload를 32×32 BF16 `ttnn.add` 한 번으로
제한했다.

실행 구성은 다음과 같다.

```text
TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4
PYTHONPATH=/home/iris_hb4/tt-metal-hb4
timeout --signal=INT --kill-after=15s 60s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -c <32x32 BF16 ttnn.add; explicit close>
```

### 관측 사실

- topology discovery와 user-mode driver open은 시작됐다.
- 마지막 주요 marker는 device/fabric 초기화 구간이었다.
- `SMOKE_VALUE`, add completion 및 `DEVICE_CLOSED` marker는 없었다.
- 60초 SIGINT 뒤 cleanup이 15초 안에 끝나지 않아 SIGKILL이 발동했고 최종 exit code는 137이었다.
- Python PID 8943은 PID 1 아래 `Z`/`<defunct>` 상태로 남았다.
- profiler 또는 `capture-release` 잔여 process는 없었다.
- 실패 뒤 MLP, SDPA, add 재시도 또는 device reopen은 수행하지 않았다.

### 해석과 현재 상태

이번 결과는 host 전체가 freeze됐다는 증거는 아니다. 그러나 사용자가 앞서 확인한 독립적인
open/close 상태와 달리 에이전트의 실제 worker workload는 completion에 도달하지 못했다. add kernel
compile/load/launch 중 정확히 어느 단계에서 멈췄는지는 marker가 부족해 확정할 수 없다.

영구 규칙에 따라 장치는 다시 격리한다. 따라서 계획했던 MLP baseline, DRAM-sharded block 8 및
W2 block 16 A/B는 실행하지 않았으며 성능 결과도 없다. 다음 격리 해제는 host/server 재시작 확인
또는 사용자의 새로운 post-incident 정상 open/close 확인이 필요하고, 어느 경우든 첫 에이전트
workload는 다시 timeout-limited 32×32 add 한 번이다.

## 10. BF16 add CB-pop BOS workaround 적용

후속 조사에서 `SDPA_DECODE_HANG_WA`는 SDPA 전용 API가 아니라 compute-side
`cb_pop_front<BOS_HANG_WA>`가 UNPACK의 `llk_pop_tiles<true>` 경로를 선택하게 하는
`[BOS-ARCH-006]` workaround임을 확인했다. Matmul과 SDPA에는 이미 이 경로가 적용돼 있었다.

동일 shape의 tensor 두 개를 더하는 BF16 `ttnn.add`는 기본 dispatch에서 binary_ng의
`eltwise_binary_no_bcast.cpp`를 선택한다. 이 compute kernel의 LHS/RHS input pop 두 곳에만
`BOS_HANG_WA=true`를 적용했다.

```text
/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/
  kernels/compute/eltwise_binary_no_bcast.cpp
```

적용 범위에는 broadcast, SFPU, legacy binary kernel과 SDPA kernel이 포함되지 않는다. `ttnncpp`
host target build와 direct source occurrence 검증은 통과했다. Device가 격리 상태이므로 Tensix JIT,
add completion, correctness와 device close는 검증하지 않았다.

07:38 add는 이 변경 전 실행됐고 device/fabric 초기화 뒤 completion marker 없이 정지했다. 따라서
당시 정지 원인이 CB pop이었다는 증거는 없으며, 이 workaround가 그 failure를 해결할 것이라는 판단은
현재 미검증 가설이다. 다음 실행에서도 격리 해제 뒤 첫 timeout-limited add 한 번으로만 검증한다.

## 11. 08:54 UTC 재부팅 뒤 recovery 검증

사용자가 server reboot 완료를 확인한 뒤 첫 device workload를 동일한 timeout-limited 32×32 BF16
`ttnn.add` 한 번으로 제한했다. 수정된 binary_ng no-broadcast compute 경로에서 결과값 2.0,
`DEVICE_CLOSED`, exit code 0을 확인했다. wall time은 약 3.5초였다. 이 결과로 이번 재부팅 이후의
격리는 해제했지만, 07:38 정지의 원인이 CB pop이었다고 소급 확정하지 않는다.

이후 profiler 없이 isolated MLP A/B 세 구성을 실행했다. 세 run 모두 PCC 0.9996 이상,
`MLP_COMPLETED`, `DEVICE_CLOSED`, exit code 0이었다. 중앙 latency는 interleaved 2.229688 ms,
DRAM-sharded auto block 8은 1.899062 ms, DRAM-sharded block 16은 1.872031 ms였다. 상세 조건과
해석은 `benchmark-results/2026-08-02-bos-mlp-w2-block-width-ab.md`에 기록했다.

이 recovery는 add 및 해당 isolated MLP 구성이 현재 정상 완료된다는 제한된 증거다. 실패한
`TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1` 경로를 재검증하거나 안전하다고 판정한 것이 아니다.

## 12. 10:24 UTC 격리 중 trapdoor Watcher 위치 확정

사용자가 재부팅 없는 `BOS_HANG_DIAGNOSTIC_TRAPDOOR 1회 승인`을 명시해, 일반 격리를 해제하지
않은 채 이미 실패한 32×32 BF16 add 한 번만 다시 실행했다. 실행 전 살아 있는 Python workload,
Tracy 또는 capture child는 없었다. 기존 `capture-release` PID 6954와 Python PID 16989·17517은
모두 PID 1 아래 zombie여서 signal을 보내지 않았다.

실행은 profiler 없이 `TT_METAL_WATCHER=100ms`와
`timeout --signal=INT --kill-after=15s 60s`만 사용했다. 결과값과 `DEVICE_CLOSED` marker 없이
SIGINT cleanup 상한도 넘겨 exit code 137로 끝났고, Python PID 19217은 PID 1 아래 zombie가 됐다.
trapdoor는 소진됐으며 장치는 계속 격리 상태다. 추가 device workload는 수행하지 않았다.

### 12.1 관측 사실

Blackhole `llk_pop_tiles()`에 다음 임시 waypoint를 삽입했다.

```text
tiles_acked update
PSW -> TTI_STALLWAIT -> PSD
TSW -> tensix_sync() -> TSD
ACKW -> TT_SETDMAREG/TT_STOREREG -> ACKD
```

Watcher Dump #1~#4는 idle이었다. 실제 add tile을 처리한 compute core `(0,0)`의 TRISC0는
Dump #5, 1.570초에 `TSW`에 도달했고 마지막 완료 Dump #655, 73.829초와 부분 Dump #656까지
동일 상태였다. 이때 core 상태는 다음과 같았다.

```text
BRISC NTW, NCRISC W, TRISC0 TSW, TRISC1 W, TRISC2 W
rmsg:D0G|BNT smsg:DGDD k_ids:4|6|5|5|5
```

`TSD`, `ACKW`, `ACKD`는 관측되지 않았다. dispatch core `(5,3)`의 `PSW`는 kernel id 1
`cq_prefetch.cpp`가 쓰는 기존 waypoint와 label이 우연히 같은 것이며, compute-side 임시 `PSW`의
증거로 해석하지 않는다. 이 add의 실제 tile work는 available BOS worker grid 5×4 전체가 아니라
core `(0,0)` 한 개에서 수행됐다.

### 12.2 소스에 근거한 결론

직접 확정할 수 있는 정지 위치는 `llk_pop_tiles()`의 `tensix_sync()` 내부다. Blackhole
`tensix_sync()`는 선행 write를 ordering하기 위해 `pc_buf_base[1]`에 먼저 write한 뒤 같은 위치를
read하며, 구현 주석은 이 read가 Tensix idle까지 block한다고 명시한다.

또한 `TTI_STALLWAIT`는 `INSTRUCTION_WORD(...)`와 `.ttinsn`으로 expand되는 inline Tensix
instruction이다. 따라서 `TSW` 도달은 TRISC RISC가 선행 `STALL_THCON/UNPACK` instruction을
발행했다는 뜻이지, 그 hardware wait condition이 이미 해제됐다는 뜻은 아니다. 현재 가장 강한
해석은 `tensix_sync()`의 PC-buffer read가 Tensix idle을 기다리고 있고, 그 앞의 UNPACK stall을
포함한 instruction stream이 drain되지 않아 영구 대기한다는 것이다.

### 12.3 비교와 미검증 가설

Wormhole `llk_pop_tiles()`는 `TT_SETDMAREG -> TTI_STALLWAIT -> TT_STOREREG` 순서이며 중간에
`tensix_sync()`가 없다. 현재 Blackhole BOS 경로는 `TTI_STALLWAIT -> tensix_sync() ->
TT_SETDMAREG/TT_STOREREG` 순서다. Blackhole LLK 주석도 automatic Tensix/TRISC access tracking이
과거에 필요했던 많은 `tensix_sync()`를 제거할 수 있다고 설명한다.

이 비교는 현재 BOS workaround의 full-idle fence가 너무 강하고 self-deadlock을 만든다는 가설을
지지하지만, Wormhole 순서를 custom BOS NPU에 그대로 적용해도 된다는 증거는 아니다. 다음 검증은
`tensix_sync()`의 PC-buffer write 직후와 blocking read 직전에 별도 waypoint를 두어 write와 read를
분리하는 것이다. 그 뒤에야 full-idle fence 제거, ACK instruction ordering만 보존하는 대체 fence,
또는 Blackhole tracking 사용을 각각 별도 수정 후보로 평가한다.

### 12.4 artifact와 원복

- Watcher dump:
  `/home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/watcher_add_waypoint_trapdoor_2026_08_02_10_24_47.log`
- run record:
  `/home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/trapdoor_run_2026_08_02_10_24_47.txt`
- Watcher SHA-256: `d5d7c726d81c06a5b71b7918d918afa17e1eae2c683aca6f9416ba2d4b01f377`
- 임시 waypoint 제거 뒤 `llk_io_unpack.h` SHA-256:
  `2c828c690e69048c16229e0b786dfe4535dd57e9c482894f30507a2eebbf4835`

## 13. 11:23 UTC 재부팅 뒤 vanilla SDPA hang-WA 성능 baseline 실패

사용자가 server reboot 완료를 다시 확인한 뒤, 격리 해제의 첫 device workload인 32x32 BF16 add는
결과 2.0, `DEVICE_CLOSED`, exit code 0으로 통과했다. 이어 `cb_pop_front` hang workaround의
SDPA runtime 영향을 측정하기 위해 profiler와 Watcher 없이 vanilla isolated paged SDPA를 실행했다.

실제 BOS Llama 3.2 3B decode 설정과 맞춘 B=1, Q heads=24, KV heads=8, head dim=128,
q/k chunk=128, program grid 5x4 구성이다. 4K correctness와 짧은 latency가 통과한 뒤에만 64K와
`SDPA_DECODE_HANG_WA=false`를 실행할 계획이었다. reduce-only helper를 포함한 모든 experimental
SDPA environment flag는 unset했다.

기본 `SDPA_DECODE_HANG_WA=true` 4K run은 device open과 K/V/Q/page-table upload 뒤
`INPUTS_READY`까지 도달했지만 첫 SDPA correctness call을 완료하지 못했다. 90초 SIGINT timeout과
15초 cleanup 상한 뒤 exit code 137로 종료됐다. `CORRECTNESS_PASSED`, `MEASUREMENT_COMPLETE`,
`DEVICE_CLOSED`는 없고 result JSON도 생성되지 않았다. Python PID 3863은 PID 1 아래 zombie이며,
살아 있는 workload child는 없다.

따라서 성능 sample은 유효하지 않으며 hang-WA의 SDPA latency 영향을 이번 run에서 수치화할 수 없다.
`WA=false`와 64K는 실행하지 않았다. source는 원래의 `WA=true` 상태와 checksum
`c64306c9297eb2173ab3e98a2be3759dd2812416401e4bd7da58b2204bf78aa7`을 유지한다. 장치는 다시
격리 상태이며 추가 open/close나 workload는 수행하지 않았다.

- Benchmark script:
  `/home/iris_hb4/benchmark_runs/sdpa_cb_pop_hang_wa_2026_08_02_11_15_00/benchmark_isolated_paged_sdpa.py`
- Exact command 및 run record:
  `/home/iris_hb4/benchmark_runs/sdpa_cb_pop_hang_wa_2026_08_02_11_15_00/run_record.md`

# BOS Llama 3.2 3B DRAM-sharded MLP prefill validation failure

- 발생일: 2026-08-02 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- workload: Llama 3.2 3B Instruct greeting smoke
- 상태: first-decode hang 재현; 초기 SDPA hang의 잔존 또는 standard reader/writer 경로 회귀 조사 필요

## 요약

vanilla SDPA와 새 MLP 후보 baseline인 DRAM-sharded W1/W3/W2 및 W2 block 16으로 짧은 greeting
demo를 실행했다. 28개 layer weight load는 완료됐지만 첫 prefill의 layer 0 W1 matmul이
`WIDTH_SHARDED` input B를 거부했다. SDPA 또는 device kernel에는 진입하지 않았다.

Python runner는 예외 뒤 mesh device를 close하지 못하고 futex wait에 남았다. 300초 timeout이
발동하기 전에 PID 8484에 SIGINT를 한 번 보내 cleanup했고 즉시 exit code 130으로 종료됐다.
SIGKILL은 사용하지 않았다. 종료 뒤 살아 있는 child와 `/dev/bos/0` 보유 process는 없지만 signal
종료 규칙에 따라 장치는 격리한다.

## 실행 구성

- model: `meta-llama/Llama-3.2-3B-Instruct`
- prompt: `안녕하세요! 오늘 잘 지내고 있나요?`
- max sequence length: 128
- max generated tokens: 16
- paged attention: enabled
- SDPA: 모든 BOS experimental env를 unset한 vanilla path
- MLP: `TT_METAL_MLP_DRAM_SHARDED=1`
- W2: `TT_METAL_MLP_W2_IN0_BLOCK_W=16`
- profiler, Watcher 및 JIT-force: disabled
- process bound: `timeout --signal=INT --kill-after=15s 300s`의 direct Python child

run artifact:

```text
/home/iris_hb4/benchmark_runs/llama32_3b_greeting_dram_sharded_w2_block16_2026_08_02_12_30_00
```

## 관측 타임라인

1. device open 및 28-layer weight load 진행
2. greeting prompt 출력
3. 첫 compile-prefill에서 layer 0 MLP W1 호출
4. host `TT_FATAL`: `Input B memory layout must be INTERLEAVED, got WIDTH_SHARDED`
5. traceback 뒤 Python main thread가 `futex_wait_queue`에 약 1분 이상 고정
6. `/dev/bos/0` fd와 79 threads 유지, 정상 close marker 없음
7. PID 8484에 SIGINT 1회
8. 즉시 exit code 130; SIGKILL 없음
9. 살아 있는 Llama/timeout child 및 `/dev/bos/0` holder 없음

응답, output file, SDPA kernel marker는 생성되지 않았다.

## 원인

### 확인된 사실

`MLP.__init__`은 `TT_METAL_MLP_DRAM_SHARDED=1`이면 W1/W3와 W2를 모두 DRAM width-sharded memory
config로 load한다. 반면 `MLP.forward`는 선택된 DRAM-sharded matmul program config를
`mode == "decode"`일 때만 `ttnn.linear`에 전달한다. prefill에서는 program config가 `None`이 되어
generic matmul validation이 interleaved input B만 허용하고 즉시 실패한다.

따라서 isolated decode MLP가 성공했다는 사실은 full demo의 prefill compatibility를 보장하지 않는다.
이번 실패는 vanilla SDPA, dual-NoC reader, tagged async, six-reader 또는 reduce-only helper와 무관하다.

### 추가 lifecycle 문제

`run_llama32.py`는 정상 함수 끝에서만 `close_mesh_device()`를 호출하고 exception-safe `finally`가
없다. 이 때문에 host validation exception 뒤에도 device와 runtime threads가 남았다. 이 lifecycle
문제는 matmul validation failure와 별개의 예방 항목이다.

## 영향과 현재 상태

- Llama greeting 응답을 얻지 못했다.
- DRAM-sharded + block 16을 현재 형태 그대로 full-demo baseline으로 사용할 수 없다.
- device-kernel hang 증거는 없고 host validation에서 중단됐다.
- 그러나 signal cleanup이 발생했으므로 recovery 확인 전까지 장치는 격리한다.

## 다음 조치

1. 사용자가 host/server 재시작 또는 사건 뒤 별도 정상 device open/close를 확인하기 전에는 장치 작업을
   실행하지 않는다.
2. 격리 해제 뒤 첫 workload는 timeout-limited 32×32 add 한 번이다.
3. 가장 안전한 demo 재검증은 MLP 두 env도 unset한 완전 vanilla 구성으로 시작한다.
4. optimized MLP full demo는 prefill용 interleaved weight와 decode용 DRAM-sharded weight를 분리하거나,
   prefill이 DRAM-sharded input B를 정식 지원하도록 program config와 validation을 함께 수정해야 한다.
5. runner의 mesh lifetime을 `try/finally`로 감싸 host exception에서도 명시적 close를 수행한다.

위 수정은 최초 incident 기록 시점에는 구현하거나 재실행하지 않았다.

## 2026-08-02 12:37 UTC 후속 수정 및 재부팅 뒤 smoke 실패

사용자가 server reboot 완료를 확인한 뒤 다음 host-side 수정을 적용했다.

1. `mlp.py`에서 prefill용 interleaved W1/W2/W3와 decode용 DRAM width-sharded W1/W2/W3를 분리했다.
   `TT_METAL_MLP_DRAM_SHARDED=1`일 때만 decode 사본을 추가하고, `forward()`는 mode에 맞는 layout을
   선택한다. 기존 `.dram_sharded` cache suffix는 decode 사본에만 유지했다.
2. `run_llama32.py`의 device 생성과 model 실행을 분리하고 `try/finally`에서
   `close_mesh_device()`를 호출하도록 변경했다.
3. 두 파일의 `python3 -m py_compile`은 성공했다. 이 시점에는 device workload를 실행하지 않았다.

수정 소스:

- `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tt/mlp.py`
  SHA-256: `7bf41474e029cb5dbe520c67ea83db0515b7b713537a2544cf9eacfde8263fc8`
- `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32.py`
  SHA-256: `02551c4e5622f2109d32e9e777a5a4e71d790c7ccb7793ff4da8a019003ac7bf`

재부팅 뒤 첫 device workload는 규칙대로 기존 32x32 BF16 `ttnn.add` 한 번으로 제한했다.
실패한 reduce-only helper, Watcher, profiler 및 다른 experimental SDPA flag는 사용하지 않았다.

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND \
  -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER \
  -u TT_METAL_SDPA_DECODE_DIRECT_BURST \
  -u TT_METAL_SDPA_DECODE_TAGGED_ASYNC \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  timeout --signal=INT --kill-after=15s 60s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  /home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/smoke_add_32.py
```

관측 결과:

- device open과 Blackhole topology/TLB 초기화 로그까지 진행
- 마지막 host log 시각: 12:37:54 UTC
- `SMOKE_VALUE`, add completion 및 `DEVICE_CLOSED` marker 없음
- 60초 SIGINT timeout 및 15초 cleanup 상한 뒤 exit code 137
- 사후 `ps`에서 살아 있는 smoke, Llama, Tracy 또는 capture workload child 없음
- host에 `fuser`와 `lsof`가 없어 `/dev/bos/0` holder는 별도 도구로 확인하지 못함
- Llama full-demo는 시작하지 않음

smoke에는 add 전용 BOS hang workaround가 빠지지 않았다. 스크립트의
`ttnn.add(..., use_legacy=False)`가 선택하는
`binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`에는
`#define BOS_HANG_WA true`가 있고 두 input pop 모두
`cb_pop_front<BOS_HANG_WA>(...)`를 호출한다. 이 WA는 runtime env가 아니라 compile-time kernel
source 설정이다. SDPA kernel의 `SDPA_DECODE_HANG_WA=true`와 동일한
`[BOS-ARCH-006]` LLK pop workaround 계열이다.

다만 마지막 로그가 device open 초기화에 머물렀으므로 compile, load, launch 또는 kernel 내부 중
정확히 어디에서 정지했는지와 WA 적용 지점까지 실제 실행이 도달했는지는 확정할 수 없다. exit 137
규칙에 따라 장치는 다시 격리하며, 새 recovery 확인 전에는 추가 open/close나 Llama workload를
실행하지 않는다.

## 2026-08-02 12:48–12:55 UTC 재부팅 뒤 smoke 성공과 full-demo first-decode hang

사용자가 server reboot 완료를 다시 확인했다. 에이전트의 첫 device workload인 동일 32x32 BF16
`ttnn.add(..., use_legacy=False)`는 add 전용 `BOS_HANG_WA=true` 상태에서
`SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit code 0으로 약 3초 안에 통과했다.

이후 아래 구성을 재검증했다.

- model: Llama 3.2 3B Instruct
- prompt: `안녕하세요! 오늘 잘 지내고 있나요?`
- max sequence/generated tokens: 128/16
- MLP: prefill interleaved + decode DRAM-sharded dual layout, W2 block 16
- SDPA environment flags: 모두 unset
- profiler/Watcher/trace: disabled
- process bound: direct Python child, SIGINT 300초, SIGKILL cleanup 상한 15초
- run directory:
  `/home/iris_hb4/benchmark_runs/llama32_3b_greeting_layout_split_revalidation_2026_08_02_12_50_00`

### 관측 사실

1. 28개 layer의 interleaved 및 DRAM-sharded MLP weight load를 완료했다.
2. 6.43 GB `model_weights.pth` 저장은 12:50:06 UTC에 완료됐다.
3. runner의 `Compile Pre-fill`과 `Inference Pre-fill` 뒤에만 출력되는 두 번째
   `Llama:` prompt가 나타났다. 따라서 이전 interleaved validation failure는 해결됐고 prefill 두
   호출은 반환했다.
4. 이후 첫 decode token은 반환되지 않았고 `output.txt`도 생성되지 않았다.
5. TT-Metal kernel cache는 12:50:11–12에 interleaved BMM, DRAM-sharded BMM 및 argmax kernel의
   `.SUCCESS`를 기록한 뒤 추가 compile artifact를 만들지 않았다.
6. 대기 중 main thread는 `futex_wait_queue`, runtime worker 한 개는 80–90% CPU를 유지했다.
   새 compile artifact가 없으므로 host JIT보다는 async device completion polling으로 해석하는 것이
   더 타당하다.
7. 사용자의 중지 요청으로 Python PID 1766에 SIGINT를 한 번 보냈다. cleanup 상한과 겹쳐 exit code
   137로 끝났고 정상 process exit 또는 device-close 완료는 검증되지 않았다.
8. 사후 살아 있는 Llama, timeout, Tracy 또는 capture child는 없었다.
9. host available memory는 117 GiB, swap 사용은 0이고 dmesg에 OOM/driver fault는 없었다.

### Root cause 분석

#### 시간축 정정: HANG_WA는 초기 hang에 대한 선행 workaround

초기 standard SDPA reducer/worker 경로에서 이미 반복 hang이 있었고, 이를 완화하기 위해 `SDPA_DECODE_HANG_WA=true`가 먼저 도입됐다. 그 뒤에 남는 4개 core를 buffering/helper로 활용하는 설계와 reduce-only helper, endpoint 및 tagged-async 실험이 진행됐다. 따라서 현재 source와 오래된 build 사본의 차이만으로 HANG_WA 자체를 이번 회귀의 일차 원인으로 지목하는 것은 시간축상 잘못된 추론이다.

HANG_WA=true는 Blackhole `llk_pop_tiles<true>`를 선택해 tile acknowledge 전에 `tensix_sync()`를 추가한다. 검증 대상이기는 하지만 초기 hang 때문에 필요한 workaround였으므로 첫 분리 실험에서 바로 끄면 안 된다. 보존된 HANG_WA=true 64K synthetic run도 28 layers와 `MODEL_READY` 뒤 완료 marker를 남기지 않았다. 이는 HANG_WA 자체가 원인이라는 증거가 아니라, 기존 completion 문제가 workaround 뒤에도 구성에 따라 남을 수 있음을 뜻한다.

#### reducer/helper 구성 정정

기본 BOS decode mapping은 8 KV heads에 head당 reducer 1개와 worker 1개, 즉 8 reducers + 8 workers = 16 active cores다. 5x4 grid의 나머지 4개 core를 helper/buffering에 쓰자는 것은 후속 설계였고 기본 reducer 수를 4개로 바꾼 것이 아니다. 실제 구현 및 실행된 POC도 `TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=1`로 선택되는 KV head 7용 reduce-only helper 1개였다.

이번 full-demo에서는 이 flag와 dual-NoC, tagged-async, pair/bank-balanced endpoint 및 six-reader flag가 모두 unset이어서 특수 helper kernel은 선택되지 않았다.

#### 현재 원인 후보

1. **기존 standard reducer/worker hang의 잔존**: HANG_WA가 모든 producer/consumer ordering을 해결하지 못했을 가능성을 먼저 분리해야 한다.
2. **standard path에 남은 후속 reader/writer 변경**: 현재 writer는 causal mask를 각 head의 reducer만 생성한다. compute 소비 조건과는 설계상 일치하지만 실제 CB producer/consumer count와 semaphore 진행은 아직 계측되지 않았다. page-table index의 `% Bkv` 변경은 B=1 greeting에서는 사실상 no-op이다.
3. **full-layer attention/MLP async interaction**: isolated DRAM-sharded MLP는 통과했지만 full model의 layer 경계와 queue completion까지 독립 검증하지 않아 완전히 배제할 수 없다.

현재 증거만으로 HANG_WA 자체, reduce-only helper 또는 MLP 중 하나를 확정 원인으로 정할 수 없다.

#### 가능성 낮음: dual-layout MLP decode

DRAM-sharded W1/W3/W2와 W2 block 16은 isolated MLP에서 PCC 0.999641, 10회 latency 반복,
NoC capture 및 정상 device close를 통과했다. 또한 layer 순서는 attention이 MLP보다 앞선다.
따라서 이번 first-decode hang의 일차 원인으로는 SDPA보다 가능성이 낮다. 다만 full-layer async
interaction을 독립적으로 분리하지 않았으므로 완전히 배제하지는 않는다.

#### 배제 또는 약화된 가설

- prefill layout validation: 해결됨
- host OOM/swap: 증거 없음
- 장시간 JIT compile: cache activity가 약 1초 뒤 멈춰 가능성 낮음
- SDPA endpoint/tagged/six-reader/reduce-helper env: 모두 unset
- add용 HANG_WA 누락: smoke에서 적용됐고 정상 통과

### 다음 검증 순서

사용자가 이 사건 뒤 server reboot 완료를 확인했다. 격리는 해제됐지만 재부팅 뒤 에이전트의 첫 device workload는 아직 실행하지 않았다. 다음 순서로 한 변수씩 분리한다.

1. 외부 timeout이 있는 필수 32x32 add smoke 한 번
2. HANG_WA=true를 유지하고 helper/tagged/dual-NoC/six-reader를 모두 끈 standard paged SDPA isolated correctness
3. 후속 무조건부 reader/writer 변경을 compile-time으로 분리해 A/B하고 causal-mask producer, CB page count 및 reducer semaphore를 계측
4. worker partial send, reducer wait, `cb_l/cb_m/cb_o` wait, correction merge와 output write 앞뒤에 Watcher waypoint 추가
5. isolated standard SDPA 통과 뒤 MLP env를 unset한 full Llama greeting, 이후에만 DRAM-sharded MLP를 다시 추가
6. HANG_WA=false는 초기 hang을 재노출할 수 있으므로 마지막 대조군으로만 검토

reduce-only helper 실패 구성은 계속 금지한다. 첫 add smoke가 실패하면 추가 open/close나 다른 workload 없이 즉시 다시 격리한다.

## 2026-08-02 13:23–13:27 UTC 재부팅 뒤 smoke 성공 및 standard SDPA 재현

사용자가 server reboot 완료를 확인했다. 첫 device workload인 32x32 BF16 `ttnn.add(..., use_legacy=False)`는 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit code 0으로 약 3초 안에 통과했다.

그 다음 profiler와 Watcher 없이 다음 isolated correctness 한 번만 실행했다.

- sequence length/current position: 4096/4095
- batch/Q heads/KV heads/head dim: 1/24/8/128
- program grid: 5x4, expected active compute cores: 16
- q/k chunk: 128/128
- paged KV block size: 32
- `SDPA_DECODE_HANG_WA=true` compile-time 유지
- reduce-only helper, dual-NoC, tagged async, endpoint balancing 및 six-reader env: 모두 unset
- warmup/repeats: 0/1, correctness call 1회
- timeout: SIGINT 90초, cleanup 상한 15초

관측은 `DEVICE_OPENED available_worker_grid=(x=5,y=4)`와 `INPUTS_READY`까지였고, correctness 호출이 반환하지 않았다. `CORRECTNESS_PASSED`, `DEVICE_CLOSED` 및 result JSON은 생성되지 않았으며 최종 exit code는 137이었다. 사후 살아 있는 benchmark, Tracy 또는 capture child는 없었다.

이 재현은 full Llama, MLP 및 reduce-only helper 없이 standard paged SDPA completion 문제가 존재함을 보여준다. 따라서 MLP와 helper는 이번 hang의 필요조건이 아니며, 우선 조사 대상은 standard reducer/worker의 CB 또는 semaphore 진행이다. 다만 HANG_WA=true인 한 번의 실패만으로 HANG_WA 자체가 원인인지, workaround가 불충분한지는 구분할 수 없다. 정확한 위치는 Watcher waypoint가 없어 미확정이다.

exit 137 규칙에 따라 장치를 다시 격리했다. 새 재부팅 또는 사용자가 사건 뒤 별도 device open/close 정상 완료를 확인하기 전에는 추가 device workload를 실행하지 않는다.

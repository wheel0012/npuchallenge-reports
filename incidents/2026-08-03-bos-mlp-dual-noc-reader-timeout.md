# BOS MLP dual-NoC reader 첫 실제 실행 timeout

발생일: 2026-08-03 UTC

## 요약

Llama 3.2 3B isolated decode MLP의 DRAM-sharded weight reader 여섯 개를 BOS의 두 NoC에
3개씩 나누는 opt-in을 처음 실제 runtime으로 실행했다. program factory log에서
`DRAM-sharded weight reader dual NoC: true`를 확인한 뒤 MLP completion, PCC 및 device close
marker가 나오지 않았다. 외부 180초 timeout의 SIGINT cleanup도 15초 안에 끝나지 않아 SIGKILL이
발동했고 최종 exit code는 137이었다.

이 사건 뒤 추가 device workload는 수행하지 않았으며 장치는 격리 상태다. 서버 재시작 또는 계약에
정의된 별도 recovery 확인 전에는 add smoke, device open/close, MLP, SDPA와 profiler를 실행하지 않는다.

## 장치와 대상

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker topology: 5×4 = 20 cores
- 대상 MLP program grid: 4×4
- active compute core 수: kernel trace로 검증하지 못함
- physical DRAM: 3 banks
- worker DRAM endpoints: bank당 2개, 총 6개
- runtime log: `Dram Interface Workers: 6`
- 모델: `meta-llama/Llama-3.2-3B-Instruct`, layer 0 isolated MLP
- weight path: DRAM-sharded
- W2 `in0_block_w=16`

표준 P100/P150 SKU 이름은 사용하지 않는다. runtime의 P150 추정 log는 custom BOS identity에 대한
authoritative 판정이 아니다.

## 변경 의도와 구현

기존 NoC capture에서는 여섯 reader의 request 수는 같지만 destination traffic이 3:2:1로 갈렸고
W1/W3/W2 aggregate는 45.40 GB/s로 86.83 GB/s microbenchmark peak의 52.28%였다. 이를 개선하기 위해
다음 두 opt-in을 만들었다.

- `TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=1`: reader assignment의 최적화 NoC 기준을 in0 NoC에서
  weight-reader NoC로 바꿈
- `TT_METAL_MLP_DRAM_SHARDED_DUAL_NOC=1`: 여섯 logical DRAM view의 endpoint 좌표를 확인해
  x={0,4,5}는 NOC0, x={1,2,3}은 NOC1 reader kernel로 3개씩 분할

두 kernel은 서로 겹치지 않는 reader core set에서 RISCV0을 사용한다. 이 경로는 기본값이 아니며
환경변수가 정확히 `1`일 때만 활성화됐다.

## 용어 및 설계 정정

위 실험의 `DUAL_NOC`는 vanilla MLP의 정상적인 reader/writer NoC 역할 분담을 뜻하지 않는다.
factory는 원래 다음처럼 두 방향을 이미 구분한다.

- `in0_noc = preferred_noc_for_dram_write(arch)`
- `in1_noc = preferred_noc_for_dram_read(arch)`
- activation/multicast data-movement kernel은 `in0_noc`
- weight reader/output writer 결합 kernel은 `in1_noc`

마지막 kernel은 weight를 읽은 뒤 compute output CB를 기다리고, 같은 kernel 안에서 output reshard
`noc_async_write`까지 수행한다. 실패한 실험은 reader traffic만 3+3으로 나눈 것이 아니라 NOC0
variant 세 core의 output write 방향까지 함께 바꿨다. 따라서 이 변경은 “여섯 reader의 위치만
분산”한 실험이 아니며, 기존 reader/writer 역할 계약을 보존하지 않았다. `dual-NoC reader`라는
이름도 이 사실을 가려 혼동을 일으켰다.

2026-08-03에 실패 patch를 `git apply -R`로 제거하고 다시 빌드·설치했다. Python이 로드하는
`build_Release/lib/_ttnncpp.so`에서 dual-NoC 환경변수와 log 문자열이 사라지고 locality-only
환경변수만 남은 것을 확인했다. 장치 격리 중이므로 교정 runtime은 아직 device에서 실행하지 않았다.

## saturation benchmark의 reader 배치

86.83 GB/s를 기록한 BOS direct read benchmark는 MLP처럼 reader와 writer NoC를 분업한 workload가
아니다. read-only kernel이므로 두 ring을 모두 DRAM read에 사용한다.

- 20 reader를 NOC0/NOC1에 10/10 배치
- NOC0 endpoint x={0,4,5}, NOC1 endpoint x={1,2,3}
- endpoint마다 reader 3개를 기본 배치하고 각 ring의 서로 다른 physical bank endpoint 하나씩에
  reader를 하나 더 배치
- 결과적으로 endpoint reader 수는 3 또는 4, physical-bank reader 수는 7/7/6
- 4-reader endpoint의 reader당 iteration을 3, 3-reader endpoint는 4의 비율로 조정해 endpoint,
  ring 및 physical-bank byte 수를 동일하게 유지
- reader와 endpoint의 horizontal route distance가 최소가 되도록 실제 worker physical coordinate를
  정렬해 짝지음
- 동일 endpoint 및 동일 NoC/worker-row route에서 VC가 겹치지 않게 4개 VC를 edge coloring
- 32 KiB block, 두 L1 slot 및 tagged transaction으로 다음 read를 먼저 발행

이 배치는 writer traffic, CB backpressure, matrix compute 및 output reshard가 없는 silicon peak
reference다. MLP 결합 kernel에 두-ring reader 배치를 그대로 복사할 수 없다.

실제 SDPA 구조에 더 가까운 reference는 six-reader disjoint sequence-shard relay다. 여섯
endpoint-local producer가 각각 다른 sequence range를 읽고 열 개 consumer에 중복 없는 shard만
전달한다. 128 KiB block에서 77.369 GB/s, 별도 5-run에서 평균 76.915 GB/s를 기록했다. 반대로 같은
block 전체를 consumer에 복제한 fanout은 51.070 GB/s에 그쳤다. 따라서 적용 원칙은 full-block
복제가 아니라 endpoint-local ownership과 disjoint work 분할이다.

stale vanilla MLP W1 trace는 NOC1만 사용했고 여섯 interface reader가 세 destination에 2/3/1로
배치됐다. 이는 `Dram Interface Workers: 6`이 six physical endpoint 사용을 뜻하지 않는 직접적인
예다.

## runtime stale 발견과 선행 결과 무효화

초기 A/B와 첫 NoC capture 뒤 Python extension의 실제 dependency를 확인한 결과, 실행은 새로 빌드된
`build_Release/ttnn/_ttnncpp.so`가 아니라 오래된
`build_Release/lib/_ttnncpp.so`를 로드하고 있었다. 따라서 다음 초기 latency는 모두 새 opt-in의
효과를 측정하지 않은 값이며 비교에 사용하지 않는다.

- baseline median: 1.909383 ms
- locality median: 1.905781 ms
- dual label median: 1.898901 ms

다음 profiler run도 exit 0, PCC, close, ops CSV 및 raw trace까지 갖췄지만 experimental validity는 없다.

```text
/home/iris_hb4/profiler_runs/mlp_decode_dram_sharded_dual_noc_2026_08_03_05_07_00
```

raw W1의 여섯 `READ_SET_STATE`가 모두 NOC1인 것은 당시 stale vanilla runtime을 사용했다는 사실과
일치한다. 이 artifact는 운영상 complete지만 dual-NoC 성능 또는 route의 증거로 인용하지 않는다.

## runtime 교정과 안전 게이트

`ttnn-runtime` install component를 적용해 Python이 실제 로드하는
`build_Release/lib/_ttnncpp.so`를 갱신했다. 설치 뒤 다음을 확인했다.

- loaded dependency: `/home/iris_hb4/tt-metal-hb4/build_Release/lib/_ttnncpp.so`
- binary strings: 두 opt-in 이름과 `DRAM-sharded weight reader dual NoC`
- 32×32 BF16 add: `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0
- 교정 runtime vanilla reader 1회: PCC 0.9996410623, 1.935495 ms, `MLP_COMPLETED`,
  `DEVICE_CLOSED`, exit 0

vanilla reader command의 핵심 설정은 다음과 같다.

```bash
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 \
  TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_DUAL_NOC=0 \
  timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## 실패 실행

동일 command에서 `TT_METAL_MLP_DRAM_SHARDED_DUAL_NOC=1`만 바꿨다.

타임라인:

1. 05:14:38 UTC: `DRAM-sharded weight reader dual NoC: true`가 W1/W3 program 생성에서 출력됨
2. 이후 PCC, sample, `MLP_COMPLETED`, `DEVICE_CLOSED`가 출력되지 않음
3. 180초 뒤 timeout SIGINT
4. 15초 cleanup 상한 뒤 SIGKILL
5. exit code 137
6. 사후 host process 점검에서 runner, Tracy, capture child가 남아 있지 않음을 확인

NoC profiler나 Watcher는 사용하지 않았다. ops CSV와 raw NoC trace도 생성되지 않았다.

## 영향

- dual-NoC의 correctness, latency 및 bandwidth는 측정하지 못했다.
- 3+3 route가 device에서 실제 발생했는지 확인하지 못했다.
- timeout cleanup 실패로 device state를 신뢰할 수 없어 장치를 격리했다.
- opt-in off인 vanilla reader는 교정 runtime에서 정상 동작했지만, 이는 사건 전 결과다.

## 원인 분석

### 관측 사실

- host factory가 dual opt-in을 인식하고 두 program 생성을 시작했다.
- 실패는 model load 전이나 host validation이 아니라 MLP program compile/launch 구간에서 발생했다.
- 마지막 host log만으로 kernel compile 완료와 device launch 사이 경계는 확정할 수 없다.
- weight reader kernel은 DRAM read만 하지 않는다. compute output을 기다린 뒤 output reshard write도
  수행하는 결합 reader/writer kernel이다.

### 유력한 추론

이번 split은 read NoC만 바꾼 것이 아니라 NOC0에 배정한 세 core의 output reshard write NoC도 함께
바꿨다. 따라서 기존 producer/compute/writer 계약을 보존한 단순 locality 변경이 아니다. split된
RISCV0 kernel 두 개, CB producer-consumer 진행 및 reshard write route 중 하나가 기존 동기화 가정을
깨뜨렸을 가능성이 높다.

### 미검증 가설

- 같은 RISCV processor에서 서로 다른 core set에 두 NoC variant를 배치하는 방식이 이 program의
  kernel/CB 계약과 호환되지 않을 수 있다.
- NOC0 variant의 tagged read transaction 또는 write-back barrier가 완료되지 않았을 수 있다.
- 최초 compile이 비정상적으로 길었을 가능성도 배제할 수 없지만, SIGINT cleanup까지 실패한 점은
  단순 host compile 지연만으로 설명하기 어렵다.

## 복구 및 다음 단계

1. 사용자가 server restart를 확인하기 전까지 장치 격리를 유지한다.
2. 재시작 뒤 첫 workload는 외부 timeout이 있는 32×32 add 한 번으로 제한한다.
3. dual-NoC opt-in은 실패 구성으로 분류했고 source와 installed runtime에서 제거했다.
4. 첫 A/B는 `TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=1`만 사용한다. 이 opt-in은 weight-read NoC를
   기준으로 generic DRAM-view-to-worker assignment를 다시 계산할 뿐 kernel NoC, runtime args, CB,
   output writer와 compute를 바꾸지 않는다.
5. 새 run에서는 source reader coordinate, destination coordinate, NoC, VC와 byte count를 반드시
   남겨 baseline 2/3/1 destination 및 route distance가 실제로 개선됐는지 확인한다.
6. program compile 완료, enqueue 직전, synchronize 직후 marker를 추가해 compile/launch 경계를
   구분한다.
7. correctness/JIT 1회와 짧은 latency가 정상 종료된 뒤에만 isolated warmup 1 + measured 1 NoC
   capture를 수행한다.

## DRAM에서 compute pipeline으로 넘어가는 판단 기준

reader locality A/B에서 아래를 순서대로 본다.

1. DRAM read destination의 2/3/1 불균형과 source-to-destination route span
2. W1/W3/W2 각각의 raw DRAM byte rate와 BRISC/NCRISC span
3. TRISC span, compute CB wait, 마지막 core와 첫 core의 completion spread

route가 개선됐는데 DRAM byte rate와 layer latency가 오르지 않거나, BRISC/NCRISC보다 TRISC 및
compute completion tail이 지배적이면 다음 병목을 math engine pipeline으로 판정한다. 그때의 목표는
reader 수를 더 늘리는 것이 아니라 K block double buffering, unpack/math/pack overlap, subblock
크기와 W1/W3/W2 work balance를 조정해 matrix engine이 CB starvation 없이 계속 동작하게 하는 것이다.

SDPA에서는 K-chunk 256이 약 70.266 GB/s이고 six-reader disjoint relay reference가 약
76.9--77.4 GB/s이므로 reader ownership으로 남은 이론적 폭은 약 10%다. locality/ownership 변경 뒤
그 폭이 재현되지 않으면 slowest head, reducer synchronization과 matrix pipeline tail을 우선한다.
MLP는 기존 projection aggregate가 45.40 GB/s라 더 큰 memory headroom이 있지만, whole-layer
effective bandwidth에는 GEMM과 elementwise 시간이 포함되므로 per-projection NoC와 TRISC를 함께
보지 않고 memory-bound라고 단정하지 않는다.

## 재발: 실제 6 endpoint × 3 reader fanout-3

같은 날 후속 실험은 이전의 generic dual-NoC 3+3 split과 달리 BOS의 6개 worker DRAM endpoint를
명시적으로 지정하고 endpoint마다 reader/compute 3개를 배정했다. host log에서 endpoint 분배
`3:3:3:3:3:3`, reader NoC `9:9`, active reader/compute 18을 확인했다.

profiler와 Watcher를 사용하지 않은 1-iteration correctness에서 W1/W2 program 생성 뒤 completion이
없었다. 180초 SIGINT와 15초 cleanup 상한 뒤 SIGKILL, exit 137이며 `MLP_PCC`, `MLP_COMPLETED`,
`DEVICE_CLOSED`는 없다. 종료 뒤 Python PID 11104는 PID 1 아래 zombie로 남았다. artifact는
`/home/iris_hb4/profiler_runs/mlp_fanout3_dual_noc_3x6_correctness_2026_08_03_19_10_00/run.log`, SHA-256은
`9f42c5f0cea9cdfc5d3ddd9b144db70b2cb125ae8afb0ddef332d9e00b0bd79f`다.

### 추가 원인 분석

관측 후 성공한 20-core/6-endpoint DRAM microbenchmark와 host-side 비교했다. microbenchmark는 같은
endpoint와 같은 `(NoC, worker-row)` route의 edge가 VC를 공유하지 않도록 4-VC edge coloring을 한다.
실패 당시 새 MLP 경로는 endpoint별 VC 하나를 고정해 endpoint당 reader 3개가 VC를 공유했다. 이 계약
차이는 확인됐으며 read completion 정지의 가장 유력한 원인이지만 Watcher waypoint가 없어 확정 원인은
아니다. RISCV0 reader/writer 결합 kernel의 NOC0 write-back 등 기존 미검증 가설도 남아 있다.

### 현재 상태

- 장치: exit 137로 격리. 재부팅 확인 전 open/close 및 smoke 포함 추가 workload 금지
- host source: endpoint/route conflict-free 4-VC coloring으로 교정
- host build/runtime: `ttnncpp` build 성공, 배포 checksum
  `87dbd2cc6d8f8f4770f23022ea875c8391e079ea2bb1b8bf2b986289df9a3950`
- 교정본 device 검증: 미수행
- 재개 gate: 사용자 재부팅 확인 → timeout 보호 32×32 add 1회 → corrected isolated correctness 1회
- latency/NoC: correctness 성공 전 금지. fanout-2 1.472280 ms보다 느리면 NoC capture 생략

## VC 교정본 재시도 결과

재시작 후 32×32 add 안전 게이트는 정상 통과했다. 그러나 conflict-free VC edge coloring 교정본도
동일한 program 생성 marker 뒤 정지했고 timeout cleanup 실패와 exit 137이 반복됐다. PCC와 close는
없고 PID 7036 zombie가 남았다.

이 결과로 고정 VC는 단독 root cause에서 제외한다. 확인된 계약 위반이므로 교정은 유지하지만,
다음 조사는 NOC0 reader variant가 같은 결합 kernel에서 output reshard write도 NOC0으로 바꾸는 점을
분리해야 한다. read/write NoC를 독립시키거나 Watcher로 read barrier와 output CB wait 중 어느 쪽에
머무는지 확인하기 전에는 dual-NoC 경로를 다시 실행하지 않는다. 현재 장치는 다시 격리 상태다.

SDPA TurboQuant opt-in은 두 실행에서 호출·활성화하지 않았으며 사건 범위 밖이다.

## 독립 writer NoC 재시도와 정상 대조군

재부팅과 add gate 성공 뒤 dual fanout-3의 output write를 reader 반대 NoC로 분리하고 Watcher 100ms를 켜서 재시도했다. endpoint `3:3:3:3:3:3`, reader NoC `9:9`, 18 reader/compute가 적용됐지만 completion 없이 exit 137이 반복됐다.

- failed artifact: `/home/iris_hb4/profiler_runs/mlp_fanout3_dual_noc_3x6_separate_writer_correctness_2026_08_03_13_31_24/run.log`
- SHA-256: `927b33cfed92bb76b6208d5b5593d5bbd9b510ed9e7357c61f52c6de581a219e`
- process: PCC/completion/close 없음, Python PID 3802 zombie
- Watcher: periodic check만 출력했고 명시적 error/waypoint 없음

다음 재부팅과 add gate 뒤 동일 build에서 dual/fanout3/helper를 끄고 기존 balanced fanout-2를 실행했다. NOC1 endpoint `4:4:4`, 12 readers/12 compute, PCC `0.9996410623374821`, 1.487526 ms, 정상 completion/close와 exit 0을 확인했다.

- success artifact: `/home/iris_hb4/profiler_runs/mlp_existing_fanout2_balanced_correctness_2026_08_03_13_41_50/run.log`
- SHA-256: `7f4b7479714c216f2222851f331fbf129873f466b7d4d2a7a9036e43023452e9`

공통 source/runtime 손상 가설은 기각한다. writer NoC 결합도 단독 root cause가 아니며, fanout-3 dual-NoC 전용 core-set/kernel instantiation과 reader completion 계약을 다음 조사 범위로 제한한다. 성공한 fanout-2 뒤 장치는 정상 close됐고 격리 상태가 아니다. TurboQuant는 사용하지 않았다.

## Split-kernel-only 재현으로 좁힌 원인 범위

정상 대조군 뒤 standard fanout-3의 mapping과 generic bank address helper를 보존하고, RISCV0 in1
reader/writer source만 NOC0/NOC1 kernel handle 두 개로 나눈 opt-in을 실행했다. 즉 이전 실패의
explicit endpoint coordinate, custom endpoint당 3-reader placement, 별도 writer NoC와 VC 변경을
모두 제거했고 reader NoC split `9:9`만 남겼다.

- run: `mlp_fanout3_split_kernel_only_correctness_2026_08_03_14_02_23`
- timeout: `timeout --signal=INT --kill-after=15s 180s`
- Watcher: `TT_METAL_WATCHER=100ms`
- 마지막 정상 host marker: W1/W2 program 생성 및 split-only topology 확인
- 누락 marker: PCC, `MLP_COMPLETED`, `DEVICE_CLOSED`
- 종료: 약 194초 뒤 SIGKILL 상한, exit 137로 분류
- child: Python PID 4737, PID 1 아래 `Z/<defunct>`
- run log SHA-256: `de7cc49e6bcf6490250a5bb9f281d13e97d9c44e92dc5fe5a9b3bc5786748115`
- Watcher SHA-256: `f1b9146f4f194c5eb931275f9d891a5a2bf599d22359913b8aa478ecb8dd713f`

Watcher는 오류를 보고하지 않고 dump를 계속했으며 마지막 dump에 같은 source의 RISCV0 kernel ID 5와
6이 공존했다. 따라서 정확한 read barrier/CB waypoint는 미확정이다. 다만 성공한 standard fanout-3
대비 의도적으로 남긴 변화가 split kernel handle뿐이므로, 원인 신뢰도는 다음처럼 갱신한다.

- 높은 신뢰도: 같은 RISCV0 processor에서 NoC 설정이 다른 두 data-movement kernel handle을 core-set별로
  생성한 방식 또는 그에 수반되는 kernel-group/runtime contract가 hang을 유발한다.
- 낮아진 가설: explicit endpoint address, custom core placement, VC coloring, output writer NoC 결합은
  각각 단독 root cause가 아니다.
- 미확정: split handle 자체의 dispatch/kernel-group 문제인지, NOC0 variant 내부의 read completion
  문제인지는 kernel waypoint 없이는 구분할 수 없다.

exit 137로 장치를 다시 격리했다. 재부팅 확인과 32×32 add gate 전에는 어떤 device workload도 실행하지
않는다. TurboQuant는 이번 실행에서도 사용하지 않았다.

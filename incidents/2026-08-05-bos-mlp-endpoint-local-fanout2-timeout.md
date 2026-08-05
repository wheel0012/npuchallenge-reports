# BOS MLP endpoint-local fanout-2 timeout

- 날짜: 2026-08-05 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- 상태: 장치 격리

## 영향

Profiler/Watcher 없는 isolated Llama 3.2 3B MLP correctness 1회가 device completion 없이 정지했다.
외부 timeout cleanup이 완료되지 않아 SIGKILL, exit code 137로 끝났다. `MLP_PCC`, `MLP_COMPLETED`,
`DEVICE_CLOSED`는 없었다. 추가 device workload를 중단했다.

## 마지막 실행 구성

- DRAM-sharded BFP8 weights
- W2 `in0_block_w=16`, 16 KiB read-page cap
- fanout-2 tagged depth-2
- physical DRAM endpoints x=0..5에 reader/compute 2개씩
- native NoC: x={0,4,5} NOC0, x={1,2,3} NOC1, reader 6:6
- conflict-free endpoint/route VC coloring
- separate opposite-NoC output write
- TurboQuant, fused gate/up, helper, profiler, Watcher off
- timeout: `timeout --signal=INT --kill-after=15s 90s`

Opt-in은 `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=1`이다. 기본 경로에는 영향 없다.

## 관측 타임라인

1. 기존 정상 장치 상태에서 host build와 runtime install 성공.
2. W1/W3와 W2 program 생성 성공.
3. 두 projection 모두 `2:2:2:2:2:2`, NOC reader `6:6`, compute workers 12 확인.
4. physical mapping은 endpoint x=0,1,2의 인접 y=2/4를 우선 사용했다. 남은 endpoint는 row-cap과
   worker 중복 방지 때문에 x=2..4, y=0/1에 배치됐다.
5. 이후 PCC/completion/close marker 없이 정지.
6. 90초 SIGINT cleanup 뒤 15초 상한에서 SIGKILL, exit 137.
7. Python PID 5920은 PID 1 아래 `Z/<defunct>`로 남았다. 살아 있는 Tracy/capture child는 없었다.

## 원인 상태

확정 원인 없음. Host mapping과 kernel compile은 성공했으므로 정지는 device dataflow launch 이후다.
동일 physical six-endpoint explicit-address와 split NOC0/NOC1 reader-kernel 계열은 fanout-3에서도 같은
no-completion 이력이 있다. Reader 수나 VC만의 문제보다 split-kernel producer/consumer 또는
explicit-endpoint writer 계약이 유력하다. Watcher waypoint가 없으므로 정확한 wait 위치는 모른다.

## 복구와 예방

- 재부팅 확인 전 add smoke 포함 모든 device workload 금지.
- 재부팅 뒤 첫 workload는 timeout 보호 32×32 add 한 번.
- 이 endpoint-local flag 재실행 금지.
- 다음 분석은 host-side kernel runtime-arg 및 CB producer/consumer 계약 비교만 수행.
- 안전한 기본값은 기존 NOC1 4:4:4, 12-compute tagged depth-2다.

## 재현·감사 정보

- source patch: `/home/iris_hb4/tmp/codex-patches/20260805-053000-mlp-fanout2-endpoint-local.patch`
- patch SHA-256: `41557528232452b3930f7d5d56f32faff454f9bf47deac68fb6936f11ba0ab8a`
- source SHA-256: `4415a1dba4018499108d1bc86bfcf588397a99ffe7f82cc265f4e3b4b4cce368`
- runtime `_ttnncpp.so` SHA-256: `935fb275e4e4d69bf8f205c1fc80df3cf2a5f7a88b25f9159b867e22ac264c78`
- 별도 profiler artifact 없음. Console transcript만 존재.

## 복구 확인

사용자가 서버 재부팅을 확인했다. 2026-08-05 05:20 UTC 첫 device workload로 기존 32×32 BF16 add를
실행했다. `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0을 확인했다. 일반 장치 격리는 해제한다.

실패한 `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=1` 구성은 계속 금지한다. 복구 성공은 해당
kernel의 correctness를 의미하지 않는다.

## single-kernel runtime-NoC 재검증

사용자가 서버 재부팅을 확인한 뒤 32×32 add gate가 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로
통과했다. 기존 split RISCV0 reader kernel handle을 endpoint-local 경로에서 제거하고, 단일 RISCV0
kernel이 endpoint x로 `reader_noc`을 runtime 선택하도록 수정했다. Address generation, tagged TRID read,
read barrier에는 같은 runtime NoC을 전달했고 output write는 반대 NoC을 사용했다.

Profiler, Watcher, TurboQuant, helper, fused 경로는 끈 채 다음 구성을 correctness 1회 실행했다.

- W2 `in0_block_w=16`, 16 KiB read-page cap
- fanout-2 tagged depth-2
- endpoint x=0..5에 reader/compute 2개씩, 총 12
- reader NoC 6:6, endpoint/route VC coloring 유지
- timeout: `timeout --signal=INT --kill-after=15s 90s`

Host log에서 W1/W3 13,056-byte read page, W2 8,704-byte read page와 두 program 모두
`endpoint-local single-kernel`, `2:2:2:2:2:2`, reader NoC `6:6`, compute 12를 확인했다. 그러나 첫 run과
같이 `MLP_PCC`, `MLP_COMPLETED`, `DEVICE_CLOSED` 없이 정지했고 exit 137이었다. Python PID 3103은
PID 1 아래 `Z/<defunct>`다. Tracy/capture child와 profiler artifact는 없다.

이 결과는 split NOC0/NOC1 kernel handle이 필요조건이라는 가설을 반증한다. 남은 공통점은 physical
explicit-endpoint addressing, endpoint별 runtime NoC, opposite-NoC output write 및 그 CB/writer 계약이다.
정확한 device wait 위치는 Watcher waypoint가 없어 여전히 모른다.

장치는 다시 격리한다. 재부팅 또는 계약에 명시된 별도 open/close 복구 확인 전에는 add 포함 device
workload를 실행하지 않는다. endpoint-local opt-in은 계속 실패 구성으로 금지한다.

### 추가 감사 정보

- core patch: `/home/iris_hb4/tmp/codex-patches/20260805-053500-mlp-endpoint-local-single-kernel.patch`
  (`9ea19cfa995df7c17a315c2f9c2663d405d4a54115909109b1a2cbe4d4fe7368`)
- TRID patch: `/home/iris_hb4/tmp/codex-patches/20260805-061000-mlp-runtime-noc-trid-set.patch`
  (`6cd0db1ada37a3d465afb18fb7917caf5e6cd8a8cb1ddfe22a66f67318184e5f`)
- read/barrier patch: `/home/iris_hb4/tmp/codex-patches/20260805-061500-mlp-runtime-noc-read-barriers.patch`
  (`e37ecd72fda668ff303627779f045b9ea80e350fb2270063d9741d48d60ad7d1`)
- factory source: `7ff64f9479f330ef8ec3b386bd7c14fbc7bd63c17a4167325b36fcab7e7e7c0f`
- reader source: `3cae94cf255e2839439ee944b685a38cd47303c2e26e3e335b78ce8732834858`
- runtime `_ttnncpp.so`: `a33011fc619de9a33e0456cd74c5897319d96e40a6aed1acfe6abfeebffb2302`

## 두 번째 복구 확인

사용자가 다시 서버 재부팅을 확인했다. 2026-08-05 05:45 UTC 첫 workload인 32×32 BF16 add가
`SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과했다. 일반 장치 격리는 해제한다.

실패한 endpoint-local flag는 계속 금지한다. 복구 뒤 추가 experimental device workload는 실행하지
않았다.

## fixed-writer 및 DRAM-view ordering 분해

GEMM_FLOPS의 고정-NoC writer 계약을 반영해 endpoint-local writer를 kernel `noc_index`로 고정했다.
Reader의 explicit endpoint와 runtime NoC은 유지했다.

첫 correctness는 timeout 없이 종료했다. `DEVICE_CLOSED`, exit 1을 확인했고 PCC는
`0.028334455265917317`로 실패했다. 반대-NoC writer를 제거하자 no-completion hang이 사라졌으므로
opposite-NoC output reshard writer가 이전 hang의 직접 원인이라는 증거가 강하다.

낮은 PCC의 원인은 endpoint-x 순서 worker 배열과 DRAM-view 순서 weight/output partition 불일치로
판단했다. Worker 배열을 DRAM view `0,0,1,1,...,5,5` 순서로 바꾼 두 번째 correctness는 mapping log 뒤
completion 없이 정지했다. 90초 SIGINT와 15초 cleanup 상한 뒤 exit 137이었다. Python PID 2866은
PID 1 아래 `Z/<defunct>`다. Tracy/capture child와 profiler artifact는 없다.

이 결과는 `compute_worker_cores_ordered`가 weight shard 순서뿐 아니라 output reshard/worker ownership
계약에도 결합되어 있음을 뜻한다. Physical placement 배열을 단순 재정렬해서 correctness를 복구할 수
없다.

장치는 다시 격리한다. 재부팅 또는 계약에 명시된 별도 open/close 복구 확인 전에는 add 포함 device
workload를 실행하지 않는다.

### 추가 감사 정보

- fixed-writer patch: `/home/iris_hb4/tmp/codex-patches/20260805-071000-mlp-endpoint-local-fixed-writer-noc.patch`
  (`f051aa1ce410f4450cca3fe4405082ca66b28bb9b0eef17712bd8a40ce658777`)
- DRAM-view-order patch: `/home/iris_hb4/tmp/codex-patches/20260805-072000-mlp-endpoint-local-dram-view-order.patch`
  (`1a784ea56f11f680e7019aba278d01c6da06169691364b2d82c0b3d7102d8853`)
- factory source: `97b4551f49ca1dade05274e625c506980b51d0fb8dadbdb9135ef13068da61fe`
- reader source: `e7fea173f8d0c8928a2865c0dc7dd51df4cff9f53c0dabfc20f651f5c80a95f6`
- runtime `_ttnncpp.so`: `70cb05fd1d316914f7b714e6552bc7007b2429512e84e39e6b7f47e7bcd66c48`

## logical writer-order 분리본 실행 차단

사용자가 서버 재부팅을 확인했다. 첫 workload인 32×32 add는 2026-08-05 06:15 UTC에
`SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과했다.

Physical endpoint-x core 순서는 유지하고 output reshard runtime args만 logical weight partition
순서로 생성하도록 분리했다. `writer_kernel_ids`는 physical worker index로 보존했다. Host build와
runtime install은 성공했다.

- patch: `/home/iris_hb4/tmp/codex-patches/20260805-074001-mlp-decouple-writer-partition-order.patch`
  (`49650cc2773cb2e9d8942a7d3d6624ac53dd08f74aa55493c28fe46d8e624e29`)
- factory source: `b34552a2b77b4ee08453d171467582073d144c7022a3aaa444c486b88f368162`
- runtime `_ttnncpp.so`: `130b1b9ddf4f90cd0466a61700eb6f6d1885ba99302af4737b0eab7dd898f73c`

사용자의 1회 실행 승인을 받은 뒤 다음 명령을 실행했다.

```bash
env -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER -u TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT3 -u TT_METAL_MLP_DRAM_SHARDED_FANOUT3_DUAL_NOC \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT3_SPLIT_KERNEL_ONLY -u TT_METAL_WATCHER \
  PATH=/home/iris_hb4/tt-metal-hb4/python_env/bin:/usr/local/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/home/iris_hb4/tt-metal-hb4/build_home_release/lib:/home/iris_hb4/tt-metal-hb4/build_home_release/tt_metal \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=1 TT_METAL_MLP_LOG_READER_MAP=1 \
  timeout --signal=INT --kill-after=15s 90s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

실행은 device kernel 전에 `CHIP_IN_USE_0_PCIe` lock에서 대기했다. Lock owner PID 3112는 다른
workspace의 NIAH 64K Python 작업이었다. 90초 SIGINT와 15초 cleanup 상한 뒤 exit 137이었다. 대상
Python PID 3996은 PID 1 아래 `Z/<defunct>`다. MLP mapping log, PCC, `MLP_COMPLETED`, `DEVICE_CLOSED`,
Tracy/capture artifact는 없다. 따라서 이번 결과는 새 writer-order 코드의 실패 증거가 아니다.

PID 3112는 다른 작업이라 종료하지 않았다. Exit 137 안전 규칙에 따라 장치는 격리한다. Lock owner가
정상 종료한 뒤 사용자 재부팅 또는 별도 device open/close 성공 확인이 필요하다.

# BOS MLP prefetch-helper NoC write-barrier Watcher abort

- Date/time (UTC): 2026-08-03 10:05:31 device abort, 10:08경 timeout SIGKILL
- System/device: custom 20-core BOS NPU, Blackhole runtime/code architecture
- Severity: 장치 격리 필요
- Status: 재부팅·안전 게이트·수정 경로 검증 완료; 성능상 기본 경로 채택 안 함

## Impact

- 6 compute owner + 6 prefetch helper MLP의 correctness, latency 및 bandwidth를 측정하지 못했다.
- Watcher가 device를 중지한 뒤 Python이 uninterruptible `D` 상태에 머물렀다.
- timeout cleanup이 완료되지 않아 exit code 137로 끝났고 장치 상태를 신뢰할 수 없게 됐다.
- 사건 뒤 add smoke, device open/close, MLP, SDPA와 profiler를 추가 실행하지 않았다.

## Device and experiment scope

- available worker grid: 5×4 = 20 cores
- physical DRAM: 3 banks
- worker NoC endpoints: bank당 2개, 총 6개
- runtime-selected DRAM interface workers: 6
- experimental active compute/output owners: 6
- dedicated prefetch helpers: 6
- total DRAM readers: 12
- weight-read NoC destination groups: 4:4:4 on NOC1
- model: `meta-llama/Llama-3.2-3B-Instruct`, layer 0 isolated decode MLP
- W2 `in0_block_w=16`, DRAM-sharded fanout-2, 16 KiB read-page cap

runtime의 P150 추정 log는 custom BOS의 authoritative board identity가 아니다. program grid와 실제
active compute core 수도 구분한다.

## Timeline

1. stale `_ttnncpp.so`를 읽은 첫 run은 기존 `readers: 6, compute workers: 12`로 실행돼 무효 처리했다.
2. 새 runtime을 명시한 run은 홀수 43-tile shard-width host guard에서 exit 1, 정상 close했다.
3. width를 44 tiles로 교정한 run은 helper runtime args 1→7 count 변경 host guard에서 exit 1,
   정상 close했다.
4. helper args를 처음부터 7개로 고정하고 Watcher 100 ms로 correctness 1회를 실행했다.
5. 10:05:30.941: `balanced endpoints ... 4:4:4`,
   `prefetch helpers: true, readers: 12, compute workers: 6` 확인.
6. 10:05:31.702: Watcher가 core `(4,0)` BRISC의 pending non-posted NoC write를 검출하고 device 중지.
7. Python이 signal abort 뒤 `D` 상태로 남고 다섯 `sh` child가 zombie가 됐다.
8. 180초 timeout의 SIGINT와 15초 cleanup 상한 뒤 SIGKILL, 최종 exit 137.
9. 사후 Python과 다섯 child는 PPID 1의 zombie로 남았다. 반복 signal은 보내지 않았다.
10. 장치를 격리하고 host-side source 수정, report 작성 및 build만 수행했다.

## Last known command and configuration

```bash
env \
  LD_LIBRARY_PATH=/home/iris_hb4/tt-metal-hb4/build_home_release/ttnn \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 \
  TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=1 \
  TT_METAL_WATCHER=100ms \
  timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

NoC/Tracy profiler는 사용하지 않았다.

## Observed symptoms

Watcher는 core `(4,0)` BRISC가 `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`를 끝낼 때
pending non-posted NoC write가 남았다고 진단했다. 마지막 waypoint는 `NKFW, W, W, W, W`였다.
PCC, measured sample, `MLP_COMPLETED`, `DEVICE_CLOSED`, ops CSV와 NoC trace는 생성되지 않았다.

## Suspected cause and confidence

직접 trigger의 신뢰도는 높다. helper는 local CB의 half-shard를 owner CB로 row-wise
`noc_async_write`한 뒤 write barrier 없이 remote valid semaphore를 보내고 local CB를 pop했다.
ordered semaphore만으로는 kernel completion의 pending non-posted write를 drain하지 않는다.

표준 sender 예제는 다음 receiver credit 전까지 local CB가 덮어써지지 않아 barrier를 생략할 수 있다.
이번 helper는 다음 credit 전에 다음 DRAM block을 prefetch하므로 ring slot이 재사용될 때 이전 async
write의 source를 덮어쓸 수도 있다. 따라서 kernel 끝의 barrier 하나보다 block별 barrier가 필요하다.
Watcher abort 뒤 Python cleanup이 끝나지 않은 원인은 별도 stack capture가 없어 중간 신뢰도다.

## Recovery and correction

- 장치 reset, PCIe reset, driver unbind/rebind 및 추가 open/close를 수행하지 않았다.
- helper row writes 뒤 `noc_async_write_barrier()`를 추가하고 그 뒤 valid semaphore와 CB pop을 수행한다.
- patch: `/home/iris_hb4/tmp/codex-patches/20260803-141500-mlp-helper-write-barrier.patch`
- `git diff --check`와 host `ttnncpp` build가 통과했다.
- dataflow kernel JIT compile과 device correctness는 격리 때문에 미검증이다.

## Artifacts

- failed Watcher run: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_correctness_2026_08_03_13_30_00/run.log`
- invalid stale-runtime run: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_correctness_2026_08_03_12_50_00/run.log`
- host-validation runs: `mlp_prefetch_helpers_correctness_2026_08_03_13_05_00`,
  `mlp_prefetch_helpers_correctness_2026_08_03_13_15_00`

## Preventive actions

1. 사용자 재시작 확인 전까지 장치 격리를 유지한다.
2. 재시작 뒤 첫 workload는 외부 timeout이 있는 32×32 add 한 번으로 제한한다.
3. add 성공 뒤 profiler 없는 isolated correctness/JIT 1회만 먼저 실행한다.
4. correctness와 짧은 latency 성공 뒤에만 warmup 1 + measured 1 NoC capture를 한다.
5. barrier가 성능을 상쇄해도 explicit completion credit 또는 별도 staging ring 없이 제거하지 않는다.
6. exit 124/137 또는 signal 종료가 재발하면 즉시 다시 격리한다.

## Post-reboot validation

사용자가 서버 재부팅을 확인한 뒤 계약 순서대로 검증했다.

- 32×32 BF16 add: `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0
- barrier correctness + 1 sample: PCC 0.999641, 1.581519 ms, 정상 close, exit 0
- profiler-free 5 samples: mean 1.556066 ms, median 1.560501 ms, min 1.523887 ms
- NoC capture: PCC/close/exit 0, ops CSV·device CSV·raw NoC JSON·host Tracy 완성
- counter capture: PCC/close/exit 0, FPU counter 완성

모든 성공 run은 profiler/Watcher-free validation을 먼저 통과한 뒤 개별 profiler pass로 실행했다.

### Final findings

- 실제 FPU-active compute core는 정확히 6개였다.
- active-core FPU utilization은 W1/W3 46.91--49.29%, W2 44.31--46.63%였다.
  12-compute balanced 경로의 약 18.26--19.46%보다 높지만 compute ceiling은 아니다.
- DRAM read destination과 request count는 기존처럼 NOC1 `4:4:4`로 균형을 유지했다.
- reconstructed projection aggregate read rate는 62.93에서 59.58 GB/s로 5.33% 감소했다.
- helper→owner remote L1 payload는 W1/W3 각 13,787,136 B, W2 13,369,344 B로,
  projection 합계 40,943,616 B의 추가 on-chip traffic과 block별 write barrier가 생겼다.

따라서 barrier 누락은 교정됐고 구성은 correctness를 만족하지만, 기존 12-compute balanced 경로보다
6-compute helper의 mean latency가 5.69% 높고 역수 처리율은 5.38% 낮다. 현 구현은 opt-in 실험으로
남기며 기본 MLP 경로로 채택하지 않는다.

success artifacts:

- correctness: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_correctness_2026_08_03_15_45_00`
- latency: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_latency_2026_08_03_15_50_00`
- NoC: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_noc_2026_08_03_15_55_00`
- counters: `/home/iris_hb4/profiler_runs/mlp_prefetch_helpers_barrier_counters_2026_08_03_16_05_00`

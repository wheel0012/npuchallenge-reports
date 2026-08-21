# BOS MLP 6-compute 20-iteration timeout

- 날짜: 2026-08-05 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- 영향: isolated MLP completion/close 없음, exit 137, 장치 격리

## 타임라인

- 재부팅 뒤 32×32 BF16 add: value 2.0, close, exit 0.
- 12-compute tagged depth-1 correctness: PCC 0.9996410623, close, exit 0.
- 12-compute depth-2 20회: mean 1.467559 ms, close, exit 0.
- 12-compute depth-1 20회: mean 1.493915 ms, close, exit 0.
- 04:24 UTC: fanout off 6-reader/6-compute 20회 시작.
- factory log: readers 6, compute workers 6, W1/W3 read page 6528 B, W2 8704 B.
- 90초 timeout 뒤 SIGINT cleanup 미완료. `--kill-after=15s` 발동, exit 137.
- `MLP_COMPLETED`, `DEVICE_CLOSED`, PCC와 samples 출력 없음.
- 종료 뒤 관련 Python/Tracy/capture child 없음. 추가 device workload 없음.

## 마지막 실행 구성

```bash
env -u TT_METAL_MLP_FUSED_GATE_UP -u TT_METAL_MLP_FUSED_GATE_UP_EPILOGUE \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH1 \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH3 \
  -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER \
  MLP_AB_ITERATIONS=20 TT_METAL_MLP_DRAM_SHARDED=1 \
  TT_METAL_MLP_W2_IN0_BLOCK_W=16 TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=0 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=0 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 \
  TT_METAL_TURBOQUANT=0 timeout --signal=INT --kill-after=15s 90s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## 원인 평가

확정 사실은 6-reader/6-compute 구성과 90초 미완료뿐이다. correctness call, measured loop 또는 close
중 어디에서 정지했는지 marker가 없어 모른다. 6 compute가 단순히 느린지 kernel deadlock인지도
구분하지 못한다. 따라서 성능 수치나 compute 부족 비율을 추론하지 않는다.

## 복구 및 예방

- exit 137 즉시 장치를 격리했다.
- 다음 device workload 전 사용자의 host/server 재시작 확인이 필요하다.
- 동일 20-iteration 6-compute 명령을 재실행하지 않는다.
- 재부팅 뒤 필요하면 1 measured call만 실행하고 host-side correctness/iteration waypoint를 추가한다.
- 첫 1회가 통과한 뒤에만 3회로 늘린다. profiler와 Watcher는 사용하지 않는다.
- endpoint admission 비교의 현행 결론은 성공한 depth-2/depth-1 A/B만 사용한다.

Artifact는 없다. console transcript만 존재한다.

## 2026-08-18 factor-1 row-reader 재현

### 목적과 변경

기존 실패가 legacy one-packet reader protocol 때문인지 분리하기 위해 2×2 실험의 첫 안전 셀로
`interleaved + 6-reader/6-compute`를 실행했다. 새 opt-in
`TT_METAL_MLP_DRAM_SHARDED_ROW_READER_FACTOR1=1`은 6개 compute/reader mapping을 유지하면서
fanout-2와 같은 contiguous row-read 및 tagged depth-2 분기를 사용한다. profiler와 Watcher는 사용하지
않았고 correctness call 뒤 measured iteration을 1회만 요청했다.

실행 전 같은 boot에서 32×32 add gate와 stable full Llama decode가 completion/close까지 성공했고,
변경한 host `ttnn` target도 빌드에 성공했다.

### 정확한 실행 구성

```bash
env -u TT_METAL_MLP_FUSED_GATE_UP -u TT_METAL_MLP_FUSED_GATE_UP_EPILOGUE \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH1 \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH3 \
  -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  MLP_PRECISION_MODE=performance MLP_PCC_THRESHOLD=0.98 MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_ROW_READER_FACTOR1=1 TT_METAL_MLP_DRAM_SHARDED_FANOUT2=0 \
  TT_METAL_MLP_FANOUT2_INTERLEAVED_WEIGHTS=1 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=0 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT16=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_DISTINCT_VC=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_OOO_COMPLETION_PROBE=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_STAGGER_CYCLES=0 TT_METAL_TURBOQUANT=0 \
  timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

### 관측 사실과 영향

- performance precision은 W1/W3 `BFLOAT4_B`, W2 `BFLOAT8_B`를 로드했다.
- factory는 W1/W3 및 W2 모두 `readers: 6, compute workers: 6`, row-reader factor 1을 만들었다.
- 마지막 안정 marker는 14:33:28 UTC의 W2 program 생성 로그다. W1/W3 read page는 13,824 B,
  W2는 8,704 B였다.
- PCC, `MLP_COMPLETED`, `DEVICE_CLOSED`는 출력되지 않았다. 첫 correctness call completion 전 정지로
  분류한다.
- 180초 SIGINT cleanup이 끝나지 않아 15초 뒤 SIGKILL이 발동했고 exit code는 137이었다.
- 종료 뒤 관련 Python, Tracy 또는 capture child는 남지 않았다.
- 장치는 즉시 격리했다. 같은 boot에서 추가 add, open/close, 12-reader 셀은 실행하지 않았다.

### 원인 분석

이번 결과로 legacy one-packet reader protocol 단독 원인 가설은 약해졌다. 동일한 row/tagged reader로
교체해도 6-compute performance 경로가 같은 단계에서 멈췄기 때문이다.

더 강한 정적 원인 후보는 non-fanout compute padding 계약이다.

1. `per_core_N_output`, `per_core_N_in1_sender`, `per_core_N_in1_reader`가 최초
   `per_core_N_compute`에서 계산된다.
2. 이후 `!use_fanout` 분기에서 `per_core_N_compute`만 subblock 폭으로 다시 padding된다.
3. compute kernel의 `in1_num_subblocks`와 output block은 변경된 compute 폭을 사용하지만, reader의
   `in1_block_num_tiles`와 CB 크기는 변경 전 sender 폭을 사용한다.
4. 따라서 performance-mode subblock padding이 실제로 폭을 바꾸면 compute가 reader가 push하지 않는
   in1 tiles를 기다릴 수 있다. accuracy-mode 6-reader 성공과 fanout-2 performance 성공은 이 분기를
   각각 변화 없이 통과하거나 건너뛴다는 설명과 양립한다.

이는 source contract와 실패 단계에 근거한 **중간-높음 신뢰도 추론**이며 device counter로 확정한
root cause는 아니다. factor-1 row-reader에 한해 legacy padding 분기를 건너뛰도록 조건을
`!use_fanout`에서 `!use_row_reader`로 바꿨고 host build는 성공했다. 장치 격리 때문에 수정본의 device
검증은 수행하지 않았다.

### Artifact, patch와 다음 안전 단계

- 로그: `/home/iris_hb4/benchmark_runs/mlp_reader_layout_2x2_2026_08_18_14_32_41/interleaved_6_safety.log`
- 로그 SHA-256: `d8ea5eabb37adb33759c7d7f233c3c82a6e68819602acd4a321469a621962e80`
- row-reader patch: `/home/iris_hb4/tmp/codex-patches/20260818-mlp-factor1-row-reader.patch`
- padding-contract patch: `/home/iris_hb4/tmp/codex-patches/20260818-mlp-factor1-padding-contract.patch`

다음 device 작업은 사용자 재부팅 확인 뒤 32×32 add gate 1회다. gate가 성공한 경우에만 수정된
interleaved factor-1 correctness+latency 1회를 다시 실행한다. 그 셀이 completion/close까지 성공해야
나머지 `interleaved 12`, `sharded 12`, `sharded 6` 셀로 진행한다.


## 2026-08-18 재부팅 후 복구 및 root-cause 확인

사용자가 서버 재부팅을 확인했다. 첫 device workload인 32×32 BF16 add는 `SMOKE_VALUE 2.0`,
`DEVICE_CLOSED`, exit 0으로 통과했다. 수정된 interleaved factor-1도 performance mode에서 PCC
`0.9869040195`, 단일 latency `2.086355 ms`, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0으로 통과했다.
이후 3회와 30회까지 정상 완료했다.

수정 전과 수정 후의 차이는 factor-1 row-reader가 non-fanout compute-only padding 분기에 들어가는지 여부다.
수정 전에는 동일 correctness call이 180초 동안 완료되지 않았고, 수정 후에는 동일 dtype, reader count,
layout 및 tagged depth로 즉시 완료됐다. 따라서 `per_core_N_compute`와 reader/CB width가 달라지는
padding-contract mismatch를 이번 6-compute performance hang의 **높은 신뢰도 직접 원인**으로 승격한다.
legacy packet reader는 별도 성능·안전 문제가 있을 수 있으나 이번 hang의 필요조건은 아니었다.

네 개 2×2 셀의 1회, 3회, 30회 실행이 모두 completion/close했으므로 이 boot의 장치 격리는 해제 상태로
유지한다. 최종 30회 평균은 interleaved 6/12가 `2.067791/1.527894 ms`, width-sharded 6/12가
`1.377647/1.040927 ms`였다. 상세 통계와 checksum은
`investigations/2026-08-17-bos-dram-characterization-to-sdpa-mlp.md`에 기록했다.

복구 add 로그:
`/home/iris_hb4/benchmark_runs/mlp_reader_layout_2x2_2026_08_18_14_32_41/add_gate_after_reboot.log`
(SHA-256 `30249bda1f02a4aa8fa2bf1a2e73ead2e11b0c45f06110ca53962b85b2e0ec3d`).

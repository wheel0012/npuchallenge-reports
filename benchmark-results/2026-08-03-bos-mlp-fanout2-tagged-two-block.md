# BOS MLP fanout-2 tagged two-block A/B

날짜: 2026-08-03 UTC

## 결론

balanced fanout-2 row-burst reader의 block별 full read barrier를 두 개의 transaction ID(TRID)가
교대하는 tagged pipeline으로 바꾸자 isolated Llama 3.2 3B MLP 평균 지연이
`1.472701 ms`에서 `1.439071 ms`로 **2.28% 감소**했다. 중앙값은 `1.460173 ms`에서
`1.428399 ms`로 **2.18% 감소**했다. 양쪽 PCC는 모두 `0.9996410623`이었고 모든 실행이
`MLP_COMPLETED`, `DEVICE_CLOSED`, exit code 0으로 끝났다.

따라서 이전 readiness profile에서 확인한 weight-late 병목의 일부는 block마다 모든 DRAM read를
완료시키던 barrier bubble이었다. 다만 개선폭이 약 2%이므로 weight delivery의 나머지 병목은
endpoint/route contention, block publish cadence, compute 소비 속도 등에 남아 있다.

## 하드웨어 및 실행 범위

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker topology: 5×4 = 20 cores
- 이 matmul 경로의 실제 구성: 12 readers, 12 compute workers
- physical DRAM topology: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime log: `Dram Interface Workers: 6`
- balanced NOC1 endpoint groups: 4:4:4

`Dram Interface Workers: 6`은 이 data path가 선택한 interface-worker 수이며 physical bank 수나
tensor shard 수와 동일한 뜻으로 해석하지 않는다. runtime의 P150 출력은 BOS에서의 heuristic
이름일 뿐 board identity로 사용하지 않는다.

## 변경 내용

새 경로는 opt-in 환경변수 `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1`일 때만 활성화된다.
기본 fanout-2 full-barrier 경로는 그대로 유지했다.

1. host program factory가 opt-in을 읽고 fanout-2이며 prefetch helper가 꺼진 경우에만
   `DRAM_SHARDED_FANOUT2_TAGGED` compile define을 reader kernel에 전달한다.
2. reader는 기존 triple-buffer CB의 slot 0/1/2를 순환한다.
3. NoC read에는 TRID 1/2를 교대로 태깅한다. CB slot 수와 TRID 수는 서로 독립적이다.
4. 현재 block read를 issue한 뒤 직전 TRID만 기다리고 직전 block을 publish한다.
5. 다음 block read가 먼저 발행되므로 정상 steady state에서는 최소 한 block의 DRAM request가
   outstanding 상태로 남는다.
6. 마지막 block 뒤에는 마지막 TRID barrier를 수행하고 최종 CB block을 publish한다.

즉 이 구현은 tech report의 “pending request를 하나 이상 유지”하는 아이디어를 fanout-2
row-burst reader에 적용한 것이다. 데이터를 검증하지 않고 미리 publish하는 방식이 아니라,
각 block은 자기 TRID completion이 확인된 뒤에만 consumer에게 공개된다.

## 수정 파일과 검증 식별자

- factory:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp`
  - SHA-256: `bf2f3a1f840f208e0f862d5b14be7f470c1cd82df56ad847bf43225be4a39575`
- reader:
  `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`
  - SHA-256: `d8b0cee059a0a19cdf807413de39d26baa2a6e3a1c4ee4992a0cdf4b9339ba6e`
- runner:
  `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py`
  - SHA-256: `3965949c09cb0112c1bf918e8d1f5541d84cc8341bcdce0b6d3e83398457e711`
- 실제 Python runtime이 로드한 library:
  `/home/iris_hb4/tt-metal-hb4/build_home_release/lib/_ttnncpp.so`
  - SHA-256: `4eafa4c0493aacf3dcd9ae21cecb0259f91c8566d1fd1eca5e5e478c91ab5865`

적용 patch는 감사 목적으로 아래에 보존했다.

- `/home/iris_hb4/tmp/codex-patches/20260803-172100-mlp-fanout2-tagged-factory-v2.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-172300-mlp-fanout2-tagged-reader-main-v2.patch`
- `/home/iris_hb4/tmp/codex-patches/20260803-172400-mlp-fanout2-tagged-reader-close-v3.patch`

## 빌드 경로 검증

처음 `build_Release/ttnn/_ttnncpp.so`를 빌드했지만 Python extension의 `ldd` 결과 실제 dependency는
`build_home_release/lib/_ttnncpp.so`였다. 잘못된 build tree로 실험하는 것을 막기 위해 다음 순서로
실제 runtime tree를 갱신했다.

```bash
ninja -C build_home_release ttnncpp
cmake --install build_home_release --component ttnn-runtime
ldd ttnn/ttnn/_ttnn.so | rg _ttnncpp
strings build_home_release/lib/_ttnncpp.so | rg FANOUT2_TAGGED
```

최종 `ldd`는 `/home/iris_hb4/tt-metal-hb4/build_home_release/lib/_ttnncpp.so`를 가리켰고,
library에서 opt-in과 compile define 문자열을 모두 확인했다.

## 측정 조건

공통 조건은 다음과 같다.

- model: `meta-llama/Llama-3.2-3B-Instruct`, layer 0 isolated MLP
- DRAM-sharded weights, fanout-2, balanced endpoints
- W2 `in0_block_w=16`
- 16 KiB read-page cap, reader locality off
- prefetch helpers off, fanout-3 off, TurboQuant off
- profiler/Watcher/NoC capture 없음
- 각 process에 `timeout --signal=INT --kill-after=15s 120s` 적용
- correctness gate 1회 뒤 각 variant 5 samples

핵심 재현 명령은 아래와 같다. `<TAGGED>`를 `0` 또는 `1`로 바꾼다.

```bash
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=5 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=<TAGGED> \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 TT_METAL_TURBOQUANT=0 \
  timeout --signal=INT --kill-after=15s 120s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## 결과

| variant | samples (ms) | mean (ms) | median (ms) | min (ms) | PCC |
|---|---|---:|---:|---:|---:|
| full barrier | 1.514896, 1.470919, 1.457718, 1.459798, 1.460173 | 1.472701 | 1.460173 | 1.457718 | 0.9996410623 |
| tagged two-block | 1.482735, 1.426772, 1.428399, 1.427181, 1.430268 | 1.439071 | 1.428399 | 1.426772 | 0.9996410623 |

- mean latency reduction: `(1.472701 - 1.439071) / 1.472701 = 2.28%`
- median latency reduction: `(1.460173 - 1.428399) / 1.460173 = 2.18%`
- minimum latency reduction: `(1.457718 - 1.426772) / 1.457718 = 2.12%`

1회 tagged correctness gate도 PCC `0.9996410623`, latency `1.488706 ms`, 정상 close/exit 0이었다.

## 관측 사실, 해석, 한계

### 관측 사실

- tagged와 full-barrier 모두 동일 PCC와 정상 종료를 보였다.
- tagged의 mean, median, min이 모두 full-barrier보다 낮았다.
- reader/compute 수와 endpoint group은 양쪽 모두 12/12 및 4:4:4였다.

### 해석

- tagged issue가 DRAM read latency와 block publish/compute를 일부 겹쳤다는 해석이 가장 타당하다.
- 이전 profile에서 weight가 W1/W3/W2 core-block pair의 76.6/68.8/75.0%에서 늦었던 사실과
  방향이 일치한다.
- 약 2% 개선은 full barrier가 병목의 일부였다는 증거이지, 모든 weight wait가 제거됐다는 뜻은 아니다.

### 한계

- 표본은 variant당 5개이며 process 시작/JIT 뒤의 짧은 isolated MLP 측정이다.
- profiler-free console 측정이므로 이번 run에는 별도 raw NoC/Tracy artifact가 없다.
- projection별 W1/W3/W2 개선폭과 실제 outstanding request 수는 아직 직접 계측하지 않았다.
- full 28-layer decode의 tok/s 효과는 아직 검증하지 않았다.
- 기본 동작은 바뀌지 않았으며 tagged 경로는 계속 opt-in이다.

다음 우선순위는 tagged 경로의 projection별 readiness profile을 한 번만 비교해 weight-late 비율이
실제로 줄었는지 확인하는 것이다. 그 뒤 full-model A/B를 해야 약 2.28% isolated MLP 개선이 전체
decode에서 얼마나 보존되는지 판단할 수 있다.

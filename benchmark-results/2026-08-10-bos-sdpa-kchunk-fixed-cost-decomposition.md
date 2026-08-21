# BOS SDPA K128/K256 chunk 고정비 분해

날짜: 2026-08-10

## 결론

64K isolated paged SDPA에서 K chunk를 128에서 256 tokens로 키워 얻은 기존 kernel 단축
151.937 us 중 150.176 us, 98.84%를 compute phase 계측으로 설명했다.

K256은 online-softmax merge를 255회에서 127회로 줄여 419.653 us를 절약했고, 현재 chunk의
max/sub/exp/sum 처리에서 219.566 us를 더 절약했다. 반대로 큰 chunk의 QK와 PV 구간은 합계
489.043 us 길어졌다. 순효과는 -150.176 us다. 따라서 개선의 핵심은 DRAM payload 감소가 아니라
chunk별 online-softmax 작업 감소다. 큰 matmul 구간의 비용 증가가 이득 대부분을 상쇄하여 K256부터
plateau가 생긴다.

## 하드웨어와 경로

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 cores
- SDPA active cores: 16; 5x4 program grid와 구분
- DRAM: 3 physical banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- KV layout: interleaved paged KV cache
- sequence/current position: 65,536 tokens
- Q heads / KV heads / head dimension: 24 / 8 / 128
- page block: 128 tokens
- K/V dtype: BFLOAT8_B
- profiler clock: 650 MHz
- six-endpoint, dual-NoC, tagged, inner-K streaming, helper opt-in: 모두 off

## 측정법

`TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE` opt-in을 추가했다.

| mode | SumN1 | SumN2 |
|---|---|---|
| `matmul` | `SDPA_PHASE_QK` | `SDPA_PHASE_PV` |
| `softmax` | `SDPA_PHASE_CURRENT_SOFTMAX` | `SDPA_PHASE_ONLINE_MERGE` |
| `empty` | `SDPA_PHASE_EMPTY` | 없음 |

zone은 매 chunk에서 timer event를 내지 않는다. `DeviceZoneScopedSumN1/N2`가 kernel 내부에서
누적한 뒤 종료 시 core/RISC당 한 번만 기록한다. 따라서 profiler buffer overflow 없이 64K 전체
누적 시간을 얻는다.

phase 경계는 다음과 같다.

- QK: `Q @ K`의 `cb_matmul_blocks`
- current softmax: mask, max reduction, subtract/exp, sum reduction
- PV: normalized score와 V의 `cb_matmul_blocks`
- online merge: 이전 chunk의 max/sum/output 재보정과 현재 결과 누적
- empty: chunk loop 안의 빈 SumN scope. zone 자체의 누적 비용 추정용

각 phase에서 core별 TRISC_0/1/2 누적값 중 최댓값을 cooperative wall-time proxy로 선택하고,
16 active cores의 평균을 냈다. 빈 zone 비용은 동일 chunk 수에 맞춰 뺐다. online merge만 첫 chunk에
없으므로 empty 값에 `(chunks - 1) / chunks`를 곱해 뺐다.

## 정적 작업량

| 항목 | K128 | K256 |
|---|---:|---:|
| chunks/core | 256 | 128 |
| QK calls/core | 256 | 128 |
| current-softmax calls/core | 256 | 128 |
| PV calls/core | 256 | 128 |
| online merges/core | 255 | 127 |
| K+V payload | 136 MiB | 136 MiB |

총 QK/PV 산술량과 K/V payload는 같다. 호출 shape와 호출 수만 변한다.

## 결과

### Correctness와 completion

| 구성 | PCC | max abs | completion |
|---|---:|---:|---|
| K128 | 0.9998791595 | 0.04526168 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |
| K256 | 0.9999178293 | 0.02732441 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |

profiler-free gate 6개와 완료 capture 6개가 correctness, explicit close, exit 0을 기록했다.

### Raw cooperative phase 누적 시간

| phase | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| QK | 555.121 us | 919.806 us | +364.685 us |
| PV | 621.781 us | 741.537 us | +119.756 us |
| current softmax | 1,018.435 us | 796.568 us | -221.867 us |
| online merge | 848.473 us | 426.520 us | -421.953 us |
| empty SumN | 4.199 us | 1.898 us | -2.301 us |

empty 값은 256회/128회 scope 전체 누적이다. 한 scope당 약 16.4 ns/14.8 ns다. 기존 151.937 us
개선에서 profiler scope 수 차이가 차지할 수 있는 값은 약 2.301 us, 1.51%다.

### Empty-overhead 보정

| phase | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| QK | 550.922 us | 917.908 us | +366.986 us |
| PV | 617.582 us | 739.639 us | +122.057 us |
| current softmax | 1,014.236 us | 794.670 us | -219.566 us |
| online merge | 844.290 us | 424.637 us | -419.653 us |
| 합계 | 3,027.030 us | 2,876.854 us | **-150.176 us** |

기존 독립 capture의 전체 device kernel은 3.482728 ms에서 3.330791 ms로 151.937 us 줄었다.
phase 합의 150.176 us 감소와 차이는 1.761 us다.

### 절약과 상쇄

128개 감소 chunk 기준으로 보면 다음과 같다.

| 성분 | eliminated chunk당 순효과 |
|---|---:|
| online merge | -3.278 us |
| current softmax | -1.715 us |
| QK 증가 | +2.867 us |
| PV 증가 | +0.954 us |
| 합계 | -1.173 us |

기존 전체-kernel 관측값은 eliminated chunk당 -1.187 us다.

## 판정

### 관측 사실

1. K128과 K256의 K/V payload는 모두 136 MiB다.
2. K256의 online merge 누적 시간은 49.72% 감소했다.
3. current-softmax 누적 시간은 21.56% 감소했다. chunk 수는 절반이지만 chunk당 token 연산이 커서
   정확히 절반이 아니다.
4. QK/PV 누적 구간은 합계 489.043 us 증가했다.
5. 네 phase의 순감소 150.176 us가 전체 kernel 감소 151.937 us의 98.84%를 설명한다.
6. 빈 SumN zone의 K128/K256 차이는 2.301 us로 작다.

### 강한 추론

K256 sweet spot의 직접 원인은 chunk별 online-softmax merge와 관련 호출 고정비 감소다. DRAM
traffic 감소가 아니다. K256의 큰 QK/PV shape는 총 산술량이 같아도 현재 LLK block schedule에서 더
비싸다. softmax 절약 639.219 us 중 matmul 증가 489.043 us가 상쇄되고 150.176 us만 남는다.

K512가 K256보다 6.542 us만 빠른 기존 결과도 이 구조와 맞는다. merge 수 추가 감소가 더 큰 CB wait와
matmul/shape 비용에 거의 상쇄된다.

### 한계

- 각 mode/K 조합은 measured capture 1회다.
- phase는 다른 capture에서 각각 측정했다. 합은 동시 단일-run waterfall이 아니다.
- cooperative wall-time proxy는 세 TRISC 중 최장 누적값이다. 세 pipeline의 세부 overlap은 분리하지 않는다.
- QK/PV zone 안에는 `cb_matmul_blocks` 내부 wait가 포함될 수 있다. 순수 math-engine cycle만 뜻하지 않는다.
- current-softmax는 token 비례 산술과 호출 고정비를 함께 포함한다. 순수 고정항을 완전히 분리하려면
  K64/128/256 이상의 회귀가 추가로 필요하다.
- isolated SDPA 결과다. full-model tokens/s 개선과 같지 않다.

## 중단 사건

첫 softmax/K128 capture는 다른 사용자의 NIAH process PID 651935가 device lock을 점유해 device open
전에 대기했다. 외부 timeout으로 exit 124가 발생했고 완성 artifact는 host Tracy 파일뿐이다. SDPA
kernel 실패로 분류하지 않았다. 안전 규칙에 따라 장치를 격리했고, 사용자 재부팅 확인 뒤 첫 workload인
32x32 BF16 add가 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과한 뒤 capture를 새 디렉터리에서
재개했다.

## 재현

```bash
env \
  TT_METAL_DEVICE_PROFILER=1 \
  TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=softmax \
  SDPA_SEQ_LEN=65536 \
  SDPA_K_CHUNK_SIZE=256 \
timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy \
  -p -r --check-exit-code --sync-host-device \
  -o <capture-dir> -n sdpa_chunk_phase_softmax_k256 \
  tests/bos_model/run_sdpa_kchunk_profile.py
```

`mode`와 `SDPA_K_CHUNK_SIZE`를 `matmul|softmax|empty`, `128|256`으로 바꿔 여섯 capture를 만든다.

## Artifact


## 2026-08-21 follow-up: 6-endpoint phase decomposition

### 목적과 판정

과거와 같은 production single-layer runner, identity page table, accuracy mode 및 device-profiler 최대-core span 집계로 다시 측정했다. 현재 source는 K128 `2.52003 ms`, K256 `2.04132 ms`, 즉 `-478.715 us (-19.00%)`를 기록했다. 과거 K128 `2.51933 ms`, K256 `2.03641 ms`, `-482.92 us (-19.17%)`와 개선율 차이는 약 `0.17 percentage point`다. 따라서 과거 `-19.17%`는 현재 production 경로에서도 사실상 재현됐다.

처음 사용한 randomized-page isolated unit-test는 K128 `2.440189 ms`, K256 `2.147965 ms`, `-11.98%`였지만 production 재현과 workload가 다르다. 아래 phase attribution은 이 randomized-page 보조 실험 자체의 감소 원리를 설명하며, production `-19.00%`를 정량 분해한 결과로 사용하지 않는다.

### 실행 조건

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4; SDPA active compute cores: 16
- context/current position: 65,536 tokens
- Q/KV heads, head dimension: 24/8, 128
- phase subexperiment K/V: BFLOAT8_B, interleaved paged cache, 32-token page, randomized page table
- reader: dual-NoC, 6 worker endpoints
- pair/bank balance, route-overlap, tagged-async, six-reader-sharded, reduce-only helper, inner-K streaming: off
- phase modes: `matmul`, `softmax`, `empty`
- profiler clock: 650 MHz

재부팅 뒤 첫 workload인 32x32 BF16 add는 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과했다.
여섯 profiler-free compile/correctness gates와 여섯 device-profiler captures도 모두 correctness, explicit
close, exit 0으로 완료됐다. 각 capture에 device CSV와 host Tracy trace가 존재한다.

### Randomized-page isolated unit-test latency

이 표는 `torch.randperm` page table을 쓰는 isolated unit-test 결과다. 각 sample은 warmup 뒤 SDPA 10회를 평균한 값이며 K별 5 samples다.

| K chunk | n | mean ± sample SD | median | K128 대비 latency | inverse-latency gain |
|---:|---:|---:|---:|---:|---:|
| 128 | 5 | 2.440189 ± 0.004005 ms | 2.438440 ms | baseline | baseline |
| 256 | 5 | 2.147965 ± 0.001790 ms | 2.147722 ms | -292.224 us (-11.98%) | +13.60% |

이 표를 production historical result와 직접 비교하지 않는다. Production runner는 identity page table과 full-layer allocation을 사용하며, 아래 동종 profiler 재현에서 `-19.00%`를 기록했다.

### Phase 집계법

기존 vanilla 보고서와 같이 각 core에서 TRISC_0/1/2의 phase 누적값 중 최댓값을 cooperative wall-time
proxy로 사용했다. Empty-zone 누적 비용을 K별로 빼고, online merge에는 `(chunks-1)/chunks`를 적용했다.
6-endpoint K256은 core 편차가 크므로 16-core 평균 phase 합을 whole-device critical path와 비교하지 않고,
K128/K256 모두 합이 최대인 logical core `(0,2)`를 critical-core attribution으로 사용했다.

| phase, logical core (0,2) | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| QK | 317.120 us | 594.786 us | +277.666 us |
| PV | 292.366 us | 390.262 us | +97.896 us |
| current softmax | 1,018.517 us | 795.660 us | -222.857 us |
| online merge | 847.007 us | 425.576 us | -421.431 us |
| **phase sum** | **2,475.010 us** | **2,206.284 us** | **-268.726 us** |

Gross softmax/merge saving은 `644.288 us`, QK/PV cost 증가는 `375.562 us`다. 순감소 `268.726 us`를
marker-free 감소 `292.224 us`와 비교하면 `91.96%`이며, zone 밖 및 cross-capture 잔차는 `23.498 us`다.
Phase 절대합이 marker-free kernel 절대시간과 정확히 같아야 하는 것은 아니다. phase는 서로 다른 single
captures를 합성했고 timestamp marker를 포함한다. 상대 차이와 기여 방향을 주된 결과로 사용한다.

참고로 16-core 평균 phase 합은 K128 `2,468.317 us`, K256 `2,068.436 us`로 `-399.881 us`였지만,
K256 core별 합 범위가 `1,790.793--2,206.283 us`로 넓다. 평균은 critical-path kernel latency의
분모가 아니므로 발표용 전체 감소 설명에는 사용하지 않는다.

### Production runner 재현

과거와 같은 production single-layer runner는 identity page table(`torch.arange`)과 full-layer allocation 상태를 사용한다. Warmup 1회와 measured 1회를 device profiler로 캡처한 뒤 각 SDPA 호출의 최대-core kernel span을 평균했다.

| 측정 | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| 2026-07-31 historical | 2.51933 ms | 2.03641 ms | -482.92 us (-19.17%) |
| 2026-08-21 current-source reproduction | 2.52003 ms | 2.04132 ms | -478.715 us (-19.00%) |

Profiler-free production runner의 35-call 평균 wall time도 K128 `2.737894 ms`, K256 `2.231074 ms`로 `-506.820 us (-18.51%)`였다. 따라서 chunk 효과는 production workload에서 재현된다. Randomized-page unit-test의 `-11.98%`는 page-table 순서, allocation 상태 및 runner가 다른 별도 결과다.

현재 phase attribution은 chunk-boundary 감소가 softmax/merge를 줄이고 큰 QK/PV가 일부 상쇄한다는 작동 원리를 확인한다. 다만 randomized-page unit-test에서 캡처했으므로 phase 수치 `-268.726 us`를 production `-478.715 us`의 직접 분해로 제시하지 않는다.

### Production memory breakdown

Reader profile은 각 active core에 `SDPA_KV_SERVICE_START/END`, `SDPA_K_READ_BARRIER`, `SDPA_V_READ_BARRIER`를 기록한다. Service span은 page-table translation, request issue, read barrier, CB reserve/publish 및 consumer backpressure를 포함한다. `service - K barrier - V barrier`는 순수 enqueue가 아니라 주소 계산, issue loop와 CB/backpressure를 포함한 non-barrier remainder다.

각 값은 warmup과 measured SDPA 호출에서 service span이 가장 긴 core를 고른 뒤 두 호출을 평균했다.

| critical-reader metric | K128 | K256 | delta |
|---|---:|---:|---:|
| max-core kernel span | 2,519.952 us | 2,035.054 us | -484.898 us (-19.24%) |
| KV service span | 2,484.052 us | 1,994.188 us | -489.864 us (-19.72%) |
| K read barrier sum | 327.622 us | 578.742 us | +251.120 us |
| V read barrier sum | 251.245 us | 585.042 us | +333.798 us |
| K+V barrier sum | 578.867 us | 1,163.785 us | +584.918 us |
| non-barrier remainder | 1,905.185 us | 830.404 us | -1,074.782 us (-56.41%) |
| aggregate effective K/V rate over service span | 57.41 GB/s | 71.51 GB/s | +24.56% |

K256은 같은 K/V payload를 읽는다. BFP8 encoded K와 V는 각각 68 MiB, 합계 136 MiB다. 각 core는 8.5 MiB를 담당하며 16 cores 합계가 136 MiB다. Identity page table은 8 KiB를 16 readers가 한 번씩 읽어 aggregate 128 KiB이고 output은 약 8 KiB다. 따라서 logical DRAM bytes는 K/V가 99.9% 이상을 차지한다. 이는 physical bus utilization이나 protocol byte accounting이 아니라 kernel이 요청한 logical payload breakdown이다.

핵심 관측은 K256의 barrier 누적이 줄지 않고 오히려 약 585 us 증가했다는 점이다. 이는 K256 barrier가 더 큰 outstanding set을 retire하는 구현과 일치하지만, 이번 marker는 barrier 호출 횟수를 기록하지 않으므로 개별 barrier latency 증가는 직접 측정값이 아닌 추론이다. 반면 chunk 수 감소로 반복되는 address/issue loop, CB reserve/publish 및 consumer backpressure를 포함한 remainder가 약 1,075 us 줄어 barrier 증가를 상쇄한다. 따라서 19% 개선을 DRAM 자체가 빨라졌다고 설명하지 않는다. 같은 136 MiB를 더 적은 chunk boundary와 더 나은 producer-consumer cadence로 전달하여 exposed non-barrier cost를 줄인 결과다.

계측 marker는 kernel 시간을 K128/K256 각각 2,519.952/2,035.054 us로 유지해 비계측 production profile의 2,520.030/2,041.315 us와 가깝다. K256 차이는 약 6.3 us이므로 방향과 breakdown은 낮은 침습도로 판단한다.

Artifact root: `/home/iris_hb4/profiler_runs/sdpa_kchunk_production_memory_breakdown_2026_08_21`

- K128 device CSV SHA-256: `d7b1a9d814a1294279e3c7a7d5c94700b1c60f190ddc3bc31f6cd34e0d83a548`
- K256 device CSV SHA-256: `9939123acf54d79374d96b0397f4dc1241140ebde99dfb0bf0c38ac0214d6fb8`

### 재현 명령

Randomized-page isolated A/B는 `TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE`과 `TT_METAL_DEVICE_PROFILER`를 unset하고 다음 공통 reader 조건에서 `SDPA_K_CHUNK_SIZE=128|256`만 바꾼다. Production 재현은 아래 명령의 unit-test script 대신 `models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py --precision-mode accuracy --context-len 65536 --kv-block-size 32 --kv-layout paged --warmup 1 --iterations 1 --repeats 1 --sdpa-k-chunk-size 128|256`을 사용한다.

```bash
env \
  TT_METAL_SDPA_DECODE_DUAL_NOC=1 \
  TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 \
  TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=0 \
  TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=0 \
  TT_METAL_SDPA_DECODE_ROUTE_OVERLAP_OPTIMIZED=0 \
  TT_METAL_SDPA_DECODE_TAGGED_ASYNC=0 \
  TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0 \
  TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0 \
  SDPA_SEQ_LEN=65536 SDPA_K_CHUNK_SIZE=128 SDPA_BLOCK_SIZE=32 \
  SDPA_LATENCY_ITERATIONS=10 SDPA_LATENCY_REPEATS=5 \
  timeout --signal=INT --kill-after=15s 180s \
  python_env/bin/python tests/bos_model/run_sdpa_kchunk_profile.py
```

Phase capture는 공통 조건에 `TT_METAL_DEVICE_PROFILER=1`과
`TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=matmul|softmax|empty`를 추가하고 K128/K256을 각각 실행한다.

### Follow-up artifact

Roots:

- randomized-page phase: `/home/iris_hb4/profiler_runs/sdpa_chunk_phase_6ep_2026_08_21`
- production reproduction: `/home/iris_hb4/profiler_runs/sdpa_kchunk_production_repro_2026_08_21`

대표 checksum:

| artifact | SHA-256 |
|---|---|
| `gates/production_k128_5run.log` | `c582384acb5b4df12c4e54526aeb23f1409a283eb2f9a3b5702e8b8c499a59be` |
| `gates/production_k256_5run.log` | `56e44f8a4fbbed8fd9278d22125a9866d03b05d511d12dc46018a1a8dfc49ee6` |
| `matmul_k128/profile_log_device.csv` | `4ae5527738dd7cc6223d23b9aa2f1d24d464bfa33629ff15192805f20440de44` |
| `matmul_k256/profile_log_device.csv` | `5f16ed7fcf01be15d714cbc3c76df91c527b13d8cc1db490928b057a69d040a8` |
| `softmax_k128/profile_log_device.csv` | `be6de5a1a49c44db1729b9365208997984e8baa5489c6a9c73d26dc0e3c870c8` |
| `softmax_k256/profile_log_device.csv` | `d7aca0ef97ffea4e0339baf3777f3e261bf2b8a2a41ac83a8db6924df15bc1e9` |
| `empty_k128/profile_log_device.csv` | `138751ccbbd97bea56337a073991708414ed3b66e5aec45adb6b6b058eeb9f4c` |
| `empty_k256/profile_log_device.csv` | `ca3bd4e5db9da23aa0e67214b9033665069340fb240de84393a293427c586a0f` |

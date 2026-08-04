# BOS MLP compute block cadence

날짜: 2026-08-04 UTC

## 결론

실제 MLP matmul은 data-only microbenchmark와 달리 compute가 포함된다. Balanced fanout-2의 measured
W1/W3/W2에서 compute consumer의 두 input CB wait 합은 kernel의 66.75--67.35%다. 16개 K block
사이의 matrix-engine service보다 다음 input을 기다리는 구간이 길다. 현재 병목은 TOPS 한계가 아니라
producer/consumer cadence다.

단, `in0`과 `in1` wait를 독립 원인 비율로 해석하면 안 된다. Kernel이 `in0`을 먼저 기다리고 `in1`을
나중에 기다린다. 두 input의 동시 지연 중 앞 wait가 먼저 흡수된다. Wait-order A/B 결과 W1은
weight-late가 일관됐고, W3는 균형, W2는 순서에 따라 late side가 바뀌었다. 전체 MLP를
activation-only 또는 weight-only 병목으로 분류할 수 없다.

## 장치와 구성

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20
- operation: 20 program cores, 12 readers, 12 compute workers
- physical DRAM: 3 banks, 2 worker endpoints/bank, 총 6 endpoints
- runtime-selected DRAM-interface workers: 6
- endpoint groups: NOC1 4:4:4
- weight path: DRAM-sharded, fanout-2, balanced endpoints, tagged two-block
- W2 `in0_block_w`: 16
- 16 KiB read-page cap: enabled; observed pages 13,056 B and 8,704 B
- prefetch helper/fanout-3/TurboQuant: disabled

## Marker

Consumer kernel:
`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

| Marker/zone | 의미 |
|---|---|
| `MLP_IN0_CB_WAIT` | activation CB wait 누적 |
| `MLP_IN1_CB_WAIT` | weight CB wait 누적 |
| `MLP_MATMUL_INPUTS_READY` | 두 CB wait 완료 |
| `MLP_MATMUL_BLOCK_RELEASED` | 해당 K block 계산 후 두 input CB pop 완료 |

추가 marker patch:
`/home/iris_hb4/tmp/codex-patches/20260804-mlp-compute-block-release-marker.patch`
(SHA-256 `7bb7678053fd4350d32c98b26496c9e76729f646cac5fa09145ef573f18194a2`)

## 결과

Measured call의 12 compute cores, projection당 16 K blocks를 집계했다. 시간은 650 MHz device cycle을
사용했다.

| projection | kernel | in0 wait/core mean | in1 wait/core mean | wait 합 | wait/kernel |
|---|---:|---:|---:|---:|---:|
| W1 | 431.935 us | 111.018 us | 177.293 us | 288.311 us | 66.75% |
| W3 | 441.600 us | 185.634 us | 109.994 us | 295.628 us | 66.94% |
| W2 | 419.628 us | 161.213 us | 121.398 us | 282.611 us | 67.35% |

Timestamp interval의 block-level 평균:

| projection | inputs-ready→block-released | released→next-inputs-ready | latter 비율 |
|---|---:|---:|---:|
| W1 | 7.772 us | 17.278 us | 68.98% |
| W3 | 7.770 us | 17.595 us | 69.37% |
| W2 | 7.486 us | 17.016 us | 69.44% |

`released→next-inputs-ready`는 CB wait 외 loop/marker 고정비도 포함한다. 원인 비율에는 accumulated
CB zones를 우선 사용한다. Marker 추가 뒤 kernel은 이전 tagged profile 대비 W1/W3에서 각각
4.060/5.106 us 느리고 W2는 0.518 us 빠르다. 계측 교란은 약 1.2% 이하다.

## 판정

관측:

1. 세 projection 모두 약 67%가 consumer input wait다.
2. Matrix block service는 약 7.5--7.8 us, 다음-input 구간은 약 17.0--17.6 us다.
3. Projection별 `in0`/`in1` 비중이 뒤집힌다. 한쪽만 항상 늦는 구조가 아니다.
4. 12-compute를 줄이는 근거가 없다. 현재 engine은 input starvation 상태다.

Fix:

1. 12-compute balanced fanout-2 유지.
2. W1 weight reader cadence를 우선 최적화.
3. W3/W2는 core/block별 late side를 사용해 producer를 선택적으로 조정.

## Wait-order A/B

같은 binary/config에서 baseline `in0→in1`과 opt-in `in1→in0`을 비교했다. 누적 wait 합은 거의
유지됐다. 순서 변경은 병목을 없애지 않고 어느 zone이 겹친 wait를 흡수하는지만 바꿨다.

| projection | in0-first: in0 + in1 | in1-first: in1 + in0 | total 변화 | kernel 변화 |
|---|---:|---:|---:|---:|
| W1 | 111.018 + 177.293 us | 234.377 + 54.444 us | +0.18% | -0.30% |
| W3 | 185.634 + 109.994 us | 178.828 + 114.998 us | -0.61% | -0.58% |
| W2 | 161.213 + 121.398 us | 162.692 + 125.992 us | +2.15% | +1.32% |

첫 항목은 먼저 기다린 CB, 둘째 항목은 뒤 CB의 residual wait다. W1은 어느 순서에서도 weight가
critical side다. W3와 W2는 양 producer의 cadence가 비슷하며 core/block별로 늦은 쪽이 달라진다.

Producer ready timestamp도 같은 판정을 지지한다.

| projection | baseline weight-late | in1-first weight-late | signed in1-in0 gap |
|---|---:|---:|---:|
| W1 | 132/192 (68.8%) | 140/192 (72.9%) | +12.009/+11.821 us |
| W3 | 90/192 (46.9%) | 96/192 (50.0%) | +0.998/+0.250 us |
| W2 | 110/192 (57.3%) | 80/192 (41.7%) | +3.147/-2.096 us |

W1만 order-independent weight-late다. W3는 균형이다. W2는 wait 순서가 producer backpressure를
바꾸면 late side도 바뀐다. Global weight-only prefetch는 W3/W2의 activation-late block을 못 고친다.

Opt-in: `TT_METAL_MLP_COMPUTE_WAIT_IN1_FIRST=1`
Factory patch SHA-256: `f95e327070855dc0ffb23872f38febd60e70e271dddaffb4d4eccf54736f1367`
Compute patch SHA-256: `d8501af1612351da082fd14c585b993ec151c62bdd1a75dc7eb49f64b096ed45`

Profile: `/home/iris_hb4/profiler_runs/mlp_compute_wait_in1_first_2026_08_04_04_52_00`
Device CSV SHA-256: `987a9be37d9d49f1d98d2c8ebbcbda521264bf40deb039da6c63308e9ae6264a`
Ops CSV SHA-256: `ead4549e9ed749bc248ff14c5332d53d24561b199f962b48d0c7dfc086bc8708`
Exit 0; PCC `0.9996410623374821`; `MLP_COMPLETED`; `DEVICE_CLOSED`.

## K-block merge-2 A/B

Producer/consumer 호출 경계 자체가 병목인지 확인하려고 opt-in
`TT_METAL_MLP_MERGE_K_BLOCKS2=1`을 추가했다. Activation storage producer는 기존 16개 block과
원래 width(W1/W3 6, W2 16)를 유지한다. Weight reader와 compute consumer만 인접한 두 block을
합쳐 8개 block, width 12/32로 처리한다. Non-compute activation sender slot은 overwrite 방지를 위해
2개에서 4개로 늘렸다.

Marker count가 구현을 확인했다.

| projection당 marker | baseline | merge-2 |
|---|---:|---:|
| `MLP_IN0_READY` | 192 | 192 |
| `MLP_IN1_READY` | 192 | 96 |
| `MLP_MATMUL_INPUTS_READY` | 192 | 96 |
| `MLP_MATMUL_BLOCK_RELEASED` | 192 | 96 |

12 cores 기준이다. Activation publish는 16회/core, weight publish와 consumer 경계는 8회/core다.

Profiler-free 20회 A/B 결과:

| 구성 | PCC | mean | median | min |
|---|---:|---:|---:|---:|
| baseline 16 blocks | 0.9996410623 | 1.436040 ms | 1.434233 ms | 1.423756 ms |
| merge-2 8 blocks | 0.9995545202 | 1.455052 ms | 1.452613 ms | 1.446722 ms |
| 변화 | - | +1.324% | +1.282% | +1.613% |

Merge-2가 느리다. PCC 차이는 accumulation grouping/order 변경과 일치한다. 두 결과 모두 정상 완료와
device close를 확인했다.

Measured cadence 비교:

| projection | baseline wait | merge-2 wait | wait 변화 | baseline kernel | merge-2 kernel | kernel 변화 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 288.311 us | 293.334 us | +1.74% | 431.935 us | 436.466 us | +1.05% |
| W3 | 295.628 us | 305.079 us | +3.20% | 441.600 us | 445.463 us | +0.87% |
| W2 | 282.611 us | 303.187 us | +7.28% | 419.628 us | 430.005 us | +2.47% |

| projection | merge-2 inputs-ready→released | merge-2 released→next-ready |
|---|---:|---:|
| W1 | 14.863 us | 33.007 us |
| W3 | 14.853 us | 34.078 us |
| W2 | 14.167 us | 33.176 us |

Block당 service와 next-ready gap이 대략 2배가 됐다. 경계 수가 절반이라 총 시간은 비슷하지만 약간
악화됐다. 따라서 `cb_wait`/`cb_pop_front` 호출 instruction overhead가 주 병목이 아니다. 큰 weight
block 전체가 준비돼야 publish할 수 있어 visibility가 늦어지고, consumer scheduling 기회도 줄어든다.
현재 16-block granularity를 유지한다. 다음 후보는 경계를 합치는 것이 아니라 다음 weight block을 더
일찍 pending 상태로 만들고 core/block별 phase를 교정하는 것이다.

Profile:
`/home/iris_hb4/profiler_runs/mlp_merge_k_blocks2_2026_08_04_05_09_20`

- device CSV SHA-256: `9547f51ebaa3e9c32081c58ccb6ced1968eb1d1ddc2f1df726dfe08acccb437c`
- ops CSV SHA-256: `74301588f205d6ee5db8ebc3db6e2f47ab3d6e118c4aee26bcda618de573fa09`
- exit 0; PCC `0.9995545201653806`; `MLP_COMPLETED`; `DEVICE_CLOSED`

## Tagged pending depth-3와 phase

16개 K-block 경계는 유지하고 weight reader만 current block과 future 2개를 동시에 pending으로 만드는
opt-in `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH3=1`을 구현했다. 기존 depth-2는 future 1개다.
Depth-3는 3개 TRID와 기존 triple-buffer 3 slot을 사용하고 마지막 2개 request를 순서대로 drain한다.

Profiler-free 20회 A/B:

| 구성 | PCC | mean | median | min |
|---|---:|---:|---:|---:|
| tagged depth-2 | 0.9996410623 | 1.436188 ms | 1.433867 ms | 1.424090 ms |
| tagged depth-3 | 0.9996410623 | 1.439408 ms | 1.435411 ms | 1.427247 ms |
| 변화 | - | +0.224% | +0.108% | +0.222% |

Depth-3는 correctness와 정상 close를 통과했지만 성능 이득이 없다.

Corrected marker profile의 consumer wait:

| projection | depth-2 in0/in1/합 | depth-3 in0/in1/합 | 합 변화 | depth-3 kernel |
|---|---:|---:|---:|---:|
| W1 | 111.018/177.293/288.311 us | 149.753/144.173/293.926 us | +1.95% | 433.086 us |
| W3 | 185.634/109.994/295.628 us | 189.802/110.076/299.878 us | +1.44% | 437.032 us |
| W2 | 161.213/121.398/282.611 us | 211.933/84.912/296.845 us | +5.04% | 426.098 us |

Weight BRISC barrier는 W1/W3/W2에서 292.969/297.624/225.679 us에서
115.520/104.625/48.153 us로 크게 줄었다. 그러나 제거된 weight wait가 activation wait로 이동했다.
즉 request가 늦게 issue되는 문제는 실제였지만 global depth 증가만으로 end-to-end critical path를
줄이지 못한다.

Block별 12-core weight-ready timestamp spread 평균은 W1/W3/W2
77.193/82.094/75.989 us에서 58.489/60.133/58.775 us로 줄었다. 그래도 physical core별 phase는
남았다. Depth-2에서 weight-minus-activation 평균이 큰 core는 `(0,2)` +44.1 us, `(1,1)` +33.1 us,
`(0,0)` +32.6 us다. Reader map상 각각 runtime DRAM view 5 lane0 VC1, view 2 lane1 VC2,
view 0 lane1 VC0다. 이는 worker endpoint/view 번호이며 3개 physical DRAM bank 번호가 아니다.

같은 DRAM view의 두 lane에 `(view + lane) % 4` VC를 강제한
`TT_METAL_MLP_DRAM_SHARDED_FANOUT2_DISTINCT_VC=1`도 검사했다. PCC는 유지됐지만 20회 latency는
mean 1.474413 ms, median 1.474434 ms, min 1.444550 ms였다. Baseline보다 각각
2.662/2.829/1.437% 느리다. 따라서 단순 VC 분리도 사용하지 않는다.

판정:

1. Depth-3와 distinct-VC는 opt-in 기본 비활성으로 유지한다.
2. Global weight prefetch 깊이 부족이나 CB 호출 횟수는 최종 병목이 아니다.
3. 다음 수정은 activation multicast와 weight read를 함께 맞추는 per-core phase scheduling이어야 한다.
4. `(0,2)/(1,1)/(0,0)` late route를 우선 보되, 단순 지연 삽입은 critical path를 줄이지 못하므로 금지한다.

Corrected profile:
`/home/iris_hb4/profiler_runs/mlp_tagged_depth3_phase_2026_08_04_05_23_00`

- device CSV SHA-256: `84a92ca90094dcf1b186025930775bed9efe75b420462590e1ed6438699e20d4`
- ops CSV SHA-256: `9396bb1842f484717c5b2c97d0752fba5b51a716472adad9aad4bd85504206c8`
- exit 0; PCC `0.9996410623374821`; `MLP_COMPLETED`; `DEVICE_CLOSED`

초기 depth-3 profile은 publish marker를 `N-1`로 기록했지만 실제 publish는 `N-2`였다. Marker를
`N-pending_depth+1`로 교정하고 profiler-free gate를 다시 통과한 뒤 위 corrected profile을 얻었다.
초기 artifact는 block phase 근거로 사용하지 않는다.

## Dependency-chain 계측

Balanced fanout-2 tagged depth-2에 activation multicast와 weight request marker를 추가했다.
`(0,2)/(1,1)/(0,0)`은 runtime DRAM core가 아니라 profile CSV의 physical worker 좌표다. 세 코어 모두
이 matmul에서 reader와 compute consumer가 활성이다.

| 경로 | 시작→끝 marker | 포함 범위 |
|---|---|---|
| activation | `MLP_IN0_PHASE_START`→`MLP_IN0_MCAST_ARRIVED` | sender semaphore rendezvous와 multicast 도착 |
| activation publish | `MLP_IN0_MCAST_ARRIVED`→`MLP_IN0_READY` | CB push/publication |
| weight issue | `MLP_IN1_ISSUE_START`→`MLP_IN1_ISSUED` | block row read request enqueue |
| weight pending | `MLP_IN1_ISSUED`→`MLP_IN1_DRAM_DONE` | tagged request issue부터 해당 TRID barrier 완료까지 |
| weight publish | `MLP_IN1_DRAM_DONE`→`MLP_IN1_READY` | CB push/publication |
| compute | `MLP_MATMUL_INPUTS_READY`→`MLP_MATMUL_BLOCK_RELEASED` | matrix-engine block service와 CB release |

Measured W1/W3/W2 각각 12 cores × 16 blocks에서 주요 marker가 192개씩 존재했다. Activation
`MCAST_ISSUED`는 sender의 자기 block에서만 발생하므로 projection당 8개다. Marker 누락은 없다.

세 late core의 block 평균은 다음과 같다. 시간 단위는 us, 괄호는 최대값이다.

| projection/core | activation phase→arrived | weight issue enqueue | weight issued→done | compute service | released→next-ready |
|---|---:|---:|---:|---:|---:|
| W1 `(0,2)` | 0.967 (1.998) | 0.563 (0.666) | 50.400 (79.938) | 7.777 (9.211) | 17.486 (32.289) |
| W1 `(1,1)` | 11.526 (33.931) | 0.563 (0.666) | 48.777 (79.502) | 7.773 (9.205) | 17.408 (35.709) |
| W1 `(0,0)` | 9.537 (19.137) | 1.780 (13.723) | 48.509 (77.523) | 7.778 (9.209) | 17.286 (31.789) |
| W3 `(0,2)` | 0.839 (0.992) | 0.561 (0.618) | 51.897 (78.848) | 7.778 (9.212) | 16.170 (36.663) |
| W3 `(1,1)` | 20.345 (40.508) | 0.558 (0.615) | 50.631 (74.934) | 7.771 (9.205) | 17.583 (26.538) |
| W3 `(0,0)` | 18.834 (38.394) | 3.409 (15.040) | 48.013 (58.554) | 7.768 (9.212) | 17.864 (30.788) |
| W2 `(0,2)` | 1.161 (1.349) | 0.720 (0.763) | 49.560 (82.974) | 7.484 (7.709) | 15.598 (31.274) |
| W2 `(1,1)` | 21.205 (38.635) | 0.730 (0.825) | 48.570 (68.880) | 7.473 (7.560) | 16.388 (24.969) |
| W2 `(0,0)` | 23.390 (51.502) | 12.534 (34.786) | 36.157 (49.957) | 7.477 (7.543) | 18.395 (36.294) |

Activation/weight CB publication 자체는 0.075--0.112 us다. 따라서 `cb_push_back` 비용은 병목이 아니다.
세 target core에서 weight가 activation보다 늦게 ready된 block은 W1 48/48, W3 48/48, W2 45/48이다.
이 target들의 stall은 helper가 activation을 가공하지 못해서가 아니라 tagged weight request completion
cadence가 주로 결정한다. 특히 `(0,2)`는 activation multicast가 0.8--1.2 us인데 weight pending이
49.6--51.9 us라 명확하다.

전체 12 cores에서는 weight-late가 W1 147/192, W3 98/192, W2 85/192다. W3는 균형이고 W2는
activation-late 107/192라서 세 target 결과를 전체 grid의 weight-only 병목으로 일반화하면 안 된다.
전체-grid activation phase 평균 상위 코어는 W1/W3에서 `(2,4)/(3,4)`, W2에서 `(1,0)/(3,4)`다.
따라서 단일 global helper보다 route/core별 activation과 weight phase 교정이 필요하다.

BRISC(weight), NCRISC(activation), TRISC(compute)의 timestamp를 교차 비교한 ready 순서는 직접 global
timer 기준이다. 다만 RISC 시작 skew와 marker overhead가 섞일 수 있으므로 절대적인 sub-us wake gap보다
동일 RISC 내부의 `phase→arrived`, `issue→done`, `inputs-ready→released` 구간을 강한 근거로 사용한다.

Marker patch:

- activation: `/home/iris_hb4/tmp/codex-patches/20260804-060200-mlp-in0-phase-markers.patch`
  (SHA-256 `0c259f5229b5413fc12706f454fd5162c1383bf1ab3fa3dcaeb6a3152c2c2878`)
- weight: `/home/iris_hb4/tmp/codex-patches/20260804-060100-mlp-in1-request-markers.patch`
  (SHA-256 `3fea82489aee8df9edd858d6af87b4b29f1df6580af9cb37b2a2e44586163610`)

Profiler-free gate는 PCC `0.9996410623374821`, latency `1.499183 ms`, 정상 completion/close로 통과했다.
Profile도 exit 0, PCC `0.9996410623374821`, measured latency `1.527993 ms`, `MLP_COMPLETED`,
`DEVICE_CLOSED`를 확인했다.

Profile: `/home/iris_hb4/profiler_runs/mlp_dependency_chain_2026_08_04_05_43_00`

- device CSV SHA-256: `6ddeb72759e2c9f61a8cf9ca7071110dda8718801767dc2f75cf0b3827d54f7e`
- ops CSV SHA-256: `9bc320cecb47d73becc8d0a717260b399c6ab4ae5eba078176ab616c326a2584`

## Activation sender credit와 depth-3

Sender marker로 receiver-ready rendezvous와 multicast delivery를 분리했다. Sender credit wait는 미래
block을 일찍 준비한 storage sender의 대기도 포함하므로 consumer critical-path에 직접 더하면 안 된다.

| projection | sender credit wait mean | ready→issued | issued→last receiver | arrival spread |
|---|---:|---:|---:|---:|
| W1 | 97.187 us | 0.250 us | 0.478 us | 0.124 us |
| W3 | 105.623 us | 0.244 us | 0.482 us | 0.121 us |
| W2 | 105.800 us | 0.367 us | 0.694 us | 0.126 us |

Multicast delivery는 1 us 미만이다. 긴 wait는 receiver CB capacity와 block phase의 credit 대기다.

| 구성 | PCC | 20회 mean | median | min |
|---|---:|---:|---:|---:|
| credit depth-2 | 0.9996410623 | 1.439223 ms | 1.436129 ms | 1.429691 ms |
| credit depth-3 | 0.9996410623 | 1.466872 ms | 1.469777 ms | 1.431659 ms |
| 변화 | - | +1.921% | +2.343% | +0.138% |

Depth-3는 sender credit wait를 W1/W3/W2 3.45/2.85/1.50% 줄였다. W1/W3의 줄어든 in0 wait는
in1 wait로 이동했고 W2 total wait는 285.752→292.431 us로 늘었다. Critical path 이득이 없다.

Opt-in `TT_METAL_MLP_IN0_CREDIT_DEPTH3=1`은 기본 비활성이다. Tile, block, accumulation 순서는 유지했다.
Patches: `20260804-activation-sender-credit-wait-markers.patch` / `20260804-activation-credit-depth3.patch`.
Profiles: `/home/iris_hb4/profiler_runs/mlp_activation_credit_wait_2026_08_04_06_57_30` / `/home/iris_hb4/profiler_runs/mlp_activation_credit_depth3_2026_08_04_07_02_00`.
Device CSV SHA-256: `9bea929f37a459116bc50168cb00260fe1b6181f4d8afe67df03a22f1eb376fb` / `0a1e0d5a1d950dc93aac9d7d4405a3472eece08a058df9ab74935cafbdc4c715`.
Ops CSV SHA-256: `fb4a78afa2009819dc99b6d419597a03f3f4dee25a8068fffac89ea177176703` / `777443f60c3827b31e252ce6cdf960ea98cccafcdc30bddd101e835d4c6b5af0`.

## 재현

Profiler-free gate가 PCC `0.9996410623374821`, latency `1.515464 ms`, `MLP_COMPLETED`,
`DEVICE_CLOSED`로 먼저 통과했다.

```bash
env PATH=/home/iris_hb4/tt-metal-hb4/python_env/bin:/usr/local/bin:/usr/bin:/bin \
  TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 TT_METAL_TURBOQUANT=0 \
  timeout --signal=INT --kill-after=15s 120s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy -p -r \
  --sync-host-device --check-exit-code \
  -o /home/iris_hb4/profiler_runs/mlp_compute_block_cadence_2026_08_04_04_38_00/perf_capture \
  -n mlp_compute_block_cadence \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## Artifact

Run: `/home/iris_hb4/profiler_runs/mlp_compute_block_cadence_2026_08_04_04_38_00`

- device CSV SHA-256: `497480d6c5ce3a88d9b1099b27547b70f593e6fefb614fa3e83d8e317f2e61cd`
- ops CSV SHA-256: `e89a565c266042a1e94705882ecbc222062592d710065d7a835243a1caf06ec0`
- exit 0; PCC/completion/close/report 모두 확인

한계: isolated layer 0, batch 1, correctness call 1회와 measured call 1회다. NoC NPE capture가 아니며
전체 64K decode latency를 직접 나타내지 않는다.

## Gate/up fused projection A/B

### 변경

`TT_METAL_MLP_FUSED_GATE_UP=1` opt-in을 추가했다. Decode에서 weight를 `[W3 | W1]`로 pack하고
W1/W3 projection을 `3072×16384` matmul 한 번으로 실행한다. `ttnn.swiglu`는
`first_half * silu(second_half)`이므로 이 순서가 기존 `silu(W1x) * W3x`와 맞는다. Dependency가 있는
W2는 그대로 남는다. 따라서 matmul 호출은 3회에서 2회로 줄지만 전체 MLP가 단일 kernel이 되지는 않는다.

Native `ttnn.swiglu`는 현재 fused L1 width-sharded tensor를 직접 처리하지 못했다.

- W2의 512-wide shard config를 직접 전달: `Number of shards along width 32 must not exceed number of cores 16`
- fused output의 1024-wide shard config 유지: `Cannot set circular buffer size to 1048576 ... bank size of 65536 B`
- DRAM-sharded matmul에 interleaved output 직접 요청: `Output memory config must be sharded for DRAM sharded program config`

세 run 모두 exit 1의 host validation/runtime exception이며 `DEVICE_CLOSED`가 확인됐다. timeout, signal,
exit 124/137 또는 device hang은 없었다.

기능 검증 경로는 fused matmul을 L1 width-sharded로 출력한 뒤 DRAM interleaved로 변환하고 native
`ttnn.swiglu`를 수행한 다음 W2 input으로 다시 width-shard한다.

### 결과

동일 process당 correctness 1회 뒤 measured 20회다. Profiler와 TurboQuant는 껐다.

| 구성 | PCC | mean ms | median ms | min ms | baseline 대비 mean |
|---|---:|---:|---:|---:|---:|
| fanout-2 baseline | 0.9996410623 | 1.463371 | 1.466316 | 1.432812 | 0% |
| fused gate/up + DRAM SwiGLU | 0.9996951562 | 1.490912 | 1.488409 | 1.482141 | +1.88% |

두 run 모두 exit 0, `MLP_COMPLETED`, `DEVICE_CLOSED`를 확인했다. Console-only A/B라 별도 profiler
artifact는 없다.

### 해석과 다음 단계

관측: weight 총량은 같고 matmul launch 하나를 줄였지만 mean latency가 1.88% 늘었다.

추론: saved launch/activation multicast보다 L1→DRAM fused output, native slice/SwiGLU, DRAM→L1 reshard
비용이 더 크다.

## Core-local SwiGLU 구현 handoff

### 현재 상태

DRAM intermediate를 제거하는 opt-in custom op를 구현했다. `TT_METAL_MLP_FUSED_GATE_UP=1`일 때
W3/W1 weight를 core별 `[W3_local | W1_local]` 순서로 pack하고 fused gate/up matmul을 한 번 실행한다.
새 `ttnn.local_swiglu` device op는 각 shard core의 L1에서 W1 tile에 SiLU를 적용하고 W3 tile과
곱한다. Gate/up activation을 DRAM으로 내보내지 않는다.

Gate/up DRAM-sharded factory는 12 readers/12 active compute workers를 선택하지만 fused output tensor는
16-core irregular width-shard grid `{(0,0)-(4,2), (0,3)}`를 사용한다. 두 수는 다른 개념이다.
`ttnn.local_swiglu`는 이 16개 L1 shard에서 실행하고 `ttnn.to_memory_config`가 W2의 16-core rectangular
`4×4` width-shard layout으로 L1→L1 NoC remap한다. 이 remap은 남지만 DRAM 왕복은 없다.

### 검증 결과

Profiler-free isolated layer-0 gate를 2026-08-04 08:24 UTC에 통과했다.

- exit: 0
- PCC: `0.999695441571227`
- measured 1회: `1.479607 ms`
- markers: `MLP_COMPLETED`, `DEVICE_CLOSED`
- TurboQuant/profiler: disabled
- device state: 정상 close; 격리 아님

이 값은 1회 sanity 결과다. 아래에 20회 isolated A/B와 full-model 64K decode 결과를 추가했다.

### 해결한 실패

1. W2 rectangular 16-core output grid 직접 지정: local input의 irregular 16-core shard grid와 달라 host validation 실패.
2. input memory config 재사용: output shard width가 절반이 아니어서 validation 실패.
3. 첫 device JIT: `silu_tile_init`/`silu_tile` 선언 누락으로 compile 실패.

모두 exit 1, `DEVICE_CLOSED`였다. timeout, signal, exit 124/137, hang은 없었다. 마지막 성공에서
`compute_kernel_api.h` include와 자동 half-width output shard spec을 확인했다.

### 주요 소스와 상태

- `models/bos_model/llama32/tt/mlp.py`: opt-in, core-local pack, fused forward, L1 reshard
- `models/bos_model/llama32/tt/model_config.py`: fused program config
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/local_swiglu_sharded_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/local_swiglu_sharded.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_composite.hpp`, `unary_pybind.cpp`

SHA-256: `mlp.py` `e653b1f0f8d16836f7224c9b19c5e3ed0af212dc4f33d3ebc3db24e4ca3db4f1`;
`model_config.py` `dc60d9b2e396a1266247dabdbf0ae770faf1425cdf4c531774dcd48532c98750`;
factory `70624a4dfe58d3381acd1d97606fcbd145d99187b7ea6f8b734984b5fa747c7c`;
kernel `b5fe48d3a03cc230ef771a49751c957270b2795355acada0358d29d2f60d023b`.

`ttnn_op_eltwise_unary`, `_ttnn.so`, Python import, `git diff --check`, `py_compile` 모두 통과했다.

### 다음 순서

Profiler-free 20회 isolated A/B, isolated profile, Llama 3.2 3B 64K full decode A/B까지 완료했다.

모든 실행은 `TT_METAL_TURBOQUANT=0`, fanout-3/helper off를 유지했다. 기존 사용자 수정
`models/bos_model/llama32/run_llama32.py`와 SDPA compute kernel은 건드리지 않는다.

### Core-local SwiGLU 20회 A/B

| 구성 | PCC | mean ms | median ms | min ms |
|---|---:|---:|---:|---:|
| baseline fanout-2 | 0.9996410623 | 1.444443 | 1.441964 | 1.421561 |
| fused + local SwiGLU | 0.9996954416 | 1.415214 | 1.412716 | 1.406332 |
| 변화 | - | -2.02% | -2.03% | -1.07% |

두 run 모두 profiler-free, 20 measured calls, exit 0, `MLP_COMPLETED`, `DEVICE_CLOSED`다. 따라서
DRAM round-trip 제거는 기존 fused+DRAM 경로의 +1.88% 악화를 뒤집고 baseline보다 빨라졌다.

### Isolated profile

- run: `/home/iris_hb4/profiler_runs/mlp_local_swiglu_2026_08_04_09_53_00`
- exit 0; PCC `0.9996954416`; `MLP_COMPLETED`; `DEVICE_CLOSED`
- ops CSV SHA-256: `462ad626fb62ef9674759af2bdc476f36d6034632bbe6f8017ce1a044a284532`
- device CSV SHA-256: `b51904e072af6995a9faec4a4d4a8fd93aa1df33f2c1c285bbc9d519dd6cbca4`
- Tracy SHA-256: `ec17dfb6dc886b449c30e061eb403e602a81e5bc9b159e73c2c70ec0b8b7db0f`

Measured call device kernel duration:

| op | duration |
|---|---:|
| fused W3/W1 matmul | 838.378 us |
| local SwiGLU | 48.145 us |
| 16-core irregular→16-core rectangular L1 remap | 4.042 us |
| W2 matmul | 418.246 us |

Reshard는 local SwiGLU의 8.4%, measured MLP latency의 약 0.3%다. 현재 우선 병목이 아니다.

### Llama 3.2 3B 64K full decode A/B

#### 무효화한 첫 A/B

첫 A/B의 6.061334/6.071251 tok/s는 SDPA K256만 설정하고 아래 네 opt-in을 누락했다.

- `TT_METAL_SDPA_DECODE_DUAL_NOC=1`
- `TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6`
- `TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1`
- `TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1`

첫 로그에는 과거 성공 run의 `BOS dual-NoC SDPA decode reader enabled` marker가 없었다. MLP reader,
compute worker, endpoint group과 read-page 설정은 같았다. 따라서 7.646→6.061 tok/s 감소는 source
회귀나 fused 영향이 아니라 SDPA 6-endpoint 경로가 비활성인 실행 명령 오류다. 이 결과는 성능 판단에서
제외한다. Artifact는 감사용으로 보존한다:
`/home/iris_hb4/profiler_runs/llama32_3b_64k_local_swiglu_ab_2026_08_04_10_00_00`.

#### corrected exact A/B

현재 동일 source/binary에서 SDPA K256, dual-NoC, 6 endpoints, pair/bank balancing과 MLP balanced
fanout-2를 공통으로 두고 `TT_METAL_MLP_FUSED_GATE_UP`만 0/1로 바꿨다. Synthetic zero paged KV,
positions 65,486--65,535, warmup 3, measured 50 tokens, profiler/Watcher/TurboQuant off 조건이다.

| 구성 | elapsed s | ms/token | tokens/s |
|---|---:|---:|---:|
| optimized 6-endpoint baseline | 6.541588 | 130.831765 | 7.643404 |
| fused + local SwiGLU | 6.521695 | 130.433901 | 7.666719 |
| 변화 | -0.3041% | -0.3041% | +0.3050% |

Corrected baseline은 과거 7.645991 tok/s와 -0.0338% 차이다. 성능 회귀가 재현되지 않았다. 두 run 모두
`BOS dual-NoC SDPA decode reader enabled`, endpoint loads `3/2/3/3/3/2`, NoC0/NoC1 `8/8`, exit 0,
`WARMUP_COMPLETE`, `RESULT_JSON`, `DEVICE_CLOSED`를 확인했다.

Baseline final token은 499, fused는 269다. Autoregressive exact는 보장하지 않으며 아래 고정 입력 logits
결과로 numerical 차이를 분리한다.

Artifact root: `/home/iris_hb4/profiler_runs/llama32_3b_64k_local_swiglu_ab_corrected_2026_08_04_11_00_00`

- baseline log SHA-256: `8abb9d0fa1e56bd0d94e843cb904fdb2991ec21426c7da9012c8e7f58144a79b`
- fused log SHA-256: `4084cffcb121d350a884ba94302f0e66af89bd746d34fad1f5c3ee1591055d7e`

### 고정 입력 full-model logits 검증

동일한 seed 0, token 1, position 65,535, zero paged KV에서 sampling 없이 전체 FP32 logits를 비교했다.

| 지표 | 결과 |
|---|---:|
| shape | `[1,1,128256]` |
| bitwise equal | false |
| differing elements | 98,518 / 128,256 |
| max / mean absolute difference | 0.3125 / 0.044205 |
| RMSE | 0.058057 |
| PCC / cosine | 0.999288 / 0.999414 |
| top-1 | both 320 |
| top-5 / top-10 overlap | 5/5 / 9/10 |

Baseline top-1/top-2 logits는 10.875/9.0, fused는 10.75/8.875이며 두 margin은 모두 1.875다.
따라서 bitwise/exact가 아니라 `fixed-input top-1 preserving numerical approximation`이다. Fused matmul의
accumulation grouping과 SwiGLU 구현 순서 차이가 원인 후보이며, 여러 실제 KV/token accuracy 검증 전에는
accuracy-preserving 결론을 내리지 않는다. 두 run 모두 exit 0, `RESULT_JSON`, `DEVICE_CLOSED`다.

Artifact root: `/home/iris_hb4/profiler_runs/llama32_3b_64k_local_swiglu_logits_ab_2026_08_04_10_30_00`

- baseline log / tensor SHA-256: `8715231d36a599c3306335a335641e635afe1eec3516f8f0623a8e9bbbe7cedd` / `9c8e4529e8fac726f71d4855d4f2729cb865331e22035d45819efd289bdd6624`
- fused log / tensor SHA-256: `ad77d5c9ce65af8f1f8ed7c21f29756cb00dffe98f735965e826cd6a50a0f488` / `3fc0dbcc9c4da68dd6e12809ba68ebbecca8b28b963a4a665e9faecc7ee4a930`

### 다음 단계

1. full decode 이득이 0.305%로 작으므로 fused matmul의 12-reader cadence와 48 us local SwiGLU를 분해한다.
2. 실제 KV/token accuracy suite 통과 전 opt-in 기본값은 off로 유지한다.
3. 재현 명령은 SDPA 6-endpoint marker를 필수 safety/performance assertion으로 검사한다.

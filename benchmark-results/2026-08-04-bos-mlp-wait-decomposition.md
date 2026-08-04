# BOS MLP wait decomposition

날짜: 2026-08-04 UTC

## 결론

MLP consumer wait 원인은 단일 CB 부족이 아니다. Direct fanout-2에서 weight DRAM publish가 주로
늦다. Tagged two-block은 BRISC barrier를 줄였지만 compute input wait는 약 70%로 유지됐다.
W1 이득을 W3/W2의 activation/weight phase 변화가 상쇄했다.

현재 6-owner + 6-helper prefetch는 해결책이 아니다. Shard 절반 DRAM read는 약 19.5--21.2 us다.
helper→owner remote-L1 전송은 5.1--5.3 us, owner의 helper residual wait는 5.8--6.5 us다.
결과 block delivery cadence는 25.9--27.2 us다. Direct reader는 24.0--24.9 us다.

## 위치

- weight producer: `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`
- activation producer: `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`
- consumer: `bmm_large_block_zm_fused_bias_activation.cpp`
- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available grid: 5×4 = 20 workers
- DRAM: 3 physical banks, 2 worker endpoints/bank, 6 endpoints
- direct tagged: 12 readers, 12 compute
- helper: 12 readers, 6 compute owners, 6 helpers
- runtime-selected DRAM-interface workers: 6
- endpoint groups: NOC1 4:4:4

## 분해 항목

| 항목 | marker |
|---|---|
| activation multicast | `MLP_IN0_MCAST_WAIT`, `MLP_IN0_READY` |
| direct weight DRAM | `MLP_IN1_READ_BARRIER`, `MLP_IN1_READY` |
| weight CB full | `MLP_IN1_CB_RESERVE` |
| consumer activation/weight | `MLP_IN0_CB_WAIT`, `MLP_IN1_CB_WAIT` |
| helper CB reserve | `MLP_HELPER_BLOCK_START` → `MLP_HELPER_RESERVED` |
| helper DRAM | `MLP_HELPER_RESERVED` → `MLP_HELPER_DRAM_READY` |
| helper request | `MLP_HELPER_DRAM_READY` → `MLP_HELPER_REQUEST_READY` |
| helper remote write | `MLP_HELPER_REQUEST_READY` → `MLP_HELPER_WRITE_DONE` |
| owner DRAM | `MLP_OWNER_RESERVED` → `MLP_OWNER_DRAM_READY` |
| owner helper wait | `MLP_OWNER_DRAM_READY` → `MLP_OWNER_HELPER_READY` |

BRISC accumulated zone은 accumulator가 2개뿐이다. Helper 세부 단계는 block별 timestamp로 측정했다.

## Full barrier 대비 tagged

### BRISC weight barrier mean

| projection | full (us) | tagged (us) | 변화 |
|---|---:|---:|---:|
| W1 | 371.616 | 293.941 | -20.90% |
| W3 | 362.089 | 295.031 | -18.52% |
| W2 | 349.145 | 220.579 | -36.82% |

Barrier 합은 compute와 겹친다. Kernel duration에 더하지 않는다.

### Consumer critical wait

| projection | full wait/kernel | tagged wait/kernel | tagged kernel 변화 |
|---|---:|---:|---:|
| W1 | 312.697/440.297 us (71.02%) | 300.260/427.875 us (70.17%) | -12.422 us |
| W3 | 300.611/430.078 us (69.90%) | 306.122/436.494 us (70.13%) | +6.416 us |
| W2 | 292.092/417.000 us (70.05%) | 295.058/420.146 us (70.23%) | +3.146 us |

### Producer arrival

| projection | full weight-late | tagged weight-late | full/tagged mean arrival gap |
|---|---:|---:|---:|
| W1 | 147/192 | 142/192 | 28.933/26.495 us |
| W3 | 132/192 | 106/192 | 27.288/21.795 us |
| W2 | 144/192 | 93/192 | 26.947/21.887 us |

Tagged가 weight lateness를 줄였다. W3/W2는 activation이 새 late side가 됐다. Consumer wait 총량은
남았다. Tagged latency A/B의 2.28% 이득이 작은 이유다.

## Helper phase

96 core-block pair 평균이다.

| projection | helper DRAM | remote write | owner DRAM | owner helper wait | owner total delivery |
|---|---:|---:|---:|---:|---:|
| W1 | 21.175 us | 5.236 us | 20.560 us | 6.485 us | 27.232 us |
| W3 | 20.318 us | 5.310 us | 19.868 us | 5.850 us | 25.903 us |
| W2 | 19.474 us | 5.069 us | 20.107 us | 5.751 us | 26.041 us |

Reserve mean은 helper 0.108 us, owner 0.110--0.111 us다. CB full이 원인 아니다. Helper request wait
median은 0.129--0.135 us다. Request handshake도 주원인 아니다. 비용은 remote write와 그 뒤 owner
residual wait다.

| projection | direct full publish cadence | direct tagged cadence | helper owner delivery |
|---|---:|---:|---:|
| W1 | 24.940 us | 24.488 us | 27.232 us |
| W3 | 24.276 us | 24.711 us | 25.903 us |
| W2 | 24.016 us | 24.189 us | 26.041 us |

Shard를 반으로 나눠 얻은 DRAM 절감은 약 3--5 us다. Remote write 약 5 us가 이를 소모한다.

Helper profiled kernel은 W1/W3/W2 463.728/451.868/460.491 us다. Full direct는
440.297/430.078/417.000 us다. Helper consumer critical wait 비율은 47.97/46.08/49.27%로 낮지만,
compute owner가 12→6으로 줄어 kernel 전체가 느려졌다.

## 판정

관측:

1. Direct critical path는 대부분 weight publish가 결정한다.
2. CB reserve는 direct 대부분 core와 helper 모든 block에서 작다.
3. Tagged는 DRAM barrier를 줄이나 consumer wait를 제거하지 못한다.
4. Current helper는 remote-L1 hop을 추가한다.
5. Current helper latency와 bandwidth는 direct보다 나쁘다.

Fix:

1. 현 `6 compute + 6 helper`를 성능 경로로 사용하지 않는다.
2. Direct 12-compute 유지. Tagged는 opt-in 유지.
3. Activation-late core의 multicast route/phase를 교정한다.

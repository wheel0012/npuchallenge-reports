# BOS MLP DRAM-nearest 12-core placement A/B

날짜: 2026-08-05

## 결론

두 번째 fanout-2 reader 6개를 base DRAM-interface reader와 물리적으로 가깝게 재배치했다.
base-to-partner Manhattan distance 합은 24에서 20으로 줄었지만 isolated MLP mean latency는
1.435805 ms에서 1.557427 ms로 8.47% 악화됐다. 코드는 전부 rollback했다.

## 장치와 실행 구성

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available workers: 5x4, 20 cores
- active MLP readers/compute workers: 12/12
- physical DRAM: 3 banks, 6 worker NoC endpoints
- selected DRAM-interface workers: 6
- MLP: DRAM-sharded, fanout-2 tagged depth 2, endpoint groups 4:4:4
- W2 `in0_block_w=16`, 16 KiB read-page cap
- helper, fanout-3, endpoint-local, fused paths, TurboQuant: off
- profiler/Watcher: off

## 변경

기존 base 6 readers는 runtime의 `get_optimal_dram_bank_to_logical_worker_assignment()`를 유지했다.
기존 추가 6 readers는 logical scan 순서와 endpoint group capacity로 선택됐다. 실험은 추가 6개만
각 base reader와의 physical Manhattan distance가 작은 순서로 선택했다. 기존 NOC, VC assignment,
tagged producer-consumer protocol, CB contract, writer path는 바꾸지 않았다.

첫 구현은 base와 같은 endpoint group만 허용했다. 실제 base 분포에서는 세 번째 reader에 partner가
남지 않아 host-side `TT_FATAL`과 exit 1이 발생했다. device kernel launch 전 실패했고
`DEVICE_CLOSED`를 확인했다. 이후 같은-group 제약을 제거하고 전역 4:4:4 capacity를 유지했다.

## 최종 배치

후보 partner physical 좌표와 base distance:

| DRAM view | base physical | partner physical | distance |
|---:|---|---|---:|
| 0 | (0,4) | (0,1) | 3 |
| 1 | (1,4) | (1,2) | 2 |
| 2 | (2,4) | (3,2) | 3 |
| 3 | (3,4) | (4,2) | 3 |
| 4 | (4,4) | (4,1) | 3 |
| 5 | (0,2) | (4,0) | 6 |

총 distance는 20이다. 기존 mapping의 계산값은 24였다.

## 성능과 정확도

correctness 검사 뒤 20회를 측정했다.

| 구성 | PCC | mean ms | median ms | min ms |
|---|---:|---:|---:|---:|
| stable fanout-2 | 0.9996410623 | 1.435805 | 1.432855 | 1.424194 |
| DRAM-nearest partners | 0.9996410623 | 1.557427 | 1.557791 | 1.536540 |

- mean: +8.47% 느림
- median: +8.72% 느림
- correctness: 동일 PCC
- 모든 실제 kernel run: exit 0, `MLP_COMPLETED`, `DEVICE_CLOSED`

거리 감소만으로 NoC service latency가 줄지 않았다. 관측 결과는 route overlap, VC 조합, endpoint별
cadence가 단순 endpoint 거리보다 중요하다는 증거다. 정확한 악화 원인 분해는 미측정 추론이다.

## Rollback과 재현

- source와 installed `_ttnncpp.so`에서 nearest opt-in 문자열 제거 확인
- rollback build/install 성공
- rollback 뒤 stable 5회: PCC 0.9996410623, mean 1.447440 ms, median 1.440242 ms
- runner: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py`
- 적용/rollback 감사 patch:
  - `/home/iris_hb4/tmp/codex-patches/20260805-101500-mlp-nearest-fanout2-partners.patch`
  - `/home/iris_hb4/tmp/codex-patches/20260805-102200-mlp-nearest-partners-contract.patch`
  - `/home/iris_hb4/tmp/codex-patches/20260805-103000-mlp-nearest-global-balanced.patch`

## 다음 방향

core distance 단독 최적화는 중단한다. 다음 placement 실험은 endpoint distance가 아니라 실제 NOC route
overlap과 VC pressure를 비용함수에 포함해야 한다. profiler 없이 먼저 host-side route model로 후보를
선별하고, stable protocol을 유지한 단일 mapping만 isolated A/B한다.

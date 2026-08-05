# BOS MLP DRAM admission depth-1 및 6-compute 비교

- 날짜: 2026-08-05 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- 대상: Llama 3.2 3B layer 0 isolated decode MLP

## 목적

12 readers가 지속적으로 DRAM request를 pending한 상태에서 endpoint queueing을 줄이면 service latency와
MLP latency가 개선되는지 확인한다. 기존 tagged depth-2, reader당 한 block만 outstanding으로 두는
depth-1, fanout을 제거한 6-reader/6-compute를 비교했다.

## 공통 조건

- BFP8 weights, DRAM width-sharded
- W2 `in0_block_w=16`
- 16 KiB read-page cap
- reader locality off
- TurboQuant, fused gate/up, prefetch helper, fanout-3 off
- profiler, Watcher off
- correctness 1회 뒤 measured 20회
- 각 process에 `timeout --signal=INT --kill-after=15s 90s`

실행기는
`/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py`다.

```bash
MLP_AB_ITERATIONS=20 TT_METAL_MLP_DRAM_SHARDED=1 \
TT_METAL_MLP_W2_IN0_BLOCK_W=16 TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
TT_METAL_TURBOQUANT=0 timeout --signal=INT --kill-after=15s 90s \
/home/iris_hb4/tt-metal-hb4/python_env/bin/python \
models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

Variant delta는 다음과 같다.

- depth-2: fanout-2, tagged, balanced endpoints
- depth-1: depth-2 + `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH1=1`
- 6-compute: fanout-2/tagged/balanced endpoints 모두 off

## 결과

| variant | readers/compute | 최대 block pending | mean ms | median ms | PCC | 종료 |
|---|---:|---:|---:|---:|---:|---|
| tagged depth-2 | 12/12 | 24 | 1.467559 | 1.468924 | 0.9996410623 | close, exit 0 |
| tagged depth-1 | 12/12 | 12 | 1.493915 | 1.495990 | 0.9996410623 | close, exit 0 |
| fanout off | 6/6 | 6 | 측정 불가 | 측정 불가 | 출력 없음 | timeout, exit 137 |

Depth-1은 depth-2보다 mean 1.796%, median 1.843% 느리다. Queue depth를 절반으로 줄여도 service
latency 이득보다 overlap 손실이 컸다. 12-reader 요청 과잉이 현재 latency의 주원인이라는 가설은
지지되지 않는다.

6-compute는 log에서 `readers: 6, compute workers: 6`, read page W1/W3 6528 B, W2 8704 B를
확인했다. 그러나 90초 안에 20 measured calls를 끝내지 못했고 completion/close가 없었다. 성공
성능값으로 사용하지 않는다.

## 구현

`TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH1=1` opt-in을 추가했다. tagged loop의
`pending_depth=1`, transaction ID 1개를 사용한다. 기존 기본값은 depth-2다. depth-1과 depth-3의
동시 사용은 host validation으로 금지했다.

Audit patches:

- `/home/iris_hb4/tmp/codex-patches/20260805-043000-mlp-tagged-depth1-admission.patch`
- `/home/iris_hb4/tmp/codex-patches/20260805-043500-mlp-tagged-depth1-validation.patch`

## 한계와 결정

- console transcript 외 별도 artifact를 생성하지 않았다.
- 6-compute timeout 뒤 장치를 격리했다. 재부팅 전 추가 device workload를 실행하지 않는다.
- depth-1은 채택하지 않는다. 기존 12-compute tagged depth-2를 유지한다.
- 다음 후보는 모든 reader의 outstanding을 줄이는 것이 아니라 endpoint별 issue phase를 작은 폭으로
  stagger하는 것이다. 재부팅 뒤 1 measured call correctness부터 시작해야 한다.

## 2026-08-05 재부팅 뒤 lane-1 start stagger

재부팅 뒤 32×32 add gate를 통과했다. Tagged fanout-2의 두 번째 reader lane만 최초 DRAM request 전에
`riscv_wait(256)`을 한 번 수행하는 opt-in을 추가했다. Reader/compute는 12/12, endpoint destination은
4:4:4, tagged pending depth는 2로 유지된다. 반복 block마다 throttle하지 않는다.

정확한 목표 조건은 W2 `in0_block_w=16`, 16 KiB read-page cap이다. Log에서 W1/W3 13,056 B,
W2 8,704 B page와 cap `true`를 확인했다.

| variant | mean ms | median ms | PCC | 종료 |
|---|---:|---:|---:|---|
| 256-cycle lane-1 stagger | 1.473408 | 1.474435 | 0.9996410623 | close, exit 0 |
| no stagger | 1.440942 | 1.436232 | 0.9996410623 | close, exit 0 |

Stagger는 mean 2.253%, median 2.660% 느리다. Endpoint별 동시 issue를 256 cycles 어긋나게 해도
service-latency 이득이 없고 startup/phase 손실만 늘었다. 채택하지 않는다.

설정 audit 중 잘못된 `TT_METAL_MLP_W2_BLOCK_W`와
`TT_METAL_MLP_DRAM_SHARDED_READ_PAGE_SIZE`를 사용한 예비 A/B도 발견했다. 실제 log는
`w2_in0_block_w=auto`, 16 KiB cap `false`였다. 목표 조건 결과에서 제외한다. Host model assertion으로
커널 전에 종료한 1회도 성능 결과에서 제외한다.

구현 env는 `TT_METAL_MLP_DRAM_SHARDED_FANOUT2_STAGGER_CYCLES`이며 0--4096 정수만 허용한다.
Tagged fanout-2가 아니면 host validation이 실패한다. 기본값은 0이다.

Audit patch:

- `/home/iris_hb4/tmp/codex-patches/20260805-045000-mlp-fanout2-start-stagger.patch`

결정은 12-compute tagged depth-2, no stagger 유지다. 6-compute timeout 결과와 함께 reader 수를
줄이거나 global/phase admission을 제한하는 방향은 중단한다.

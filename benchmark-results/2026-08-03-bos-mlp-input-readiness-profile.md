# BOS MLP activation/weight readiness profile

## 결론

Balanced fanout-2 isolated decode MLP에서 projection critical input-wait의 1차 병목은 weight다.
Measured W1/W3/W2의 192개 core-block 쌍 가운데 weight가 activation보다 늦게 publish된 쌍은 각각
147개(76.6%), 132개(68.8%), 144개(75.0%)였다. 각 projection에서 input wait가 가장 긴 core도
모두 16/16 block이 weight-late였다.

다만 activation 경로도 균일하지 않다. 물리 worker core `2:4`, `3:4`에서는 세 projection 모두
14/16 block이 activation-late였고 activation이 평균 약 38--44 us 늦었다. 따라서 전체 critical
path는 weight reader가 결정하지만 특정 코어에는 activation multicast 병목이 별도로 존재한다.

## 장치와 실행 구성

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4 = 20 cores
- active MLP reader/compute cores: 12/12
- physical DRAM: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- runtime selected DRAM-interface workers: 6
- weight: BFP8, DRAM width-sharded, balanced fanout-2, NOC1 destination groups `4:4:4`
- W2 `in0_block_w`: 16, weight read page cap: 16 KiB
- prefetch helper, fanout-3, TurboQuant: off
- Watcher와 NoC trace: 사용하지 않음

## 계측 계약

기존 순차 wait는 아래 구조라 개별 wait만으로 실제 arrival order를 확정할 수 없었다.

```text
wait activation CB
wait weight CB
matmul
```

이번에는 profiler 비활성 빌드에서 no-op인 timestamp를 block마다 기록했다.

| RISC | Marker | 의미 |
|---|---|---|
| NCRISC | `MLP_IN0_READY` | activation multicast 완료 뒤 `cb_push_back` 직후 |
| BRISC | `MLP_IN1_READY` | weight DRAM barrier 완료 뒤 `cb_push_back` 직후 |
| TRISC0 | `MLP_MATMUL_INPUTS_READY` | 두 input `cb_wait_front`가 모두 끝난 직후 |

NCRISC에는 `MLP_IN0_CB_RESERVE`와 `MLP_IN0_MCAST_WAIT` accumulated zone도 추가했다. 기존
BRISC weight reserve/barrier와 TRISC input-wait zone은 유지했다. CSV의 `TS_DATA.data`에는 block
index 0--15가 기록되므로 동일 global call count, physical core, block index로 세 timestamp를
결합했다. Blackhole clock 650 MHz를 사용해 cycle을 us로 변환했다.

## Projection 매핑

| Projection | Global call count | Device kernel duration |
|---|---:|---:|
| W1 | 7168 | 440.297 us |
| W3 | 8192 | 430.078 us |
| W2 | 11264 | 417.000 us |

Warmup call 1024/2048/5120은 marker completeness와 JIT 확인에는 사용했지만 아래 measured 통계에서는
제외했다. 각 measured projection은 12 cores x 16 blocks = 192개 비교쌍이며 누락 marker는 0개다.

## 결과

### Producer arrival order

`signed mean`은 `activation_ready - weight_ready`다. 음수이면 weight가 평균적으로 더 늦다.

| Projection | Activation late | Weight late | Weight-late 비율 | Signed mean | Mean 절대차 | 최대 절대차 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 45 | 147 | 76.6% | -13.631 us | 28.933 us | 80.291 us |
| W3 | 60 | 132 | 68.8% | -10.043 us | 27.288 us | 74.483 us |
| W2 | 48 | 144 | 75.0% | -11.266 us | 26.947 us | 73.045 us |

`MLP_MATMUL_INPUTS_READY - max(IN0_READY, IN1_READY)`는 평균 0.048--0.059 us, 최대
0.105--0.112 us였다. 즉 producer timestamp와 compute가 두 CB를 확보한 시점이 정확히 맞물리며,
marker pairing이 input readiness를 잘 설명한다.

### Input-wait critical core

| Projection | Core | Total input wait | Activation wait | Weight wait | Arrival 판정 |
|---|---|---:|---:|---:|---|
| W1 | `0:2` | 312.697 us | 1.358 us | 311.338 us | weight 16/16 late |
| W3 | `1:0` | 300.611 us | 4.302 us | 296.309 us | weight 16/16 late |
| W2 | `0:2` | 292.092 us | 1.311 us | 290.782 us | weight 16/16 late |

따라서 이전 보고서에서 순차 wait 때문에 남아 있던 모호성은 projection critical input-wait에
대해서는 해소됐다. 가장 오래 기다리는 compute core는 모두 weight arrival에 막힌다.

### Activation-local imbalance

Core `2:4`, `3:4`는 세 projection에서 모두 activation 14/16, weight 2/16 late였다.

| Projection | `2:4` signed mean | `3:4` signed mean |
|---|---:|---:|
| W1 | +40.901 us | +40.935 us |
| W3 | +43.559 us | +38.404 us |
| W2 | +42.275 us | +40.446 us |

반대로 `0:0`, `0:2`, `0:4`, `1:0`, `1:1`, `1:4`는 모든 measured block에서 weight-late였다.
이는 단일 global 병목만 있는 것이 아니라 core placement에 따라 weight DRAM 경로와 activation
multicast 경로의 늦은 쪽이 갈린다는 뜻이다.

### Aggregate zone 교차검증

| Projection | Activation mcast wait mean/max | Weight barrier mean/max | Activation reserve mean/max | Weight reserve mean/max |
|---|---:|---:|---:|---:|
| W1 | 253.205 / 364.206 us | 371.616 / 416.017 us | 135.313 / 373.132 us | 19.810 / 125.326 us |
| W3 | 259.517 / 359.991 us | 362.089 / 404.069 us | 119.097 / 372.417 us | 18.172 / 119.408 us |
| W2 | 246.013 / 350.620 us | 349.145 / 391.662 us | 122.613 / 357.586 us | 22.341 / 146.857 us |

Activation reserve가 길다는 사실은 double-buffer가 자주 full이라는 뜻이지만, 이것만으로 activation이
consumer critical path라고 결론내릴 수 없다. 실제 publish timestamp에서는 대부분의 critical core에
activation이 먼저 도착했다. 더 큰 activation CB는 producer backpressure를 흡수할 수 있으나 weight
service rate가 그대로면 projection critical path를 직접 줄이지 못할 가능성이 크다.

## 판정과 다음 우선순위

### 관측 사실

1. 세 measured projection 모두 전체 core-block 쌍의 약 69--77%에서 weight가 늦다.
2. input-wait가 가장 긴 core는 모두 weight 16/16 late다.
3. `2:4`, `3:4`에서는 activation multicast가 일관되게 늦다.
4. Weight CB는 기존 분석대로 triple-buffered지만 fanout reader는 block별 barrier를 사용한다.
5. profiler-free와 profiled run 모두 PCC 0.999641, completion, device close, exit 0을 통과했다.

### 최적화 우선순위

1. 전체 critical path에는 weight reader의 block별 completion을 먼저 개선한다. 다음 실험은 여러 future
   block을 실제 outstanding으로 두는 tagged two-block issue를 한 projection opt-in으로 검증하는 것이
   가장 직접적이다.
2. activation은 global 변경보다 `2:4`, `3:4`의 multicast route/phase를 표적 분석한다. sender placement,
   multicast rectangle 및 semaphore arrival 순서를 host-side mapping과 standard zone으로 먼저 확인한다.
3. 단순 CB depth 증가는 우선순위가 낮다. Weight는 이미 triple buffer이고 activation double buffer의
   reserve backpressure가 critical consumer lateness와 동일하지 않다.

### 아직 주장하지 않는 것

- 모든 MLP core에서 weight만 병목이라는 주장
- activation double buffer 확대가 무효라는 확정 주장
- isolated layer의 비율이 full-model tokens/s에 그대로 적용된다는 주장
- timestamp profiler의 절대 latency가 profiler-free latency와 동일하다는 주장

## 실행과 안전성

Profiler-free gate는 PCC 0.9996410623, 1.529151 ms, `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0이었다.
Profiler capture는 PCC 0.9996410623, 1.508256 ms, 동일 completion/close, exit 0이었다. ops CSV,
device CSV, host Tracy가 완성됐고 잔존 Python/Tracy/capture process는 없었다. timeout, signal,
exit 124/137 또는 hang은 발생하지 않았다.

재현 명령의 핵심 환경은 다음과 같다.

```bash
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct MLP_AB_ITERATIONS=1 \
  TT_METAL_MLP_DRAM_SHARDED=1 TT_METAL_MLP_W2_IN0_BLOCK_W=16 \
  TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 TT_METAL_TURBOQUANT=0 \
  timeout --signal=INT --kill-after=15s 120s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy \
  -p -r --sync-host-device --check-exit-code \
  -o /home/iris_hb4/profiler_runs/mlp_input_readiness_2026_08_03_17_10_00/perf_capture \
  -n mlp_input_readiness \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## Artifact와 checksum

Artifact root:

```text
/home/iris_hb4/profiler_runs/mlp_input_readiness_2026_08_03_17_10_00
```

- device CSV: `bb675bc78dbb5db60f9e1e74d15e79cb86aff16eb4e8bae5733e16caf43417d5`
- ops CSV: `ea0588b9b48abae8003f24e579dcb777e97c5800dec07839870b2add03e1da2f`
- host Tracy: `b83bc2ad039063a7b57d85848f493717ed3648a074842d4b71fd64cf506aa9a8`
- main-path patch: `/home/iris_hb4/tmp/codex-patches/20260803-170400-mlp-readiness-main-path-v5.patch`
  (`71a31c36d51dce0eb710313767d4c8a3dfa1352f6798d1f2c535771c27d22fd0`)
- activation receiver-path patch:
  `/home/iris_hb4/tmp/codex-patches/20260803-170500-mlp-readiness-receiver-path-v6.patch`
  (`447bee4a91db41f43100eaaf1d112cdab276c42d5202cfc744d3dd220ab59cfb`)

Raw CSV와 Tracy trace는 `reports/`로 복제하지 않고 위 timestamped run에 유지한다.

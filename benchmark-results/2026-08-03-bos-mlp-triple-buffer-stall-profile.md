# BOS MLP triple-buffer stall profile

## 결론

Balanced fanout-2 MLP의 weight input CB는 이미 block 3개를 담는 triple buffer다. 최종 device-zone
capture에서 W1/W3/W2 critical compute core의 input-CB wait 합은 각각 302.177, 313.675,
294.966 us였고 projection kernel의 70.42%, 70.90%, 70.61%였다.

반면 reader의 weight-CB `reserve_back` 중앙값은 0.787, 0.773, 1.223 us였다. 각 projection에서
12 reader 중 10개는 약 1.3 us 이내였고 두 reader만 54--156 us의 backpressure를 보였다. 따라서
전체 reader가 CB-full로 막히는 구조가 아니며, **CB depth를 3보다 더 늘리는 것만으로 critical path가
크게 개선될 근거는 없다.** 현재 우선순위는 block별 DRAM completion과 activation/weight arrival의
core별 불균형을 줄이는 것이다.

## 장치와 측정 범위

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4 = 20 cores
- physical DRAM topology: 3 banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- operation: Llama 3.2 3B layer 0 isolated decode MLP
- active MLP readers/compute: balanced fanout-2, 12/12
- weight layout: BFP8, DRAM width-sharded, fanout-2 cache
- weight read NoC: NOC1, physical-bank destination `4:4:4`
- W2 `in0_block_w`: 16
- read-page cap: 16 KiB
- helper, fanout-3, TurboQuant: 모두 off
- Watcher와 NoC trace: 사용하지 않음

## 기존 buffering 계약

Program factory는 `B * num_blocks > 1`이면 activation CB를 double buffer, weight CB를 triple buffer로
할당한다.

```cpp
in0_CB_tiles = in0_block_tiles * 2;
in1_CB_tiles = in1_block_tiles * 3;
```

Fanout reader는 block마다 아래 순서를 반복한다.

```text
cb_reserve_back(weight block)
row burst read issue
noc_async_read_barrier()
cb_push_back(weight block)
```

따라서 storage slot은 3개지만 DRAM block 여러 개를 barrier 너머로 동시에 outstanding 상태로 두지는
않는다. Block `i`를 publish한 뒤 compute가 이를 소비하는 동안 block `i+1`을 읽는 producer/consumer
overlap은 가능하지만, reader는 `i+1` completion 전에는 `i+2`를 issue하지 않는다.

## 계측 방법

일반 실행에서 no-op인 `DeviceZoneScopedSumN1/N2`를 아래 네 지점에 추가했다.

| RISC | Zone | 범위 |
|---|---|---|
| BRISC | `MLP_IN1_CB_RESERVE` | weight block `cb_reserve_back` |
| BRISC | `MLP_IN1_READ_BARRIER` | issued row reads의 completion barrier |
| TRISC0 | `MLP_IN0_CB_WAIT` | activation block `cb_wait_front` |
| TRISC0 | `MLP_IN1_CB_WAIT` | weight block `cb_wait_front` |

Compute는 in0을 먼저 기다린 뒤 in1을 기다리므로 in0/in1 개별 wait 비율은 arrival order에 의존한다.
두 값을 합한 `input wait total`을 compute가 입력 준비를 기다린 안정적인 지표로 사용했다. Zone의
`data` cycle을 capture CSV가 기록한 Blackhole clock 650 MHz로 변환했다.

Measured projection은 ops CSV와 global call count로 연결했다.

| Projection | Global call count | Kernel duration |
|---|---:|---:|
| W1 | 7168 | 429.114 us |
| W3 | 8192 | 442.428 us |
| W2 | 11264 | 417.718 us |

## 결과

### Compute input wait

아래 수치는 12 active reader/compute core의 TRISC0 accumulated zone이다.

| Projection | Input wait mean | Input wait median | Critical input wait | Kernel 대비 critical wait |
|---|---:|---:|---:|---:|
| W1 | 288.184 us | 289.918 us | 302.177 us | 70.42% |
| W3 | 296.529 us | 291.382 us | 313.675 us | 70.90% |
| W2 | 278.906 us | 278.248 us | 294.966 us | 70.61% |

Critical core에서는 weight wait가 각각 300.958, 311.660, 292.045 us로 대부분을 차지했다. 하지만
다른 core에서는 activation wait가 249 us까지 증가하고 weight wait가 줄었다. 따라서 이것을 모든
core의 순수 DRAM stall 70%라고 해석하지 않는다. 확실한 관측은 두 input 가운데 늦게 도착한 쪽을
기다리는 시간이 projection critical path의 약 70%라는 것이다.

### Reader reserve와 DRAM barrier

| Projection | Reserve mean | Reserve median | Reserve max | Barrier mean | Barrier max |
|---|---:|---:|---:|---:|---:|
| W1 | 17.765 us | 0.787 us | 121.515 us | 366.776 us | 405.598 us |
| W3 | 22.404 us | 0.773 us | 155.589 us | 368.617 us | 417.057 us |
| W2 | 18.508 us | 1.223 us | 120.909 us | 351.282 us | 394.514 us |

Barrier zone은 BRISC가 DRAM completion을 기다린 누적 시간이며 compute와 겹칠 수 있으므로 kernel
duration에 더하지 않는다. Reserve median이 매우 작다는 사실과 두 core에만 큰 reserve가 나타난 것은
triple buffer가 전체적으로 부족한 것이 아니라 reader service rate와 compute consumption이 core별로
다르다는 증거다.

## 판정

### 관측 사실

1. Weight CB는 이미 triple-buffered다.
2. 대부분의 reader는 free CB slot을 기다리지 않는다.
3. Compute critical core는 input readiness를 kernel 시간의 약 70--71% 기다린다.
4. BRISC read barrier가 길고 core별 편차가 크다.
5. Profiler-free와 profiled run은 모두 PCC 0.999641 및 정상 device close를 통과했다.

### 추론

- 단순 CB depth 3→4 확대는 steady-state DRAM service rate를 높이지 않는다.
- 현재 buffer는 jitter를 흡수하지만 block별 barrier 때문에 여러 future block을 동시에 in-flight로
  만들지는 못한다.
- 다음 최적화는 더 큰 CB보다 direct reader의 two-block tagged issue, block 크기 조정 또는
  activation/weight arrival phase 정렬을 검토하는 편이 타당하다.
- Tagged issue는 producer/consumer 계약을 바꾸므로 별도 opt-in, correctness-first, 한 projection
  isolated test로만 시작해야 한다.

### 아직 주장하지 않는 것

- 70% input wait 전체가 DRAM latency라는 주장
- CB depth 증가가 성능을 반드시 악화하거나 개선한다는 주장
- 이번 isolated layer 결과가 full-model tokens/s에 그대로 적용된다는 주장

## 실행과 안전성

Profiler-only zone 교정 뒤 profiler-free gate:

- PCC: 0.9996410623
- latency: 1.509136 ms
- `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0

최종 zone capture:

- PCC: 0.9996410623
- profiler-instrumented latency: 1.527677 ms
- profiler-free 대비 약 +1.23%
- `MLP_COMPLETED`, `DEVICE_CLOSED`, exit 0
- ops CSV, device CSV, host Tracy 완성
- 잔존 Python/Tracy/capture process 없음

첫 시도는 runtime kernel define이 persistent compile/cache에 반영되지 않아 정상 종료했지만 custom zone이
없었다. 교정 과정의 한 run은 device launch 전 TRISC JIT compile error로 exit 1이었고
`DEVICE_CLOSED`를 확인했다. timeout, signal, exit 124/137 또는 device hang은 없었다.

## 재현 명령

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
  -o /home/iris_hb4/profiler_runs/mlp_triple_buffer_stall_2026_08_03_16_16_25/perf_capture_v2 \
  -n mlp_triple_buffer_stall_v2 \
  /home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## Artifact와 checksum

Artifact root:

```text
/home/iris_hb4/profiler_runs/mlp_triple_buffer_stall_2026_08_03_16_16_25
```

- profiler-free log: `baseline_profiler_zones_v2.log`, SHA-256
  `87c1742ecdb99226a405f87cb3044b21c4164e5c6550925b81da535262f2097f`
- final profiler log: `stall_profile_v2.log`, SHA-256
  `8f09137f4be51786204df5900e23ba4df06940625c48e601c8abae2752e079ca`
- device CSV: SHA-256 `9b5ab50f6e26167cfe1d5bad08f51e1f0cf65e0a6ad988581bb5749a3bb97cca`
- ops CSV: SHA-256 `774100aa31cde12255119c3dcd5e699490385907352b4880a7a9423bfa20162b`
- host Tracy: SHA-256 `a4c1da74b323e14a0a498420c8e81c295b13a0280261857367dab56671ece0cb`
- final instrumentation patches:
  - `/home/iris_hb4/tmp/codex-patches/20260803-mlp-stall-zones-profiler-only-v2.patch`
    (`7648e98d42fd5260d511fcc582ca77b6a61f330ed27b119b009d7b1327da570c`)
  - `/home/iris_hb4/tmp/codex-patches/20260803-mlp-compute-profiler-include.patch`
    (`b30ecdc4d7740d088cd4a90fc08d61aa8f81ce39a998ffc10928c2aca89f52e0`)

Raw CSV, Tracy trace와 logs는 `reports/`로 복제하지 않고 위 timestamped run에 유지한다.

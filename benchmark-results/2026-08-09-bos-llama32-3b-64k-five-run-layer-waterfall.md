# BOS Llama 3.2 3B 64K 5-run 및 layer-0 waterfall

## 결론

같은 source/build와 host session에서 네 누적 구성을 각각 5회 측정했다. Best-stable은
`8.20450 ± 0.00336 tok/s`이며 vanilla-equivalent `5.12297 ± 0.00309 tok/s` 대비 `+60.15%`다.
모든 구성의 CV는 `0.065%` 이하다. 단일 측정 우연으로 설명하기 어려운 차이다.

signpost로 분리한 layer 0의 device FW duration 합은 `7579.298→4663.924 us`, `-38.46%`다.
가장 큰 절감은 SDPA `3497.640→2037.246 us`와 MLP sublayer `3129.095→1779.843 us`다.

## 장치와 측정 조건

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- DRAM: 3 physical banks, 2 worker endpoints/bank, 6 endpoints
- model: `meta-llama/Llama-3.2-3B-Instruct`, batch 1
- context: zero-initialized synthetic paged KV, 65,536-token window
- full model: 28 layers, warmup 3 tokens, measured 50 tokens × 5 windows
- layer profile: TransformerBlock layer 0, warmup 1회, signpost measured call 1회
- profiler: full-model 반복 run은 off; layer 0은 Tracy device profiler, NoC trace off
- SnapKV, TurboQuant, reduce-only helper, experimental tagged SDPA: off

각 full-model process는 한 번 open한 뒤 warmup하고 같은 50-token curpos 구간을 5회 다시 실행했다.
각 반복은 시작 token과 curpos를 동일하게 초기화했다. 모든 4개 full-model run과 4개 layer gate/profile은
정상 device close와 exit 0을 기록했다. timeout, signal, exit 124/137은 없었다.

## Full-model 5-run waterfall

표의 `±`는 sample standard deviation이다. 95% CI는 자유도 4의 t 값 2.776을 사용했다.

| 단계 | tok/s mean ± SD | CV | 95% CI | 직전 대비 | vanilla 대비 |
|---|---:|---:|---:|---:|---:|
| Vanilla K128 | 5.122967 ± 0.003086 | 0.0602% | ±0.003831 | 기준 | 기준 |
| + SDPA K256/6EP | 6.415537 ± 0.004162 | 0.0649% | ±0.005167 | +25.23% | +25.23% |
| + MLP DRAM-sharded | 7.637903 ± 0.000543 | 0.0071% | ±0.000675 | +19.05% | +49.09% |
| + QKV/Wo stable | 8.204496 ± 0.003359 | 0.0409% | ±0.004170 | +7.42% | +60.15% |

샘플 범위는 각각 `5.11745--5.12443`, `6.40809--6.41754`, `7.63699--7.63836`,
`8.19877--8.20764 tok/s`다. 인접 단계 95% CI가 겹치지 않는다.

## Layer-0 detailed waterfall

아래 값은 signpost 사이 device FW duration 합이다. warmup, model load, JIT는 제외했다.

| 구간 | Vanilla | + SDPA | + MLP | Final | Final vs vanilla |
|---|---:|---:|---:|---:|---:|
| 전체 TransformerBlock | 7579.298 us | 6108.996 us | 4788.464 us | 4663.924 us | -38.46% |
| Attention sublayer | 4450.203 us | 2987.587 us | 3009.972 us | 2884.081 us | -35.19% |
| MLP sublayer | 3129.095 us | 3121.409 us | 1778.492 us | 1779.843 us | -43.15% |
| QKV matmul | 434.942 us | 433.046 us | 435.283 us | 273.228 us | -37.18% |
| SDPA | 3497.640 us | 2037.246 us | 2043.082 us | 2039.428 us | -41.69% |
| Wo matmul | 263.197 us | 262.143 us | 261.778 us | 165.362 us | -37.17% |
| W1 matmul | 557.103 us | 554.563 us | 245.408 us | 249.711 us | -55.18% |
| W3 matmul | 583.411 us | 579.642 us | 432.943 us | 437.097 us | -25.08% |
| SwiGLU | 605.388 us | 601.629 us | 236.722 us | 235.734 us | -61.06% |
| W2 matmul | 683.003 us | 684.146 us | 433.205 us | 430.562 us | -36.96% |

누적 layer FW 합 변화는 `-19.40%`, 추가 `-21.62%`, 추가 `-2.60%`다.

### 해석

- SDPA 단계의 layer 절감 `1470.302 us` 중 `1460.394 us`가 SDPA op 자체에서 나온다.
- MLP 단계에서 MLP sublayer가 `3121.409→1778.492 us`, `-43.02%` 감소한다.
- Final 단계에서 QKV는 `435.283→273.228 us`, Wo는 `261.778→165.362 us`로 감소한다.
- Final의 QKV 이후 SDPA 전 변환 합은 `107.031→213.250 us`로 증가한다. QKV matmul 절감 일부가
  변환/reshard 비용으로 이동했다.
- Final의 Wo 뒤 attention tail도 `11.061→171.190 us`로 증가한다. Wo matmul 단독 수치만으로
  end-to-end 이득을 과대평가하면 안 된다.
- 그래서 final attention sublayer 순절감은 `125.891 us`, layer 합 추가 절감은 `124.540 us`다.

## 구성

1. Vanilla: K128, SDPA endpoint override off, grouped concat/QKV/Wo/MLP optimization off.
2. SDPA: K256, dual-NoC, 6 endpoints, endpoint load `3/2/3/3/3/2`, NoC load `8/8`.
3. MLP: 6 DRAM interface workers, 12 reader/compute workers, balanced fanout-2 tagged,
   W2 block16, 16 KiB read-page cap.
4. Final: grouped concat, DRAM-sharded QKV, L1 Wo program, DRAM-sharded balanced fanout-2.

`CORE COUNT=20` 같은 op-level 범위와 실제 MLP reader/compute workers 12를 구분한다. SDPA도 8 KV
heads × 2 cores/head라 active readers는 16이다.

## Artifact

Full-model 5-run:

`/home/iris_hb4/benchmark_runs/llama32_3b_64k_waterfall_repeats5_2026_08_09_17_20_12`

- `01_vanilla_k128.log`부터 `04_final_stable.log`
- `repeat_summary.csv`: samples, mean, SD, CV, 95% CI, incremental/cumulative 개선폭

Layer-0 profiles:

`/home/iris_hb4/profiler_runs/llama32_3b_64k_layer0_waterfall_2026_08_09_17_26_00`

- `01_vanilla`부터 `04_final`: gate log, Tracy log, raw reports와 trace
- 각 단계 `measured_ops.csv`: 두 signpost 사이 full-column op rows
- `layer0_waterfall_summary.csv`: semantic operator 구간 집계

## 한계

- 실제 64K prompt prefill이 아니라 synthetic zero KV다.
- 5회 반복은 같은 process 안의 연속 window다. process-to-process/JIT 변동을 측정하지 않는다.
- layer profile은 layer 0 한 개의 대표값이다. 28개 layer 각각의 별도 capture가 아니다.
- device FW duration 합은 op overlap이 있으면 device wall latency와 다르다.
- full-model 단계별 final token은 서로 다르다. 기존 fixed-input gate의 PCC 약 0.9993과 top-1/top-5
  동일 결과에 의존하며 bit-exact는 아니다.

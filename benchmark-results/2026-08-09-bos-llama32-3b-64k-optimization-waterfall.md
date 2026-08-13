# BOS Llama 3.2 3B 64K optimization waterfall

## 발표용 결론

같은 서버 session에서 vanilla-equivalent부터 best-stable까지 누적 A/B를 다시 측정했다.

- throughput: `5.124290→8.207842 tok/s`, `+60.18%`
- token latency: `195.148988→121.834702 ms`, `-37.57%`
- 가장 큰 단일 단계: SDPA K256/6-endpoint, `+25.18%`
- 다음 단계: DRAM-sharded MLP, 직전 대비 `+19.16%`
- QKV/Wo projection: 직전 대비 `+7.39%`

발표 headline은 vanilla 대비 최종 누적 `+60.18%`를 사용한다. 각 기법 설명에는 직전 단계 대비
incremental 수치를 사용한다.

## 조건

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- DRAM topology: 3 physical banks, 2 worker endpoints/bank, 6 endpoints
- model: `meta-llama/Llama-3.2-3B-Instruct`
- batch: 1
- context/curpos range: synthetic zero-initialized paged KV, 65,536 window, 65,486--65,535
- page block: 32 tokens
- warmup: 3 decode tokens
- measured: 50 decode tokens
- trace/profiler: off
- process: 단계마다 새 process/open/close, 같은 host session과 source/build
- TurboQuant, SnapKV, experimental helper/tagged SDPA, threshold override: off

각 run은 `WARMUP_COMPLETE`, `RESULT_JSON`, `DEVICE_CLOSED`, exit 0을 확인했다. 재부팅 뒤 첫 workload인
32×32 add도 value 2.0, 정상 close, exit 0으로 통과했다.

## Waterfall

| 단계 | 추가 변경 | ms/token | tok/s | 직전 대비 | vanilla 대비 |
|---|---|---:|---:|---:|---:|
| 1 | vanilla-equivalent, K128 | 195.148988 | 5.124290 | 기준 | 기준 |
| 2 | SDPA K256 + dual-NoC + 6 endpoints + pair/bank balance | 155.897821 | 6.414458 | +25.18% | +25.18% |
| 3 | + MLP DRAM-sharded fanout-2/tagged, W2 block16 | 130.834474 | 7.643245 | +19.16% | +49.16% |
| 4 | + grouped concat, DRAM-sharded QKV/Wo | 121.834702 | 8.207842 | +7.39% | +60.18% |

### 단계 1: vanilla-equivalent

- K chunk 128
- dual-NoC/endpoint override off
- grouped concat/QKV/Wo projection optimization off
- DRAM-sharded MLP off

### 단계 2: SDPA

- K chunk 256
- 16 active readers for 8 KV heads
- endpoint loads `3/2/3/3/3/2`
- NoC reader loads `8/8`
- pair-balanced, physical-bank-balanced

SDPA 단계만으로 full-model throughput이 `+25.18%` 증가했다. 이 값은 isolated SDPA bandwidth 증가와
다른 end-to-end 지표다. 발표에서 둘을 같은 막대에 섞지 않는다.

### 단계 3: MLP

- 6 DRAM interface workers
- 12 reader/compute workers
- fanout-2, endpoint groups `4:4:4`
- tagged weight reads
- W2 `in0_block_w=16`
- 16 KiB read-page cap

MLP 추가 이득은 직전 대비 `+19.16%`, vanilla 대비 누적 `+49.16%`다.

### 단계 4: attention projections

- grouped SDPA concat
- QKV DRAM-sharded balanced fanout-2
- Wo L1 input program + DRAM-sharded balanced fanout-2
- exact Wo-input fallback off

QKV/Wo 추가 이득은 직전 대비 `+7.39%`; 최종 누적은 `+60.18%`다.

## 과거 독립 run 재현성

| 구성 | 과거 tok/s | 이번 tok/s | 편차 |
|---|---:|---:|---:|
| vanilla K128 | 5.126868 | 5.124290 | -0.050% |
| SDPA+MLP stable | 7.645991 | 7.643245 | -0.036% |
| final stable | 8.213245 | 8.207842 | -0.066% |

세 기준점 모두 편차가 0.07% 이하다. 이번 waterfall의 양 끝과 주요 중간점이 독립 run에서 재현됐다.

## 발표 그래프 권장

### Throughput waterfall

```text
Vanilla       5.124 tok/s
+ SDPA        6.414 tok/s  (+25.18% incremental)
+ MLP         7.643 tok/s  (+19.16% incremental)
+ QKV/Wo      8.208 tok/s  (+7.39% incremental)
```

그래프 마지막 막대에 `+60.18% vs vanilla`를 크게 표시한다. 각 단계 위에는 incremental 값을 작게
표시한다.

### 별도 기술 슬라이드

- SDPA: isolated latency와 effective K/V GB/s
- MLP: projection latency, input wait, weight cadence
- QKV/Wo: 각 kernel latency 감소

end-to-end tok/s와 isolated GB/s를 같은 축에 놓지 않는다.

## 정확성

이번 run은 throughput 측정이다. 단계별 final token은 동일하지 않으므로 이 run 자체를 correctness
증거로 사용하지 않는다. 기존 fixed-input gate에서 final stable 계열은 다음을 보였다.

- optimized-vs-vanilla full logits PCC: 0.999326, top-1 동일, top-5 5/5
- QKV/Wo stable-vs-exact-fallback PCC: 0.9992725, top-1 동일, top-5 5/5
- bitwise exact 아님

발표에는 `PCC≈0.9993, same top-1/top-5, not bit-exact`를 함께 적는다.

## 한계

- 각 단계 50-token measurement 1회다. error bar가 없다.
- actual 64K prefill이 아니라 synthetic zero KV다.
- 단계별 새 process라 model load/JIT는 측정 구간에서 제외되지만 process-local runtime 상태는 다르다.
- stage 4의 QKV/Wo 묶음은 개별 operator 기여도를 분리하지 않는다.
- threshold8 후보는 포함하지 않았다.

## 재현

runner:

```text
/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/benchmark_llama32_3b_64k_decode.py
```

공통 command:

```bash
timeout --signal=INT --kill-after=15s 300s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/benchmark_llama32_3b_64k_decode.py --label <stage>
```

단계별 exact environment와 모든 stdout은 artifact log에 있다.

## Artifact

- run root: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_waterfall_2026_08_09_16_57_00`
- vanilla: `01_vanilla_k128.log`
- SDPA: `02_sdpa_k256_6ep.log`
- SDPA+MLP: `03_sdpa_plus_mlp.log`
- final stable: `04_final_stable.log`

모든 log는 정상 close와 exit 0 결과다. Tracy/Visualizer artifact는 생성하지 않았다.

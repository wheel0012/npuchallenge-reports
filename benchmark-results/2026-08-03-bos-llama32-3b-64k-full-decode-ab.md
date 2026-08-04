# BOS Llama 3.2 3B 64K full-model decode A/B

측정일: 2026-08-03 UTC

## 결과

재부팅 뒤 32×32 BF16 add 안전 게이트가 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과했다.
그 뒤 동일한 28-layer Llama 3.2 3B full-model decode runner로 vanilla와 검증된 SDPA+MLP 최적화
구성을 각각 한 process에서 측정했다.

| 구성 | 50 tokens elapsed (s) | ms/token | tokens/s |
|---|---:|---:|---:|
| vanilla K128 | 9.759053 | 195.181062 | 5.123448 |
| SDPA 6-endpoint K256 + MLP balanced fanout-2 | 6.539374 | 130.787490 | 7.645991 |

- latency 감소: 32.9917%
- throughput 증가: 49.2353%
- speedup: 1.49235×

두 run 모두 warmup, measured 50 tokens, `RESULT_JSON`, `DEVICE_CLOSED`와 정상 driver close를
완료했다. timeout, signal 종료, Watcher 또는 profiler는 없었다.

## 공통 조건

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- model: `meta-llama/Llama-3.2-3B-Instruct`, 28 layers
- batch: 1
- context: zero-initialized synthetic paged KV, max sequence 65,536
- measured decode positions: 65,486--65,535
- paged KV block size: 32
- warmup: 3 tokens
- measured: 50 tokens
- trace/profiler/Watcher: off
- timeout: `timeout --signal=INT --kill-after=15s 300s`, direct child Python

runner:

```text
/home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/benchmark_llama32_3b_64k_hang_wa.py
```

공통 실행 형태:

```bash
cd /home/iris_hb4/tt-metal-hb4
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
    PYTHONPATH=/home/iris_hb4/tt-metal-hb4 \
    HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
    <A/B 환경변수> \
    timeout --signal=INT --kill-after=15s 300s \
    /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
    /home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/benchmark_llama32_3b_64k_hang_wa.py \
    --label <label>
```

## A/B 차이

vanilla는 `TT_METAL_LLAMA32_SDPA_DECODE_K_CHUNK_SIZE=128`이고 SDPA dual-NoC 및 MLP DRAM-sharded
opt-in을 모두 껐다.

optimized는 다음 검증 구성을 사용했다.

- SDPA K chunk 256
- active SDPA reader 16, endpoint load `3/2/3/3/3/2`, NoC0/NoC1 `8/8`
- pair-balanced 및 bank-balanced 6-endpoint address mapping
- MLP DRAM-sharded weights, W2 `in0_block_w=16`
- 16 KiB read-page cap; 실제 W1/W3 및 W2 page는 12,672/8,704 bytes
- balanced fanout-2: NOC1 logical endpoint groups `4:4:4`, 12 readers/12 compute
- fanout-3, split-kernel-only, prefetch helper, reduce-only helper, six-reader relay와 TurboQuant off

full-model K256 설정을 위해 기본값을 보존하는
`TT_METAL_LLAMA32_SDPA_DECODE_K_CHUNK_SIZE` opt-in을 `model_config.py`에 추가했다. 값은 power-of-two
및 32 이상을 검증한다. source checksum은
`7c05b3b52072e6a6025c1bdca56b852f40e8da2e5faf8c71ce66366557c48af6`이다.

## Artifact

- vanilla: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_decode_ab_2026_08_03_14_30_00/vanilla/run.log`
- vanilla SHA-256: `b0e6726af556d6d6ff44737ecb49d96952a4803034809be18c62f14bb4450dc5`
- optimized: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_decode_ab_2026_08_03_14_30_00/optimized/run.log`
- optimized SHA-256: `93b11e8d5ed65c728164f92bcc6303c2d775c9dbe0158fb6dbba5fe66fa12f3c`
- K-chunk opt-in patch: `/home/iris_hb4/tmp/codex-patches/20260803-144500-llama32-sdpa-k-chunk-optin.patch`
- patch SHA-256: `6ac3b114b60f62c096d61f47e8d702b85c004c032c981b61dab9999bcded0397`

## 한계와 해석

- 한 process씩 수행한 50-token sample이므로 session 반복 분산은 측정하지 않았다.
- synthetic zero KV로 실제 64K prefill 비용과 실제 prompt의 KV locality는 포함하지 않는다.
- `enable_trace=False`이므로 measured loop에는 각 token의 host-side 호출 및 token readback이 함께 포함된다.
- vanilla final token은 1131, optimized final token은 499로 달랐다. isolated single-layer A/B는 이전에
  bit-exact였지만, 이번 run은 full-model logits PCC나 생성 token 동등성을 검증하지 않았다. 따라서
  49.24%는 동일 shape/workload의 성능 결과이며 full-model correctness 승인으로 사용하지 않는다.
- optimized weight 적재는 layer당 별도 DRAM-sharded decode weight를 준비해 vanilla보다 오래 걸렸지만,
  모델 적재와 JIT/warmup 시간은 measured 50-token 구간에서 제외했다.

## 후속: 고정 입력 full-model logits 검증

autoregressive token feedback의 연쇄 효과를 제거하기 위해 별도 process에서 page-table seed 0, token ID
1, current position 65,535, zero-initialized paged KV를 고정했다. sampling과 trace를 끄고 한 decode
step의 전체 128,256-vocabulary FP32 host logits를 저장했다. vanilla와 optimized는 성능 A/B와 동일한
환경 차이만 사용했다.

| 지표 | 결과 |
|---|---:|
| shape/dtype | `[1,1,128256]` / FP32 |
| bitwise equal | false |
| differing elements | 105,754 / 128,256 (82.4554%) |
| max absolute difference | 0.3125 |
| mean absolute difference | 0.0548843 |
| RMSE | 0.0691399 |
| PCC | 0.9993256 |
| cosine similarity | 0.9991258 |
| top-1 | both token 320 |
| top-5 set overlap | 5/5 |
| top-10 set overlap | 9/10 |

vanilla top-1/top-2는 320/482, logits 10.75/8.75, margin 2.0이다. optimized는 같은 320/482,
logits 10.875/9.0, margin 1.875다. 따라서 새 quantization을 추가하지 않았더라도 K-chunk, reduction 및
matmul block 순서 차이 때문에 bitwise exact는 아니다. 이 한 샘플에서는 top-1과 top-5가 보존되고
PCC가 높으므로 `fixed-input top-1 preserving numerical approximation`으로 분류한다. 모델 전체의
accuracy-preserving 결론에는 여러 token/position과 실제 KV prompt 평가가 더 필요하다.

artifact:

- runner: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/tests/run_llama32_3b_64k_full_logits.py`
- runner SHA-256: `00d78fffeeb51f72c6a5f74487ee046ae8560045919862fbc8c9dcff7a378fd1`
- vanilla log: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_logits_ab_2026_08_03_15_20_00/vanilla/run.log`
- vanilla log SHA-256: `0a89b576f4fa11ee264d19e9dd95019b2aada5e0f3eea05ca0c7904be6f02878`
- vanilla logits SHA-256: `aad172cf09b14df0aae957c7238fb6f2486d8662dbe9d5858a6d2d3614fbc037`
- optimized log: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_logits_ab_2026_08_03_15_20_00/optimized/run.log`
- optimized log SHA-256: `405dfc3f939d6f8df7aff6ca7ecd8ceffc4942bda06383e98a70363c536948c3`
- optimized logits SHA-256: `9c8e4529e8fac726f71d4855d4f2729cb865331e22035d45819efd289bdd6624`

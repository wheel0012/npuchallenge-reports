# BOS Llama 3.2 3B 64K best stable vs vanilla toks/s

## 결론

동일한 28-layer full-model curpos-only decode runner에서 현재 최고 stable 조합은 vanilla보다
throughput이 51.79% 높았다.

| 구성 | 50 tokens elapsed | ms/token | tokens/s |
|---|---:|---:|---:|
| vanilla K128 | 9.752543 s | 195.050857 | 5.126868 |
| best stable | 6.424898 s | 128.497954 | 7.782225 |
| 변화 | -3.327645 s | **-34.12%** | **+51.79%** |

Speedup은 약 `1.518×`다. 두 run 모두 warmup 3 tokens, measured 50 tokens, exit 0,
`RESULT_JSON`, `DEVICE_CLOSED`를 완료했다.

## Hardware와 workload

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20`
- physical DRAM: 3 banks
- worker NoC endpoints: bank당 2, 총 6
- model: `meta-llama/Llama-3.2-3B-Instruct`, 28 layers
- batch: 1
- maximum context: 65,536
- measured positions: 65,486--65,535
- KV: paged, block 32, zero-initialized synthetic context
- profiler, Watcher, trace: off
- timeout: SIGINT 300 s, SIGKILL grace 15 s

Runtime P100/P150 문자열은 custom BOS board identity로 사용하지 않았다.

## Vanilla

```text
SDPA K chunk: 128
SDPA dual-NoC/endpoint override: off
SDPA active cores: vanilla 8 KV heads × 2 cores/head = 16
grouped concat: off
MLP DRAM-sharded: off
TurboQuant: off
```

Result:

```json
{
  "elapsed_s": 9.752542873,
  "ms_per_token": 195.05085746,
  "tokens_per_second": 5.126868002644258,
  "final_token": 1131
}
```

## Best stable

SDPA:

```text
K chunk: 256
active readers/compute: 16
dual-NoC: 8/8
6 endpoint load: 3/2/3/3/3/2
pair-balanced: on
bank-balanced: on
three-way 12-core: off
six-reader relay/helper: off
```

Post-SDPA:

```text
12-core, 2 heads/core grouped concat: on
exact Wo-input bridge: on
L1-sharded Wo and gather-Wo prototypes: off
```

MLP:

```text
DRAM-sharded weights
W2 in0_block_w: 16
16 KiB read-page cap
balanced fanout-2: 12 readers / 12 compute
NOC1 destination groups: 4:4:4
tagged two-block, depth 2
helper/fanout-3/fanout-16/endpoint-local: off
```

Result:

```json
{
  "elapsed_s": 6.424897676,
  "ms_per_token": 128.49795351999998,
  "tokens_per_second": 7.782225106366037,
  "final_token": 499
}
```

## Correctness 의미

Grouped exact Wo-input 경로는 stable optimized generic-concat 경로와 비교한 별도 28-layer gate에서
final logits bit-exact와 PCC 1.0을 통과했다.

그러나 vanilla와 전체 optimized 조합은 reduction/block 순서가 다르다. 기존 fixed-input gate는
PCC 0.9993256과 동일 top-1을 보였지만 bitwise exact가 아니었다. 이번 autoregressive run도 final token이
vanilla 1131, optimized 499로 다르다. 따라서 이 문서의 51.79%는 동일 shape의 throughput 비교이며
모델 품질 동등성 승인이 아니다.

## 재현 runner

```text
/home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/benchmark_llama32_3b_64k_hang_wa.py
```

공통 실행:

```bash
timeout --signal=INT --kill-after=15s 300s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  /home/iris_hb4/benchmark_runs/llama32_3b_64k_hang_wa_2026_08_02/benchmark_llama32_3b_64k_hang_wa.py \
  --label <label>
```

## Artifact와 한계

- 이번 run은 profiler-free stdout 측정이다. 별도 raw log 파일은 만들지 않았다.
- Exact `RESULT_JSON` 값은 위에 보존했다.
- 모델 load, weight conversion, JIT와 warmup은 elapsed 구간에서 제외했다.
- 실제 65,535-token prefill은 하지 않았다.
- 한 process씩 한 번 측정했다. session 간 분산은 미측정이다.
- 기존 2026-08-03 최고 stable 값은 7.645991 tok/s였다. Grouped exact를 포함한 이번 값은
  7.782225 tok/s다. 서로 다른 process/day 단일 sample이므로 이 1.78% 차이만 독립 효과로 단정하지 않는다.

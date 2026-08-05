# BOS SDPA grouped concat full-model exact gate

- Date: 2026-08-05
- Device: custom 20-core BOS NPU
- Runtime/code architecture: Blackhole
- Available worker grid: `5×4 = 20 cores`
- Physical DRAM: 3 banks, 6 worker NoC endpoints
- Status: grouped concat rejected from stable full-model configuration

## Question

Single-layer에서 실제 batch row가 bit-exact이고 layer wall이 줄었던 12-core, 2-heads/core grouped concat가
stable 6-endpoint SDPA와 12-compute MLP를 포함한 Llama 3.2 3B 28-layer decode에서도 exact한지 확인했다.

## Configuration

```text
model: meta-llama/Llama-3.2-3B-Instruct
batch: 1
current_pos: 65535
token_id: 1
seed: 0
page_block_size: 32
SDPA K chunk: 256 tokens
SDPA endpoint loads: 3/2/3/3/3/2
SDPA NoC0/NoC1 loads: 8/8
MLP fanout: 2
MLP readers/compute workers: 12/12
MLP endpoint groups: 4:4:4
MLP read-page cap: 16 KiB
TurboQuant: off
profiler: off
```

Grouped concat 이외 환경변수와 입력은 동일했다. OFF control을 독립 process로 두 번 실행했다. 모든 run은
exit code 0과 `DEVICE_CLOSED`를 확인했다.

## Observations

### Stable 6-endpoint single-layer interaction

두 독립 reverse-order pair, 각 warmup 3회, 5 iterations, 7 repeats의 pooled 결과다.

| 경로 | mean layer | median layer | mean SDPA wall |
|---|---:|---:|---:|
| generic concat | 5.343060 ms | 5.350916 ms | 2.225453 ms |
| grouped concat | 5.211358 ms | 5.247701 ms | 2.230472 ms |
| 변화 | -2.46% | -1.93% | +0.23% |

실제 batch row는 bit-exact였다. 이 개선은 KV reader가 아니라 post-SDPA concat/layout 경로에서 나왔다.

### Full 28-layer fixed logits

| 비교 | exact | PCC | cosine | max abs | mean abs | changed | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF vs OFF repeat | true | 1.0 | 1.0 | 0 | 0 | 0 | 동일 |
| OFF vs ON | false | 0.999314 | 0.999355 | 0.3125 | 0.046367 | 99,701/128,256 | 동일, 320 |

OFF SHA-256는 두 run 모두
`0231c08b883443bbb55bf3ee13e32bcb3419935550a6c24cd4fadebae75c5241`이었다. ON SHA-256는
`179462fa2a2fdcb784e1ad66c72bf440af49ef48c14915254058c9aa8dc3ddcd`였다.

## Interpretation

### Fact

- Independent-process baseline은 deterministic하고 bit-exact다.
- Grouped concat를 켠 full-model logits만 변한다.
- top-1 token은 유지되지만 exact baseline 요구를 통과하지 못한다.
- 따라서 full-model tokens/s A/B는 진행하지 않았다.

### Inference

Single-layer actual row exact 검사는 full-model consumer 계약을 완전히 검증하지 못했다. 작은 layer-local 차이,
padding row 값 또는 shard/layout 해석 차이가 Wo와 residual stream을 거치며 누적될 수 있다.

### Unverified hypotheses

- Wo reader가 actual row 외 padding 영역 일부를 읽는다.
- grouped output shard width 또는 tile ownership이 generic concat와 완전히 같지 않다.
- 첫 layer부터 차이가 있으나 single-layer probe가 Wo 이후 residual output을 비교하지 않았다.

## Decision and next experiment

`TT_METAL_SDPA_DECODE_GROUPED_CONCAT=1`은 opt-in 상태로만 유지한다. stable full-model configuration에는
넣지 않는다. 다음 실험은 fixed input으로 layer 0부터 27까지 Wo 이후 residual을 저장해 first-divergence
layer를 찾고, 해당 layer에서 grouped output 전체 32 rows와 Wo reader tile 범위를 대조하는 것이다.

## Artifacts

```text
/home/iris_hb4/profiler_runs/sdpa_l1_output_ab_2026_08_05/
/home/iris_hb4/profiler_runs/llama32_3b_64k_grouped_concat_full_ab_2026_08_05/
  grouped_off_logits.pt
  grouped_off_repeat_logits.pt
  grouped_on_logits.pt
```

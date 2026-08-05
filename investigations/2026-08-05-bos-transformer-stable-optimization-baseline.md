# BOS Llama 3.2 3B transformer stable optimization baseline

날짜: 2026-08-05 UTC

## 1. 결론

현재 가장 강한 통합 실행 증거를 가진 64K decode 기준선은 다음 조합이다.

1. SDPA: K chunk 256, 16 active readers, dual NoC, 6 DRAM worker endpoints.
2. MLP: DRAM-sharded weights, W2 `in0_block_w=16`, 16 KiB read-page cap.
3. MLP reader/compute: balanced fanout-2, 12 readers/12 compute, destination `4:4:4`.
4. MLP weight pipeline: tagged two-block, pending depth 2.
5. 모든 helper, fanout-3, endpoint-local MLP, six-reader SDPA relay, TurboQuant, fused matmul epilogue는 off.

이 문서에서 `stable`은 production 기본값이나 광범위한 모델 정확도 승인을 뜻하지 않는다. 아래 조건을
모두 만족한 현재의 **stable evidence baseline**을 뜻한다.

- profiler/Watcher 없는 full-model 64K decode가 warmup과 measured loop를 완료했다.
- `RESULT_JSON`, `DEVICE_CLOSED`, 정상 driver close가 있었다.
- isolated SDPA·MLP correctness와 latency가 재현됐다.
- 실패 또는 미검증 경로와 독립된 opt-in 조합이다.

Full-model 성능은 vanilla K128의 5.123448 tok/s에서 7.645991 tok/s로 증가했다. Throughput 증가는
49.2353%, latency 감소는 32.9917%다. 다만 fixed-input full-model logits는 bitwise exact가 아니며
PCC 0.9993256이다. 따라서 이 기준선은 `fixed-input top-1 preserving numerical approximation`이지
exact transformer 구현이 아니다.

## 2. 하드웨어와 용어

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20 cores`
- physical DRAM: 3 banks
- worker-visible DRAM NoC endpoints: bank당 2개, 총 6개
- 모델: `meta-llama/Llama-3.2-3B-Instruct`, 28 decoder layers
- 주 workload: batch 1, 64K paged decode, head dim 128

Program grid, active compute core, physical DRAM bank, logical DRAM view, endpoint와 interface worker는 서로
다른 값이다. 특히 `Dram Interface Workers: 6`은 선택된 MLP data path의 interface-worker 수다. Physical
bank가 6개이거나 active compute가 6개라는 뜻이 아니다. Runtime의 P100/P150 문자열도 custom BOS의
authoritative board identity가 아니다.

### 실제 active core

| stage | available/program 범위 | 실제 active reader/compute | 근거 |
|---|---:|---:|---|
| SDPA | 5×4 가능 | 16 | 8 KV heads × 2 cores/head |
| MLP stable fanout-2 | 20 program cores | 12 readers / 12 compute | active FPU counter와 runtime assignment |

20-core 설정값만 보고 20 active compute라고 쓰지 않는다. SDPA의 남은 4 grid 위치와 MLP의 non-worker
program cores는 해당 operation의 math-active core가 아니다.

## 3. Stable SDPA 구성

### 채택 구성

- paged/interleaved K/V cache
- K chunk 256 tokens
- 16 active readers 유지
- reader를 NoC0/NoC1에 `8/8` 분배
- 6 endpoint load `3/2/3/3/3/2`
- pair-balanced 및 bank-balanced endpoint assignment
- reducer/worker 구조 유지

대표 opt-in은 다음과 같다.

```bash
TT_METAL_LLAMA32_SDPA_DECODE_K_CHUNK_SIZE=256
TT_METAL_SDPA_DECODE_DUAL_NOC=1
TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6
TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1
TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1
TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0
```

Pair/bank balancing은 isolated SDPA에서 baseline보다 각각 0.38%/0.18% 느려 독립 성능 이득은 없었다.
하지만 검증된 full-model 기준선은 두 flag를 포함했고 endpoint route를 결정적으로 고정한다. 따라서
현재 재현 recipe에서는 유지하되 핵심 speedup 원인으로 주장하지 않는다.

### SDPA 성능

| 구성 | effective K/V BW | 대표 critical span | 판정 |
|---|---:|---:|---|
| vanilla K128 | 41.12 GB/s | 약 3.47 ms | 기능 기준점 |
| dual-NoC, 6 endpoint, K128 | 56.61 GB/s | 2.519 ms | 유효 |
| dual-NoC, 6 endpoint, K256 | 70.03 GB/s | 2.036 ms | stable 후보 |
| dual-NoC, 6 endpoint, K512 | 71.22 GB/s | 2.002 ms | 실험 옵션 |

K256은 vanilla K128 대비 effective bandwidth가 약 70.3% 증가했다. K512는 K256보다 1.7% 빠르지만
검증 부담과 L1/CB pressure가 증가해 stable 기준선으로 채택하지 않는다. 20-reader direct DRAM
microbenchmark의 86.83 GB/s는 compute와 paged addressing이 없는 saturation ceiling이므로 SDPA 수치와
같은 workload로 취급하지 않는다.

### SDPA에서 제외한 경로

- tagged cross-chunk prefetch: 성능 중립, 기본 off
- generic DRAM-sharded K/V: 약 24 GB/s대로 악화
- contiguous K/V: paging 제거 이득 약 0.4%, 실제 workload 대표성 부족
- six-reader full fanout/relay: ACK와 remote-CB traffic 증가, timeout 이력
- reduce-only helper: deadlock 이력, 금지
- TurboQuant 4-bit: 별도 operation opt-in, stable baseline에 포함하지 않음

## 4. Stable MLP 구성

### 채택 구성

```bash
TT_METAL_MLP_DRAM_SHARDED=1
TT_METAL_MLP_W2_IN0_BLOCK_W=16
TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1
TT_METAL_MLP_DRAM_SHARDED_READER_LOCALITY=0
TT_METAL_MLP_DRAM_SHARDED_FANOUT2=1
TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=1
TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=1
TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0
TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0
TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=0
TT_METAL_MLP_FUSED_GATE_UP=0
TT_METAL_TURBOQUANT=0
```

6 logical DRAM views마다 두 worker가 shard width 절반을 직접 읽고 같은 core에서 계산한다. 총
12 readers/12 compute다. NOC1 destination source를 `6:5:1`에서 `4:4:4`로 고친 것이 fanout-2의 핵심
개선이다. Helper relay 없이 consumer가 직접 읽으므로 remote-L1 왕복과 helper barrier가 없다.

Tagged depth-2는 현재 block과 다음 block의 request를 겹쳐 reader마다 future request 하나를 유지한다.
Depth-1은 overlap이 부족했고 depth-3은 이득이 없었다.

### MLP 성능 이력

| 구성 | mean latency | stable 12-compute 대비 | 판정 |
|---|---:|---:|---|
| vanilla interleaved | 2.213819 ms | +53.8% | 기능 baseline |
| DRAM-sharded 6 readers/6 compute | 1.879179 ms | +27.6% | 중간 단계 |
| 12 readers/6 compute helper | 1.556066 ms | +5.7% | 느림 |
| balanced 12 readers/12 compute | 1.472280 ms | +2.3% | fanout 기준 |
| balanced fanout-2 tagged depth-2 | 1.439071 ms | 기준 | stable |
| 18 readers/18 compute fanout-3 | 1.703471 ms | +18.4% | 폐기 |

Endpoint balance 뒤 W1/W3/W2 direct bandwidth는 63.10/62.63/63.07 GB/s, 합산 62.93 GB/s다. 기존
`6:5:1` fanout의 48.60 GB/s보다 29.47% 높다. Microbenchmark 86.83 GB/s의 72.48%이며 남은 차이는
activation multicast, CB cadence, compute와 output reshard가 포함된 실제 matmul 비용이다.

### 왜 12 compute인가

12는 하드웨어 최대가 아니다. 실측에서 6 direct는 compute 직렬화, 6 helper는 remote L1 hop, 18 direct는
padding·route·multicast 비용으로 12보다 느렸다. Balanced 12-compute는 다음을 동시에 만족한 유일한
구성이다.

1. PCC 0.999641.
2. 12 active FPU cores 확인.
3. Destination `4:4:4`.
4. Direct weight ownership.
5. 정상 completion과 device close.

Consumer는 kernel 시간의 약 67%를 input CB에서 기다린다. W1은 주로 weight-late지만 W3/W2는 activation과
weight의 late side가 phase별로 바뀐다. Matrix engine TOPS ceiling보다 producer-consumer cadence가 먼저
막힌 상태다. Reader 수를 더 늘리는 것만으로 해결되지 않는다.

### MLP에서 제외한 경로

- 6-compute helper: remote L1 write와 compute 감소로 느림
- fanout-3 18-compute: stable fanout-2보다 느림
- fanout-3 dual-NoC: timeout 이력
- reader-packed multiblock: sustained 이득 없음
- K-block merge-2: mean 1.324% 악화
- tagged depth-3: mean 0.224% 악화
- tagged depth-1: mean 1.796% 악화
- activation credit depth-3: mean 1.921% 악화
- lane start stagger 256 cycles: mean 2.253% 악화
- fused gate/up 뒤 DRAM SwiGLU: mean 1.88% 악화
- fused matmul epilogue: 실제 device launch timeout, 미검증
- endpoint-local fanout-2: fixed-writer 뒤 PCC 실패, ordering 교정본 device 미검증

## 4A. 코드 dataflow와 논문 기법

### Decoder layer 전체 흐름

현재 Llama decoder layer의 큰 흐름은 다음과 같다.

```text
hidden x
  │
  ├─ RMSNorm
  │    └─ fused QKV linear → reshape/head split → RoPE(Q,K)
  │                                  │
  │                                  ├─ new K,V를 paged KV cache에 기록
  │                                  └─ Q + page table + cached K,V
  │                                         ↓
  │                              Flash Decode SDPA
  │                              chunk partial (m,l,O)
  │                                         ↓
  │                              reducer merge → concat heads → Wo
  │                                         ↓
  └──────────────────────────── residual add
                                           │
                                      RMSNorm
                                           │
                    W1 gate ── SiLU ─┐     │
                    W3 up ────────────⊙ ←───┘
                                      │
                                     W2
                                      │
                              residual add → next layer
```

Attention 내부의 `(m,l,O)` reducer와 tensor-parallel projection 뒤의 reduce를 같은 것으로 취급하지
않는다. 전자는 여러 K/V chunk 또는 worker가 만든 online-softmax state를 합치는 단계다. 후자는 Wo 등
projection shard의 부분 출력을 모델 hidden dimension 기준으로 합치는 단계다.

### SDPA host assignment

`sdpa_decode_program_factory.cpp`가 shape에서 active core 수와 reducer/output 역할을 계산한다. 현재
64K, 8 KV-head, 2 cores/head 구성은 16 active cores다. Stable dual-NoC 경로는 compute core 수를
줄이지 않는다. 각 active reader에 endpoint x를 할당하고 endpoint x에 대응하는 NoC0 또는 NoC1을
선택한다. 성공 run의 결과가 endpoint `3/2/3/3/3/2`, NoC `8/8`이다.

이 단계는 tensor layout을 6-way DRAM shard로 바꾸는 것이 아니다. Paged/interleaved K/V tensor는
유지하고, 각 reader가 사용할 worker-visible DRAM endpoint와 route를 바꾼다. 따라서 16 readers가 모두
자기 K/V chunk를 직접 local CB로 가져온다. Six-reader owner/relay POC와 다른 구조다.

### Paged K/V reader

`reader_decode_all.cpp`의 stable paged path는 다음 순서다.

1. Page table을 DRAM 또는 sharded L1에서 CB `c9`로 읽는다.
2. Logical sequence tile row를 page-table entry와 block offset으로 physical K/V tile ID에 변환한다.
3. K는 page mapping이 쉬운 row-major 순서로 요청하되 local CB에는 matmul이 원하는 transposed tile
   순서로 배치한다.
4. V도 같은 logical chunk 범위에서 physical page를 따라 local CB로 읽는다.
5. `cb_reserve_back`으로 producer 공간을 확보하고 NoC read barrier 뒤 `cb_push_back`으로 compute에
   chunk 완성을 알린다.

Paged KV의 logical 순서는 연속이어도 physical block은 비연속일 수 있다. TensorAccessor와 page-table
translation이 정확한 주소를 만들지만, DRAM row locality와 큰 contiguous burst는 보장하지 않는다.
이번 SDPA 최적화가 endpoint 분산과 K chunk 확대에 집중한 이유다.

### Flash Decode compute

`sdpa_flash_decode.cpp`는 64K 전체 attention score matrix를 DRAM이나 L1에 만들지 않는다. K256이면
sequence tile 8개씩 K/V를 streaming한다. 각 chunk에서 다음을 수행한다.

1. `Q @ K_chunk`로 score tile을 만든다.
2. 마지막 causal chunk 또는 dynamic boundary에 mask를 적용한다.
3. Chunk row max `m_i`를 구한다.
4. `exp(score - m_i)`와 chunk sum `l_i`를 구한다.
5. `P_chunk @ V_chunk`로 chunk output `O_i`를 만든다.
6. 이전 running `(m,l,O)`를 새 max 기준으로 rescale하고 현재 chunk와 합친다.

개념식은 다음과 같다.

```text
m_new = max(m_prev, m_i)
l_new = exp(m_prev-m_new)·l_prev + exp(m_i-m_new)·l_i
O_new = exp(m_prev-m_new)·O_prev + exp(m_i-m_new)·O_i
output = O_new / l_new
```

실제 kernel은 BF16/BFP8 tile, CB와 matrix engine API로 이 recurrence를 수행한다. 알고리즘은 full
softmax와 같은 값을 목표로 하지만 tile reduction과 accumulation 순서가 바뀌므로 hardware 결과가
bitwise identical일 필요는 없다.

### SDPA reducer와 writer

각 worker가 담당 chunk 범위의 partial `(m,l,O)`를 만든다. `writer_decode_all.cpp`는 semaphore와 NoC
주소로 partial state를 reducer core에 전달한다. Reducer는 local partial과 remote partial을 읽고 global
max 기준으로 다시 rescale해 합친다. Output core가 여러 reducer 결과를 모아 최종 head output을 만든다.

이 구조 때문에 reader bandwidth만 높여도 전체 latency가 같은 비율로 줄지 않는다. K/V read, QK/PV
matmul, CB wait, partial-state transfer와 reducer tail이 같은 critical path에 있다.

### MLP activation dataflow

Stable MLP는 W1, W3, W2 각각을 DRAM-sharded matmul로 실행한다. In0 activation sender kernel은 다음
K block을 local CB로 읽은 뒤 receiver semaphore를 확인하고 destination cores에 NoC multicast한다.
Blackhole에서는 multicast source data가 재사용되기 전에 `noc_async_writes_flushed()`로 command 발행을
보장한다. Receiver semaphore가 block availability를 나타내며 compute는 in0 CB와 in1 CB가 모두 준비된
뒤 matmul block을 소비한다.

W1과 W3는 같은 normalized activation에 의존하지만 서로 독립 projection이다. SwiGLU 뒤 W2는
`SiLU(W1) ⊙ W3` 결과에 의존한다. 이 dependency는 projection 사이에 존재한다. 같은 projection의
activation block과 weight block은 overlap할 수 있으므로 전체 MLP가 필연적인 full barrier인 것은 아니다.

### MLP weight dataflow

`matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp`는 6 logical DRAM views와 12 worker의
weight/output ownership을 만든다. Fanout-2에서 view마다 lane 0/1 두 reader가 shard row의 서로 다른
연속 column 구간을 직접 읽는다. `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`의
`source_row_tile + reader_lane * in1_block_w`가 두 lane의 분할을 만든다.

Tile별 짧은 command 대신 K-row의 `in1_block_w` tiles를 한 NoC burst로 읽는다. W2 block 16은 activation
K block 폭과 weight row burst를 키워 command·barrier 빈도를 줄인다. Read-page cap은 최대 16 KiB이며
실제 page는 tensor shape와 tile size에 맞춰 나뉜다.

### Tagged two-block pipeline

Stable tagged depth-2는 단순한 동기식 double buffer가 아니다. Reader kernel은 세 L1 buffer slot과 두
transaction IDs를 사용한다.

```text
issue block n (TRID 1) ───────────────┐
issue block n+1 (TRID 2) ────────┐   │  최대 2개 DRAM transaction outstanding
wait TRID 1 → CB push block n    │   │
issue block n+2 in freed slot ───┼───┘
wait TRID 2 → CB push block n+1 ─┘
```

Completion은 TRID별 barrier로 확인하고 CB에는 program order로 publish한다. Compute가 현재 block을
소비하는 동안 reader가 다음 block을 DRAM에 요청할 수 있다. Depth-1은 이 overlap이 부족했고 depth-3은
추가 outstanding request가 service latency를 줄이지 못했다.

### Matmul compute와 output reshard

각 worker는 직접 읽은 weight lane과 multicast 받은 activation을 matrix engine으로 누산한다. Compute
output은 per-core N partition이다. In1 reader와 writer가 결합된 dataflow kernel은 output CB를 기다린 뒤
factory가 계산한 storage-core offset과 byte count에 따라 NoC write-back한다. Worker output 폭과 최종
sharded tensor 폭이 다르면 한 worker 결과가 하나 또는 두 storage cores로 나뉜다.

Stable fanout-2는 physical reader 배치, logical weight lane, output-column ownership과 reshard 순서가
이미 일치하는 검증 경로다. Endpoint-local 실험에서 이 네 계약을 한 vector 순서로 함께 바꾸자 PCC 또는
completion이 깨졌다. Reader를 endpoint 가까이 옮기는 일은 주소 route만의 문제가 아니다.

### 주요 코드 위치

| 역할 | repository 상대 경로 |
|---|---|
| Llama attention orchestration | `models/bos_model/llama32/tt/attention.py` |
| Llama MLP orchestration | `models/bos_model/llama32/tt/mlp.py` |
| SDPA core/endpoint assignment | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` |
| paged K/V reader | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp` |
| Flash Decode online softmax | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` |
| partial-state reducer/writer | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp` |
| MLP DRAM-sharded factory | `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp` |
| activation reader/multicast | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` |
| weight reader/tagged pipeline/output writer | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` |

### 논문 기법과 현재 구현의 관계

| 논문 기법 | 핵심 아이디어 | 현재 BOS 코드에서의 역할 | 이번 최적화와 관계 |
|---|---|---|---|
| Transformer | attention과 position-wise gated MLP를 residual block으로 반복 | `attention.py`, `mlp.py`가 decoder layer 구성 | 최적화 대상의 상위 계산 그래프 |
| FlashAttention | SRAM tiling과 online softmax로 full attention matrix의 HBM 왕복 제거 | `sdpa_flash_decode.cpp`가 chunk `(m,l,O)` recurrence 수행 | K256은 chunk당 work와 CB granularity 확대 |
| PagedAttention | logical KV block을 page table로 non-contiguous physical block에 매핑 | `reader_decode_all.cpp`가 virtual tile row를 physical tile ID로 변환 | 메모리 절약·유연성 대신 주소 변환과 locality 비용 존재 |
| GQA | 여러 Q heads가 더 적은 KV heads를 공유 | 24 Q heads가 8 KV heads를 공유하고 KV head당 2 cores 사용 | KV cache/read traffic을 MHA보다 줄이지만 16-core mapping 결정 |
| RoPE | Q/K vector에 position-dependent rotation 적용 | 새 Q/K 생성 뒤 attention과 KV-cache update 전에 적용 | 64K 위치 정보를 score에 반영, DRAM 최적화와 직교 |
| RMSNorm | mean subtraction 없이 RMS로 hidden state scale 정규화 | attention 전과 MLP 전에 hidden input 정규화 | 작은 reduction/scale stage, 주 DRAM weight 병목과 별개 |
| SwiGLU | `SiLU(xW1) ⊙ xW3` gated MLP 뒤 W2 projection | stable MLP의 W1/W3/SwiGLU/W2 dependency | local SwiGLU opt-in은 intermediate DRAM 왕복을 L1로 치환 |

Primary references:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

FlashAttention과 PagedAttention은 같은 기법이 아니다. FlashAttention은 **어떻게 계산할지**를 바꾸고,
PagedAttention은 **KV를 어디에 저장하고 어떻게 주소를 찾을지**를 바꾼다. 현재 Flash Decode는 두 기법을
결합한다. Page table로 K/V 위치를 찾고, 찾은 K/V를 chunk streaming하면서 online softmax로 합친다.

Decode는 query length가 1이라 training/prefill FlashAttention처럼 여러 Q row가 같은 KV tile을 크게
재사용하기 어렵다. SRAM은 64K KV 전체를 cache하는 공간보다 현재 K/V chunk, QK intermediate와 running
`(m,l,O)` state를 유지하는 streaming scratchpad로 쓰인다. 그래서 chunk를 무작정 MB 단위로 키우는 것보다
NoC request granularity, CB capacity, matrix-engine cadence와 reducer tail의 균형이 중요하다.

## 5. Full transformer 결과

### 64K full-model decode

공통 조건은 28 layers, batch 1, synthetic zero paged KV, positions 65,486--65,535, warmup 3,
measured 50 tokens, profiler/Watcher off다.

| 구성 | elapsed | ms/token | tokens/s |
|---|---:|---:|---:|
| vanilla K128 | 9.759053 s | 195.181062 | 5.123448 |
| stable SDPA K256 + stable MLP | 6.539374 s | 130.787490 | 7.645991 |

- latency: -32.9917%
- throughput: +49.2353%
- speedup: 1.49235×

두 run 모두 warmup, measured loop, `RESULT_JSON`, `DEVICE_CLOSED`, 정상 driver close를 완료했다. 모델
load, weight preparation과 JIT는 측정 구간에서 제외했다. Synthetic zero KV라 실제 64K prefill 비용과
실제 prompt KV locality는 포함하지 않는다.

### Single-layer 교차검증

Vanilla K256 대비 SDPA 6-endpoint는 layer latency를 18.65% 줄였다. MLP까지 결합하면 26.66% 줄었다.
Single-layer output 비교는 SDPA와 MLP 모두 `MAX_ABS=0`, `PCC=1.0`이었다. 이 결과는 해당 isolated
shape에서 bit-exact였다는 뜻이며 full-model autoregressive exact를 보장하지 않는다.

## 6. 정확도와 exactness

Stable full-model 기준선은 quantization을 새로 추가하지 않았지만 K chunk, reduction과 matmul block
순서가 바뀐다. Fixed token/position, zero paged KV 한 step의 전체 FP32 logits 결과는 다음과 같다.

| 지표 | 결과 |
|---|---:|
| shape | `[1,1,128256]` |
| bitwise equal | false |
| differing elements | 105,754 / 128,256 |
| max / mean absolute difference | 0.3125 / 0.0548843 |
| RMSE | 0.0691399 |
| PCC / cosine | 0.9993256 / 0.9991258 |
| top-1 | 양쪽 token 320 |
| top-5 / top-10 overlap | 5/5 / 9/10 |

따라서 `exact`, `bitwise exact`, `accuracy-preserving production`이라고 부르지 않는다. 현재 증거는
한 fixed-input sample의 top-1 보존과 높은 PCC다. 실제 prompt, non-zero KV, 여러 token/position과
task-level accuracy suite가 필요하다.

## 7. Core-local SwiGLU 선택 옵션

`TT_METAL_MLP_FUSED_GATE_UP=1`은 W3/W1 matmul을 결합하고 SwiGLU intermediate를 DRAM에 쓰지 않고
L1에서 처리한다. Stable baseline에 대한 추가 성능은 작다.

| 구성 | isolated MLP mean | full decode |
|---|---:|---:|
| stable baseline | 1.444443 ms | 7.643404 tok/s |
| fused + local SwiGLU | 1.415214 ms | 7.666719 tok/s |
| 변화 | -2.02% | +0.305% |

Fixed-input full-model logits PCC는 0.999288, top-1은 같고 top-5 overlap은 5/5다. Bitwise exact가 아니며
실제 accuracy suite가 없다. 따라서 기능적으로 정상 종료하는 **approximation opt-in**으로 유지하고
stable exact baseline에는 포함하지 않는다.

## 8. 현재 소스와 실행 상태

Stable benchmark artifact 뒤에도 여러 default-off 실험 경로가 소스에 추가됐다. 현재 installed runtime에는
endpoint-local fixed-writer와 logical writer-order 분리 코드도 포함된다. 이 코드는
`TT_METAL_MLP_DRAM_SHARDED_FANOUT2_ENDPOINT_LOCAL=0`이면 stable path에 들어가지 않는다.

최신 endpoint-local ordering 교정본의 device 검증 시도는 다른 PID가 `CHIP_IN_USE_0_PCIe` lock을 잡아
kernel launch 전에 종료됐다. 이 exit 137은 교정본의 correctness나 hang 증거가 아니다. Stable baseline의
기존 성공 artifact도 무효화하지 않는다. Device availability와 code-path stability를 구분한다.

현재 working tree에는 별도 사용자 변경이 있을 수 있다. 재현 전에는 source/runtime checksum, 실제
loaded `_ttnncpp.so`, 모든 opt-in 값과 runtime marker를 다시 기록해야 한다. TurboQuant는 별도 operation이며
이 baseline에 자동 포함하지 않는다.

## 9. 재현 및 채택 gate

새 source/binary에서 stable baseline을 다시 주장하려면 다음 순서가 필요하다.

1. 다른 process의 `CHIP_IN_USE_0_PCIe` lock이 없는지 host-side 확인.
2. 안전 계약이 요구하면 32×32 add gate 통과.
3. profiler/Watcher 없는 isolated SDPA correctness 1회.
4. profiler/Watcher 없는 isolated MLP correctness 1회.
5. single-layer 64K SDPA+MLP A/B와 output 비교.
6. full-model 50-token latency run과 fixed-input logits 비교.
7. 위 단계가 모두 통과한 뒤에만 isolated NoC profile.

실행 로그에서 최소한 다음 marker를 확인한다.

- SDPA: dual-NoC enabled, endpoint count/load, NoC0/NoC1 load
- MLP: DRAM-sharded, W2 block 16, read-page size, readers 12, compute workers 12, endpoint `4:4:4`
- 종료: warmup completion, result/PCC, `DEVICE_CLOSED`

다음 조건 중 하나라도 있으면 stable 결과로 분류하지 않는다.

- profiler artifact 불완전
- stale `_ttnncpp.so` 로드
- PCC/result/close marker 누락
- timeout, signal 종료, unexplained exit 124/137
- endpoint-local, helper, fanout-3, TurboQuant 또는 fused epilogue가 의도치 않게 활성

## 10. 근거 문서와 artifact

### 중앙 보고서

- `benchmark-results/2026-08-03-bos-llama32-3b-64k-full-decode-ab.md`
- `benchmark-results/2026-08-03-bos-64k-six-endpoint-sdpa-mlp-ab.md`
- `investigations/2026-08-01-sdpa-dram-performance-optimization-history.md`
- `investigations/2026-08-01-vanilla-sdpa-vs-dram-saturation-gap.md`
- `investigations/2026-08-04-bos-mlp-optimization-investigation.md`
- `investigations/2026-08-04-bos-mlp-vanilla-to-12-compute.md`
- `benchmark-results/2026-08-04-bos-mlp-compute-block-cadence.md`

### 주요 artifact

- Full decode A/B: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_decode_ab_2026_08_03_14_30_00`
- Full logits A/B: `/home/iris_hb4/profiler_runs/llama32_3b_64k_full_logits_ab_2026_08_03_15_20_00`
- Single-layer A/B: `/home/iris_hb4/benchmark_runs/llama32_3b_64k_6endpoint_ab_2026_08_02_18_30_00`
- SDPA K256: `/home/iris_hb4/benchmark_runs/sdpa_16reader_6endpoint_2026_08_03`
- MLP balanced fanout-2: `/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_balanced_noc_2026_08_03_09_15_00`
- MLP tagged depth-2: 관련 수치와 artifact는
  `benchmark-results/2026-08-03-bos-mlp-fanout2-tagged-two-block.md` 참조
- Local SwiGLU full decode:
  `/home/iris_hb4/profiler_runs/llama32_3b_64k_local_swiglu_ab_corrected_2026_08_04_11_00_00`

## 11. 최종 상태표

| 영역 | 구성 | 상태 |
|---|---|---|
| SDPA | dual-NoC + 6 endpoint + K256 | stable evidence baseline |
| SDPA | K512 | 실험 옵션 |
| SDPA | six-reader relay/reduce helper | 실패·금지 |
| MLP | DRAM-sharded + block16 + balanced fanout-2 + tagged depth-2 | stable evidence baseline |
| MLP | local SwiGLU | 정상 종료한 approximation opt-in |
| MLP | fanout-3/helper/packed/depth3/credit/stagger | 폐기 또는 보류 |
| MLP | endpoint-local writer-order 교정 | device 미검증 |
| MLP | fused matmul epilogue | launch timeout, 미검증 |
| Full transformer | SDPA stable + MLP stable | 7.645991 tok/s, numerical approximation |
| Full transformer | local SwiGLU 추가 | 7.666719 tok/s, accuracy 검증 부족 |

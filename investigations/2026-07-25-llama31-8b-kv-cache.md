# Llama 3.1 8B KV-cache 최적화 분석

- 분석일: 2026-07-25
- 실제 진입점: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32/run_llama32.py`
- 대상 모델: `meta-llama/Llama-3.1-8B-Instruct`
- 대상 코드: BOS `llama32` 모델 및 TTNN paged-cache/SDPA decode runtime
- 분석 방식: 소스 정적 추적 + 보존된 실제 Llama 3.1 8B 실행 artifact 교차 확인

## 1. 문서 정정 사항

이전 분석은 `test_demo_llama32.py`와 Llama 3.2 1B를 주 기준으로 삼았다. 실제 사용할 진입점이 `run_llama32.py`이고 대상 모델이 Llama 3.1 8B이므로 다음 사항을 바로잡는다.

1. `run_llama32.py`는 Llama 3.2 1B 전용이 아니다.
2. `load_model()`의 `model_id="meta-llama/Llama-3.2-1B-Instruct"` 기본값은 코드에서 사용되지 않는다.
3. 실제 모델은 `HF_MODEL` 또는 `LLAMA_DIR` 환경변수로 결정된다.
4. 저장소에는 Llama 3.1 8B용 config와 params가 존재하며, BOS 코드도 해당 모델을 명시적으로 처리한다.
5. `run_llama32.py`의 physical block 수는 `ceil(max_seq_len / 32)`이므로 기본 1K 실행에는 32 blocks만 할당된다.
6. 따라서 `test_demo_llama32.py`의 “1K 실행에 1024 blocks를 할당하여 32배 과할당”이라는 결론은 runner에는 적용되지 않는다.
7. 보존된 실제 실행 기록상 Llama 3.1 8B, batch 1, P150 단일 칩, 32K context 실행이 성공했다.

이전 `test_demo_llama32.py` 전용 분석은 다음 파일로 보존한다.

- `/home/iris_hb4/2026npu/readme_llama32_test_demo_analysis.md`

## 2. 핵심 결론

`run_llama32.py`로 Llama 3.1 8B를 실행할 때 적용되는 주요 KV-cache 최적화는 다음과 같다.

| 영역 | 적용 기능 | runner에서의 상태 |
|---|---|---|
| Attention 구조 | GQA: Q 32 heads, KV 8 heads | 활성 |
| 저장 정밀도 | BFP8_B KV-cache | 활성 |
| 저장 구조 | Paged KV-cache, 32-token block | 항상 활성 |
| Cache 수명 | 레이어별 K/V 사전 할당 및 영속 재사용 | 활성 |
| Cache 갱신 | prefill/decode in-place fill/update | 활성 |
| 멀티칩 | TP별 local KV-head 저장 | 멀티칩에서 활성 |
| Prefill | multicore bulk `paged_fill_cache` | 활성 |
| Prefill | 실제 prompt page까지만 cache write | 활성 |
| Prefill | 짧은 prompt에서 cache 즉시 read-back 회피 | 조건부 활성 |
| 장문 Prefill | chunked prefill + cache-backed chunked SDPA | prompt/device 조건부 |
| Decode 입력 | 현재 K/V의 L1 height sharding | 활성 |
| Decode 저장 | update kernel 내부 BFP8 pack/tilize | 활성 |
| 주소 변환 | 장치 측 virtual-to-physical page 해석 | 활성 |
| Decode Attention | Flash-Decode/online softmax cache streaming | 활성 |
| 런타임 | TTNN program cache 재사용 | 활성 |
| Trace | decode trace capture/replay | 기본 비활성, CLI 버그로 조건부 활성 |
| Layout | 32-token page와 TT tile 높이 정렬 | 활성 |

Llama 3.1 8B의 exact raw TT BFP8 tile payload는 다음과 같다.

- 기본 `max_seq_len=1024`: 전체 32 layers K+V 합계 **68 MiB**
- `max_seq_len=32768`: 전체 32 layers K+V 합계 **2,176 MiB = 2.125 GiB**
- 위 수치는 allocator metadata와 page-table storage를 제외한 TT packed tile payload다.

## 3. 실제 모델 선택 방식

### 3.1 `model_id`는 미사용

`run_llama32.py`의 함수 signature에는 다음 기본값이 있다.

```python
def load_model(
    mesh_device,
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    ...
):
```

하지만 `model_id`는 함수 내부에서 한 번도 참조되지 않는다.

근거:

- `models/bos_model/llama32/run_llama32.py:79-121`

따라서 source에 적힌 1B 문자열만 보고 실제 실행 모델을 판단하면 안 된다.

### 3.2 실제 모델은 환경변수가 결정

실제 checkpoint 선택 우선순위는 다음과 같다.

```text
LLAMA_DIR 또는 HF_MODEL
```

- 두 환경변수를 동시에 설정하면 assertion 실패
- `HF_MODEL`: Hugging Face repo ID 또는 HF-compatible 경로
- `LLAMA_DIR`: 로컬 Meta/HF checkpoint 절대 경로
- 둘 다 없으면 일반 runner 실행은 assertion으로 중단

근거:

- `models/bos_model/llama32/tt/model_config.py:459-486`

Llama 3.1 8B 실행 예:

```bash
cd /home/iris_hb4/tt-metal-hb4
source env_set.sh

HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python models/bos_model/llama32/run_llama32.py --live -n 5
```

로컬 checkpoint 실행 예:

```bash
LLAMA_DIR=/absolute/path/to/Llama-3.1-8B-Instruct \
python models/bos_model/llama32/run_llama32.py --live -n 5
```

### 3.3 저장소의 8B 지원 근거

Llama 3.1 8B local params:

- `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/params.json`
- `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`

BOS model config의 8B 처리:

- local params mapping: `models/bos_model/llama32/tt/model_config.py:404-409`
- 8B chunk threshold: `models/bos_model/llama32/tt/model_config.py:545-582`
- Meta checkpoint 이름 판정: `models/bos_model/llama32/tt/model_config.py:1548-1550`

저장소 README도 Llama 3.1 8B를 지원 모델로 명시한다.

- `models/bos_model/llama32/README.md:4-7`

## 4. Llama 3.1 8B 구조와 KV-cache 차원

Llama 3.1 8B text model의 주요 파라미터는 다음과 같다.

| 파라미터 | 값 |
|---|---:|
| Hidden dimension | 4096 |
| Decoder layers | 32 |
| Query heads | 32 |
| KV heads | 8 |
| Head dimension | 128 |
| HF max position embeddings | 131072 |
| GQA query/KV ratio | 4:1 |

근거:

- `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/params.json:1`
- `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json:14-22`

Llama 3.2 1B와 3.1 8B는 모두 Q 32/KV 8이지만 다음 값이 다르다.

| 모델 | Layers | Head dimension | Q heads | KV heads |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 16 | 64 | 32 | 8 |
| Llama 3.1 8B | 32 | 128 | 32 | 8 |

따라서 Q/KV head 수만으로 모델을 판별하거나 KV-cache 크기를 계산하면 안 된다. 8B는 layer 수가 2배이고 head dimension도 2배라 같은 sequence length에서 1B보다 KV-cache가 4배 크다.

## 5. `run_llama32.py`의 실제 실행 설정

### 5.1 기본값

| 설정 | runner 기본값 |
|---|---:|
| Batch | 1 |
| Data parallel | 1 |
| Max sequence length | 1024 |
| Max generated tokens | 256 |
| Page block size | 32 |
| Physical blocks | `ceil(max_seq_len/32)` |
| Paged Attention | 항상 구성 |
| KV dtype | BFP8_B |
| Precision profile | performance |
| Sampling | temperature 0, device greedy argmax |
| Context mode | `LOW_CONTEXT` |
| Command queues | 2 |
| Trace region | 10,419,200 bytes |
| Decode trace | CLI 기본 실행에서 비활성 |

근거:

- CLI defaults: `models/bos_model/llama32/run_llama32.py:29-62`
- batch 1 assertion: `models/bos_model/llama32/run_llama32.py:263-265`
- model load: `models/bos_model/llama32/run_llama32.py:277-281`
- mesh/CQ: `models/bos_model/llama32/run_llama32.py:182-200`
- main config: `models/bos_model/llama32/run_llama32.py:562-578`

### 5.2 Batch와 DP

CLI에 `--batch_size`는 있지만 runner는 다음 assertion으로 1만 허용한다.

```python
assert batch_size in [1]
```

`load_model()`의 `data_parallel` 기본값도 1이고 `llama_runner()`가 다른 값을 전달하지 않는다.

따라서 이 entrypoint의 실제 상태는:

```text
batch = 1
DP = 1
full visible mesh = model/tensor parallel mesh
```

### 5.3 Paged cache는 사실상 항상 활성

`load_model()`은 호출될 때마다 다음 설정을 직접 만든다.

```text
block_size = 32
max_num_blocks = ceil(max_seq_len / 32)
```

근거:

- `models/bos_model/llama32/run_llama32.py:87-91`

Main config의 `paged_attention=True` 여부와 별개로 모델 load 자체가 항상 `PagedAttentionConfig`와 page table을 생성한다. 따라서 직접 `llama_runner(..., paged_attention=False)`를 호출하더라도 검증 분기만 달라지고 모델은 여전히 paged 경로를 사용한다.

## 6. KV-cache shape와 physical page pool

### 6.1 일반 shape

각 decoder layer는 K와 V를 각각 다음 shape으로 가진다.

```text
[num_physical_blocks, n_local_kv_heads, block_size, head_dim]
```

Llama 3.1 8B TP1에서는:

```text
[ceil(max_seq_len/32), 8, 32, 128]
```

속성:

```text
dtype  = BFP8_B
layout = TILE
memory = DRAM interleaved
```

근거:

- `models/bos_model/llama32/tt/attention.py:345-399`

### 6.2 기본 1K 실행

```text
max_seq_len = 1024
blocks      = ceil(1024/32) = 32
K shape     = [32, 8, 32, 128]
V shape     = [32, 8, 32, 128]
```

### 6.3 32K 실행

```text
max_seq_len = 32768
blocks      = ceil(32768/32) = 1024
K shape     = [1024, 8, 32, 128]
V shape     = [1024, 8, 32, 128]
```

보존된 실제 실행 기록:

- `/home/iris_hb4/profiler_runs/llama31_8b_32k_decode_perf_2026_07_22_08_02_00/RUN.md`
- 모델: Llama 3.1 8B
- Batch: 1
- 실제 prompt: 32,032 tokens
- 32 layers × 10 decode steps 성공

실제 생성된 zero-cache tensor artifact도 shape와 dtype을 확인해 준다.

```text
model_cache/meta-llama/Llama-3.1-8B-Instruct/P150/
  tensor_cache_instruct_bfp8/
  kvcache_torch.Size([1024, 8, 32, 128])_dtype_BFLOAT8_B_layout_TILE.tensorbin
```

### 6.4 Page table

Page table은 모델 load 시 한 번 생성된다.

1. `max_num_blocks` 크기의 random permutation 생성
2. virtual block을 physical block으로 대응
3. batch/DP에 맞춰 reshape

근거:

- `models/bos_model/llama32/run_llama32.py:65-76`
- `models/bos_model/llama32/run_llama32.py:116-120`

현재 batch 1/DP1에서는:

```text
page_table shape = [1, max_num_blocks]
```

32K 실행에서는 `[1, 1024]`이다.

주의:

- page table은 모든 질문에서 같은 mapping을 재사용한다.
- 질문마다 새 physical page를 동적으로 할당하지 않는다.
- page free, eviction, compaction도 없다.
- Paged라는 이름과 달리 runner 자체는 수요 기반 memory allocator가 아니다.

## 7. Llama 3.1 8B KV-cache 메모리 계산

### 7.1 TT BFP8 tile 크기

TT TILE은 32×32 elements다.

```text
BF16 tile = 2048 bytes
BFP8 tile = 1088 bytes
```

근거:

- `tt_metal/api/tt-metalium/tt_backend_api_types.hpp:90-99`

따라서 BFP8은 BF16 대비 정확히 2배가 아니라 다음 비율로 raw tile payload를 줄인다.

```text
2048 / 1088 = 약 1.882배
```

BFP8 tile에는 exponent metadata가 포함되기 때문이다.

### 7.2 Block당 tile 수

8B의 한 KV head block은:

```text
32 tokens × 128 dimensions
```

TT tile 기준:

```text
1 tile high × 4 tiles wide = 4 tiles
```

한 layer의 K 또는 V tensor raw payload 공식:

```text
blocks × local_KV_heads × 4 tiles × 1088 bytes
```

전체 model K+V 공식:

```text
2 × 32 layers × blocks × local_KV_heads × 4 × 1088 bytes
```

### 7.3 Sequence length별 exact TT packed payload

TP1, local KV heads 8 기준:

| max_seq_len | Blocks | K 한 개/layer | K+V/layer | 32 layers 전체 |
|---:|---:|---:|---:|---:|
| 1,024 | 32 | 1.0625 MiB | 2.125 MiB | **68 MiB** |
| 32,768 | 1,024 | 34 MiB | 68 MiB | **2,176 MiB = 2.125 GiB** |
| 49,152 | 1,536 | 51 MiB | 102 MiB | **3,264 MiB = 3.1875 GiB** |
| 131,072 | 4,096 | 136 MiB | 272 MiB | **8,704 MiB = 8.5 GiB** |

32K exact 계산:

```text
2(K,V)
× 32 layers
× 1024 blocks
× 8 KV heads
× 4 tiles/block/head
× 1088 bytes/tile
= 2,281,701,376 bytes
= 2,176 MiB
= 2.125 GiB
```

실제 32K tensorbin 하나의 파일 크기:

```text
35,651,952 bytes
= 34 MiB tile payload + 368-byte serialization header
```

따라서 source 계산과 materialized artifact가 일치한다.

### 7.4 MHA/BF16과의 비교

동일한 Llama 3.1 8B layer 수, head dimension, 32K capacity를 가정한다.

| 구성 | 32K KV-cache raw payload |
|---|---:|
| MHA 32 KV heads + BF16 | 16 GiB |
| MHA 32 KV heads + BFP8 | 8.5 GiB |
| GQA 8 KV heads + BF16 | 4 GiB |
| GQA 8 KV heads + BFP8 | **2.125 GiB** |

절감 관계:

```text
GQA:               4배 절감
BFP8 vs BF16 tile: 약 1.882배 절감
결합:              약 7.529배 절감
```

즉 사용자가 최초 정리한 “GQA로 MHA 대비 약 4배 절감”은 맞으며, 현재 구현에서는 BFP8 tile 저장이 추가로 결합된다.

### 7.5 `test_demo`의 32배 과할당과 구분

`test_demo_llama32.py`:

```text
max_seq_len    = 1024
max_num_blocks = 1024
capacity       = 32768 tokens
```

`run_llama32.py` 기본 실행:

```text
max_seq_len    = 1024
max_num_blocks = 32
capacity       = 1024 tokens
```

`run_llama32.py --max_seq_len 32768`:

```text
max_seq_len    = 32768
max_num_blocks = 1024
capacity       = 32768 tokens
```

따라서 runner의 32K 실행에서 1024 blocks는 과할당이 아니라 요청한 max sequence와 정확히 일치한다. `max_seq_len`이 32의 배수가 아닐 때도 초과량은 마지막 partial block의 최대 31 tokens뿐이다.

## 8. Tensor Parallel KV-head 분할

Attention은 다음 값을 사용한다.

```text
n_local_kv_heads = n_kv_heads / num_devices_per_group
```

근거:

- `models/bos_model/llama32/tt/attention.py:55-78`
- `models/bos_model/llama32/tt/attention.py:223-262`

Non-TG에서 TP degree가 1/2/4/8이면 32K 전체 model cache의 장치당 크기는 다음과 같다.

| TP degree | Local KV heads | 장치당 K/V shape | 장치당 전체 cache |
|---:|---:|---|---:|
| TP1 | 8 | `[1024,8,32,128]` | 2,176 MiB |
| TP2 | 4 | `[1024,4,32,128]` | 1,088 MiB |
| TP4 | 2 | `[1024,2,32,128]` | 544 MiB |
| TP8 | 1 | `[1024,1,32,128]` | 272 MiB |

Non-TG에서는 TP group 전체 aggregate가 2,176 MiB로 같다. 즉 TP는 데이터를 추가 압축하는 것이 아니라 KV heads를 장치 사이에 분산해 장치당 capacity와 bandwidth를 줄인다.

TG 32-chip은 구현상 8-device group이 4개다.

- local KV heads: 1
- 장치당 32K cache: 272 MiB
- 4개 group replication을 포함한 physical aggregate: 8,704 MiB = 8.5 GiB

보존된 실제 32K profiler run은 P150 단일 칩이므로 TP1이며 전체 2.125 GiB가 한 장치에 존재했다.

## 9. 현재 활성화된 KV-cache 최적화 상세

### 9.1 GQA cache head 축소

Llama 3.1 8B는 32 query heads에 대해 8 KV heads만 저장한다. Query head 네 개가 하나의 K/V head group을 공유한다.

효과:

- MHA 대비 K cache head 수 4배 감소
- MHA 대비 V cache head 수 4배 감소
- cache read bandwidth도 head dimension/sequence 조건이 같으면 약 4배 감소

### 9.2 BFP8 KV 저장

Runner는 `create_tt_model()`에 `ttnn.bfloat8_b`와 performance precision profile을 전달한다.

근거:

- `models/bos_model/llama32/run_llama32.py:101-109`
- `models/bos_model/llama32/tt/model_config.py:223-252`
- `models/bos_model/llama32/tt/model_config.py:2518-2525`
- `models/bos_model/llama32/tt/attention.py:131-133`

Cache tensor는 BFP8_B/TILE/DRAM으로 생성된다. 이는 단순 논리적 1 byte/element보다 tile exponent metadata만큼 크지만 BF16보다 약 1.882배 작다.

### 9.3 Cache 사전 할당과 영속적 수명

각 layer는 모델 생성 시 K와 V buffer를 한 번 할당한다. 이후 모든 질문과 decode token이 같은 buffer를 재사용한다.

근거:

- `models/bos_model/llama32/tt/attention.py:345-399`
- `models/bos_model/llama32/tt/common.py:608-617`
- `models/bos_model/llama32/tt/model.py:384-394`

효과:

- decode token마다 tensor 재할당 없음
- 기존 cache 전체를 concat하거나 복사하지 않음
- layer cache 주소를 안정적으로 유지
- TTNN program/trace에서 동일 buffer address 재사용 가능

### 9.4 In-place fill/update/reset

Prefill:

```text
paged_fill_cache(K)
paged_fill_cache(V)
```

Decode:

```text
paged_update_cache(K)
paged_update_cache(V)
```

새 질문 시작:

```text
ttnn.mul(cache, 0, output_tensor=cache)
```

근거:

- reset: `models/bos_model/llama32/run_llama32.py:359-364`
- decode update: `models/bos_model/llama32/tt/attention.py:525-542`
- prefill fill: `models/bos_model/llama32/tt/attention.py:791-829`
- update in-place runtime: `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:197-206`
- fill in-place runtime: `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:65-74`

주의:

- buffer 재할당은 피하지만 질문마다 전체 pool을 zeroing한다.
- 사용된 page만 선택적으로 reset하는 최적화는 없다.

### 9.5 Prefill multicore bulk page fill

Prefill K/V는 token별 update loop가 아니라 prompt 구간을 `paged_fill_cache`로 한 번에 physical pages에 기록한다.

TTNN fill program은:

- `num_heads × seq_len/32` tile rows를 core grid에 분산
- page table을 사용해 destination physical block 계산
- circular buffer double buffering 사용
- 임시 contiguous cache 생성 후 scatter하는 복사 단계 제거

근거:

- `models/bos_model/llama32/tt/attention.py:791-829`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:36-50`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:76-96`

### 9.6 Compute padding의 cache write 억제

Prefill compute tensor는 하드웨어 효율을 위해 128 또는 더 큰 정렬 단위로 padding될 수 있다. 그러나 cache 기록용 page table은 실제 prompt가 차지하는 block 수로 자른다.

```text
num_blocks = ceil(actual_prompt_len / block_size)
```

K/V도 이 page-table width까지만 slice한 후 fill한다.

근거:

- `models/bos_model/llama32/tt/common.py:476-487`
- `models/bos_model/llama32/tt/generator.py:76-103`
- `models/bos_model/llama32/tt/generator.py:1240-1244`
- `models/bos_model/llama32/tt/attention.py:818-829`

효과:

- compute용 대규모 padding 전체를 cache에 쓰지 않음
- 마지막 32-token page 경계까지만 기록
- 마지막 partial page에는 최대 31개의 padded positions가 남을 수 있음

### 9.7 짧은 prefill의 즉시 cache read-back 회피

Non-chunked prefill에서는:

1. 향후 decode를 위해 K/V를 cache에 기록
2. 현재 prefill attention은 방금 계산한 K/V tensor를 직접 사용

따라서 cache에 쓴 K/V를 같은 prefill 단계에서 DRAM으로부터 바로 다시 읽지 않는다.

근거:

- `models/bos_model/llama32/tt/attention.py:846-869`

이 최적화는 기본 1K prompt처럼 chunked prefill 임계값보다 짧은 경로에 해당한다.

### 9.8 장문 chunked prefill

Padded prefill length가 장치별 `max_prefill_chunk_size`를 초과하면 generator가 prompt를 여러 chunk로 나눈다.

근거:

- `models/bos_model/llama32/tt/generator.py:119-178`
- `models/bos_model/llama32/tt/model_config.py:545-582`

각 chunk는:

1. 자기 구간 page table만 사용해 K/V를 cache에 fill
2. `chunk_start_idx`를 attention에 전달
3. 이미 cache에 누적된 이전 K/V와 현재 chunk를 대상으로 chunked SDPA 수행

근거:

- `models/bos_model/llama32/tt/attention.py:822-859`

Llama 3.1 8B의 설정값:

| 장치 이름 | 기본 max prefill chunk |
|---|---:|
| N150 | 4K |
| N300 | 64K |
| T3K | 128K |
| TG | 128K |
| P150x4 | 128K |
| 알 수 없는 장치 이름 fallback | 4K |

P150 단일 칩은 이 table에 직접 키가 없어 fallback 4K가 적용될 수 있다. 따라서 보존된 32,032-token P150 실행은 chunked prefill 경로에 들어간다.

구분:

- 기본 1K: non-chunked prefill, freshly-computed K/V 직접 사용
- 실제 32K P150: chunked prefill, 이전 chunk의 누적 cache를 읽는 cache-backed SDPA 활성

### 9.9 Decode K/V 입력의 L1 sharding

현재 token의 K/V는 `nlp_create_qkv_heads_decode`에서 L1 height-sharded 형태로 생성된다.

근거:

- `models/bos_model/llama32/tt/attention.py:493-502`
- `models/bos_model/llama32/tt/model_config.py:871-881`

Paged update worker는 input shard grid를 직접 이용한다. Cache/intermediate circular buffer도 double-buffered다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp:110-147`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp:223-243`

중요한 구분:

- 새 현재-token K/V 입력: L1 height-sharded
- 누적 K/V cache: DRAM interleaved

즉 누적 cache tensor 자체를 chip 내부 core에 shard하는 최적화는 아니다.

### 9.10 Update kernel 내부 BFP8 pack/tilize

Rotary 이후 BF16 K/V를 별도의 Python-level BFP8 tensor로 materialize하지 않고 곧바로 paged update에 전달한다.

Update compute kernel은 내부에서:

- unpack
- tile patch/update
- tilize
- cache dtype에 맞춘 pack
- physical cache address에 직접 write

를 수행한다.

근거:

- `models/bos_model/llama32/tt/attention.py:537-542`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp:40-85`

### 9.11 장치 측 page 주소 해석

Update kernel은 다음 입력을 장치에서 읽는다.

- current-position tensor
- user page-table row
- block/tile offsets

그리고 virtual block을 physical block/tile 주소로 변환한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:60-95`

효과:

- host가 token마다 physical K/V destination을 계산하지 않음
- host scatter 제거
- position `-1`인 lane/user를 kernel에서 skip 가능

### 9.12 Flash-Decode/online softmax cache streaming

Paged decode attention의 실제 compute kernel은 Flash Attention 방식이다.

- current position으로 유효 KV prefix 결정
- K/V를 chunk 단위로 stream
- running maximum/running sum 유지
- chunk별 online softmax와 lazy rescaling
- full attention score matrix materialization 방지
- K/V circular buffer double buffering
- 작업 없는 core early exit

근거:

- model 호출: `models/bos_model/llama32/tt/attention.py:556-568`
- program config: `models/bos_model/llama32/tt/model_config.py:883-888`
- compute: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:218-275`
- online softmax: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:338-460`
- reader/early exit: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:112-125`
- cache page streaming: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:263-360`

이는 Paged Attention이라는 상위 기능의 구체적인 runtime 최적화다.

### 9.13 TTNN program cache

TT-Metal program cache는 기본 활성이다.

근거:

- `tt_metal/api/tt-metalium/program_cache.hpp:106-133`

Paged update의 hash는 실제 current-position 값을 포함하지 않고, fill hash도 실제 batch index를 포함하지 않는다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:209-223`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:77-85`

따라서 token position과 buffer runtime address가 바뀌어도 같은 tensor spec이면 KV update/fill program을 다시 컴파일하지 않고 runtime arguments만 교체한다.

### 9.14 32-token page와 TT tile 정렬

Block size 32는 TT tile height 32와 일치한다. 8B head dimension 128은 tile width 네 개와 일치한다.

```text
한 block/head = 1 × 4 tiles
```

효과:

- page boundary가 tile row boundary와 일치
- page fill/update address 계산 단순화
- 한 block이 여러 tile-height에 걸쳐 분리되지 않음

### 9.15 초기 zero-cache tensor의 disk cache

Cache 생성 시 shape/dtype/layout을 포함한 `cache_file_name`을 사용해 초기 zero tensor의 TT 포맷 결과를 disk에 저장하고 다음 load에서 재사용한다.

근거:

- `models/bos_model/llama32/tt/attention.py:385-399`

실제 32K 8B artifact가 존재한다.

이 기능은:

- 모델 시작 시 zero tensor 변환/생성 시간 절감: 해당
- 실제 prompt KV prefix 저장: 아님
- 요청 간 KV-cache 재사용: 아님

## 10. Runner의 실제 KV-cache 수명

### 10.1 모델 load

```text
checkpoint/config 선택
  → model/weight cache load
  → layer별 K/V physical pool 한 번 할당
  → host page table 한 번 생성
  → Generator 생성
```

### 10.2 각 질문 시작

```text
기존 K/V 전체 pool in-place zeroing
  → 이전 질문을 문자열 prompt에 다시 포함(LOW/HIGH_CONTEXT)
  → tokenization
  → prompt 전체 prefill
```

근거:

- context prompt 생성: `models/bos_model/llama32/run_llama32.py:323-326`
- 전체 cache reset: `models/bos_model/llama32/run_llama32.py:359-364`

중요:

`LOW_CONTEXT`와 `HIGH_CONTEXT`는 KV-cache memory mode가 아니다. 이전 질문/답변 일부를 새 prompt 문자열에 넣고 매 질문마다 전체를 다시 prefill하는 host-level conversation reconstruction이다.

따라서:

- 이전 질문의 KV-cache 재사용 없음
- 공통 system/context prefix 재사용 없음
- 대화가 길수록 prefill 작업 반복 증가

### 10.3 Prefill 두 번 실행

각 질문마다 prefill이 두 번 호출된다.

1. compile/warm-up prefill
2. 측정 대상 inference prefill

근거:

- warm-up: `models/bos_model/llama32/run_llama32.py:368-374`
- inference: `models/bos_model/llama32/run_llama32.py:376-386`

두 실행은 같은 physical pages를 같은 prompt K/V로 다시 덮어쓴다. 첫 호출은 TTNN program compilation/warm-up 목적이지만, 이미 program cache가 warm 상태인 후속 질문에도 현재 source는 prefill을 계속 두 번 호출한다.

### 10.4 Decode loop

매 token마다:

```text
out token + current_pos
  → decode input 준비
  → Q/K/V + rotary
  → K paged_update_cache
  → V paged_update_cache
  → paged Flash-Decode가 cache prefix read
  → greedy argmax 또는 host sampling
  → current_pos + 1
```

근거:

- `models/bos_model/llama32/run_llama32.py:399-439`
- `models/bos_model/llama32/run_llama32.py:484-504`

## 11. Decode trace와 2-CQ 상태

### 11.1 CLI trace 플래그가 반대로 동작

Parser:

```python
parser.add_argument("--no_trace", action="store_false", ...)
```

Main config:

```python
"trace": not args.no_trace
```

근거:

- `models/bos_model/llama32/run_llama32.py:48`
- `models/bos_model/llama32/run_llama32.py:570`

실제 결과:

| CLI | `args.no_trace` | 실제 trace |
|---|---:|---:|
| flag 생략 | `True` | OFF |
| `--no_trace` 지정 | `False` | ON |

즉 flag 이름과 help 설명이 실제 동작의 반대다.

현재 source 그대로 사용할 경우:

```bash
# Trace OFF
python models/bos_model/llama32/run_llama32.py --live

# Trace ON: 이름과 반대로 이 flag가 필요
python models/bos_model/llama32/run_llama32.py --live --no_trace
```

### 11.2 Trace 활성 시 KV 관련 효과

Trace가 활성화되면 첫 decode에서:

1. compile run
2. CQ0 trace capture
3. device input/output buffer와 trace ID 저장

이후 token에서는:

1. token/current position/page table 내용을 기존 device buffer에 복사
2. 동일 captured graph를 blocking=False로 replay
3. 같은 persistent KV-cache 주소에 update

근거:

- `models/bos_model/llama32/tt/generator.py:294-325`
- `models/bos_model/llama32/tt/generator.py:327-355`
- `models/bos_model/llama32/tt/generator.py:370-385`

효과:

- token별 operation dispatch 감소
- cache update/attention graph 재구성 감소
- device input buffers와 cache addresses 재사용

보존된 실제 8B profiler run들은 반대로 구현된 `--no_trace`를 생략하여 trace OFF 상태였다.

### 11.3 2 command queues

Runner는 항상 두 command queues를 연다.

```text
num_command_queues = 2
```

Token loop에는 CQ0/CQ1 event handshake가 있다.

근거:

- `models/bos_model/llama32/run_llama32.py:195-199`
- `models/bos_model/llama32/run_llama32.py:408`
- `models/bos_model/llama32/run_llama32.py:484-487`
- `models/bos_model/llama32/run_llama32.py:504`

하지만 KV update와 trace는 CQ0에 있고 CQ1에 별도 KV operation을 명시적으로 제출하지 않는다. 따라서 이것을 “KV compute-transfer overlap” 또는 독립적인 KV-cache 최적화라고 단정하면 안 된다.

## 12. 구현은 있지만 기본 runner에서 비활성 또는 조건부

### 12.1 Chunked prefill

- 기본 1K prompt: 비활성
- 32K P150 prompt: 활성
- 장치와 prompt length에 따라 달라짐

따라서 일반적인 “항상 활성/항상 비활성” 분류가 아니라 실행 조건부다.

### 12.2 Speculative multi-token shared cache

`TT_PAGED_MULTI_TOKEN_DECODE=1`일 때만 update와 paged SDPA에 `share_cache=True`가 전달된다.

근거:

- `models/bos_model/llama32/tt/attention.py:534-568`

기본 `run_llama32.py`는 speculative runner가 아니며 이 환경변수를 설정하지 않는다.

### 12.3 External runtime/vLLM-owned cache

Attention에는 `use_paged_kv_cache=True`일 때 내부 cache 할당을 건너뛰는 hook이 있다.

근거:

- `models/bos_model/llama32/tt/attention.py:336-338`

현재 `run_llama32.py → create_tt_model()` 경로는 이를 활성화하지 않으므로 내부 persistent cache를 사용한다.

### 12.4 Non-paged long-prefill K/V sharding

긴 non-paged prefill을 위한 별도 K/V write sharding 경로가 있으나 `page_table`이 없는 조건에서만 사용된다.

근거:

- `models/bos_model/llama32/tt/attention.py:799-844`

Runner는 항상 page table을 제공하므로 이 경로는 비활성이다.

### 12.5 Sliding-window runtime 기능

SDPA runtime 내부에는 sliding-window 관련 지원이 있지만 BOS Llama 3.1 8B 호출은 window argument를 전달하지 않는다. 따라서 decode는 current position까지의 전체 prefix를 대상으로 한다.

## 13. 런타임에는 있지만 모델이 사용하지 않는 기능

### 13.1 Fused K+V cache update

TTNN에는 `paged_fused_update_cache`가 존재한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.hpp:26-38`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.hpp:60-62`

하지만 현재 attention은:

```text
paged_update_cache(K)
paged_update_cache(V)
```

를 별도로 호출한다.

근거:

- `models/bos_model/llama32/tt/attention.py:537-542`

따라서 fused K/V update의 공통 dispatch/page-index 처리 절감은 적용되지 않는다.

### 13.2 `serialize_paged_updates` dead plumbing

`TT_PAGED_MULTI_TOKEN_DECODE`와 `TT_TREE_SPECULATIVE_DECODE`에서 `serialize_paged_updates` 값을 계산하지만 이후 실제 operation에 사용하지 않는다.

근거:

- `models/bos_model/llama32/tt/attention.py:534-536`

Tree speculative flag만 설정해도 이 값에 의한 update serialization은 발생하지 않는다.

## 14. 현재 없는 KV-cache 최적화

### 14.1 Prefix와 요청 간 재사용

- Prefix/prompt KV-cache reuse
- 공통 system prompt cache
- Cross-request KV reuse
- 대화 continuation cache
- Prefix deduplication
- Page-level copy-on-write

현재 질문마다 cache 전체를 지우므로 모두 적용되지 않는다.

### 14.2 동적 page 관리

- Demand-based physical page allocation
- Page free/reclamation
- Eviction
- Compaction
- Watermark management
- 사용 page만 선택적으로 reset

현재는 max sequence에 맞춘 전체 pool을 시작 시 할당하고 정적 random page mapping을 재사용한다.

### 14.3 Long-context cache 정책

- Sliding-window/ring KV-cache
- Attention sink
- 오래된 token eviction
- Recent-token hot cache

### 14.4 추가 압축

- INT8 KV + per-channel scale
- FP8 scale 기반 cache
- K/V 별도 precision
- Layer/token-age adaptive precision
- Low-rank KV compression

현재 확인된 저장 정밀도 최적화는 BFP8_B다.

### 14.5 Offload와 계층화

- Host/CPU offload
- NVMe offload
- DRAM↔L1 hot-page migration
- Hierarchical KV cache

누적 cache는 장치 DRAM interleaved tensor에 유지된다.

### 14.6 Chip 내부 누적-cache sharding

KV heads는 TP로 chip 간 분할되지만 한 chip의 누적 cache tensor 자체는 DRAM interleaved다. Sharded인 것은 decode의 새 K/V input이다.

Runtime validation도 interleaved cache를 요구한다.

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:55-59`

## 15. `run_llama32.py` 실행 전 주의사항

### 15.1 매 시작 시 거대한 `model_weights.pth` 저장

`load_model()`은 model을 만든 뒤 매번 다음 코드를 실행한다.

```python
torch.save(state_dict, "model_weights.pth")
```

근거:

- `models/bos_model/llama32/run_llama32.py:114`

현재 checkout에 실제 생성된 파일은:

```text
/home/iris_hb4/tt-metal-hb4/model_weights.pth
6,425,581,487 bytes
```

이는 KV-cache 최적화와 무관하며 8B startup 시간, disk write, 저장 공간에 큰 부담을 줄 수 있다.

### 15.2 8B 전용 performance decoder override 경로 불일치

8B 전용 파일은 다음 위치에 있다.

```text
models/tt_transformers/model_params/
  Llama-3.1-8B-Instruct/performance_decoder_config.json
```

이 파일은 decoder 31의 `FF1_FF3`를 BFP8, fidelity를 HIFI2_FP16으로 보정한다.

그러나 BOS `model_config.py`의 precision factory는 다음 위치를 찾는다.

```text
models/bos_model/llama32/model_params/<model_name>/
```

근거:

- `models/bos_model/llama32/tt/model_config.py:2543-2565`

현재 그 BOS 경로에는 파일이 없어 override가 적용되지 않고 모든 layer에 일반 performance profile이 사용된다.

영향:

- KV-cache dtype BFP8에는 영향 없음
- 마지막 decoder MLP/출력 정확도에는 영향 가능

### 15.3 `--device_id`가 실제 open call에 전달되지 않음

`llama_runner()`는 `device_ids=[device_id]`를 helper에 전달하지만 `get_mesh_device()`가 계산 후 `ttnn.open_mesh_device()`에 해당 ID list를 전달하지 않는다.

근거:

- `models/bos_model/llama32/run_llama32.py:182-200`
- `models/bos_model/llama32/run_llama32.py:269-275`

따라서 `--device_id`가 물리 장치 선택을 확실히 보장한다고 보기 어렵다.

### 15.4 알 수 없는 CLI 옵션이 조용히 무시됨

Parser가 `parse_args()`가 아니라 `parse_known_args()`를 사용한다.

근거:

- `models/bos_model/llama32/run_llama32.py:61`

옵션 오타가 즉시 오류가 되지 않고 무시될 수 있으므로 실행 log의 실제 config를 확인해야 한다.

### 15.5 `--live`도 queries 파일을 먼저 읽음

Main은 live 여부를 확인하기 전에 `args.queries` 파일을 연다.

근거:

- `models/bos_model/llama32/run_llama32.py:553-561`

기본 queries 파일이 존재하면 문제없지만 custom working tree에서는 `--live`만 사용해도 파일 누락으로 먼저 실패할 수 있다.

### 15.6 Generation 종료 조건

Decode 종료 검사가 다음과 같다.

```python
if iteration > max_generated_tokens:
```

근거:

- `models/bos_model/llama32/run_llama32.py:498`

`>=`가 아니라 `>`이므로 EOS가 나오지 않는 고정 길이 실행에서 의도보다 추가 decode step이 발생할 수 있다. 실제 짧은 profiler run에서도 requested 8 tokens에 대해 10 decode steps가 기록됐다.

## 16. 권장 실행 구분

### 16.1 기본 1K, trace OFF

현재 source에서 trace를 끄려면 `--no_trace`를 넣지 않는다.

```bash
cd /home/iris_hb4/tt-metal-hb4
source env_set.sh

HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python models/bos_model/llama32/run_llama32.py \
  --live \
  --max_seq_len 1024 \
  -g 256 \
  -n 5
```

KV-cache:

```text
32 blocks
K/V each [32,8,32,128] per layer
32-layer aggregate 68 MiB
non-chunked prefill
trace OFF
```

### 16.2 32K, trace OFF

```bash
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python models/bos_model/llama32/run_llama32.py \
  --live \
  --max_seq_len 32768 \
  -g 256 \
  -n 5
```

KV-cache TP1:

```text
1024 blocks
K/V each [1024,8,32,128] per layer
32-layer aggregate 2.125 GiB
P150에서는 4K chunked prefill 가능/활성
trace OFF
```

### 16.3 Trace ON

현재 역전된 CLI 구현에서는 `--no_trace`를 넣어야 trace가 켜진다.

```bash
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python models/bos_model/llama32/run_llama32.py \
  --live \
  --max_seq_len 32768 \
  --no_trace \
  -g 256 \
  -n 5
```

이 명령 의미는 flag 이름과 반대이므로 source 수정 전까지만 유효하다.

## 17. `test_demo_llama32.py`와 runner 비교

| 항목 | `test_demo_llama32.py` | `run_llama32.py` |
|---|---|---|
| 주 용도 | pytest/demo 검증 | interactive/file runner |
| 기본 분석 모델 | 1B로 해석하기 쉬움 | 환경변수 모델, 실제 8B 가능 |
| `model_id` | 실질 미사용 | 실질 미사용 |
| Max sequence | load 내부 1024 | CLI 값이 model load에 전달 |
| Physical blocks | 1024 고정 | `ceil(max_seq_len/32)` |
| 기본 cache capacity | 32768 tokens | 1024 tokens |
| Batch | 1 | 1 assertion |
| DP | 1 | 1 고정 |
| Command queues | 1 | 2 |
| Decode trace | 명시적 OFF | 기본 OFF, `--no_trace` 버그로 ON 가능 |
| Conversation mode | 없음 | 문자열 기반 LOW/HIGH context |
| Query reset | 전체 cache zeroing | 전체 cache zeroing |
| Dynamic pages | 없음 | 없음 |

가장 중요한 차이:

- `test_demo` 1K: 1024 blocks는 32배 과할당
- runner 1K: 32 blocks로 정확히 할당
- runner 32K: 1024 blocks가 정확한 capacity

## 18. 최종 적용/비적용 분류표

| 기능 | 구현 | Llama 3.1 8B runner 상태 |
|---|---:|---:|
| Autoregressive KV caching | 있음 | 활성 |
| GQA 32Q/8KV | 있음 | 활성 |
| BFP8_B KV storage | 있음 | 활성 |
| Paged KV cache | 있음 | 항상 활성 |
| Paged Attention | 있음 | 항상 활성 |
| Persistent preallocated cache | 있음 | 활성 |
| In-place prefill/decode update | 있음 | 활성 |
| TP local KV-head partition | 있음 | 멀티칩에서 활성 |
| Multicore bulk prefill fill | 있음 | 활성 |
| Padding cache-write trimming | 있음 | 활성 |
| Non-chunked prefill read-back 회피 | 있음 | 짧은 prompt에서 활성 |
| Chunked cache-backed prefill | 있음 | 긴 prompt/device 조건부 |
| Decode K/V L1 sharding | 있음 | 활성 |
| Update 내부 BFP8 pack | 있음 | 활성 |
| Device-side page translation | 있음 | 활성 |
| Flash-Decode online softmax | 있음 | 활성 |
| TTNN program cache | 있음 | 활성 |
| Decode trace replay | 있음 | 기본 OFF, CLI 조건부 |
| Fused K+V cache update | runtime에 있음 | 모델 미사용 |
| Speculative shared cache | 있음 | 기본 runner 비활성 |
| External runtime-owned cache | 있음 | 비활성 |
| Dynamic page allocation/free | 없음 | 비활성 |
| Prefix/prompt cache reuse | 없음 | 비활성 |
| Page copy-on-write/dedup | 없음 | 비활성 |
| Sliding-window/ring cache | 호출 경로에 없음 | 비활성 |
| Cache offload | 없음 | 비활성 |
| INT8/scale-based 추가 압축 | 없음 | 비활성 |
| Chip 내부 누적-cache sharding | 없음 | DRAM interleaved |

## 19. 개선 우선순위

다음은 현재 적용 기능이 아니라 분석 결과상 개선 효과가 클 후보이다.

1. 매 질문 전체 cache zeroing 대신 used-page validity/reset 도입
2. conversation prefix page 재사용 및 copy-on-write 검토
3. 두 번째 질문부터 불필요한 compile용 prefill 중복 호출 제거 검토
4. `paged_fused_update_cache`로 K/V update 통합 가능성 측정
5. trace CLI 논리와 flag 이름 수정
6. 32K decode에서 trace ON/OFF latency를 동일 조건으로 비교
7. `model_weights.pth` 무조건 저장 제거 또는 명시적 debug option화
8. 8B performance decoder config lookup 경로 수정
9. dynamic page allocator/free가 필요한 multi-request runtime과 runner cache 소유권 분리
10. long-context에서 sliding window가 허용되는 workload라면 cache capacity/SDPA read 절감 검토

## 20. 최종 판정

Llama 3.1 8B를 `run_llama32.py`로 실행할 때 KV-cache는 다음과 같이 정리된다.

```text
Model: Llama 3.1 8B
Layers: 32
Q heads: 32
KV heads: 8
Head dim: 128
Cache dtype: BFP8_B
Layout: TILE
Memory: DRAM interleaved
Block size: 32 tokens
Blocks: ceil(max_seq_len/32)
Batch: 1
DP: 1
```

실제 32K TP1 실행:

```text
K per layer: [1024,8,32,128], 34 MiB
V per layer: [1024,8,32,128], 34 MiB
K+V per layer: 68 MiB
32-layer aggregate: 2176 MiB = 2.125 GiB
```

이 runner에는 GQA, BFP8, paged layout, persistent/in-place cache, TP KV-head 분할, multicore prefill fill, padding write trimming, 장치 측 page translation, L1-sharded decode update, Flash-Decode/online softmax, TTNN program cache가 적용되어 있다.

반면 요청 간 prefix reuse, dynamic allocation/free, selective reset, eviction, sliding-window cache, offload, fused K/V update는 적용되지 않는다. 대화 memory mode도 KV를 유지하는 방식이 아니라 매 질문 cache를 지우고 이전 문맥을 문자열로 다시 prefill하는 방식이다.

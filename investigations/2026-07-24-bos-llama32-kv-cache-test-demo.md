# BOS Llama 3.2 KV-cache 최적화 구현 분석

- 분석일: 2026-07-24
- 분석 대상: `/home/iris_hb4/tt-metal-hb4/models/bos_model/llama32`
- 주 테스트 경로: `test_demo_llama32.py`
- 연관 범위: BOS Llama 3.2 모델 코드, 공용 generator/model 코드, TTNN paged-cache 및 SDPA decode 구현

## 1. 분석 목적과 범위

이 문서는 다음과 같이 이미 확인된 기능 외에, 현재 BOS Llama 3.2 구현에 포함된 KV-cache 관련 최적화를 소스 코드 기준으로 빠짐없이 구분하는 것을 목적으로 한다.

1. Autoregressive KV caching
2. Grouped-Query Attention(GQA)
   - Llama 3.2 1B 기준 32 query heads
   - 8 KV heads
   - 동일 정밀도와 동일 token capacity의 MHA 대비 KV-cache 약 4배 절감
3. Paged KV cache
4. KV-cache BFP8 저장
5. Paged Attention

분석 결과는 다음 세 범주로 나눈다.

- 현재 `test_demo_llama32.py` 경로에서 실제로 활성화되는 기능
- 소스에는 구현되어 있으나 현재 테스트 조건에서는 비활성인 기능
- 런타임에는 존재하지만 모델이 사용하지 않거나, 현재 구현에 없는 기능

> 이 문서는 정적 소스 분석 결과이다. 분석 시점의 셸에는 `HF_MODEL`, `LLAMA_DIR`, `MESH_DEVICE` 및 관련 KV 환경변수가 설정되어 있지 않았으므로 실제 장치 실행 프로파일 결과를 포함하지 않는다.

## 2. 핵심 결론

현재 코드에는 사용자가 이미 확인한 다섯 가지 외에 다음 최적화가 적용되어 있다.

| 구분 | 최적화 | 현재 테스트 경로 |
|---|---|---|
| 메모리 수명 | 레이어별 KV-cache 사전 할당 및 영속 재사용 | 활성 |
| 갱신 방식 | prefill/decode cache in-place 갱신 | 활성 |
| 멀티칩 | Tensor Parallel 단위 local KV-head 저장 | 멀티칩에서 활성 |
| Prefill | multicore bulk paged fill | 활성 |
| Prefill | 연산 padding의 불필요한 cache write 억제 | 활성 |
| Prefill | 방금 기록한 cache를 즉시 다시 읽지 않음 | 활성 |
| Decode | 현재 K/V 입력의 L1 height sharding | 활성 |
| Decode | update kernel 내부 BFP8 pack/tilize | 활성 |
| Decode | 장치 측 virtual-to-physical page 주소 계산 | 활성 |
| Attention | Flash-Decode/online-softmax cache streaming | 활성 |
| 런타임 | KV 연산 프로그램 컴파일 cache 재사용 | 활성 |
| Layout | 32-token page와 TT tile 높이 정렬 | 활성 |

다만 현재 demo 설정에는 중요한 역효과가 있다.

- 모델이 사용하는 최대 sequence length는 1,024인데 cache pool은 32,768 token 크기로 사전 할당된다.
- Llama 3.2 1B 단일 칩 기준 raw KV tile 저장량은 약 **544 MiB**다.
- 1,024 token에 맞게 pool을 구성하면 약 **17 MiB**이므로, 현재 pool은 필요한 크기의 **32배**다.
- 매 prompt 시작 시 사용된 page만 지우는 것이 아니라 이 pool 전체를 zeroing한다.

따라서 현재 설정의 Paged KV cache는 메모리 절약형 수요 할당기라기보다, 고정 크기 physical page pool에 대한 주소 재매핑 및 런타임 기능 검증에 가깝다.

## 3. 먼저 확인해야 할 실행 설정

### 3.1 `model_id`가 실제 모델을 결정하지 않음

`test_demo_llama32.py`의 `load_model()`은 `model_id` 인자를 받지만 해당 값을 모델 선택에 사용하지 않는다. 실제 checkpoint/config는 `HF_MODEL` 또는 `LLAMA_DIR` 환경변수로 결정된다.

근거:

- `models/bos_model/llama32/tt/model_config.py:460-486`

따라서 “32 query heads / 8 KV heads”는 실제 환경변수가 Llama 3.2 1B를 가리킬 때 확정된다.

- Llama 3.2 1B: 32 query heads, 8 KV heads
- Llama 3.2 3B: 24 query heads, 8 KV heads

1B의 로컬 모델 파라미터 근거:

- `models/tt_transformers/model_params/Llama-3.2-1B-Instruct/params.json:2-5`
- `models/tt_transformers/model_params/Llama-3.2-1B-Instruct/config.json:13`

### 3.2 테스트 fixture 일부가 모델 생성에 반영되지 않음

테스트는 fixture로 `max_seq_len`, `paged_attention`, `page_params` 등을 받지만 모델을 만들 때 `load_model(mesh_device)`만 호출한다.

`load_model()` 내부 기본값/고정값은 다음과 같다.

```text
max_seq_len = 1024
batch_size = 1
data_parallel = 1
block_size = 32
max_num_blocks = 1024
```

근거:

- `models/bos_model/llama32/test_demo_llama32.py:40-80`
- `models/bos_model/llama32/test_demo_llama32.py:267`

즉, 현재 소스에서 Paged Attention 활성 여부는 테스트 인자로 모델 생성에 전달되는 구조가 아니라 `load_model()` 내부 설정에 의해 사실상 고정된다. `paged_attention` fixture는 뒤쪽 assertion 조건에만 사용된다.

### 3.3 Decode trace는 꺼져 있음

현재 테스트 parametrization은 `enable_trace=False`이며 command queue도 하나를 사용한다.

근거:

- `models/bos_model/llama32/test_demo_llama32.py:214`
- `models/bos_model/llama32/test_demo_llama32.py:232`
- `models/bos_model/llama32/test_demo_llama32.py:369-378`

따라서 이후 설명하는 TTNN program cache는 활성 상태지만, trace capture/replay에 의한 decode 제출 비용 절감은 현재 테스트에 적용되지 않는다.

## 4. 현재 활성화된 추가 KV-cache 최적화

### 4.1 레이어별 KV-cache 사전 할당

각 attention layer는 시작 시 K cache와 V cache를 각각 한 번 할당한다.

Paged 설정의 cache shape은 다음과 같다.

```text
[max_num_blocks, n_local_kv_heads, block_size, head_dim]
```

현재 주요 속성은 다음과 같다.

- Layout: TT TILE layout
- Memory: DRAM interleaved
- Dtype: BFP8
- Page size: 32 tokens
- Head dimension: Llama 3.2 1B 기준 64

근거:

- `models/bos_model/llama32/tt/attention.py:345-400`

이 방식은 decode token마다 cache tensor를 새로 생성하거나 기존 전체 cache를 복사하는 비용을 제거한다.

### 4.2 동일 cache tensor의 alias 전달과 in-place 수명 관리

`create_tt_model()`이 반환하는 cache는 각 layer의 `layer_past`를 가리키는 handle이다. 별도 복사본이 아니다. 이후 model과 decoder도 이 handle을 attention layer까지 전달한다.

근거:

- `models/bos_model/llama32/tt/common.py:608-617`
- `models/bos_model/llama32/tt/model.py:384-394`
- `models/bos_model/llama32/tt/attention.py:525-530`
- `models/bos_model/llama32/tt/attention.py:792-795`

TTNN의 paged update/fill 연산도 output cache를 새로 반환하는 함수형 복사가 아니라 전달받은 cache buffer를 직접 변경한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:197-206`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:65-74`

### 4.3 Tensor Parallel 기반 local KV-head 저장

Attention 초기화 과정에서 KV head 수를 device group 크기로 나눈다.

```text
n_local_kv_heads = n_kv_heads / num_devices_per_group
```

Llama 3.2 1B의 8 KV heads를 기준으로 하면 다음과 같다.

| TP group 크기 | 장치당 KV heads |
|---:|---:|
| 1 | 8 |
| 2 | 4 |
| 4 | 2 |
| 8 | 1 |

Q/K/V weight도 head 단위로 분할되어 mesh에 배치되고, cache shape에는 `n_local_kv_heads`만 들어간다.

근거:

- `models/bos_model/llama32/tt/attention.py:71-79`
- `models/bos_model/llama32/tt/attention.py:235-262`
- `models/bos_model/llama32/tt/attention.py:350-365`

효과:

- 장치당 KV-cache capacity 감소
- 장치당 cache read/write bandwidth 감소
- 장치당 attention에서 읽어야 하는 KV head 수 감소

주의:

- 이는 mesh 전체 합산 KV 데이터 자체를 추가로 압축하는 기법은 아니다.
- 단일 칩에서는 이 추가 절감이 없다.
- 저장된 cache buffer는 chip 내부 core/L1 sharded가 아니라 DRAM interleaved다.

### 4.4 Prefill의 multicore bulk paged fill

Prefill 시에는 K와 V를 토큰별 `paged_update_cache`로 반복 저장하지 않는다. prompt 전체 K/V를 cache dtype으로 처리한 뒤 K와 V 각각 한 번씩 `paged_fill_cache`에 전달한다.

근거:

- `models/bos_model/llama32/tt/attention.py:791-829`

TTNN fill 프로그램은 다음 방식으로 동작한다.

- `num_heads × seq_len/32` tile row 작업을 compute/storage grid에 분산
- physical page table에 따라 cache 목적지를 직접 계산
- circular buffer double buffering 사용
- 중간 contiguous cache를 만들고 다시 scatter하는 단계 없음

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:36-50`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:76-96`

### 4.5 Prefill 연산 padding을 cache에 불필요하게 저장하지 않음

Prefill 계산 tensor는 하드웨어 효율을 위해 실제 prompt보다 크게 padding될 수 있다.

- 최소 128 token 정렬
- 길이에 따라 power-of-two 정렬
- 긴 구간에서는 2048 배수 정렬

근거:

- `models/bos_model/llama32/tt/common.py:476-487`
- `models/bos_model/llama32/tt/generator.py:81-89`

하지만 cache 기록에 쓰는 page table은 다음 크기로 잘린다.

```text
ceil(actual_prompt_len / block_size)
```

근거:

- `models/bos_model/llama32/tt/generator.py:1240-1244`
- `models/bos_model/llama32/tt/generator.py:90-94`

Attention에서도 K/V를 `page_table_width × block_size`까지만 slice한 뒤 fill한다.

근거:

- `models/bos_model/llama32/tt/attention.py:818-829`

따라서 연산을 위해 추가된 큰 padding 구간 전체가 KV-cache에 기록되지는 않는다. 다만 page granularity가 32이므로 마지막 partial page에는 최대 31개의 padding 위치가 포함될 수 있다.

### 4.6 일반 prefill에서 cache write 직후 read-back 회피

일반적인 non-chunked prefill 경로는 다음 두 작업을 수행한다.

1. 생성한 K/V를 향후 decode를 위해 paged cache에 기록
2. 현재 prefill attention은 동일한 freshly-computed K/V tensor를 직접 사용

즉, 방금 K/V를 DRAM cache에 기록한 뒤 같은 prefill attention에서 다시 cache로부터 읽는 왕복을 하지 않는다.

근거:

- `models/bos_model/llama32/tt/attention.py:846-869`

현재 1K max-sequence 테스트는 chunked prefill 임계값보다 훨씬 작기 때문에 이 경로를 사용한다.

### 4.7 Decode K/V 입력의 L1 height sharding

Decode의 `nlp_create_qkv_heads_decode` 출력 중 현재 token의 K/V는 L1 height-sharded memory config를 사용한다.

근거:

- `models/bos_model/llama32/tt/attention.py:493-502`
- `models/bos_model/llama32/tt/model_config.py:871-881`

Paged update 프로그램은 이 input shard grid를 worker 배치에 사용하며 cache/intermediate circular buffer를 double-buffered로 구성한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp:110-147`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp:223-243`

여기서 구분해야 할 점은 다음과 같다.

- 새로 생성된 현재 token의 K/V 입력: L1 sharded
- 누적된 장기 KV-cache: DRAM interleaved

### 4.8 Decode update 내부 BFP8 pack/tilize 결합

Rotary embedding 이후의 decode K/V는 BF16 형태다. Python/model 단계에서 별도의 BFP8 cache tensor를 먼저 만들지 않고 곧바로 `paged_update_cache`에 전달한다.

근거:

- `models/bos_model/llama32/tt/attention.py:537-542`

Update compute kernel은 내부에서 다음 작업을 수행한다.

- input unpack
- cache tile format에 맞춘 tilize
- cache output dtype에 맞춘 pack
- cache destination에 직접 기록

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp:40-85`

따라서 별도 Python-level typecast 결과를 materialize한 뒤 다시 cache에 복사하는 경로를 피한다.

### 4.9 장치 측 page 주소 해석

Decode update 시 host가 매 token의 physical cache 주소를 직접 계산하지 않는다.

장치 kernel이 다음 정보를 사용한다.

- current-position tensor
- 해당 user의 page-table row
- block size와 tile offset

이를 통해 virtual block index를 physical block/tile 주소로 변환한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:60-95`

Position이 `-1`인 user/lane은 장치에서 skip할 수 있다.

효과:

- host-side token별 scatter 제거
- physical page가 연속적이지 않아도 동일 update program 사용 가능
- batch/user별 서로 다른 page mapping 지원

### 4.10 Flash-Decode와 online softmax

이는 사용자가 이미 확인한 Paged Attention 내부의 핵심 세부 최적화다.

모델은 paged decode attention에 현재 position tensor를 전달한다.

근거:

- `models/bos_model/llama32/tt/attention.py:556-568`

현재 program config는 다음과 같은 chunk 단위를 사용한다.

- Wormhole: K chunk size 256
- Blackhole: K chunk size 128
- Core grid: 5 × 4

근거:

- `models/bos_model/llama32/tt/model_config.py:883-888`

실제 runtime kernel은 Flash Attention 형태로 동작한다.

- K/V cache를 chunk 단위로 stream
- chunk별 QK 계산
- running maximum과 running sum 유지
- lazy rescaling을 포함한 online softmax
- full attention score matrix materialization 방지
- K/V circular buffer double buffering

근거:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:218-275`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:338-460`

Reader는 current position으로 필요한 prefix chunk 범위를 결정하고, 작업이 없는 core는 조기 종료한다.

근거:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:112-125`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:263-360`

주의:

- physical read는 tile/chunk 경계로 반올림될 수 있다.
- “정확히 현재 token까지만 byte 단위로 읽는다”기보다는, 유효 prefix까지만 계산에 포함하고 나머지는 mask하며 필요한 chunk 범위까지만 stream한다고 표현하는 것이 정확하다.

### 4.11 TTNN program cache 재사용

TT-Metal program cache는 기본 활성 상태다.

근거:

- `tt_metal/api/tt-metalium/program_cache.hpp:106-133`

Paged update의 프로그램 hash에는 실제 update index 값이 포함되지 않고, paged fill의 hash에도 실제 batch index가 포함되지 않는다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:209-223`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:77-85`

따라서 token position이나 buffer address가 달라져도 동일 tensor spec/program 조건이면 다시 컴파일하지 않고 runtime arguments와 주소만 교체할 수 있다.

테스트는 실제 inference 전에 prefill을 한 번 추가 실행해 관련 프로그램을 warm-up한다.

근거:

- `models/bos_model/llama32/test_demo_llama32.py:323-339`

이는 cache 저장 용량을 줄이는 최적화는 아니지만 KV update/fill 및 attention의 반복 실행 준비 비용을 줄인다.

### 4.12 Page와 TT tile의 하드웨어 정렬

현재 block size는 32이고 TT tile의 높이도 32다. Llama 3.2 1B의 head dimension은 64이므로 한 KV head의 한 block은 다음 tile 구조가 된다.

```text
32 tokens × 64 dimensions
= tile 높이 1개 × tile 너비 2개
```

효과:

- 한 page가 tile row 경계와 일치
- page update/fill에서 token block이 여러 tile-height에 걸쳐 분리되지 않음
- physical page 주소 계산 단순화

이는 독립적인 cache 알고리즘이라기보다 현재 paged layout을 TT tile geometry에 맞춘 구현상 최적화다.

### 4.13 초기 zero-cache 포맷의 디스크 cache

Attention cache 생성 시 `cache_file_name`을 지정해 최초 zero tensor의 장치 포맷 변환 결과를 재사용할 수 있다.

근거:

- `models/bos_model/llama32/tt/attention.py:385-396`

하지만 이것은 다음과 구분해야 한다.

- 모델 시작/초기 tensor 변환 최적화: 해당
- 실제 prompt의 KV prefix 재사용: 해당하지 않음
- 요청 간 cache 공유: 해당하지 않음

## 5. Prefill 및 Decode 데이터 흐름

### 5.1 Prefill

```text
실제 prompt
  → compute용 sequence padding
  → Q/K/V 및 rotary 계산
  → 실제 prompt에 필요한 page-table 범위만 선택
  → K/V를 physical page에 multicore bulk fill
  → 현재 prefill attention은 freshly-computed K/V를 직접 사용
  → 이후 decode에서 사용할 cache는 DRAM에 유지
```

주요 절감 지점:

- token별 cache update 호출 제거
- compute padding 전체의 cache write 방지
- cache write 직후 동일 K/V read-back 방지
- physical page scatter를 device에서 처리

### 5.2 Decode

```text
현재 token
  → fused QKV projection 및 rotary
  → 현재 K/V를 L1 height-sharded 형태로 생성
  → K paged_update_cache
  → V paged_update_cache
  → 장치가 current position과 page table로 목적지 계산
  → BFP8/tile 포맷으로 기존 cache에 직접 기록
  → paged Flash-Decode가 유효 prefix의 K/V chunk를 stream
  → online softmax로 output 계산
```

주요 절감 지점:

- 전체 cache 복사 없이 현재 position만 in-place 갱신
- 별도 BFP8 cache-conversion tensor 제거
- host-side physical address/scatter 제거
- full score matrix 제거
- program 재컴파일 제거

### 5.3 Prompt 간 cache 초기화

새 prompt 시작 전 cache는 다음과 같이 같은 tensor에 zero를 곱해 in-place 초기화한다.

근거:

- `models/bos_model/llama32/test_demo_llama32.py:316-321`

장점:

- cache buffer를 해제하고 다시 할당하지 않음
- tensor/program layout을 유지

단점:

- 사용한 page만 선택적으로 초기화하지 않음
- 32,768-token pool 전체를 매 prompt마다 scan/write
- prefix cache를 재사용하지 않음

## 6. KV-cache 메모리 정량 분석

### 6.1 현재 physical pool capacity

```text
block_size     = 32 tokens
max_num_blocks = 1024

physical capacity
= 32 × 1024
= 32,768 tokens
```

모델 생성에 전달되는 최대 sequence length는 1,024이므로:

```text
32,768 / 1,024 = 32배
```

현재 test의 page table은 이 physical block들을 기반으로 한 정적 mapping이며, 요청이 들어올 때 필요한 만큼만 page를 동적으로 할당하는 allocator가 아니다.

### 6.2 BFP8 tile 실제 크기

TT backend에서:

```text
BF16 tile = 2048 bytes
BFP8 tile = 1088 bytes
```

근거:

- `tt_metal/api/tt-metalium/tt_backend_api_types.hpp:90-99`

따라서 BFP8의 실제 raw tile 절감률은:

```text
2048 / 1088 ≈ 1.882배
```

BFP8 tile에는 exponent metadata가 있으므로 정확히 2배 절감은 아니다.

### 6.3 Llama 3.2 1B 단일 칩의 현재 pool

가정:

- K와 V: 2개
- Layers: 16
- Blocks: 1024
- Local KV heads: 8
- Block/head당 tiles: `32 × 64`이므로 2개
- BFP8 tile: 1088 bytes

계산:

```text
2 × 16 × 1024 × 8 × 2 × 1088
= 570,425,344 bytes
= 544 MiB
```

이는 allocator metadata 등을 제외한 raw tile 저장량이다.

### 6.4 1,024-token 크기로 맞춘 pool

1,024 token에 필요한 block 수:

```text
1024 / 32 = 32 blocks
```

계산:

```text
2 × 16 × 32 × 8 × 2 × 1088
= 17,825,792 bytes
= 17 MiB
```

따라서:

| 구성 | Raw KV 저장량 |
|---|---:|
| 현재 32,768-token BFP8 GQA pool | 544 MiB |
| 1,024-token BFP8 GQA pool | 17 MiB |
| 차이 | 32배 |

### 6.5 동일 1K 조건의 MHA BF16과 비교

32 KV heads인 MHA, BF16, 1,024-token pool을 가정하면:

```text
2 × 16 × 32 blocks × 32 heads × 2 tiles × 2048
= 128 MiB
```

올바르게 1K로 구성된 GQA+BFP8 pool은 17 MiB이므로:

```text
128 / 17 ≈ 7.53배 절감
```

이는 다음 두 효과의 결합이다.

```text
GQA:       4배
BFP8 tile: 약 1.882배
결합:      약 7.53배
```

하지만 현재 구현은 capacity를 32배 크게 잡았으므로:

```text
현재 pool 544 MiB / MHA BF16 1K 128 MiB
= 4.25배
```

즉, 현재 demo 설정의 실제 사전 할당량은 “1K MHA BF16 cache”보다도 약 4.25배 크다. 이는 GQA나 BFP8 구현의 문제가 아니라 `max_num_blocks=1024`라는 과도한 physical pool 설정 때문이다.

### 6.6 멀티칩 해석

Tensor Parallel에서는 local KV head 수가 줄어들므로 장치당 위 계산값이 TP group 크기에 따라 감소한다.

예를 들어 8-chip TP에서는 장치당 KV head가 1개이므로 장치당 cache는 단일 칩 값의 약 1/8이다.

그러나 동일 TP group 전체를 합산하면 원본 8 KV heads를 모두 저장하므로 aggregate raw KV 데이터는 대체로 동일하다. 따라서 TP 분할은 주로 다음 목적의 최적화다.

- 장치당 메모리 footprint 감소
- 장치당 DRAM bandwidth 감소
- attention 계산의 병렬화

## 7. 구현되어 있지만 현재 테스트에서는 비활성

### 7.1 Automatic chunked prefill

Generator는 padded sequence length가 `max_prefill_chunk_size`를 초과하면 prompt를 여러 chunk로 나눌 수 있다.

근거:

- `models/bos_model/llama32/tt/generator.py:119-178`

각 chunk는:

- 해당 구간의 page-table slice 사용
- start position 전달
- 현재 chunk의 K/V를 cache에 fill
- 이전 chunk까지 누적된 cache를 이용해 chunked SDPA 실행

Attention 근거:

- `models/bos_model/llama32/tt/attention.py:822-859`

기본 임계값과 환경변수 override:

- `models/bos_model/llama32/tt/model_config.py:545-582`

Llama 3.2 1B의 기본 임계값은 장치 조건상 128K 수준이고 현재 테스트 max sequence는 1K이므로 이 경로는 실행되지 않는다.

### 7.2 Non-paged long-prefill K/V fill input sharding

긴 sequence에서 non-paged cache를 채울 때 사용하는 K/V input sharding 경로가 존재한다.

근거:

- `models/bos_model/llama32/tt/attention.py:799-844`

하지만 이 분기는 긴 sequence이면서 `page_table`이 없는 경우에 해당하므로 현재 paged test 경로에서는 비활성이다.

### 7.3 Speculative decode의 cache sharing

환경변수 `TT_PAGED_MULTI_TOKEN_DECODE=1`이면 여러 speculative lane이 동일 paged cache를 공유하도록 update와 paged SDPA에 `share_cache=True`를 전달한다.

근거:

- `models/bos_model/llama32/tt/attention.py:534-568`
- `models/bos_model/llama32/run_speculative_llama32.py:91-107`

현재 `test_demo_llama32.py`에서는 이 환경변수를 설정하지 않으므로 비활성이다.

### 7.4 외부 runtime/vLLM 소유 cache

Attention에는 `use_paged_kv_cache=True`일 때 내부 cache 초기화를 건너뛰고 외부에서 전달받은 cache를 사용하는 hook이 있다.

근거:

- `models/bos_model/llama32/tt/attention.py:336-338`

현재 `create_tt_model()` 경로는 이를 true로 전달하지 않으므로 내부 사전 할당 cache를 사용한다.

### 7.5 Decode trace capture/replay

Generator에는 decode trace를 capture하고 반복 replay하는 경로가 있다.

근거:

- `models/bos_model/llama32/tt/generator.py:282-387`

Trace가 활성화되면:

- 반복 command submission 비용 감소
- 고정 device input/address 재사용
- page table/current-position update 경로의 host overhead 감소

현재 test는 `enable_trace=False`이므로 적용되지 않는다.

## 8. 런타임에는 있지만 현재 모델이 사용하지 않는 기능

### 8.1 Fused K+V paged update

TTNN 런타임에는 K와 V를 한 operation으로 갱신하는 `paged_fused_update_cache` API가 있다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.hpp:26-38`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.hpp:60-62`

하지만 BOS Llama 3.2 attention은 현재 다음 두 operation을 각각 호출한다.

```text
paged_update_cache(K)
paged_update_cache(V)
```

근거:

- `models/bos_model/llama32/tt/attention.py:537-542`

따라서 fused K/V update에 의한 operation dispatch 및 공통 index/page-table 처리 절감은 현재 적용되지 않는다.

### 8.2 `serialize_paged_updates` 미사용

Attention은 다음 환경변수로 `serialize_paged_updates` 값을 계산한다.

- `TT_PAGED_MULTI_TOKEN_DECODE`
- `TT_TREE_SPECULATIVE_DECODE`

근거:

- `models/bos_model/llama32/tt/attention.py:534-536`

그러나 계산된 값이 이후 실제 update 호출이나 program config에 사용되지 않는다. 따라서 현재는 dead/incomplete plumbing으로 판단된다.

특히 tree speculative POC가 관련 환경변수를 설정하더라도 이 변수 자체로 update serialization이 적용되지는 않는다.

## 9. 현재 구현에서 확인되지 않은 KV-cache 기법

다음 기능은 현재 BOS Llama 3.2 경로에서 확인되지 않았다.

### 9.1 요청 및 prefix 재사용

- Prefix/prompt cache
- 요청 간 KV-cache 재사용
- 동일 prefix deduplication
- Page-level copy-on-write
- 대화 세션의 기존 cache continuation

현재 새 prompt 시작 시 cache 전체를 zeroing하므로 이전 prompt cache는 보존되지 않는다.

### 9.2 동적 physical page 관리

- 실제 사용량 기반 page allocation
- Page free/reclamation
- Eviction
- Compaction
- Watermark 기반 pool 관리
- 사용한 page만 선택적으로 reset

현재 테스트는 시작 시 1024 blocks를 모두 할당하고 정적 page table을 구성한다.

### 9.3 장기 context 제어

- Sliding-window KV-cache
- Ring/circular KV-cache
- Attention sink
- 오래된 token의 선택적 제거

TTNN runtime 차원의 유사 기능 가능성과 별개로, 현재 BOS Llama 경로는 관련 파라미터를 paged attention에 전달하지 않는다.

### 9.4 추가 cache 압축

- INT8 KV-cache와 per-channel scale
- FP8 scale 기반 cache
- K와 V에 서로 다른 정밀도
- Layer별 adaptive precision
- 최근/과거 token별 혼합 정밀도
- Low-rank KV compression

현재 확인된 cache 저장 정밀도 최적화는 BFP8이다.

### 9.5 Cache offload 및 계층화

- Host/CPU KV-cache offload
- NVMe offload
- DRAM↔L1 hot-page migration
- 계층형 cache

누적 cache는 장치 DRAM interleaved buffer에 유지된다.

### 9.6 Chip 내부 cache sharding

Tensor Parallel로 KV heads가 chip 사이에 분할되지만, 한 chip 안에서 누적 cache tensor 자체는 core-sharded memory config가 아니다.

Runtime paged-update validation도 현재 interleaved cache를 전제로 한다.

근거:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:55-59`

## 10. 최적화로 오해하기 쉬운 항목

### 10.1 Fused QKV projection

Q/K/V weight를 결합한 projection은 attention 연산 최적화지만 KV-cache의 저장량, 수명, paging 또는 cache update 자체를 최적화하는 기능은 아니다. 따라서 본 문서의 KV-cache 최적화 목록에서는 별도 항목으로 세지 않았다.

### 10.2 Data Parallel submesh별 cache

Data Parallel submesh마다 독립 cache pool을 갖는 것은 병렬 확장 구조다. 요청 처리량을 높일 수 있지만 요청 하나의 KV-cache 저장량을 줄이는 압축 기법은 아니다.

현재 테스트 기본값은 `data_parallel=1`이다.

### 10.3 Disk에 저장되는 초기 zero tensor

초기 zero cache의 포맷 결과를 파일 cache로 재사용하는 것은 모델 시작 비용 최적화다. 실제 사용자 prompt의 KV 값을 디스크에 저장하거나 prefix cache로 재사용하는 기능은 아니다.

## 11. 현재 구현의 장점과 병목 요약

### 장점

- GQA로 KV head 수 4배 감소
- BFP8 저장으로 BF16 대비 raw tile 약 1.88배 감소
- cache tensor의 영속 사전 할당 및 in-place 갱신
- TP 환경에서 장치당 local KV head만 저장
- Prefill의 bulk multicore physical-page fill
- 계산용 padding의 불필요한 cache write 억제
- 일반 prefill의 즉시 cache read-back 방지
- Decode update 입력의 L1 sharding
- update kernel 내부 cache-format pack
- device-side page translation
- Flash-Decode 및 online softmax
- TTNN program cache 재사용

### 현재 병목 및 비효율

- 1K 모델 실행에 32K physical pool을 할당
- 매 prompt마다 전체 pool zeroing
- 동적 page allocator/free가 없음
- Prefix cache 및 요청 간 재사용이 없음
- K와 V update를 별도 operation으로 실행
- Decode trace가 비활성
- 누적 cache가 DRAM interleaved이며 chip 내부 sharding 없음
- no-trace 경로에서 decode용 page-table/device input 관리 비용이 반복될 수 있음
- `serialize_paged_updates`가 실제로 사용되지 않음

## 12. 우선순위가 높은 개선 후보

다음은 현재 코드에 이미 적용된 기능이 아니라, 분석 결과상 효과가 클 가능성이 높은 개선 후보이다.

1. `max_num_blocks`를 실제 동시 batch와 최대 context 요구량으로 산정
   - 현재 1024 blocks를 32 blocks로 줄이면 1B 단일 칩 raw cache가 544 MiB에서 17 MiB로 감소한다.
2. 전체 pool zeroing을 사용 page reset 또는 generation/validity metadata 방식으로 변경
3. `paged_fused_update_cache`로 K/V update 통합 가능성 검증
4. `enable_trace=True` 조건에서 decode latency 및 page-table 전달 비용 측정
5. 실제 요청 수명에 맞춘 page allocator/free 도입
6. 반복 system prompt를 위한 prefix page reuse 및 copy-on-write 검토
7. `serialize_paged_updates` plumbing을 연결하거나 dead code 제거
8. 장치별 DRAM bandwidth가 병목일 경우 cache sharding 또는 hot-page 계층화 검토

## 13. 최종 분류표

| 기능 | 구현 여부 | 현재 `test_demo_llama32.py` 활성 여부 | 비고 |
|---|---:|---:|---|
| Autoregressive KV caching | 있음 | 활성 | 기본 |
| GQA | 있음 | 활성 | 실제 checkpoint에 따라 head 수 변동 |
| BFP8 KV storage | 있음 | 활성 | BF16 대비 raw tile 약 1.88배 |
| Paged KV cache | 있음 | 활성 | 고정 사전 할당 pool |
| Paged Attention | 있음 | 활성 | Flash-Decode 기반 |
| Persistent preallocated cache | 있음 | 활성 | 레이어별 K/V |
| In-place cache update/fill | 있음 | 활성 | 재할당/전체 복사 없음 |
| TP local KV-head partition | 있음 | 조건부 활성 | 멀티칩에서 효과 |
| Multicore bulk prefill fill | 있음 | 활성 | K/V 각각 한 번 |
| Padding write trimming | 있음 | 활성 | page granularity까지 |
| Prefill cache read-back 회피 | 있음 | 활성 | non-chunked prefill |
| Decode K/V input L1 sharding | 있음 | 활성 | 누적 cache는 DRAM interleaved |
| Update 내부 BFP8 packing | 있음 | 활성 | 별도 cast tensor 회피 |
| Device-side page lookup | 있음 | 활성 | current position + page table |
| Online-softmax cache streaming | 있음 | 활성 | Paged Attention 내부 |
| TTNN program cache | 있음 | 활성 | trace와 별개 |
| Chunked prefill | 있음 | 비활성 | 현재 길이가 임계값 미만 |
| Speculative shared cache | 있음 | 비활성 | 환경변수 필요 |
| External runtime-owned cache | 있음 | 비활성 | 생성 경로에서 미사용 |
| Decode trace replay | 있음 | 비활성 | 테스트에서 false |
| Fused K+V cache update | 런타임에 있음 | 미사용 | 모델은 두 번 호출 |
| Dynamic page allocation/free | 없음 | 비활성 | 정적 pool/table |
| Prefix/prompt cache | 없음 | 비활성 | 전체 reset |
| Copy-on-write/dedup | 없음 | 비활성 | 확인되지 않음 |
| Sliding-window/ring cache | 없음 | 비활성 | 확인되지 않음 |
| Cache offload | 없음 | 비활성 | 확인되지 않음 |
| 추가 INT8/FP8-scale 압축 | 없음 | 비활성 | 현재 BFP8만 사용 |
| Chip 내부 누적-cache sharding | 없음 | 비활성 | DRAM interleaved |

## 14. 주요 코드 위치

### BOS Llama 3.2

- `models/bos_model/llama32/test_demo_llama32.py`
- `models/bos_model/llama32/tt/attention.py`
- `models/bos_model/llama32/tt/common.py`
- `models/bos_model/llama32/tt/generator.py`
- `models/bos_model/llama32/tt/model.py`
- `models/bos_model/llama32/tt/model_config.py`

### TTNN paged cache

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.hpp`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/`

### TTNN paged SDPA decode

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`

### TT-Metal runtime

- `tt_metal/api/tt-metalium/program_cache.hpp`
- `tt_metal/api/tt-metalium/tt_backend_api_types.hpp`

## 15. 분석 시 주의사항

1. 실제 모델은 `model_id` 문자열이 아니라 환경변수 경로로 선택된다.
2. 따라서 실제 checkpoint의 `n_heads`, `n_kv_heads`, layer 수를 실행 전에 확인해야 한다.
3. 이 문서의 544 MiB/17 MiB 계산은 Llama 3.2 1B, 단일 칩, 16 layers, 8 local KV heads를 기준으로 한다.
4. Tensor Parallel에서는 장치당 수치는 local KV head 수에 비례해 줄어든다.
5. 메모리 계산은 raw tile payload 기준이며 allocator/runtime metadata는 제외했다.
6. 소스에 존재하는 조건부 경로와 현재 test에서 실제 실행되는 경로를 구분해야 한다.
7. 정확한 latency, DRAM bandwidth 및 core utilization은 장치 실행 profiler로 별도 검증해야 한다.

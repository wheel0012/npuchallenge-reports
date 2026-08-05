# BOS DRAM Bandwidth Benchmark Report

## 1. 목적

BOS NPU에서 현재 UMD가 노출하는 단일 4 GiB DRAM bank의 실효 대역폭을 측정하고,
Qwen2.5-3B decode-stage MatMul이 memory-bound인지 확인한다.

현재 runtime topology는 다음과 같다.

```text
Worker grid:          5 x 4
Visible DRAM channel: 1
DRAM grid:            1 x 1
DRAM bank size:       4 GiB
NPU clock:            650 MHz
```

공급사 답변에 따르면 최종 구성은 4 GiB bank 3개, 총 12 GiB일 가능성이 있으나,
현재 UMD는 bank 하나만 노출한다. 따라서 본 보고서의 측정값은 현재 노출된 단일
논리 DRAM 경로에 한정된다.

## 2. 측정 코드

원본 TT-Metal microbenchmark를 BOS용 디렉터리로 복사하여 사용했다.

- 원본: `tests/tt_metal/tt_metal/perf_microbenchmark`
- BOS 포트: `tests/tt_metal/tt_metal/perf_microbenchmark_bos`
- 단일-bank read:
  `perf_microbenchmark_bos/8_dram_adjacent_core_read/test_dram_read.cpp`
- DRAM→remote-L1 후보:
  `perf_microbenchmark_bos/9_dram_adjacent_read_remote_l1_write`

BOS 포트에는 다음을 반영했다.

- runtime-visible DRAM channel 범위 검사
- 기본 bank를 bank 0 하나로 변경
- Blackhole 512 GB/s 상수 기반 pass criterion 제거
- 예외 발생 시에도 `CloseDevice` 실행
- BF16, BFP8_B, BFP4_B 지원
- 같은 bank의 서로 다른 구간을 읽는 `--readers-per-bank` 옵션
- source archive 환경에서 유효한 JIT cache 식별자 사용

## 3. 기존 TTNN device memory roof

TTNN `to_memory_config`를 20 cores에서 실행하고 device kernel duration으로 계산한
결과다. 이 측정은 host enqueue와 synchronize 시간을 제외한다.

| dtype | 방향 | Device bandwidth |
| --- | --- | ---: |
| BF16 | DRAM→L1 read | 33.53 GB/s |
| BF16 | L1→DRAM write | 37.11 GB/s |
| BFP8_B | DRAM→L1 read | 약 27.1 GB/s |
| BFP8_B | L1→DRAM write | 약 36.9 GB/s |
| BFP4_B | DRAM→L1 read | 약 17.25 GB/s |
| BFP4_B | L1→DRAM write | 약 23.9 GB/s |

원본 결과:

```text
~/profiler_runs/bos_memory_roof_large_2026_07_19_03_41_57/memory_roof_device.csv
~/profiler_runs/bos_memory_roof_2026_07_19_03_40_39/memory_roof_device.csv
```

37 GB/s는 BF16 DRAM read가 아니라 BF16 L1→DRAM write 결과다. Qwen weight streaming과
비교할 때는 dtype별 DRAM→L1 read ceiling을 사용해야 한다.

### 3.1 `ttnn.to_memory_config`란 무엇인가

`ttnn.to_memory_config(tensor, target_memory_config)`는 tensor의 값은 유지하면서 device
내 저장 위치와 분할 방식을 목표 `MemoryConfig`에 맞게 변환하는 TTNN operation이다.
단순히 Python tensor의 속성만 바꾸는 함수가 아니며, source와 destination 구성이
다르면 새로운 device tensor를 할당하고 실제 data-movement program을 실행한다.

내부 operation 선택은 대략 다음과 같다.

| Source | Destination | 내부 경로 |
| --- | --- | --- |
| Interleaved | Sharded | `InterleavedToShardedDeviceOperation` |
| Sharded | Interleaved | `ShardedToInterleavedDeviceOperation` |
| Sharded | 다른 sharding | `ReshardDeviceOperation` |
| Interleaved | Interleaved | `CopyDeviceOperation` |
| 동일한 memory config | 동일한 memory config | 복사 없이 기존 tensor 반환 |

따라서 이 benchmark에서 `to_memory_config`는 다음 두 개의 서로 다른 실제 device
복사를 발생시킨다.

```text
read:  DRAM interleaved tensor ──InterleavedToSharded──▶ 20-core L1 width shards
write: 20-core L1 width shards ──ShardedToInterleaved──▶ DRAM interleaved tensor
```

### 3.2 Tensor와 shard를 만드는 방법

먼저 host의 BF16 random tensor를 선택한 packed dtype과 tile layout으로 변환해 DRAM에
배치한다. 이 `from_torch` 업로드는 측정 전에 끝나므로 bandwidth 시간에 포함되지 않는다.

```python
source = ttnn.from_torch(
    torch.randn(shape, dtype=torch.bfloat16),
    dtype=DTYPES[dtype_name],
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

20 cores 조건에서는 grid를 5×4로 잡고 tensor 폭을 `20 × 32 = 640` elements로 만든다.
그 뒤 width sharding을 지정해 각 core가 폭 32 elements, 즉 한 tile column씩 담당하게
한다.

```python
shard_shape = (shape[2], shape[3] // 20)  # (full height, 32)
l1_config = ttnn.create_sharded_memory_config(
    shape=shard_shape,
    core_grid=ttnn.CoreGrid(x=5, y=4),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

`create_sharded_memory_config` 자체는 이동을 실행하지 않고 배치 규칙만 기술한다. 실제
DRAM→L1 전송은 다음 호출에서 시작된다.

```python
l1_tensor = ttnn.to_memory_config(source, l1_config)
```

이 구성은 전체 tensor를 모든 core에 복제하는 cache 동작이 아니다. tensor의 서로 다른
width shard를 20개 core L1에 나눠 저장한다. shape은 tile 및 core 수에 맞춰 올림되므로
CSV의 `bytes`는 사용자가 요청한 MiB와 약간 다를 수 있다.

### 3.3 DRAM→L1 read의 device 동작

DRAM source는 tile-page 단위 interleaved tensor이고 destination은 각 worker core의 L1
shard다. `InterleavedToShardedDeviceOperation`은 core별 runtime argument에 담당 page
범위를 설정하고 다음 data-movement pipeline을 실행한다.

```text
DRAM interleaved pages
        │ reader data-movement kernel + TensorAccessor
        ▼
core-local circular buffer
        │ writer data-movement kernel
        ▼
각 core의 width-sharded L1 tensor
```

program factory는 interleaved source를 읽는 reader kernel, sharded destination을 쓰는
writer kernel, 그리고 core-local circular buffer를 생성한다. `TensorAccessor`가 page
ID를 source DRAM의 bank/NoC 주소로 변환한다. 각 core는 자기 shard에 필요한 page만
가져오므로 정상적인 20-core fan-out이 발생한다.

현재 runtime에는 논리 DRAM bank가 하나만 노출되어 있다. 따라서 20 core가 병렬로
동작하더라도 source page는 모두 bank 0 경로에 매핑되며, 측정값은 여러 reader가 단일
논리 channel을 공유했을 때의 aggregate payload bandwidth다.

### 3.4 L1→DRAM write의 device 동작

read 측정이 끝난 뒤 별도로 DRAM→L1 변환을 한 번 실행하고 synchronize하여 source L1
tensor를 준비한다. 이 준비 전송은 write 측정 구간에 포함되지 않는다. 이후 다음 호출을
반복 측정한다.

```python
dram_tensor = ttnn.to_memory_config(l1_tensor, ttnn.DRAM_MEMORY_CONFIG)
```

`ShardedToInterleavedDeviceOperation`은 반대 방향으로 동작한다.

```text
각 core의 width-sharded L1 tensor
        │ sharded reader kernel / core-local CB
        ▼
interleaved writer kernel + TensorAccessor
        │
        ▼
DRAM interleaved destination pages
```

각 core는 자기 L1 shard를 읽고, writer data-movement kernel이 destination의 interleaved
page ID에 맞춰 DRAM write를 발행한다. 현재 구성에서는 이 write 역시 bank 0으로
집중된다. read와 write가 서로 다른 kernel과 protocol 경로를 사용하므로 두 bandwidth가
같을 필요는 없다.

### 3.5 시간과 bandwidth 계산

host-synchronized 측정은 각 iteration에서 operation을 enqueue한 직후부터
`ttnn.synchronize_device(device)`가 반환될 때까지 잰다.

```python
start = time.perf_counter_ns()
output = operation()
ttnn.synchronize_device(device)
seconds = (time.perf_counter_ns() - start) / 1e9
ttnn.deallocate(output)  # 측정 종료 뒤 실행
```

이 값에는 Python 호출, output allocation, host dispatch 및 device 완료 대기가 포함된다.
반면 본 보고서의 roof는 Tracy의 `DEVICE KERNEL DURATION [ns]`를 사용하므로 host enqueue
비용을 제외한 device data-movement program의 시간이다. warmup 이후 반복 결과의 median을
사용한다.

분자는 실제 packed tile payload를 한 번 이동한 byte 수로 계산한다.

```text
payload bytes = tile count × packed bytes per tile
effective GB/s = payload bytes / device kernel seconds / 10^9
```

예를 들어 BF16 20 MiB DRAM→L1 결과는 다음과 같다.

```text
payload:       20,971,520 bytes
kernel median:    625,547.5 ns
bandwidth:     20,971,520 / (625,547.5 × 10^-9) / 10^9
             = 33.53 GB/s
```

L1→DRAM은 같은 payload와 565,114.5 ns를 사용해 37.11 GB/s다. 이 수치는 payload를
한 번만 센 effective bandwidth이며 NoC packet header, address/control traffic, padding
및 protocol overhead는 분자에 넣지 않는다.

### 3.6 이 측정값이 의미하는 범위

이 방식은 TTNN이 실제 사용하는 sharding operation, TensorAccessor, circular buffer 및
data-movement kernels를 통과하므로 단순 host memory copy가 아니다. 또한 모든 core에
tensor 전체를 넣는 것이 아니라, 20개의 L1 shard와 단일 interleaved DRAM tensor 사이의
aggregate 전송량을 측정한다.

다만 MatMul reader가 수행하는 multicast, compute와의 CB back-pressure, weight 재사용 및
output write를 그대로 재현하지는 않는다. 그러므로 이 결과는 TTNN layout conversion의
empirical DRAM roof로 사용하고, Qwen MatMul이 실제로 얻은 `DRAM` 수치와 비교해야 한다.
MatMul 자체의 bandwidth라고 해석해서는 안 된다.

benchmark의 device lifecycle은 context manager의 `finally`에서 `ttnn.close_device`를
호출하도록 구성되어 있다. 단, native `synchronize_device`가 process 내부에서 완전히
멈추면 Python의 `finally`도 그 호출이 반환되기 전에는 실행될 수 없다는 한계가 있다.

## 4. Low-level single-bank read

### 4.0 측정 메커니즘

#8 benchmark는 TTNN operation을 사용하지 않고 TT-Metal host API와 data-movement
kernel을 직접 사용한다. 전체 데이터 경로는 다음과 같다.

```text
Host random packed tiles
        │ WriteToBuffer
        ▼
Interleaved DRAM buffer, bank 0
        │ NCRISC NoC async read
        ▼
Reader Tensix core의 L1 circular buffer
```

연산 kernel이나 output tensor는 없으며, 측정 대상은 DRAM에서 reader core의 L1으로
packed tile을 공급하는 경로다.

#### 4.0.1 Host-side 입력 생성

`k`와 `n`은 실제 MatMul을 수행하기 위한 행렬 차원이 아니라 전송할 tile 수를
결정하는 편리한 shape parameter다.

```text
tile count = (k / 32) × (n / 32)
```

packed tile 크기는 다음과 같다.

| dtype | Bytes per 32×32 tile |
| --- | ---: |
| BF16 | 2,048 B |
| BFP8_B | 1,088 B |
| BFP4_B | 576 B |

따라서 입력 크기는 다음과 같이 계산한다.

```text
input bytes = tile count × packed tile bytes
```

host에서 dtype별 random packed vector를 생성하고, page size를 packed tile 크기로 한
interleaved DRAM buffer를 생성한 뒤 `WriteToBuffer`로 한 번 업로드한다. 이 초기
host→DRAM 업로드 시간은 bandwidth 측정 구간에 포함되지 않는다.

현재 UMD는 bank 하나만 노출하므로 buffer의 모든 page는 논리 bank 0에 배치된다.

#### 4.0.2 Block과 L1 CB 구성

전체 tile을 `readers_per_bank`개의 연속 구간으로 나누고, 각 reader 구간을 다시
`num_blocks`개 block으로 나눈다.

```text
tiles per reader = total tiles / readers
tiles per block  = tiles per reader / num_blocks
CB bytes         = tiles per block × packed tile bytes
```

25.5 MiB BFP8_B, reader 4개, block 128개 조건에서는:

```text
total tiles:      24,576
tiles per reader:  6,144
tiles per block:      48
L1 CB per core:   52,224 bytes
```

따라서 전체 25.5 MiB를 L1에 올려놓는 것이 아니라 약 51 KiB짜리 CB window를
반복해서 사용한다. 이 방식으로 L1 용량보다 큰 DRAM payload를 streaming할 수 있다.

#### 4.0.3 Bank-local 주소 변환

kernel은 runtime argument로 `bank_id`, NoC virtual channel 및 reader별 byte offset을
받는다. TT-Metal firmware가 준비한 다음 두 table을 사용해 bank-local address를 실제
NoC transaction으로 변환한다.

```cpp
dram_bank_to_noc_xy[noc_index][bank_id]
bank_to_dram_offset[bank_id]
```

개념적인 source address는 다음과 같다.

```text
DRAM buffer base
+ bank_to_dram_offset[bank_id]
+ reader_offset
+ block/page offset
```

여러 reader를 사용할 때 모든 reader의 `bank_id`는 0이지만, `reader_offset`을
달리해 서로 겹치지 않는 연속 구간을 읽는다. VC는 reader별로 0~3을 순환시켜 동일
NoC command path의 불필요한 충돌을 줄인다.

현재 구현에서는 UMD에 기록된 6개 DRAM NoC port를 runtime argument로 직접 선택하지
않는다. 따라서 6-reader 실험은 “6개 물리 port 실험”이 아니라 “동일한 대표 bank
endpoint에 여러 reader가 요청을 발행하는 실험”이다.

#### 4.0.4 Device kernel

reader kernel은 BRISC/NCRISC data-movement path에서 다음 작업을 반복한다.

1. L1 CB의 write pointer를 얻는다.
2. DRAM source address와 L1 destination address로 asynchronous NoC read를 발행한다.
3. page 단위로 source와 destination address를 증가시킨다.
4. transaction ID 1~3을 순환해 최대 세 block의 read request를 pipeline한다.
5. 해당 transaction ID의 barrier를 기다려 DRAM read 완료를 보장한다.

compute kernel은 실행되지 않는다. 또한 #8에서는 소비자 kernel이 없으므로 각 block은
동일한 작은 CB window를 재사용한다. 측정값에는 DRAM controller, NoC read response,
NCRISC issue/barrier 및 L1 write 비용이 함께 포함된다.

#### 4.0.5 측정 구간

각 반복의 host timer 범위는 다음과 같다.

```cpp
start = steady_clock::now();
EnqueueProgram(...);
Finish(command_queue);
ReadDeviceProfilerResults(device);
end = steady_clock::now();
```

따라서 현재 low-level 시간에는 다음이 포함된다.

- host enqueue
- command dispatch
- device kernel
- device completion synchronization
- profiler 결과 readback 호출

반대로 input 생성, DRAM buffer allocation, 초기 `WriteToBuffer`, program compile은
측정 구간 밖이다. 첫 반복이 느린 이유는 측정 구간 안에 남아 있는 cold dispatch 및
runtime overhead 때문이다. 표에서는 첫 반복을 제외한 중앙값을 사용했다.

최종 발표용 순수 device bandwidth는 Tracy의 `DEVICE KERNEL DURATION [ns]`를
분모로 다시 계산해야 한다.

#### 4.0.6 Bandwidth 식과 단위

현재 #8 소스의 계산식은 다음과 같다.

```text
(input_bytes / 1024³) / elapsed_seconds
```

따라서 log에는 `GB/s`라고 표시되지만 수치의 실제 단위는 **GiB/s**다.

```text
decimal GB/s = reported GiB/s × 1.073741824
```

예를 들어 log의 25.23은 25.23 GiB/s이며, decimal 단위로는 약 27.09 GB/s다.
TTNN memory roof와 Qwen effective weight bandwidth는 decimal GB/s를 사용하므로,
서로 비교할 때 반드시 환산해야 한다.

#### 4.0.7 검증 경로의 현재 한계

기존 benchmark는 `device->allocator()->get_base_allocator_addr(L1)`를 CB 주소로
간주하고 host에서 해당 주소를 읽는다. BOS에서는 이 위치에서 0이 반환되어 실제
program이 할당한 동적 CB 주소와 일치하지 않는 것으로 보인다.

따라서 현재 readback 실패만으로 device kernel이 DRAM을 읽지 않았다고 단정할 수는
없지만, packed 결과의 correctness도 아직 입증되지 않았다. 이하 low-level 결과는
`--bypass-check`를 사용한 provisional throughput이며, #9 포팅 또는 profiler marker를
통해 device-side 완료와 실제 CB 주소를 다시 검증해야 한다.

### 4.1 Payload 크기 영향

BFP8_B, reader 1개로 측정했다.

| Payload | Steady-state bandwidth | 해석 |
| ---: | ---: | --- |
| 1.062 MiB | 약 15–18 GiB/s | 고정 host/dispatch overhead 영향이 큼 |
| 17.0 MiB | 약 23.2–23.9 GiB/s | 안정 구간 |
| 25.5 MiB | 25.23 GiB/s = 27.09 GB/s | reader 1개 기준 중앙값 |

68 MiB, `num_blocks=512`에서는 360–1,400 GB/s라는 물리적으로 불가능한 값이
출력됐다. transaction/barrier 또는 반복 enqueue 동작의 유효 범위를 벗어난 것으로
판단하며 결과에서 제외한다.

### 4.2 Reader-count sweep

동일한 25.5 MiB BFP8_B payload를 bank 0에서 서로 다른 주소 구간으로 나누어 읽었다.
첫 실행을 제외한 중앙값이다.

| Readers on bank 0 | Reported | Decimal bandwidth |
| ---: | ---: | ---: |
| 1 | 25.23 GiB/s | 27.09 GB/s |
| 2 | 25.18 GiB/s | 27.04 GB/s |
| 4 | 26.38 GiB/s | 28.33 GB/s |
| 6 | 24.30 GiB/s | 26.09 GB/s |

reader 수를 늘려도 대역폭이 증가하지 않았다. 현재 UMD의 대표 DRAM endpoint는
reader 하나에서 이미 거의 포화되며, reader 경쟁이 증가하면 오히려 성능이 감소한다.

이 실험은 UMD에 기록된 6개 물리 NoC port를 개별적으로 선택한 실험이 아니다. 모든
reader는 동일한 `bank_id=0`과 firmware가 선택한 대표 endpoint를 사용한다.

### 4.3 Qwen dtype sweep

전송 payload를 약 24–25.5 MiB로 맞추고 reader 4개에서 측정했다. 첫 실행을 제외한
중앙값이다.

| dtype | Payload | Reported | Decimal bandwidth |
| --- | ---: | ---: | ---: |
| BF16 | 24.0 MiB | 23.97 GiB/s | 25.74 GB/s |
| BFP8_B | 25.5 MiB | 25.18 GiB/s | 27.04 GB/s |
| BFP4_B | 24.19 MiB | 20.28 GiB/s | 21.78 GB/s |

같은 byte 수에서 BFP4_B가 더 낮은 이유는 처리해야 하는 packed tile 수가 많아져
tile별 명령, 주소 계산 및 NoC transaction overhead가 증가하기 때문이다.

동일한 논리 tensor를 전송한다고 가정하면 압축에 의한 이동량 감소를 포함한 예상
전송시간 개선은 BF16 대비 BFP8_B 약 1.98배, BFP4_B 약 3.0배다.

## 5. Qwen decode MatMul의 실제 weight bandwidth

Qwen single-layer Tracy profile에서 다음 memory placement를 사용한 MatMul을 분석했다.

```text
Activation A: L1 interleaved
Weight B:     DRAM interleaved
Output C:     L1 interleaved
Core count:   4
M:            32
```

packed weight bytes를 device kernel duration으로 나눈 결과다.

| MatMul M×K×N | Weight dtype | Device time | FLOPS | Effective weight BW |
| --- | --- | ---: | ---: | ---: |
| 32×2048×2560 | BFP8_B | 259.0 µs | 1.30 TFLOPS | 21.51 GB/s |
| 32×2048×2048 | BFP8_B | 208.4 µs | 1.29 TFLOPS | 21.38 GB/s |
| 32×2048×11008 | BFP4_B | 912.1 µs | 1.58 TFLOPS | 13.90 GB/s |
| 32×11008×2048 | BFP8_B | 1087.9 µs | 1.33 TFLOPS | 22.02 GB/s |

Arithmetic intensity는 BFP8_B 약 60.24 OP/byte, BFP4_B 약 113.78 OP/byte다.

| dtype | TTNN DRAM-read roof | Qwen GEMM BW | BW utilization |
| --- | ---: | ---: | ---: |
| BFP8_B | 약 27.1 GB/s | 21.4–22.0 GB/s | 약 79–81% |
| BFP4_B | 약 17.25 GB/s | 13.90 GB/s | 약 81% |

반면 4-core empirical compute roof 대비 utilization은 BFP8 HiFi2 약 28%,
BFP4 LoFi 약 20%다.

따라서 다음 결론을 얻는다.

> Qwen's decode-stage MatMul is clearly memory-bound, achieving about 80% of
> the empirical DRAM-read ceiling but only 20–28% of the compute ceiling.

## 6. 일반 GEMM benchmark와의 차이

원본 GEMM benchmark는 다음 범위의 비교적 큰 행렬을 사용한다.

```text
Minimum: 256 x 320 x 320
Maximum: 8192 x 10240 x 10240
```

20-core sweep의 최대 effective weight bandwidth는 다음과 같다.

| GEMM dtype | Maximum weight BW |
| --- | ---: |
| BF16×BFP8_B | 19.16 GB/s |
| BFP8_B×BFP8_B | 19.02 GB/s |
| BF16×BFP4_B | 13.69 GB/s |

최대 TOPS가 나오는 큰 GEMM에서는 weight 재사용률이 높아 effective weight bandwidth가
BFP8 약 1.5 GB/s, BFP4 약 7.7 GB/s에 불과하다. 이는 DRAM이 느린 것이 아니라
L1에 들어온 weight를 여러 M tile row에서 재사용하기 때문이다.

tile-level weight 재사용 기회는 대략 `M/32`다.

| Workload | M | M tile rows |
| --- | ---: | ---: |
| Qwen decode | 32 | 1 |
| GEMM minimum | 256 | 8 |
| BFP4 peak | 1536 | 48 |
| BFP8 peak | 8192 | 256 |

Qwen batch-1 decode에서 M=32는 tile alignment에 의한 padding을 포함하므로 유효한
weight 재사용은 사실상 1회에 가깝다.

## 7. 측정 한계

1. Low-level #8 시간은 host의 `EnqueueProgram + Finish + profiler read`를 포함한다.
   최종 발표용 ceiling은 device kernel duration으로 다시 계산해야 한다.
2. #8의 기존 validation은 allocator base를 CB 주소로 가정한다. BOS에서는 해당
   주소를 읽을 때 0이 반환되어 실제 동적 CB 주소와 일치하지 않는 것으로 보인다.
   따라서 reader/dtype sweep은 `--bypass-check` 기반 provisional 결과다.
3. `Test Passed`는 bypass 모드에서 실행 완료와 device close를 의미하며, packed
   readback correctness를 의미하지 않는다.
4. 현재 UMD는 DRAM bank 하나만 노출하므로 3-bank aggregate bandwidth를 측정할 수 없다.
5. 3-bank UMD를 받기 전에는 `NUM_DRAM_BANKS`나 NoC 좌표를 추측해 수정하지 않는다.

## 8. 다음 단계

1. #9 DRAM→reader L1→remote compute L1 benchmark를 BOS 1-bank topology에 맞게 포팅
2. BF16/BFP8_B/BFP4_B에 대해 device-profiler duration 기반 bandwidth 측정
3. #8 local-L1 roof와 #9 remote-L1 supply roof 비교
4. Qwen GEMM의 13.9–22.0 GB/s와 비교하여 손실 구간 분리
5. 실제 3-bank UMD 수령 후 bank별 alias/capacity 검사
6. bank 0/1/2 단독 및 동시 bandwidth 측정

## 9. 재실행 예시

```bash
cd /home/iris_hb4/tt-metal-prof-src

cmake --build build --target test_dram_read_bos -j2

TT_METAL_HOME=/home/iris_hb4/tt-metal-prof-src \
build/test/tt_metal/perf_microbenchmark_bos/8_dram_adjacent_core_read/test_dram_read_bos \
  --k 8192 --n 3072 \
  --num-blocks 128 --num-tests 10 \
  --data-type 0 \
  --num-banks 1 --bank-start-id 0 \
  --readers-per-bank 4 \
  --bypass-check
```

`test_dram_read_bos`의 dtype 번호는 현재 다음과 같다.

| Option | dtype |
| ---: | --- |
| 0 | BFP8_B |
| 1 | BF16 |
| 2 | BFP4_B |

## 10. Qwen decode가 skinny GEMM인 이유

MatMul을 다음과 같이 표기한다.

```text
A[M, K] × W[K, N] → O[M, N]
```

Qwen2.5-3B의 주요 decode-stage MatMul은 다음처럼 `M`만 매우 작고 `K`, `N`은 수천에서
만 단위로 크다.

```text
QKV:      [32,  2048] × [2048,  2560]
O proj:   [32,  2048] × [2048,  2048]
MLP up:   [32,  2048] × [2048, 11008]
MLP down: [32, 11008] × [11008, 2048]
```

이처럼 한쪽 차원은 한 tile row로 얇고 나머지 두 차원은 매우 넓은 직사각형인 연산을
skinny GEMM이라고 부른다. 여기서 `M=32`가 실제로 새 token 32개를 뜻하는 것은 아니다.
batch-1 autoregressive decode는 한 번에 새 token 하나를 처리하지만, Tensix tile 크기인
32에 맞추기 위해 token 축을 한 tile row까지 padding한 결과다. 따라서 유효 workload는
GEMV에 가깝지만 device에서는 tile-aligned GEMM으로 실행된다.

### 10.1 왜 weight 재사용이 낮은가

동일한 weight tile은 서로 다른 `M` tile row를 계산할 때 재사용할 수 있다. 이상적인
tile-level 재사용 기회는 대략 다음과 같다.

```text
weight reuse opportunity ≈ M / 32
```

큰 정사각 GEMM에서 `M=8192`라면 같은 weight tile을 최대 256개 activation tile row에
사용할 수 있다. 반면 Qwen decode의 `M=32`에는 tile row가 하나뿐이므로 DRAM에서 읽어온
weight를 사실상 한 번 계산한 뒤 다음 token에서 다시 읽게 된다. 모델 weight 전체를
L1에 유지할 수 없기 때문에 token마다 대규모 weight streaming이 반복된다.

### 10.2 왜 compute peak에 도달하기 어려운가

작은 `M`은 다음 문제를 동시에 만든다.

- weight byte당 수행할 MAC 수가 적어 arithmetic intensity가 낮다.
- M 방향으로 나눌 tile row가 하나뿐이라 core 간 작업 분할 선택지가 줄어든다.
- K/N 방향으로만 길게 분할하면서 multicast, reduction 및 synchronization 비용의 비중이
  커질 수 있다.
- tile padding으로 실제 token 계산에 쓰이지 않는 row도 처리한다.
- kernel launch와 CB/NoC pipeline의 고정 비용을 충분한 연산량으로 상쇄하기 어렵다.

결과적으로 DRAM에서 weight를 공급하는 시간이 Tensix compute 시간보다 먼저 한계에
도달한다. 충분히 큰 `K×N` weight matrix라고 해서 compute-bound가 되는 것은 아니다.
행렬의 총 원소 수뿐 아니라 `M` 방향 재사용 깊이가 중요하다.

### 10.3 Prefill과 decode의 차이

prefill은 여러 input token을 한꺼번에 처리하므로 `M`이 sequence chunk 크기만큼 커진다.
같은 weight를 여러 token row에 재사용할 수 있어 arithmetic intensity와 core utilization이
상승한다. 반면 decode는 한 iteration에서 새 token 하나만 처리하므로 `M`이 다시 1,
tile 표현으로는 32가 된다.

```text
Prefill: M이 큼  → weight 재사용 많음 → compute-bound에 가까워질 수 있음
Decode:  M≈1     → weight 재사용 거의 없음 → memory-bound가 되기 쉬움
```

따라서 본 보고서에서 Qwen MatMul의 약 20 GB/s를 평가할 때 정사각 GEMM의 peak TFLOPS만
분모로 쓰면 병목을 잘못 해석할 수 있다. Qwen-like streaming ceiling과 동일 shape의
skinny GEMM empirical ceiling을 함께 비교해야 한다.

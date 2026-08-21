# BOS 64K SDPA 3-endpoint vs 6-endpoint wait 분해

날짜: 2026-08-11

## 결론

K-chunk를 128로 고정하고 dual-NoC를 양쪽 모두 켠 상태에서 endpoint 수만 3에서 6으로 늘렸다.
16 reader의 endpoint load는 `5/0/5/6/0/0`에서 `3/3/2/3/3/2`로 바뀌었다.

- reader K+V barrier 평균 합: `1669.096 us` → `556.299 us`, **66.67% 감소**
- compute TRISC0 K+V input wait 평균 합: `396.584 us` → `22.817 us`, **94.25% 감소**
- profiled critical kernel span: `3014.568 us` → `2408.938 us`, **20.09% 감소**, `1.251x`
- 같은 payload로 계산한 profiled effective K/V bandwidth: `47.306 GB/s` → `59.199 GB/s`
- PCC: 양쪽 모두 `0.9998791594607118`

6-endpoint는 장식 옵션이 아니다. 3-endpoint에서 노출되던 DRAM service wait를 크게 숨기고
critical kernel을 약 20% 줄였다. 다만 barrier/wait 감소율을 kernel 개선율로 직접 더하면 안 된다.
대부분의 wait가 core 사이 또는 compute와 겹치기 때문이다.

## 장치와 실행 범위

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: `5×4 = 20 cores`
- active SDPA compute/reader cores: 16
- idle worker cores: 4
- physical DRAM: 3 banks
- worker NoC endpoints: bank당 2개, 총 6개
- workload: paged SDPA decode, batch 1, Q heads 24, KV heads 8, head dim 128
- sequence/current position: 65536/65535
- Q/K chunk: 128/128
- Q dtype: BF16
- K/V dtype: BFP8_B
- page block: 128 tokens인 PCC runner

`6-endpoint`는 reader core가 6개라는 뜻이 아니다. 16 reader core 각각이 6개 endpoint 중 하나를
선택한다. program grid 20과 active core 16도 구분한다.

## 통제 변수

양쪽 공통:

```text
TT_METAL_SDPA_DECODE_DUAL_NOC=1
TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=0
TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=0
TT_METAL_SDPA_DECODE_ROUTE_OVERLAP_OPTIMIZED=0
TT_METAL_SDPA_DECODE_TAGGED_ASYNC=0
TT_METAL_SDPA_DECODE_SIX_READER=0
TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0
TT_METAL_SDPA_DECODE_INNER_K_CHUNK_SIZE=0
SDPA_K_CHUNK_SIZE=128
```

변수는 `TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=3|6` 하나다. TurboQuant opt-in은 사용하지 않았다.

## 계측 정의

새 계측은 `TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE`의 opt-in mode다. unset이면 production path에
zone이 추가되지 않는다.

- `reader`: NCRISC의 각 `noc_async_read_barrier()` 누적을 K/V별로 기록
  - `SDPA_K_READ_BARRIER`
  - `SDPA_V_READ_BARRIER`
- `input_wait`: compute unpack thread의 K/V CB wait 누적을 기록
  - `SDPA_K_CB_WAIT`
  - `SDPA_V_CB_WAIT`

reader barrier는 outstanding read 완료 대기만 측정한다. address 계산, read issue, CB reserve 시간은
포함하지 않는다. input wait는 TRISC0 수치를 사용한다. TRISC1/2의 동일 macro 값은 수 us 수준이며
consumer readiness의 대표값으로 합산하지 않았다.

## 결과

### Critical path

| capture mode | 3EP critical kernel | 6EP critical kernel | 감소 | speedup |
|---|---:|---:|---:|---:|
| reader | 3014.568 us | 2408.938 us | 20.09% | 1.251x |
| input_wait | 3024.066 us | 2404.265 us | 20.50% | 1.258x |

두 독립 compile mode에서 critical span이 약 10 us 이내로 재현됐다. endpoint 효과는 phase-zone 종류에
민감한 우연이 아니다.

### Reader barrier

| metric, 16-core 평균 | 3EP | 6EP | 감소 |
|---|---:|---:|---:|
| K read barrier | 823.411 us | 269.709 us | 67.24% |
| V read barrier | 845.685 us | 286.590 us | 66.11% |
| K+V | 1669.096 us | 556.299 us | 66.67% |

endpoint별 K+V barrier 평균:

| 구성 | endpoint load | endpoint별 평균 K+V barrier |
|---|---|---|
| 3EP | x0=5, x2=5, x3=6 | x0=1416.374 us, x2=1360.139 us, x3=2137.161 us |
| 6EP | x0=3, x1=3, x2=2, x3=3, x4=3, x5=2 | 581.175, 504.449, 491.755, 541.636, 502.979, 763.280 us |

3EP tail은 `2138.600 us`다. 6EP tail은 x5의 `768.006 us`다. endpoint를 늘려도 x5 tail은 남지만,
최악값은 64% 줄었다.

### Compute input wait

| metric, TRISC0 16-core 평균 | 3EP | 6EP | 감소 |
|---|---:|---:|---:|
| K CB wait | 158.808 us | 14.712 us | 90.74% |
| V CB wait | 237.776 us | 8.105 us | 96.59% |
| K+V CB wait | 396.584 us | 22.817 us | 94.25% |

reader barrier는 3배 빨라졌지만 exposed compute wait는 17.38배 줄었다. 6EP service가 compute보다
충분히 앞서 도착하면서 wait 대부분이 critical path 밖으로 이동한 결과로 해석한다.

### NoC와 endpoint mapping

- 3EP: endpoint `x0/x2/x3 = 5/5/6`, reader NoC load `NOC0/NOC1 = 5/11`
- 6EP: endpoint `x0..x5 = 3/3/2/3/3/2`, reader NoC load `NOC0/NOC1 = 8/8`

따라서 이번 A/B가 입증하는 것은 “6 endpoint를 허용한 현재 mapping 전체”의 효과다. endpoint service
parallelism과 그 결과 생긴 NoC load 균형을 분리하지 않는다. dual-NoC는 양쪽 공통이므로 dual-NoC
단독 효과도 이 결과에서 주장하지 않는다.

## 관측과 추론

### 관측 사실

- 3EP와 6EP 모두 동일 PCC를 통과했다.
- 3EP는 endpoint x3와 NOC1에 load가 몰렸다.
- 6EP는 endpoint load와 NOC load가 거의 균등해졌다.
- reader barrier, compute input wait, critical kernel span이 모두 감소했다.
- 모든 four profiler capture는 exit 0, `SDPA_CORRECT`, `DEVICE_CLOSED`, ops CSV와 device CSV를 남겼다.

### 강한 추론

K128의 3EP 경로는 compute-only 병목이 아니다. DRAM endpoint service tail이 compute CB wait로 직접
노출된다. 6EP는 그 wait를 거의 제거한다. 6EP 이후 남은 약 2.4 ms는 matmul/softmax/reduction 및
숨겨지지 않은 reader issue/service가 섞인 다음 병목이다.

### 미검증 가설

- x5 endpoint의 6EP tail을 줄이면 추가 개선이 가능하다.
- 6EP에서 endpoint mapping을 유지하고 NOC0/NOC1만 일부러 불균형하게 만들면 NoC balance의 독립
  효과를 분리할 수 있다.
- K256에서는 iteration 수가 반으로 줄어 barrier 호출 수도 변하므로 이번 K128 비율이 그대로
  유지되지 않는다.

## 유효 bandwidth 해석

16 cores가 읽는 K+V payload `142,606,336 bytes`를 reader-mode critical span으로 나눴다.

```text
3EP: 142,606,336 B / 3.014568 ms = 47.306 GB/s
6EP: 142,606,336 B / 2.408938 ms = 59.199 GB/s
```

이는 DRAM controller counter가 아니라 SDPA critical-span 기반 effective bandwidth다. compute와
reduction을 포함하므로 physical DRAM peak와 직접 비교하지 않는다. profiler zone overhead도 포함한다.

## 검증 및 안전 상태

- `ttnncpp`, `ttnn` build: 성공
- deployed `_ttnncpp.so` SHA-256:
  `0e4a89a0b79a3e43461d963cfdcd6ff697418e1f506cdbcfaf3fcf1e83f16376`
- 재부팅 후 최초 32×32 BF16 add: `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0
- profiler 없는 3EP PCC gate: exit 0
- profiler 없는 6EP PCC gate: exit 0
- four profiler captures: 전부 exit 0, 정상 close

legacy 상수-V latency runner는 correctness threshold `0.1`에서 공통 max error `0.6875`를 거부해
측정 전에 exit 1했다. device는 정상 close했다. 이 run은 performance 결과로 사용하지 않았다.

## Artifact

Profiler root:

```text
/home/iris_hb4/profiler_runs/sdpa_endpoint_3ep_vs_6ep_metrics_2026_08_11/
├── reader_3ep/
├── reader_6ep/
├── input_wait_3ep/
└── input_wait_6ep/
```

각 하위 디렉터리는 `.logs/profile_log_device.csv`, Tracy host trace, ops report를 포함한다.

적용 patch:

```text
/home/iris_hb4/tmp/codex-patches/20260811-003000-sdpa-endpoint-metrics.patch
SHA-256: 82535195cde96fffec9692797e12cf89300affbe9e34fa73a52c415f17c474c0
```

## 재현 명령 형태

```bash
env \
  TT_METAL_DEVICE_PROFILER=1 \
  TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=reader \
  SDPA_K_CHUNK_SIZE=128 \
  TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=3 \
  TT_METAL_SDPA_DECODE_DUAL_NOC=1 \
  TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=0 \
  TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=0 \
  TT_METAL_SDPA_DECODE_ROUTE_OVERLAP_OPTIMIZED=0 \
timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy \
  -p -r --check-exit-code --sync-host-device \
  -o <capture-dir> -n <capture-name> \
  tests/bos_model/run_sdpa_kchunk_profile.py
```

`reader`를 `input_wait`, endpoint count를 `3|6`으로 바꿔 four captures를 만든다.


## 2026-08-19 K/V issued-byte accounting 검증

### 결론

기존의 SDPA 60--70 GB/s는 physical DRAM bus counter가 아니라 **K/V payload bytes를 SDPA reader 또는
kernel elapsed time으로 나눈 effective delivery rate**다. 이번에는 reader kernel이 실제로 실행하는 loop
bounds와 tile encoding 크기로 코어별 issued K/V bytes를 기록하고, 전체 reader service 시작/종료 timestamp를
동시에 기록해 분자와 분모를 직접 맞췄다.

Stable K256·6-endpoint isolated SDPA 64K에서 다음을 얻었다.

| Metric | Value |
|---|---:|
| Active SDPA reader cores | 16 |
| K/V encoded bytes issued per core | 8,912,896 B |
| Aggregate K/V encoded bytes issued | 142,606,336 B |
| Logical useful K/V bytes, 1 B/element | 134,217,728 B |
| BFP8 tile encoding overhead | 6.25% |
| Global reader service envelope | 1,338,066 cycles = 2,058.563 us |
| Encoded issued rate / reader envelope | **69.275 GB/s** |
| Logical useful rate / reader envelope | **65.200 GB/s** |
| Device kernel duration | 2,098.660 us |
| Encoded issued rate / whole kernel | **67.951 GB/s** |
| Logical useful rate / whole kernel | **63.954 GB/s** |

따라서 60--70 GB/s 범위 자체는 산술적으로 재현된다. 이전 K128 6-endpoint 보고서의 59.199 GB/s도
같은 142.606 MB encoded K/V payload를 당시 2.409 ms critical kernel span으로 나눈 값과 일치한다.
다만 명칭은 `DRAM bandwidth`보다 `effective encoded K/V delivery rate`가 정확하다.

### 계측 정의

`TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=reader`일 때만 다음 timestamp data를 기록하도록 했다.
Unset production path에는 marker와 byte 산식이 컴파일되지 않는다.

- `SDPA_KV_SERVICE_START`: page-table 준비 뒤, 첫 head/chunk의 CB reserve·주소 계산·K/V issue 직전
- `SDPA_KV_SERVICE_END`: 마지막 K/V completion barrier와 CB publication 뒤
- marker data: 해당 reader가 실행할 K/V NoC payload byte 수

일반 paged BFP8 path의 코어별 issued bytes는 runtime assignment로 계산한다.

```text
assigned_chunks × dynamic_chunk_tile_rows × assigned_streams
  × (DHt × encoded_K_tile_bytes + vDHt × encoded_V_tile_bytes)
```

이번 구성에서는 16개 reader가 각각 8,912,896 B를 기록했고 start/end marker도 각각 16개씩 존재했다.
총 142,606,336 B는 64K × 8 KV heads × 128 head dim × K/V 2개 × 1 B의 logical
134,217,728 B보다 6.25% 크다. BFP8 tile의 encoded transfer size를 사용했기 때문이다.

`SERVICE_START→END`는 순수 DRAM service time이 아니다. 다음을 포함한다.

- page translation 이후 K/V address generation
- NoC read issue와 completion barrier
- CB reserve/publication
- consumer가 CB를 비울 때까지의 backpressure

Q, page table, mask, output traffic은 issued K/V bytes에서 제외했다. NoC packet/header overhead, controller command,
retry 및 physical bus occupancy도 세지 않는다. 따라서 이 값은 exact K/V payload accounting이지만 physical DRAM
utilization counter는 아니다.

### 실행 구성

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- active SDPA reader/compute cores: 16
- physical DRAM: 3 banks, 6 worker endpoints
- endpoint load x0..x5: 3/2/3/3/3/2
- reader NoC load: NoC0/NoC1 = 8/8
- sequence/current position: 65536/65535
- Q heads/KV heads/head dim: 24/8/128
- K/V dtype/layout: BFP8_B, DRAM interleaved, paged KV
- K chunk/page block: 256/32 tokens
- correctness: PCC 0.9943993275067777
- profiler: one isolated operation, no NoC trace

K/V barrier 누적은 16-reader 평균 K 537.190 us, V 545.405 us, 합 1,082.595 us였다. 이는 service
envelope보다 작다. Barrier 합만으로는 address generation, issue, CB backpressure와 K/V-compute overlap을 설명하지
못한다. Reader envelope가 whole-kernel duration의 98.09%인 것도 “DRAM이 98% busy”라는 뜻이 아니라, reader가
compute와 함께 거의 kernel 전 구간에 걸쳐 살아 있다는 뜻이다.

Profiler CSV의 `PM REQ I BW=51.536`도 별도 performance-model 입력 정의를 사용하므로 67.951 GB/s와 같은
metric이 아니다. 발표에서는 아래 세 숫자를 섞지 않는다.

1. logical useful K/V rate: 63.954 GB/s over whole kernel
2. encoded issued K/V rate: 67.951 GB/s over whole kernel
3. physical DRAM utilization: 이번 계측으로는 미측정

### Microbenchmark와의 관계

Interleaved DRAM microbenchmark의 8 KiB·20-reader 최고 57.715 GB/s와 SDPA의 67.951 GB/s를 동일 ceiling의
utilization 비율로 비교하지 않는다. 양쪽 모두 payload/time 형태지만 request geometry, address mapping,
reader lifetime, working-set 반복과 CB/compute coupling이 다르다. 이번 결과는 SDPA 수치의 byte/time accounting을
검증한 것이며, physical controller throughput이나 microbenchmark 대비 utilization을 검증한 것은 아니다.

### 재현 명령

```bash
env TT_METAL_DEVICE_PROFILER=1 TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=reader SDPA_SEQ_LEN=65536 SDPA_K_CHUNK_SIZE=256 SDPA_BLOCK_SIZE=32 TT_METAL_SDPA_DECODE_DUAL_NOC=1 TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1 TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1 TT_METAL_SDPA_DECODE_SIX_READER_SHARDED=0 TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER=0 timeout --signal=INT --kill-after=15s 180s /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy -p -r --check-exit-code --sync-host-device -o /home/iris_hb4/profiler_runs/sdpa_kv_issued_envelope_final_2026_08_19_03_24_00 -n sdpa_kv_issued_envelope_k256_6ep_final tests/bos_model/run_sdpa_kchunk_profile.py
```

### Artifacts와 변경

- final run: `/home/iris_hb4/profiler_runs/sdpa_kv_issued_envelope_final_2026_08_19_03_24_00`
- device CSV SHA-256: `1869a5bc6ce22501815b1a59418f6c6ff5986af430a053ba60c15938a149850e`
- ops CSV SHA-256: `b969b73a5eaaa0c5fd81558d43769b19b8b05e35f575de209e4be2a2741da603`
- source patch: `/home/iris_hb4/tmp/codex-patches/20260819-033000-sdpa-kv-envelope.patch`
- byte-formula fix: `/home/iris_hb4/tmp/codex-patches/20260819-034000-sdpa-kv-byte-fix.patch`

첫 capture는 DHt/vDHt tile-column multiplier를 누락해 issued bytes를 정확히 4배 과소계산했다. 즉시 공식을
수정하고 correctness를 다시 통과한 뒤 final capture를 생성했다. 첫 capture의 byte 수치는 폐기하며 final
artifact만 canonical 결과로 사용한다.

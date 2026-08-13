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

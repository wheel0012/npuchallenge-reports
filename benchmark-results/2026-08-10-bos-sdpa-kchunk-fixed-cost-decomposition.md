# BOS SDPA K128/K256 chunk 고정비 분해

날짜: 2026-08-10

## 결론

64K isolated paged SDPA에서 K chunk를 128에서 256 tokens로 키워 얻은 기존 kernel 단축
151.937 us 중 150.176 us, 98.84%를 compute phase 계측으로 설명했다.

K256은 online-softmax merge를 255회에서 127회로 줄여 419.653 us를 절약했고, 현재 chunk의
max/sub/exp/sum 처리에서 219.566 us를 더 절약했다. 반대로 큰 chunk의 QK와 PV 구간은 합계
489.043 us 길어졌다. 순효과는 -150.176 us다. 따라서 개선의 핵심은 DRAM payload 감소가 아니라
chunk별 online-softmax 작업 감소다. 큰 matmul 구간의 비용 증가가 이득 대부분을 상쇄하여 K256부터
plateau가 생긴다.

## 하드웨어와 경로

- board: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5x4, 20 cores
- SDPA active cores: 16; 5x4 program grid와 구분
- DRAM: 3 physical banks, bank당 2 worker NoC endpoints, 총 6 endpoints
- KV layout: interleaved paged KV cache
- sequence/current position: 65,536 tokens
- Q heads / KV heads / head dimension: 24 / 8 / 128
- page block: 128 tokens
- K/V dtype: BFLOAT8_B
- profiler clock: 650 MHz
- six-endpoint, dual-NoC, tagged, inner-K streaming, helper opt-in: 모두 off

## 측정법

`TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE` opt-in을 추가했다.

| mode | SumN1 | SumN2 |
|---|---|---|
| `matmul` | `SDPA_PHASE_QK` | `SDPA_PHASE_PV` |
| `softmax` | `SDPA_PHASE_CURRENT_SOFTMAX` | `SDPA_PHASE_ONLINE_MERGE` |
| `empty` | `SDPA_PHASE_EMPTY` | 없음 |

zone은 매 chunk에서 timer event를 내지 않는다. `DeviceZoneScopedSumN1/N2`가 kernel 내부에서
누적한 뒤 종료 시 core/RISC당 한 번만 기록한다. 따라서 profiler buffer overflow 없이 64K 전체
누적 시간을 얻는다.

phase 경계는 다음과 같다.

- QK: `Q @ K`의 `cb_matmul_blocks`
- current softmax: mask, max reduction, subtract/exp, sum reduction
- PV: normalized score와 V의 `cb_matmul_blocks`
- online merge: 이전 chunk의 max/sum/output 재보정과 현재 결과 누적
- empty: chunk loop 안의 빈 SumN scope. zone 자체의 누적 비용 추정용

각 phase에서 core별 TRISC_0/1/2 누적값 중 최댓값을 cooperative wall-time proxy로 선택하고,
16 active cores의 평균을 냈다. 빈 zone 비용은 동일 chunk 수에 맞춰 뺐다. online merge만 첫 chunk에
없으므로 empty 값에 `(chunks - 1) / chunks`를 곱해 뺐다.

## 정적 작업량

| 항목 | K128 | K256 |
|---|---:|---:|
| chunks/core | 256 | 128 |
| QK calls/core | 256 | 128 |
| current-softmax calls/core | 256 | 128 |
| PV calls/core | 256 | 128 |
| online merges/core | 255 | 127 |
| K+V payload | 136 MiB | 136 MiB |

총 QK/PV 산술량과 K/V payload는 같다. 호출 shape와 호출 수만 변한다.

## 결과

### Correctness와 completion

| 구성 | PCC | max abs | completion |
|---|---:|---:|---|
| K128 | 0.9998791595 | 0.04526168 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |
| K256 | 0.9999178293 | 0.02732441 | `SDPA_CORRECT`, `DEVICE_CLOSED`, exit 0 |

profiler-free gate 6개와 완료 capture 6개가 correctness, explicit close, exit 0을 기록했다.

### Raw cooperative phase 누적 시간

| phase | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| QK | 555.121 us | 919.806 us | +364.685 us |
| PV | 621.781 us | 741.537 us | +119.756 us |
| current softmax | 1,018.435 us | 796.568 us | -221.867 us |
| online merge | 848.473 us | 426.520 us | -421.953 us |
| empty SumN | 4.199 us | 1.898 us | -2.301 us |

empty 값은 256회/128회 scope 전체 누적이다. 한 scope당 약 16.4 ns/14.8 ns다. 기존 151.937 us
개선에서 profiler scope 수 차이가 차지할 수 있는 값은 약 2.301 us, 1.51%다.

### Empty-overhead 보정

| phase | K128 | K256 | K256 - K128 |
|---|---:|---:|---:|
| QK | 550.922 us | 917.908 us | +366.986 us |
| PV | 617.582 us | 739.639 us | +122.057 us |
| current softmax | 1,014.236 us | 794.670 us | -219.566 us |
| online merge | 844.290 us | 424.637 us | -419.653 us |
| 합계 | 3,027.030 us | 2,876.854 us | **-150.176 us** |

기존 독립 capture의 전체 device kernel은 3.482728 ms에서 3.330791 ms로 151.937 us 줄었다.
phase 합의 150.176 us 감소와 차이는 1.761 us다.

### 절약과 상쇄

128개 감소 chunk 기준으로 보면 다음과 같다.

| 성분 | eliminated chunk당 순효과 |
|---|---:|
| online merge | -3.278 us |
| current softmax | -1.715 us |
| QK 증가 | +2.867 us |
| PV 증가 | +0.954 us |
| 합계 | -1.173 us |

기존 전체-kernel 관측값은 eliminated chunk당 -1.187 us다.

## 판정

### 관측 사실

1. K128과 K256의 K/V payload는 모두 136 MiB다.
2. K256의 online merge 누적 시간은 49.72% 감소했다.
3. current-softmax 누적 시간은 21.56% 감소했다. chunk 수는 절반이지만 chunk당 token 연산이 커서
   정확히 절반이 아니다.
4. QK/PV 누적 구간은 합계 489.043 us 증가했다.
5. 네 phase의 순감소 150.176 us가 전체 kernel 감소 151.937 us의 98.84%를 설명한다.
6. 빈 SumN zone의 K128/K256 차이는 2.301 us로 작다.

### 강한 추론

K256 sweet spot의 직접 원인은 chunk별 online-softmax merge와 관련 호출 고정비 감소다. DRAM
traffic 감소가 아니다. K256의 큰 QK/PV shape는 총 산술량이 같아도 현재 LLK block schedule에서 더
비싸다. softmax 절약 639.219 us 중 matmul 증가 489.043 us가 상쇄되고 150.176 us만 남는다.

K512가 K256보다 6.542 us만 빠른 기존 결과도 이 구조와 맞는다. merge 수 추가 감소가 더 큰 CB wait와
matmul/shape 비용에 거의 상쇄된다.

### 한계

- 각 mode/K 조합은 measured capture 1회다.
- phase는 다른 capture에서 각각 측정했다. 합은 동시 단일-run waterfall이 아니다.
- cooperative wall-time proxy는 세 TRISC 중 최장 누적값이다. 세 pipeline의 세부 overlap은 분리하지 않는다.
- QK/PV zone 안에는 `cb_matmul_blocks` 내부 wait가 포함될 수 있다. 순수 math-engine cycle만 뜻하지 않는다.
- current-softmax는 token 비례 산술과 호출 고정비를 함께 포함한다. 순수 고정항을 완전히 분리하려면
  K64/128/256 이상의 회귀가 추가로 필요하다.
- isolated SDPA 결과다. full-model tokens/s 개선과 같지 않다.

## 중단 사건

첫 softmax/K128 capture는 다른 사용자의 NIAH process PID 651935가 device lock을 점유해 device open
전에 대기했다. 외부 timeout으로 exit 124가 발생했고 완성 artifact는 host Tracy 파일뿐이다. SDPA
kernel 실패로 분류하지 않았다. 안전 규칙에 따라 장치를 격리했고, 사용자 재부팅 확인 뒤 첫 workload인
32x32 BF16 add가 `SMOKE_VALUE 2.0`, `DEVICE_CLOSED`, exit 0으로 통과한 뒤 capture를 새 디렉터리에서
재개했다.

## 재현

```bash
env \
  TT_METAL_DEVICE_PROFILER=1 \
  TT_METAL_SDPA_DECODE_CHUNK_PHASE_PROFILE=softmax \
  SDPA_SEQ_LEN=65536 \
  SDPA_K_CHUNK_SIZE=256 \
timeout --signal=INT --kill-after=15s 180s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python -m tracy \
  -p -r --check-exit-code --sync-host-device \
  -o <capture-dir> -n sdpa_chunk_phase_softmax_k256 \
  tests/bos_model/run_sdpa_kchunk_profile.py
```

`mode`와 `SDPA_K_CHUNK_SIZE`를 `matmul|softmax|empty`, `128|256`으로 바꿔 여섯 capture를 만든다.

## Artifact

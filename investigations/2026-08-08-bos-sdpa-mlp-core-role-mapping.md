# BOS stable SDPA·MLP core-role mapping 복원

날짜: 2026-08-08

## 결론

장치 재실행 없이 현재 source, stable opt-in, 기존 device profile을 결합해 core 배치를 복원했다.

- SDPA: 5×4 program grid, 16 active reader/compute, 4 idle math core
- MLP: 5×4 program grid, 12 active weight-reader/compute, 8 non-worker math core
- SDPA는 core별 DRAM endpoint와 reader/writer NoC까지 결정적으로 복원됨
- MLP stable fanout-2는 core별 DRAM view와 lane까지 복원됨. Generic addrgen이라 explicit endpoint-x를
  runtime arg로 넘기지는 않으며, 모든 weight read는 NoC1을 사용함

Board는 custom 20-core BOS NPU다. Runtime/code architecture는 Blackhole이다. Physical DRAM은 3 banks,
bank당 2 worker NoC endpoints, 총 6 endpoints다.

## 근거와 범위

Source:

- `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- `/home/iris_hb4/tt-metal-hb4/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp`
- `/home/iris_hb4/tt-metal-hb4/tt_metal/common/core_coord.cpp`

Stable recipe:

- `/home/iris_hb4/tt-metal-hb4/README.md`
- `/home/iris_hb4/reports/investigations/2026-08-05-bos-transformer-stable-optimization-baseline.md`

Physical coordinate evidence:

- `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_device_profile_2026_07_31/.logs/profile_log_device.csv`
- `/home/iris_hb4/profiler_runs/mlp_fanout2_rowburst_balanced_noc_2026_08_03_09_15_00`

이번 작업은 host-side 복원이다. Device, Watcher, NoC profiler를 실행하지 않았다.

## Logical-to-physical worker grid

Logical 5×4의 x는 유지된다. Logical y=3은 harvested physical gap 때문에 physical y=4로 보인다.

| logical row | physical row | physical cores |
|---:|---:|---|
| 0 | 0 | `(0,0)` ... `(4,0)` |
| 1 | 1 | `(0,1)` ... `(4,1)` |
| 2 | 2 | `(0,2)` ... `(4,2)` |
| 3 | 4 | `(0,4)` ... `(4,4)` |

기존 SDPA profile의 2.035 ms operation은 physical 20 cores 모두에서 BRISC/NCRISC/TRISC kernel zone을
남겼다. 이것은 20 active math를 뜻하지 않는다. Program factory가 idle core에도 kernel을 생성하고
compute runtime arg `{65, 0, 0, 0, 0, 0, 0, 0}`을 넣기 때문이다.

## Stable SDPA mapping

조건:

```text
8 KV heads × 2 cores/head = 16 active cores
K chunk = 256 tokens
endpoint loads x0..x5 = 3/2/3/3/3/2
NoC0/NoC1 reader loads = 8/8
pair-balanced = on
bank-balanced = on
six-reader-sharded = off
```

Endpoint NoC는 `x0/x4/x5 → NoC0`, `x1/x2/x3 → NoC1`이다. Writer는 해당 reader와 반대 NoC를
선택한다. Physical bank pair는 source 계약상 `x{0,1}`, `x{2,5}`, `x{3,4}`다.

| id | logical | physical | role | head | reader endpoint | reader NoC | writer NoC |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | `(0,0)` | `(0,0)` | output + reducer | 0 | x2 | 1 | 0 |
| 1 | `(1,0)` | `(1,0)` | worker | 0 | x4 | 0 | 1 |
| 2 | `(2,0)` | `(2,0)` | reducer | 1 | x4 | 0 | 1 |
| 3 | `(3,0)` | `(3,0)` | worker | 1 | x3 | 1 | 0 |
| 4 | `(4,0)` | `(4,0)` | reducer | 2 | x1 | 1 | 0 |
| 5 | `(0,1)` | `(0,1)` | worker | 2 | x0 | 0 | 1 |
| 6 | `(1,1)` | `(1,1)` | reducer | 3 | x4 | 0 | 1 |
| 7 | `(2,1)` | `(2,1)` | worker | 3 | x2 | 1 | 0 |
| 8 | `(3,1)` | `(3,1)` | reducer | 4 | x3 | 1 | 0 |
| 9 | `(4,1)` | `(4,1)` | worker | 4 | x5 | 0 | 1 |
| 10 | `(0,2)` | `(0,2)` | reducer | 5 | x0 | 0 | 1 |
| 11 | `(1,2)` | `(1,2)` | worker | 5 | x3 | 1 | 0 |
| 12 | `(2,2)` | `(2,2)` | reducer | 6 | x2 | 1 | 0 |
| 13 | `(3,2)` | `(3,2)` | worker | 6 | x5 | 0 | 1 |
| 14 | `(4,2)` | `(4,2)` | reducer | 7 | x1 | 1 | 0 |
| 15 | `(0,3)` | `(0,4)` | worker | 7 | x0 | 0 | 1 |

Idle math cores:

| logical | physical |
|---|---|
| `(1,3)` | `(1,4)` |
| `(2,3)` | `(2,4)` |
| `(3,3)` | `(3,4)` |
| `(4,3)` | `(4,4)` |

복원 방법은 source의 DFS를 그대로 적용했다. Target load `3/2/3/3/3/2`, pair별 반대 NoC 제약,
Manhattan route cost, output core의 NoC0 reader penalty를 사용했다. 최소 cost assignment는 endpoint
sequence `[2,4,4,3,1,0,4,2,3,5,0,3,2,5,1,0]`이다.

## Stable MLP mapping

조건:

```text
DRAM-sharded + 16 KiB read page
fanout-2 + balanced endpoint groups
tagged depth-2
prefetch helper = off
reader locality = off
12 reader = 12 compute
```

Base reader 6개는 runtime의 `get_optimal_dram_bank_to_logical_worker_assignment(in0_noc)` 결과다.
Partner는 full 5×4 후보를 `corerange_to_cores(..., row_wise=false)`의 column-major 순서로 훑어
NoC1 destination group을 최종 `4:4:4`로 맞춘다. 각 base/partner pair는 같은 DRAM view의 shard를
절반씩 직접 읽고 같은 core에서 계산한다.

| reader id | logical | physical | DRAM view | lane | NoC1 destination group | role |
|---:|---|---|---:|---:|---:|---|
| 0 | `(0,3)` | `(0,4)` | 0 | 0 | 0 | base reader + compute |
| 1 | `(0,0)` | `(0,0)` | 0 | 1 | 0 | partner reader + compute |
| 2 | `(1,3)` | `(1,4)` | 1 | 0 | 1 | base reader + compute |
| 3 | `(1,0)` | `(1,0)` | 1 | 1 | 1 | partner reader + compute |
| 4 | `(2,3)` | `(2,4)` | 2 | 0 | 0 | base reader + compute |
| 5 | `(1,1)` | `(1,1)` | 2 | 1 | 1 | partner reader + compute |
| 6 | `(3,3)` | `(3,4)` | 3 | 0 | 1 | base reader + compute |
| 7 | `(4,0)` | `(4,0)` | 3 | 1 | 2 | partner reader + compute |
| 8 | `(4,3)` | `(4,4)` | 4 | 0 | 2 | base reader + compute |
| 9 | `(4,1)` | `(4,1)` | 4 | 1 | 2 | partner reader + compute |
| 10 | `(0,2)` | `(0,2)` | 5 | 0 | 0 | base reader + compute |
| 11 | `(4,2)` | `(4,2)` | 5 | 1 | 2 | partner reader + compute |

Base group count는 `3:2:1`, partner 추가분은 `1:2:3`, 최종은 `4:4:4`다. Base-to-partner physical
Manhattan distance 합은 `4+4+4+5+3+4=24`다. 이 값은 이후 DRAM-nearest A/B 보고서가 기록한 기존
mapping distance 24와 일치한다.

Non-worker math cores는 `(2,0)`, `(3,0)`, `(0,1)`, `(2,1)`, `(3,1)`, `(1,2)`, `(2,2)`, `(3,2)`다.
Program은 전체 20-core rectangle에 kernel을 생성하지만 이 8개에는 `is_worker_core=false` runtime arg가
들어간다.

MLP stable path의 weight reader는 RISC0에서 NoC1을 쓴다. Activation input multicast는 별도 RISC1
kernel이다. Output reshard/write도 in1 reader-writer kernel의 runtime args로 각 output storage core를
받는다. 따라서 `12 active compute`와 `20 program cores`를 구분해야 한다.

## 한계

- SDPA endpoint mapping은 현재 source와 stable flags에서 결정적으로 복원했지만, 과거 profile CSV에는
  endpoint id 자체가 기록되지 않았다.
- MLP의 DRAM view는 allocator의 6 logical views다. Physical bank 6개라는 뜻이 아니다.
- MLP stable path는 explicit endpoint-x mode가 아니다. `NoC1 destination group 4:4:4`는 raw NoC
  source→destination 집계와 source mapping으로 검증된 분류다.
- Cache hit로 program creation log가 생략될 수 있다. 새 실행에서 확인할 때는
  `TT_LOGGER_TYPES=Op TT_LOGGER_LEVEL=DEBUG`와 `TT_METAL_MLP_LOG_READER_MAP=1`을 사용하되, device safety
  gate를 먼저 적용한다.

## 다음 계측

NoC trace 전에 host log에서 아래를 저장한다.

1. SDPA `core_group`, `core_group_idle`, `reader_endpoint_xs`, reducer/output physical arrays
2. MLP `MLP reader map: index ..., logical ..., physical ..., DRAM view ..., lane ..., vc ...`
3. Device CSV의 run host ID별 physical core set와 kernel duration

이 세 결과가 현재 표와 다를 때만 짧은 isolated NoC capture를 수행한다.

## 2026-08-09 actual 28-layer run 검증

사용자가 server reboot 완료를 확인했다. 필수 `32×32 ttnn.add` gate는 `ADD_COMPLETED`,
`DEVICE_CLOSED`, exit 0으로 통과했다.

Artifact:

- directory: `/home/iris_hb4/profiler_runs/llama32_3b_64k_layer_core_map_2026_08_09_03_35_00`
- full run: `run.log`
- short corrected-library verification: `short_verify.log`
- layer table: `layer_mapping.csv`

Checksums:

- `run.log`: `c167f5f98924dc88f9ad330cf787409b47164d6a4ebe70605316cf26ea8484de`
- `short_verify.log`: `07725c123ac5a010545ae630d482f40ccd6aee309b2fa39a3513e806bd86bc89`
- `layer_mapping.csv`: `c7ddef78c891c95b43fbb536eb8b7077539f119e4ecdbcbfd96d18dafdb1e246`

### 실행 결과

| run | warmup/measured tokens | ms/token | tok/s | completion |
|---|---:|---:|---:|---|
| full mapping | 3/50 | 128.713013 | 7.769222 | `WARMUP_COMPLETE`, `RESULT_JSON`, `DEVICE_CLOSED`, exit 0 |
| short verification | 1/1 | 128.944409 | 7.755280 | `WARMUP_COMPLETE`, `RESULT_JSON`, `DEVICE_CLOSED`, exit 0 |

두 run 모두 batch 1, paged KV, 64K curpos-only decode, K256, stable 6-endpoint SDPA, stable fanout-2
MLP를 사용했다. Full run 성능은 fresh-clone stable 기록 7.769802 tok/s와 사실상 같다.

### 관측된 mapping

- Layer 0--27 모두 SDPA와 MLP marker가 각각 정확히 28개 기록됨
- SDPA: 16 active core, 4 idle core
- SDPA endpoint load: `3/2/3/3/3/2`
- SDPA reader NoC load: `8/8`; 각 reducer/worker pair는 반대 NoC
- MLP: 12 reader = 12 compute, 8 non-worker program cores
- MLP logical DRAM view마다 lane 0/1 reader 두 개
- MLP NoC1 destination group: `4:4:4`

`short_verify.log`의 SDPA 16 active + 4 idle per-core 출력은 이 문서의 복원표와 전부 일치했다.
MLP reader 출력은 program variant 두 개에서 12개씩, 총 24줄이다. Core/view/lane/VC는 동일하고
W1/W3 계열 read page는 12,672 bytes, W2는 8,704 bytes였다.

첫 full run은 source-package `_ttnn.so`가 RUNPATH의 stale `build_home_release/lib/_ttnncpp.so`를 먼저
선택해 새 SDPA per-core 줄만 빠졌다. Layer 0--27 marker, SDPA aggregate와 MLP map은 정상이다.
Short verification은 `build_home_release/ttnn`을 `LD_LIBRARY_PATH` 첫 항목으로 두어 새 C++ log를 확인했다.

### 코드 계측

- opt-in: `TT_METAL_LOG_LAYER_CORE_MAP=1`
- `attention.py`: layer별 첫 decode SDPA marker
- `mlp.py`: layer별 첫 decode MLP marker
- `sdpa_decode_program_factory.cpp`: actual logical/physical core, role, endpoint와 reader/writer NoC
- audit patch: `/home/iris_hb4/tmp/codex-patches/20260809-001700-layer-core-map-logging.patch`

NoC profiler와 Watcher는 사용하지 않았다.

# BOS interleaved bank-run burst 및 L1 reorder A/B

## 결론

MLP W1/W3와 같은 `22 × 576 B` 블록에서는 interleaved tensor를 논리 tile 순서대로 한 장씩 읽는
방식이 중요한 병목이었다. 같은 interleaved allocation을 유지한 채 6 logical DRAM endpoint별
연속 run으로 먼저 읽고 L1에서 원래 tile 순서로 재배열하자, 12-reader read-only microbenchmark의
평균 latency가 `3.541 ms`에서 `2.164 ms`로 38.90% 감소했다. effective delivered bandwidth는
`24.76 GB/s`에서 `40.48 GB/s`로 증가했다.

측정 payload를 run당 `0.088 GB`에서 `0.350 GB`로 4배 늘린 30회 ABBA에서도 결론은 유지됐다.
latency는 `13.992 ms`에서 `8.433 ms`로 39.73% 감소했고, effective delivered bandwidth는
`25.06 GB/s`에서 `41.55 GB/s`로 65.79% 증가했다. 따라서 초기 결과는 짧은 실행의 host 고정비가
만든 차이가 아니다.

반면 W2와 같은 `8 × 1088 B` 블록에서는 DRAM 명령이 8개에서 6개로만 감소하면서 L1 reorder 8개가
추가되어 단일 방향성 샘플이 9.28% 낮아졌다. 따라서 이 방식은 interleaved layout의 일반적인 대체가
아니며, tile 수가 많은 W1/W3에만 선택적으로 검토할 후보이다. 현재 stable MLP에는 적용하지 않았다.

실제 stable SDPA의 주소·core·endpoint·barrier 계약을 복제한 별도 read-only control에서는 64K K/V
단발(q1)이 `60.397 ± 0.353 GB/s`, 반복 steady-state(q16)가 `62.349 ± 0.046 GB/s`였다. 처음의 단순
SDPA-like tile geometry가 보인 `40.97 GB/s`는 실제 SDPA reader rate를 대표하지 않는다. 특히 실제
threshold-18/final-barrier cadence를 적용하자 동일 exact control q4가 `45.22`에서 `61.86 GB/s`로
상승했다. 요청 병렬성을 무조건 키우는 것보다 workload에 맞는 issue cadence가 중요하다는 직접 증거다.

## 질문과 가설

실제 MLP interleaved reader는 한 K-row의 weight tile을 논리 순서대로 읽는다. interleaved address
generator에서는 연속 logical tile이 6 logical DRAM endpoint에 round-robin으로 배치되므로, reader는
W1/W3 블록마다 22개의 작은 DRAM read 명령을 발행한다.

검증 가설은 다음과 같다.

1. 같은 endpoint에 속하는 logical page ID는 6씩 떨어져 있지만 그 endpoint의 local DRAM 주소에서는
   연속한다.
2. 따라서 22개 tile을 endpoint residue별로 묶으면 최대 6개의 연속 DRAM burst로 읽을 수 있다.
3. staging L1의 endpoint-major 순서를 local-L1 NoC read로 logical tile 순서로 복원하더라도 DRAM
   request issue 감소가 충분히 크면 전체 latency가 낮아진다.

correctness readback이 모든 실행에서 통과했으므로, 이 실험 geometry에서는 endpoint-local 연속성 및
L1 reorder mapping이 실제 allocation과 일치했다.

## 구현

소스:

- `tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc.cpp`
- `tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/kernels/reader_dram.cpp`

새 opt-in은 `--access-layout bank-run-reorder`다.

- DRAM 단계: block의 page를 `page_id % 6` residue별로 묶어 최대 6개 one-packet read로 staging L1에
  적재한다.
- 파이프라인: 기존 tagged depth-2 DRAM prefetch를 유지한다.
- reorder 단계: DRAM TRID가 retire된 slot을 별도 TRID 15의 local-L1 read로 원래 logical page 순서로
  복원한다.
- L1: final pipeline slot과 staging pipeline slot을 별도로 두므로 bank-run 모드에서는 block storage가
  2배 필요하다.
- 안전 제약: interleaved tensor accessor, controlled 3-bank/6-endpoint reader layout, tagged mode,
  `coalesce_pages=1`, profiler/breakdown 비활성 조건에서만 opt-in을 허용한다.

## 공통 실행 조건

| 항목 | 값 |
|---|---:|
| board | custom 20-core BOS NPU |
| runtime/code architecture | Blackhole |
| physical DRAM topology | 3 banks, 2 worker endpoints/bank, 6 endpoints |
| active readers | 12 |
| reader/endpoint accounting | 2 readers per logical endpoint |
| physical-bank traffic accounting | 4 readers per physical bank |
| NoC accounting | NoC0 6 readers, NoC1 6 readers |
| pipeline | tagged depth-2 |
| consumer | 없음; final L1 scratch에 반복 덮어쓰기 |
| correctness | 마지막 block의 모든 word를 host readback으로 검증 |
| timeout | `SIGINT 60 s`, `SIGKILL 15 s` cleanup 상한 |
| profiler | 사용하지 않음 |

표의 bandwidth는 동일한 logical payload를 elapsed time으로 나눈 effective delivered rate다. physical DRAM
bus utilization이나 stable MLP kernel bandwidth로 해석하지 않는다. 12 readers 중 endpoint별 두 reader는
같은 tensor slot을 읽으므로 unique tensor byte 수가 아니라 reader-delivered byte 수를 분모로 사용한다.
이 분모는 A/B에서 동일하다.

## W1/W3형 결과: 22 × 576 B

block payload는 12,672 B이며 `pages_per_core=528`, `iteration_quanta=1`을 사용했다. A1-B1-B2-A2
순서로 각 cell을 30회 측정했다.

| 경로 | DRAM reads/block | local-L1 reorder reads/block | n | latency mean ± 95% CI | bandwidth mean ± 95% CI |
|---|---:|---:|---:|---:|---:|
| A, natural tile-wise interleaved | 22 | 0 | 60 | 3.541233 ± 0.027426 ms | 24.757083 ± 0.192279 GB/s |
| B, bank-run + reorder | 6 | 22 | 60 | 2.163583 ± 0.002989 ms | 40.484383 ± 0.056366 GB/s |
| source-order-matched uniform tile-wise | 22 | 0 | 30 | 3.630233 ± 0.038902 ms | 24.148833 ± 0.261921 GB/s |

B는 pooled natural A 대비 latency 38.903% 감소, 1.6367× speedup, delivered-rate 63.526% 증가였다.
reader별 시작 phase 차이를 제거한 uniform baseline과 비교해도 latency 40.401% 감소와 1.6779× speedup이
유지됐다. 따라서 결과를 단순 phase-stagger 차이로 설명할 수 없다.

### Read 양 4배 검증

`iteration_quanta`만 1에서 4로 높여 동일 kernel invocation의 reader payload를 `0.088 GB`에서
`0.350 GB`로 늘렸다. A1-B1-B2-A2 순서로 각 cell을 30회 측정했으며, 나머지 geometry와
correctness 검증은 동일하다.

| 경로 | n | latency mean ± 95% CI | bandwidth mean ± 95% CI |
|---|---:|---:|---:|
| A, natural tile-wise interleaved | 60 | 13.992317 ± 0.101254 ms | 25.058950 ± 0.179452 GB/s |
| B, bank-run + reorder | 60 | 8.432983 ± 0.003741 ms | 41.545783 ± 0.018483 GB/s |

B는 A 대비 latency 39.731% 감소, 1.6592× speedup, delivered-rate 65.792% 증가였다. A1과 A2의
평균 bandwidth는 각각 `25.130`, `24.988 GB/s`, B1과 B2는 각각 `41.529`, `41.562 GB/s`로
run order에 따른 방향 전환은 없었다. payload 증가 뒤에도 두 경로의 절대 bandwidth가 기존 결과와
가까우므로, 차이는 invocation 고정비보다 steady-state request organization에서 발생한 것으로 본다.

## W2형 방향성 결과: 8 × 1088 B

block payload는 8,704 B다. 이 cell은 각 경로 1회만 실행했으므로 통계적 A/B가 아니라 적용 경계를
찾기 위한 방향성 결과다.

| 경로 | DRAM reads/block | local-L1 reorder reads/block | latency | bandwidth |
|---|---:|---:|---:|---:|
| tile-wise interleaved | 8 | 0 | 3.654 ms | 43.903 GB/s |
| bank-run + reorder | 6 | 8 | 4.028 ms | 39.829 GB/s |

bank-run은 latency가 10.235% 증가하고 delivered rate가 9.280% 감소했다. 8개 DRAM 명령 중 2개만
제거하는 대신 8개 local reorder 명령과 staging traffic을 추가했기 때문이다.

## SDPA-like transport 결과

SDPA performance KV의 BFP8 tile 크기 `1,088 B`를 사용했다. K128은 K 또는 V 한 chunk의
`4 sequence tiles × 4 head-dimension tiles = 16 tiles`, K256은 `8 × 4 = 32 tiles`로 구성했다.
두 조건 모두 전체 measured payload를 `0.642 GB/run`으로 고정했으므로 K128/K256 차이는 byte 수가
아니라 block 경계와 endpoint-local run 길이의 차이다. 각 cell은 A1-B1-B2-A2 순서로 30회 측정했다.

| Chunk geometry | 경로 | DRAM reads/block | L1 reorder reads/block | n | latency mean ± 95% CI | bandwidth mean ± 95% CI |
|---|---|---:|---:|---:|---:|---:|
| K128, 16 × 1,088 B | tile-wise interleaved | 16 | 0 | 60 | 15.719800 ± 0.206086 ms | 40.934050 ± 0.552339 GB/s |
| K128, 16 × 1,088 B | bank-run + reorder | 6 | 16 | 60 | 13.870067 ± 0.011790 ms | 46.267283 ± 0.039281 GB/s |
| K256, 32 × 1,088 B | tile-wise interleaved | 32 | 0 | 60 | 15.706783 ± 0.205504 ms | 40.967333 ± 0.550483 GB/s |
| K256, 32 × 1,088 B | bank-run + reorder | 6 | 32 | 60 | 13.633583 ± 0.005483 ms | 47.069967 ± 0.018934 GB/s |

K128 bank-run은 latency 11.767% 감소, 1.1334× speedup, delivered rate 13.029% 증가였다. K256은
latency 13.199% 감소, 1.1521× speedup, delivered rate 14.896% 증가였다. Tile-wise latency CV는
K128/K256 각각 `5.181%/5.171%`였지만 bank-run은 `0.336%/0.159%`로 낮아졌다.

이 결과는 **SDPA packet/chunk geometry만 닮은 12-reader read-only transport control**이다. 실제 stable
comparison runner의 16 active cores, endpoint load `3/2/3/3/3/2`, row-major first-16 core map,
identity page table, reverse chunk ownership 및 threshold-18/final-barrier cadence를 포함하지 않는다.
따라서 표의 GB/s를 실제 SDPA K/V bandwidth로 직접 대체하지 않는다.

재현할 때 K128은 `--page-size 1088 --pages-per-block 16`, K256은
`--page-size 1088 --pages-per-block 32`를 사용한다. 공통 조건은 `--pages-per-core 512`,
`--iteration-quanta 4`, `--total-readers 12`, interleaved tensor accessor 및 tagged depth-2다.
A/B는 `--access-layout packed`와 `bank-run-reorder`만 바꾼다.

## Exact stable-SDPA paged-read control

앞 절의 geometry-only control이 실제 SDPA의 약 70 GB/s와 다른 이유를 분리하기 위해 opt-in
`--access-layout sdpa-paged-exact`를 추가했다. 이 경로는 다음 계약을 고정한다.

- 64K context, 32-token physical pages, 8 KV heads, head dimension 128
- K와 V를 합친 encoded payload `142,606,336 B` (`136 MiB`)
- BFP8 tile `1,088 B`, K256당 `8 × 4 = 32 tiles`
- actual row-major first-16 logical cores와 head당 2 cores
- actual endpoint sequence `2/4, 4/3, 1/0, 4/2, 3/5, 0/3, 2/5, 1/0`
- endpoint loads `3/2/3/3/3/2`, NoC0/NoC1 `8/8`
- `get_runtime_args`와 같은 reverse sequence ownership: core-in-head 0은 뒤 32K, 1은 앞 32K
- identity page table과 실제 paged tile-id 식
- stable reader와 같은 18-tile intermediate barrier 및 14-tile tail 뒤 final barrier

consumer, QK/PV compute, CB reserve/push/pop, page-table DRAM read, online softmax와 reducer는 제거했다.
따라서 이것은 실제 SDPA 전체 kernel이 아니라 **주소·배치·issue cadence를 맞춘 read-only control**이다.

| 조건 | payload/run | n | latency mean ± SD | encoded delivered rate mean ± SD |
|---|---:|---:|---:|---:|
| exact q1 | 0.143 GB | 30 | 2.361233 ± 0.013823 ms | 60.397000 ± 0.353288 GB/s |
| exact q16 steady-state | 2.282 GB | 30 | 36.595400 ± 0.027210 ms | 62.349333 ± 0.046330 GB/s |

비교 기준인 instrumented stable K256 SDPA는 aggregate issued bytes가 같은 `142,606,336 B`이고, global
reader service envelope가 `2.058563 ms`, encoded issued rate가 `69.275 GB/s`였다. Exact q1은 이보다
latency가 `0.302670 ms` 길고 rate가 `8.878 GB/s` 낮다. q16 plateau도 stable reader rate보다 10.00%
낮다. 두 수치의 분자와 주소 geometry는 맞췄지만 측정 대상은 여전히 동일하지 않다. 실제 reader에는
두 개의 독립 K/V buffers, CB/compute가 만드는 chunk pacing, 실제 page-table load 및 kernel 내부 instruction
interleaving이 있고 control은 한 buffer를 반복 순회한다. 남은 약 10%를 physical DRAM utilization 차이로
해석하지 않는다.

교정 과정 자체가 중요한 결과다.

| 단계 | q4 방향성 rate | 해석 |
|---|---:|---|
| geometry-only 12-reader K256 | 40.97 GB/s | 실제 mapping/cadence가 없는 비동등 control |
| 16-reader exact 주소/core map, 32-tile depth-2 issue | 45.22 GB/s | 과도한 outstanding issue와 service tail |
| + actual threshold-18/final barrier | 61.86 GB/s | stable reader cadence에 가까워지며 36.8% 상승 |

따라서 실제 SDPA가 60--70 GB/s를 보이는 것은 모순이 아니다. 40 GB/s대 control은 packet size만 닮았고,
실제 reader는 core/head ownership, NoC direction, endpoint assignment와 barrier cadence가 함께 맞춰져 있다.
또한 이 결과는 SDPA에서 compute가 단순 방해물만은 아니며, CB/compute pacing이 memory request burst를
조절할 가능성을 남긴다. 이 마지막 인과는 아직 미검증 가설이다.

재현 명령:

```bash
timeout --signal=INT --kill-after=15s 60s \
  build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-reader-3bank-sharded --active-banks 3 --readers-per-bank 2 --total-readers 16 \
  --memory-layout interleaved --addressing-mode tensor-accessor --access-layout sdpa-paged-exact \
  --pipeline-mode tagged --pipeline-depth 2 --page-size 1088 --pages-per-block 32 \
  --pages-per-core 8192 --coalesce-pages 1 --address-pattern sequential --iteration-quanta 1 --num-tests 30

# steady-state control: 위 명령에서 --iteration-quanta 16만 변경
```

## 해석

### 관측 사실

- 같은 interleaved allocation과 W1/W3 payload에서 DRAM command count를 22개에서 6개로 줄이자,
  local-L1 reorder 비용을 포함해도 latency가 약 39--40% 감소했다.
- source page 순서를 맞춘 uniform tile-wise control에서도 차이가 유지됐다.
- W2에서는 8개에서 6개로 줄이는 정도로는 reorder 비용을 상쇄하지 못했다.
- 모든 device 실행은 correctness 통과, 정상 device close, exit code 0이었다.
- sharded row-reference를 같은 12-reader override로 실행하려 한 명령은 host option validation에서
  exit code 1로 종료됐으며 device를 열지 않았다. 측정 결과로 사용하지 않는다.

### 강한 추론

W1/W3 interleaved 경로의 낮은 성능에는 physical DRAM 자체뿐 아니라 per-tile address generation 및
NoC read-command issue가 큰 비중을 차지한다. endpoint-local burst는 이 issue pressure를 줄이며,
L1 reorder를 추가하고도 이득이 남는다.

### 아직 검증되지 않은 부분

- compute/CB consumer가 결합된 실제 matmul에서도 같은 개선폭이 유지되는가
- bank-run staging과 matmul compute를 겹치면 reorder 비용을 더 숨길 수 있는가
- W1과 W3에만 선택 적용했을 때 full MLP latency와 PCC가 어떻게 변하는가
- DRAM-sharded one-row request와 비교했을 때 bank-run interleaved가 어느 정도까지 근접하는가

## 다음 실험

1. W1/W3 reader에만 opt-in bank-run staging을 이식하고 isolated MLP correctness+latency A/B를 한다.
2. W2는 기존 sharded row-read를 유지한다.
3. 실제 MLP에서는 DRAM issue, DRAM-done, L1 reorder-done, CB-ready marker를 분리해 reorder가 compute와
   overlap되는지 확인한다.
4. isolated MLP에서 이득이 확인된 경우에만 full layer/full decode로 확장한다.

## 재현 명령

아래 두 명령에서 `--access-layout`만 다르다. A/B cell은 `--num-tests 30`으로 실행했다.
read 양 4배 검증은 두 명령 모두 `--iteration-quanta 4`로 바꾸고 같은 ABBA 순서로 실행했다.

```bash
timeout --signal=INT --kill-after=15s 60s \
  build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-reader-3bank-sharded --active-banks 3 --readers-per-bank 2 --total-readers 12 \
  --memory-layout interleaved --addressing-mode tensor-accessor --access-layout packed \
  --pipeline-mode tagged --pipeline-depth 2 --page-size 576 --pages-per-block 22 \
  --pages-per-core 528 --coalesce-pages 1 --iteration-quanta 1 --num-tests 30

timeout --signal=INT --kill-after=15s 60s \
  build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-reader-3bank-sharded --active-banks 3 --readers-per-bank 2 --total-readers 12 \
  --memory-layout interleaved --addressing-mode tensor-accessor --access-layout bank-run-reorder \
  --pipeline-mode tagged --pipeline-depth 2 --page-size 576 --pages-per-block 22 \
  --pages-per-core 528 --coalesce-pages 1 --iteration-quanta 1 --num-tests 30
```

## Artifact

디렉터리:

`/home/iris_hb4/benchmark_runs/interleaved_bank_run_reorder_abba_2026_08_20_1313`

| 파일 | SHA-256 |
|---|---|
| `A1_tilewise.log` | `6b238c546313b9fcc6a9d4588c007e05986ea24db88d2ad1034d7c6d969440cd` |
| `B1_bank_run_reorder.log` | `f276459b95744a4c22f03ea0796b7046a8f2ca76011fdf121f7c42f8bf119a0e` |
| `B2_bank_run_reorder.log` | `ab5c57026f858f99d9efdda39c1b8328436bf6959f3308c7e663b7a2730b4284` |
| `A2_tilewise.log` | `7e06195ae77448360f4f95d53e49947b3751b36f2e24986512b9eb368a5f3e21` |
| `A3_uniform_tilewise.log` | `8b4e4f8e7ce5948e11eebc7ba92543f65946f19d2175162f088a6239a9d8fd3f` |
| `q4_A1_tilewise_30.log` | `b33d1796f8f8df355b0031c7baabbc232b549af088acf81d92e19c21a07ed0c8` |
| `q4_B1_bank_run_30.log` | `6595f91e0fcc1dc30fb58e44de5dae2c5b0024df201c407cdc31a0c7c00e893e` |
| `q4_B2_bank_run_30.log` | `bebd78ac3eb4a50df753892577fdadf4c4248156c9f63c3d5a9c95570b1e21da` |
| `q4_A2_tilewise_30.log` | `5b11d02f598145c0bc5d237a3f8425f6d5a5910ccc0a4deb8022775827960c69` |

W2 단일 방향성 run은 터미널 capture만 남았고 timestamped log 파일로 저장하지 않았다. 따라서 W2
수치는 다음 재현에서 반복 측정·artifact화해야 한다.

SDPA-like artifact 디렉터리:

`/home/iris_hb4/benchmark_runs/sdpa_like_bank_run_ab_2026_08_20_1340`

| 파일 | SHA-256 |
|---|---|
| `k128_A1_tilewise_30.log` | `ede79ac43ab9563f6b984e76904a18a0ca5a38d56bcd42f657c96e58b8f66d66` |
| `k128_B1_bank_run_30.log` | `dc17b12029b1b8d913c0516636b3bc8596211d5b978bf555039807aa67e9111c` |
| `k128_B2_bank_run_30.log` | `f088fc5a2be1db1d3d37486bfb0107c4e4f60ca40536c731f5bc45a884822743` |
| `k128_A2_tilewise_30.log` | `cb0fa4ee21f3ed270514990cad7aaec240c325a63b98038b4815ebe3a15a1632` |
| `k256_A1_tilewise_30.log` | `6dc484b18e8ad36daa1ccedca3438f62a1cc7de6312ff9668243f3c79c21f7a0` |
| `k256_B1_bank_run_30.log` | `117a0530633f595f7ef9df7e6a522e687410b8cf0a3fd69688b07472b716c7cc` |
| `k256_B2_bank_run_30.log` | `bbf31b3d51dc8644e5e63ad3be628d162bc5039b903610de27d1dc4258e92721` |
| `k256_A2_tilewise_30.log` | `1b987b7eadb051d1aaf1510f06d42a2f2883fc76c301b778338c9a65fb54b85a` |

Exact stable-SDPA paged-read artifact 디렉터리:

`/home/iris_hb4/benchmark_runs/sdpa_paged_exact_read_2026_08_20_1408`

| 파일 | SHA-256 |
|---|---|
| `q1.log` | `42d819d1b8a015a551ad449ca6d1b11e8df4bdff9421802c10ac999e157bea87` |
| `q16.log` | `109f275adf159b5c9091c69468dcf3f298202c2fa534fbd67438307ef8ea2e13` |

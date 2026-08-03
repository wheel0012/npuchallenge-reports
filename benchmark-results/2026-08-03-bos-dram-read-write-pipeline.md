# BOS 3-bank read/write split-NoC pipeline benchmark

측정일: 2026-08-03 UTC

## 요약

read-only saturation benchmark의 86.83 GB/s가 실제 operation의 reader/writer 역할 분담에도
근접하게 유지되는지 확인했다. physical bank당 owner core 하나를 두고 한 data-movement RISC/NoC가
DRAM을 읽어 local double-buffered CB에 넣고, 반대 RISC/NoC가 같은 CB를 소비해 같은 physical
bank의 반대 endpoint로 DRAM write하는 독립 benchmark를 구현했다.

128 KiB block에서 NOC0-read/NOC1-write는 read+write aggregate 77.312 GB/s,
NOC1-read/NOC0-write는 78.413 GB/s를 기록했다. 방향당 useful copy rate는 각각 38.656,
39.207 GB/s다. 모든 run은 output DRAM 전수 검증과 정상 device close를 통과했다.

따라서 단일 NoC 방향의 속도가 SDPA/MLP의 낮은 bandwidth를 직접 설명하지는 않는다. 정상적인
read/write 동시 traffic도 read-only peak의 약 89--90% aggregate를 유지한다.

## 장치 및 배치

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: 5×4 = 20 cores
- active owner cores: 3
- physical DRAM banks: 3
- worker endpoints: bank당 2개, 총 6개
- reader: bank당 가장 가까운 owner 하나, reader ring의 native endpoint 하나
- writer: 같은 owner의 반대 data-movement RISC, 반대 ring의 동일 physical-bank endpoint
- handoff: 2-slot local circular buffer
- 기본 page: 4 KiB
- pages/owner: 512 = 2 MiB
- iterations: 32
- measured traffic: 방향당 약 201.327 MB/run
- warmup 1회 뒤 measured 3회 또는 5회

NOC0-read 배치는 다음과 같다.

| Bank | Owner physical | Read endpoint/view | Write endpoint/view |
|---:|---|---|---|
| 0 | (0,2) | NOC0 x0 / view0 | NOC1 x1 / view3 |
| 1 | (4,2) | NOC0 x5 / view2 | NOC1 x2 / view4 |
| 2 | (4,4) | NOC0 x4 / view1 | NOC1 x3 / view5 |

역방향에서는 owner physical coordinate가 (1,2), (2,2), (3,2)이고 endpoint/view가 반대로 매핑된다.

## 결과

| Reader→writer | Block | Read-equivalent | Read+write aggregate | Read-only peak 대비 |
|---|---:|---:|---:|---:|
| NOC0→NOC1 | 32 KiB | 36.378 GB/s | 72.756 GB/s | 83.79% |
| NOC0→NOC1 | 64 KiB | 38.571 GB/s | 77.142 GB/s | 88.84% |
| NOC0→NOC1 | 128 KiB | 38.656 GB/s | 77.312 GB/s | 89.04% |
| NOC0→NOC1 | 256 KiB | 38.652 GB/s | 77.304 GB/s | 89.03% |
| NOC1→NOC0 | 128 KiB | 39.207 GB/s | 78.413 GB/s | 90.31% |

32→64 KiB에서 aggregate가 약 6.0% 상승했지만 64 KiB 이상은 plateau다. 역방향은 같은 128 KiB
조건에서 약 1.4% 높아 작은 route/placement 비대칭이 있으나 결론을 바꿀 정도는 아니다.

Read-equivalent는 input에서 output으로 복사된 useful byte를 시간으로 나눈 값이다. 각 byte가
DRAM read와 DRAM write를 한 번씩 유발하므로 실제 off-chip traffic은 그 두 배인 aggregate로 함께
보고한다. 동일 physical bank가 read와 write를 동시에 처리하므로 방향당 수치가 read-only peak의
절반에 가까운 것은 자연스럽다.

## 구현 및 재현

Host source:
tests/tt_metal/tt_metal/perf_microbenchmark/13_dram_read_write_pipeline/test_dram_read_write_pipeline.cpp

Reader/writer kernels:
tests/tt_metal/tt_metal/perf_microbenchmark/13_dram_read_write_pipeline/kernels/

reader는 CB slot을 reserve하고 DRAM read barrier 뒤 push한다. writer는 같은 CB를 wait한 뒤 반대
endpoint로 write하고 write barrier 뒤 pop한다. 이 계약 때문에 reader가 writer보다 앞서 L1 slot을
덮어쓸 수 없다. target 이름은 test_dram_read_write_pipeline이며 reader-noc, page-size,
pages-per-owner, pages-per-block, iterations와 num-tests option을 제공한다.

## SDPA/MLP에 주는 의미

- read-only 86.83 GB/s와 정상 read/write 77--78 GB/s의 차이는 약 10--11%다.
- SDPA chunk-256의 약 70.27 GB/s는 이 정상 경로 reference의 약 90%라 memory-side 잔여 폭이 작다.
- SDPA reader ownership/locality 뒤에도 개선이 작다면 slowest-head reduction과 math-engine
  pipeline tail을 우선 분석한다.
- MLP의 projection aggregate 45.40 GB/s는 더 낮지만 whole-layer effective 수치에 GEMM이 포함된다.
  reader locality A/B 뒤 BRISC/NCRISC와 TRISC span을 분리해 memory-bound 여부를 판정해야 한다.

이 benchmark에는 matrix compute, softmax/reducer, paged address translation 및 inter-core relay가
없다. 따라서 77--78 GB/s는 operation 목표값이지 실제 SDPA/MLP가 자동으로 달성해야 하는 값은 아니다.

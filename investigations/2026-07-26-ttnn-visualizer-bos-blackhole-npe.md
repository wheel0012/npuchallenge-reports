# TT-NPE BOS 및 TTNN Visualizer BOS Blackhole 적용 보고서

- 작성일: 2026-07-26
- NPE 저장소: `/home/iris_hb4/tt-npe-bos`
- Visualizer 저장소: `/home/iris_hb4/ttnn-visualizer-plus`
- 범위: BOS Blackhole 물리 NoC 6×5 모델과 trace-active 20-worker 시각화

## 1. 결론

BOS는 표준 Blackhole 17×12/130-worker topology가 아니다. 실제 UMD SoC descriptor 기준 물리 NoC는 6×5이며, worker endpoint는 24개, DRAM endpoint는 y=3 행의 6개다. 현재 확인한 SDPA decode trace에서 zone을 실행한 worker는 이 24개 중 20개다.

따라서 구현을 두 층으로 분리했다.

- `tt-npe-bos`: 물리 라우팅 모델을 6×5로 계산
- `ttnn-visualizer-plus`: 물리 asset은 6×5/24-worker로 유지하고, NPE 화면의 `T` 표시는 trace zone에 참여한 20개로 제한

이 방식은 물리 topology를 임의로 20코어에 맞춰 줄이지 않으면서 workload의 실제 참여 코어를 정확히 표현한다.

## 2. 근거 데이터

### 2.1 권위 있는 BOS SoC descriptor

확인한 파일:

`/home/iris_hb4/tt-metal-hb4/tt_metal/third_party/umd/tests/soc_descs/blackhole_6x4_BOS.yaml`

핵심 내용:

- `grid.x_size`: 6
- `grid.y_size`: 5
- functional worker: x=0..5, y=0,1,2,4 — 총 24개
- DRAM NoC endpoint: x=0..5, y=3 — 총 6개
- DRAM descriptor의 바깥 channel/controller 그룹: 1개
- ARC/PCIe/Ethernet/router-only endpoint: 없음

물리 배치는 다음과 같다.

```text
        x=0  x=1  x=2  x=3  x=4  x=5
 y=0     T    T    T    T    T    T
 y=1     T    T    T    T    T    T
 y=2     T    T    T    T    T    T
 y=3     d    d    d    d    d    d
 y=4     T    T    T    T    T    T
```

### 2.2 실제 profiler trace

확인한 원본:

`/home/iris_hb4/profiler_runs/sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33/perf_capture/.logs/noc_trace_ID1_merged.json`

- 이벤트 수: 139,296
- 고유 zone core: 20개
- zone 좌표: x=0..4, y=0,1,2,4
- trace에는 zone이 없는 물리 worker `(5,4)`를 목적지로 하는 read도 존재
- DRAM 행 `(0,3)`, `(4,3)` 대상 write도 존재

즉, x=5 worker를 물리 asset에서 삭제하면 실제 route endpoint를 worker로 분류하지 못한다. 20개는 물리 core 수가 아니라 해당 workload의 active zone core 수다.

## 3. `tt-npe-bos` 구현

### 3.1 물리 모델

`BlackholeDeviceModel`을 BOS 전용 물리 모델로 변경했다.

- rows: 12 → 5
- cols: 17 → 6
- worker: 24개
- DRAM endpoint: 6개
- DRAM controller ID: 6개 endpoint 모두 0
- NOC0/NOC1 torus wrap은 6×5 크기로 계산
- 기존 bandwidth/latency 상수는 topology 작업 범위를 넘는 추정을 피하기 위해 유지

기존 `DRAMHarvestingConfig` 인자는 upstream 호출 호환을 위해 남겼지만 BOS controller 수에는 영향을 주지 않는다.

### 3.2 factory와 CLI

다음 single-chip alias가 BOS 모델을 선택한다.

- `blackhole`
- `blackhole_bos`
- `BOS`
- `P100`
- `P150`

기존 profiler가 사용하는 `P100` alias를 유지했으므로 capture command를 바꿀 필요가 없다. Python CLI의 `--device` choice에도 `BOS`, `blackhole_bos`를 추가했다.

### 3.3 layout descriptor

`tt_npe/data/device/layout/arch-blackhole.yaml`을 UMD BOS descriptor와 맞췄다.

- grid 6×5
- 24 functional workers
- y=3의 6개 DRAM endpoint를 한 controller group으로 정의
- BOS L1/DRAM address metadata 포함

### 3.4 Python CLI 호환성

실제 trace 실행 후 마지막 통계 출력에서 기존 CLI가 `Stats.wallclock_runtime_us`를 참조하지만 바인딩에는 해당 top-level property가 없는 upstream 불일치를 발견했다. `per_device_stats`의 최대 runtime을 반환하는 read-only 호환 property를 추가해 CLI와 예제 코드를 유지했다.

## 4. Visualizer 구현

### 4.1 물리 Blackhole asset

`src/assets/data/arch-blackhole.json`을 BOS descriptor에 맞췄다.

- grid: 6×5
- worker: 24개
- DRAM: y=3의 6개 endpoint
- ARC/PCIe/Ethernet/router-only: 빈 목록

### 4.2 NPE grid의 source of truth

`NPEViewComponent`가 정적 asset보다 `common_info.num_cols`와 `common_info.num_rows`를 우선 사용하도록 변경했다. 따라서 `tt-npe-bos`가 출력한 6×5 메타데이터가 실제 NoC canvas 크기를 결정한다. 구형/누락 데이터에는 architecture asset을 fallback으로 사용한다.

### 4.3 trace-active 20코어 표시

`getNpeWorkerCoresByDevice()`를 추가했다.

- `zones[].core`를 chip별로 수집
- BRISC/NCRISC 등 여러 root zone이 같은 core에 있어도 dedupe
- grid 밖 좌표 제외
- 물리 worker 목록에 없는 DRAM/기타 endpoint 제외
- row/column 순으로 안정 정렬

zone이 존재하는 chip은 이 active 목록을 `EmptyChipRenderer`의 worker label로 사용한다. 확인한 trace에서는 정확히 20개의 `T`가 표시된다. zone 정보가 없는 trace는 물리 24-worker asset으로 fallback한다.

## 5. 변경 파일

### `tt-npe-bos`

- `README.md`
- `tt_npe/cpp/include/device_models/blackhole.hpp`
- `tt_npe/cpp/include/npeDeviceModelFactory.hpp`
- `tt_npe/cpp/pybind/bindings.cpp`
- `tt_npe/cpp/test/test_npe_device.cpp`
- `tt_npe/data/device/layout/arch-blackhole.yaml`
- `tt_npe/py/pycli/tt_npe.py`
- `tt_npe/py/pytest/test_bindings.py`

### `ttnn-visualizer-plus`

- `src/assets/data/arch-blackhole.json`
- `src/components/npe/NPEViewComponent.tsx`
- `src/functions/getNpeWorkerCoresByDevice.ts`
- `tests/blackholeArchitecture.spec.ts`
- `tests/getNpeWorkerCoresByDevice.spec.ts`

## 6. 검증 결과

### `tt-npe-bos`

- CMake Debug 구성: 성공, Clang 17
- C++ unit tests: 48/48 통과
- BOS 신규 테스트:
  - 6×5, 24 worker, 6 DRAM endpoint 확인
  - 모든 DRAM endpoint → controller 0 확인
  - NOC0 east wrap 및 NOC1 north wrap 확인
  - `BOS`, `blackhole_bos`, `P100` factory alias 확인
- Python binding rebuild: 성공
- `Stats.wallclock_runtime_us` 직접 회귀 실행: 성공
- 실제 139,296-event trace 처리: 성공
- 생성된 전체 timeline 및 8개 split file의 공통 metadata:
  - `arch=blackhole`
  - `mesh_device=BOS`
  - `num_cols=6`
  - `num_rows=5`

### `ttnn-visualizer-plus`

- BOS Vitest: 2 files, 4 tests 통과
- 전체 TypeScript `tsc --noEmit`: 통과
- 수정 파일 ESLint: 통과
- 수정 파일 Prettier check: 통과
- SPDX 검사: 통과
- Vite production build: 통과
- `git diff --check`: 통과

## 7. 전달 디렉터리 계약

이번 변경은 report ingestion과 전달 디렉터리를 수정하지 않았다. 계약은 그대로 유지된다.

```text
<run>/
├── memory_report_visualizer/
│   ├── config.json
│   └── db.sqlite
└── perf_report_visualize/
    ├── ops_perf_result.csv
    ├── profile_log_device.csv
    └── tracy_profile_log_host.tracy
```

## 8. 주의 사항

- 이 포크의 Blackhole 모델/asset은 BOS 전용이다. 표준 P100/P150 17×12 표현이 필요한 범용 배포판과 혼용하면 안 된다.
- topology와 route 좌표는 실측 descriptor로 검증했지만 bandwidth/latency 상수는 기존 Blackhole 값을 유지했다. BOS 성능 보정값이 확보되면 별도 calibration 작업이 필요하다.
- Blackhole multichip 모델도 내부적으로 동일한 compact single-chip model을 사용한다. 이 BOS 포크에서 multichip alias를 사용할 경우 chip당 6×5로 동작한다.
- performance report의 130-core heuristic은 NPE NoC 화면과 별도 영역이므로 이번 변경에 포함하지 않았다.

## 9. 최종 판단

NoC grid를 실제로 바꾸려면 `tt-npe`를 수정해야 한다는 판단이 맞다. 이번 구현은 `tt-npe-bos`에서 route 계산과 timeline metadata를 6×5로 변경하고, Visualizer는 그 metadata를 따라 canvas를 그리도록 연결했다. 동시에 물리 24코어와 workload-active 20코어를 분리해 route 정확성과 UI 목적을 모두 만족시킨다.

# Technical report archive

이 디렉터리는 `/home/iris_hb4`에서 작성한 기술 보고서를 유형별로 찾기 위한 중앙 저장소다.
코드와 가까이 있어야 하는 원본 문서는 기존 위치에 그대로 두며, 여기에는 원문 그대로의
스냅샷을 보관한다. 원본 경로와 스냅샷 체크섬은 [MANIFEST.md](MANIFEST.md)에 기록한다.

## 분류

- `benchmark-results/`: 실행 조건, 수치, 비교 결과가 있는 측정 보고서
- `investigations/`: 소스·프로파일 분석과 원인 가설
- `incidents/`: 장애 타임라인, 영향, 복구 및 재발 방지 기록
- `handoffs/`: 다른 작업자나 서버로 넘기는 인수인계 문서
- `guides/`: 반복 가능한 측정·환경 구성 절차
- `indexes/`: 외부 run 및 artifact를 가리키는 색인
- `_templates/`: 새 보고서 작성 템플릿
- `archive/`: 더 이상 현행이 아니지만 보존할 문서

## 현재 문서

### Benchmark results

- [Llama 3.2 3B 64K SDPA DRAM benchmark](benchmark-results/2026-07-26-llama32-3b-64k-sdpa-dram.md)
- [Llama 3.2 3B 20-core GEMM benchmark](benchmark-results/2026-07-26-llama32-3b-gemm-20core.md)
- [BOS 20-core, 6-endpoint DRAM saturation benchmark](benchmark-results/2026-07-31-bos-dram-saturation-20core-6-endpoint.md)
- [BOS MLP DRAM-sharded 및 W2 block-width A/B](benchmark-results/2026-08-02-bos-mlp-w2-block-width-ab.md)
- [BOS Llama 3.2 3B 64K SDPA·MLP 6-endpoint A/B](benchmark-results/2026-08-03-bos-64k-six-endpoint-sdpa-mlp-ab.md)
- [BOS Llama 3.2 3B 64K full-model decode A/B](benchmark-results/2026-08-03-bos-llama32-3b-64k-full-decode-ab.md)
- [BOS 3-bank read/write split-NoC pipeline benchmark](benchmark-results/2026-08-03-bos-dram-read-write-pipeline.md)
- [BOS MLP 6-endpoint fanout-2 및 row-burst](benchmark-results/2026-08-03-bos-mlp-six-endpoint-fanout2-row-burst.md)
- [BOS MLP triple-buffer stall profile](benchmark-results/2026-08-03-bos-mlp-triple-buffer-stall-profile.md)
- [BOS MLP activation/weight readiness profile](benchmark-results/2026-08-03-bos-mlp-input-readiness-profile.md)
- [BOS MLP fanout-2 tagged two-block A/B](benchmark-results/2026-08-03-bos-mlp-fanout2-tagged-two-block.md)
- [BOS MLP wait decomposition](benchmark-results/2026-08-04-bos-mlp-wait-decomposition.md)
- [BOS MLP reader-packed weight layout A/B](benchmark-results/2026-08-04-bos-mlp-reader-packed-layout-ab.md)
- [BOS MLP compute block cadence](benchmark-results/2026-08-04-bos-mlp-compute-block-cadence.md)

### Investigations

- [BOS Llama 3.2 test-demo KV-cache 분석](investigations/2026-07-24-bos-llama32-kv-cache-test-demo.md)
- [Llama 3.1 8B KV-cache 분석](investigations/2026-07-25-llama31-8b-kv-cache.md)
- [TTNN Visualizer BOS Blackhole NPE 분석](investigations/2026-07-26-ttnn-visualizer-bos-blackhole-npe.md)
- [BOS 64K SDPA 성능 최적화 이력](investigations/2026-08-01-sdpa-dram-performance-optimization-history.md)
- [BOS 64K vanilla SDPA와 DRAM saturation 격차](investigations/2026-08-01-vanilla-sdpa-vs-dram-saturation-gap.md)
- [BOS Flash Decode 4 idle-core helper 설계 조사](investigations/2026-08-01-sdpa-four-helper-buffering-design.md)
- [BOS Llama 3.2 3B MLP 개선 조사](investigations/2026-08-04-bos-mlp-optimization-investigation.md)

### Incidents

- [Blackhole worker-FW timeout 및 호스트 freeze](incidents/2026-07-31-blackhole-worker-fw-host-freeze.md)
- [BOS MLP NoC profiling 후 worker-FW initialization failure](incidents/2026-08-01-bos-mlp-noc-profile-fw-init-failure.md)
- [BOS SDPA reduce-only helper deadlock 및 timeout SIGKILL](incidents/2026-08-02-bos-sdpa-reduce-helper-deadlock.md)
- [BOS Llama 3.2 3B DRAM-sharded MLP prefill validation failure](incidents/2026-08-02-bos-llama32-dram-sharded-prefill-validation-failure.md)
- [BOS MLP dual-NoC reader 첫 실제 실행 timeout](incidents/2026-08-03-bos-mlp-dual-noc-reader-timeout.md)
- [BOS MLP prefetch-helper NoC write-barrier Watcher abort](incidents/2026-08-03-bos-mlp-prefetch-helper-write-barrier.md)

### Handoffs, guides, and indexes

- [Qwen2.5-3B 32K profiling handoff](handoffs/2026-07-21-qwen25-3b-32k-profiling.md)
- [GEMM benchmark 측정 메커니즘](guides/gemm-benchmark-measurement.md)
- [TT-Metal venv 경로 구성](guides/tt-metal-venv-path.md)
- [Profiler run index](indexes/profiler-runs.md)

## 관리 방법

1. 파일 이름은 날짜가 있는 결과 문서라면 `YYYY-MM-DD-주제.md`를 사용한다.
2. 새 문서를 해당 유형 디렉터리에 넣고 이 색인과 `MANIFEST.md`를 함께 갱신한다.
3. 저장소 내부 원본을 갱신했다면 중앙 스냅샷을 다시 복사하고 SHA-256을 갱신한다.
4. CSV, SQLite, Tracy trace 같은 대형 산출물은 이곳에 복제하지 않는다. `indexes/` 문서에서
   timestamped run 디렉터리를 가리킨다.
5. 현행 결론이 바뀐 문서는 삭제하지 않고 `archive/`로 옮기며 대체 문서를 명시한다.

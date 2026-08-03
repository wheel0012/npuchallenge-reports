# 20-core, 6-endpoint BOS DRAM saturation benchmark

This benchmark implements the pipelined DRAM-read method described in
`tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` for the
BOS topology:

- 20 logical worker cores (`5 x 4`)
- 3 physical DRAM banks
- 2 worker NoC endpoints per bank, for 6 endpoints total
- native ring split: NOC0 uses x={0,4,5}, NOC1 uses x={1,2,3}
- four virtual channels edge-colored to prevent reuse at either an endpoint or
  within the same NoC/worker-row route
- two L1 block slots and transaction IDs 1/2, so the next block is issued
  before the previous block's barrier

Every endpoint receives at least three readers. One endpoint from each native
ring, on different banks, receives a fourth reader. This produces ten readers
on each NoC and a 7/7/6 reader split across physical banks. The host chooses
those endpoints and pairs readers with endpoint slots to minimize horizontal
route distance for the device's actual worker placement.

To keep traffic balanced, four-reader endpoints run `3 * iteration_quanta`
iterations per reader and three-reader endpoints run `4 * iteration_quanta`.
Consequently every endpoint, NoC, and physical bank receives the same byte
count despite the unequal reader counts.

The x=5 DRAM endpoint overlaps the PCIe-assigned location that the generic BOS
address generator normally avoids. It is intentionally included because this
benchmark exercises all six worker endpoints. Each endpoint is accessed from
only one NoC ring, avoiding the same-endpoint/two-ring condition documented by
SYS-1419. This is a topology-specific silicon benchmark, not a portable DRAM
addressing example.

## Build

```bash
./build_metal.sh --build-dir build_home_release --release --build-metal-tests --configure-only
cmake --build build_home_release --target test_dram_20_core_6_noc -j
```

## Run

```bash
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc
```

Defaults are a 4 KiB page, 512 pages per reader, 8 pages per block, iteration
quanta 4, and 5 measured runs after one warmup. The default transfers about
0.604 GB per measured run and reports bandwidth without imposing a platform
peak assumption.
Custom page sizes must be multiples of the Blackhole 64-byte DRAM alignment.


Set a known board peak explicitly to enforce the report's 90% saturation
criterion:

```bash
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --target-bandwidth-gbps <board-peak-gbps> \
  --target-utilization-percent 90
```

## Vanilla SDPA reader topology

Use `--reader-config vanilla-sdpa` to keep the active logical-core set and
NOC1 direction from the 64K Llama decode SDPA while retaining this benchmark's
pipelined saturation kernel. Endpoint binding in this mode is synthetic; it is
not copied from vanilla SDPA:

- the first 16 logical cores in row-major order on the 5x4 grid
- NOC1 only, using endpoints x={1,2,3}
- a synthetic physical-x heuristic that fixes readers to three endpoints,
  producing a 3/7/6 split across the three physical banks
- per-reader iteration counts scaled by the LCM of 3, 7, and 6, so every bank
  transfers exactly the same byte count
- virtual channels selected from worker physical x modulo four

```bash
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config vanilla-sdpa
```

The default reader configuration remains `six-endpoint`.

## Why vanilla SDPA does not reach the saturation result

The `vanilla-sdpa` microbenchmark mode copies the SDPA active logical-core
set and NOC1 direction. It does **not** copy vanilla SDPA's per-tile DRAM
endpoint selection or K/V access cadence. The mode's fixed 3/7/6 endpoint
assignment is a synthetic physical-x grouping introduced for this benchmark.
This distinction explains why its bandwidth is substantially higher than the
real decode operation.

The measurements below used the same P100 device and five timed runs after one
warmup. The SDPA datapoint is the 64K paged-decode case with `B=1`, `NH=24`,
`NKV=8`, `D=128`, BF16 Q, BF8 K/V, a 5x4 program grid, and `cur_pos=65535`.

| Case | Reader/endpoint configuration | Bandwidth | Per-bank equivalent |
| --- | --- | ---: | ---: |
| Six-endpoint saturation benchmark | 20 readers, NOC0/NOC1=10/10, six endpoints | 86.83 GB/s | 28.94 GB/s |
| Synthetic SDPA-core saturation benchmark | 16 readers, NOC1 only, fixed three endpoints | 66.47 GB/s | 22.16 GB/s |
| Vanilla SDPA decode | Real paged K/V reader and compute pipeline | 41.12 GB/s effective K/V | 13.71 GB/s effective |

Restricting the saturation benchmark to this synthetic three-endpoint layout
costs about 23.5% relative to six endpoints. Vanilla SDPA reaches about 61.9%
of that 66.47 GB/s reference. Because the benchmark fixes each reader to one
endpoint while SDPA does not, this ratio is a useful bound, not a like-for-like
measurement of the real SDPA endpoint topology.

### 1. SDPA drains each K/V burst with a full barrier

For this shape each SDPA reader handles a 128-token chunk as 16 BF8 tiles. The
reader issues the K reads, calls `noc_async_read_barrier()`, publishes the K
circular buffer, then repeats the same sequence for V. The next K/V burst is
not issued while the preceding burst is completing.

The saturation kernel instead uses two L1 slots and transaction IDs 1/2. It
issues the current block before waiting for the preceding transaction, keeping
DRAM work in flight across block boundaries. Its default block is eight 4 KiB
pages, or 32 KiB, compared with the roughly 17 KiB SDPA tile burst.

Relevant code:

- SDPA K read and full barrier:
  `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
- tagged, two-slot saturation pipeline:
  `tests/tt_metal/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/kernels/reader_dram.cpp`

### 2. Readers at different physical x positions finish at different times

The vanilla NoC trace contains 15 complete reader traces; the output core's
reader trace is incomplete and is excluded from the physical-x averages below.
The complete traces show the following read spans and time inside read
barriers:

| Worker physical x | Average read span | Barrier wait / span |
| --- | ---: | ---: |
| x0 | 2.226M cycles | 47.8% |
| x1 | 2.099M cycles | 45.9% |
| x2 | 1.816M cycles | 39.6% |
| x3 | 1.689M cycles | 35.0% |
| x4 | 1.553M cycles | 14.7% |

The physical-x trend is real, but it cannot be relabeled as fixed
seven-reader, six-reader, and three-reader endpoint queues in vanilla SDPA.
Each SDPA reader uses `TensorAccessor` and visits DRAM views according to the
current tile ID. The trace therefore demonstrates position-dependent read
completion and synchronization latency, not which single endpoint owns a
reader.

This is not evidence that aggregate NoC link bandwidth is continuously
saturated. tt-npe reported 0% modeled congestion impact and only moderate peak
link demand, while the device trace directly shows large, physical-x-dependent
barrier completion times.

### 3. The 3/7/6 count is not the DRAM byte distribution

K and V are DRAM-interleaved tensors. A vanilla reader visits DRAM views
according to tile address mapping. The `3/7/6` count describes only the fixed
reader placement introduced by the synthetic benchmark; it must not be
interpreted as vanilla SDPA's bank or endpoint traffic distribution.

The `vanilla-sdpa` microbenchmark deliberately introduces the fixed 3/7/6
assignment, then scales per-reader iterations by the LCM of 3, 7, and 6. This
gives every physical bank exactly 352.322 MB per default measured run and
provides a synthetic three-endpoint reference. It is not an exact reproduction
of paged SDPA's interleaved address sequence.

### Full-barrier A/B result

The benchmark supports `--pipeline-mode tagged` (default) and
`--pipeline-mode full-barrier`. The latter drains every block with an untagged
`noc_async_read_barrier()` before issuing the next block.

| Packet and burst | Tagged | Full barrier | Full-barrier delta |
| --- | ---: | ---: | ---: |
| 4096 B x 8 = 32 KiB | 66.50 GB/s | 66.43 GB/s | -0.10% |
| 4096 B x 4 = 16 KiB | 66.64 GB/s | 66.72 GB/s | +0.11% (noise) |
| 1088 B x 16 = 17 KiB | 53.60 GB/s | 52.87 GB/s | -1.37% |

The 1088-byte case matches a BF8 tile packet and a 16-tile K or V burst much
more closely than the default 4 KiB pages. Two conclusions follow:

1. a full barrier by itself does not explain the 41.12 GB/s SDPA result;
2. reducing the packet from 4096 B to 1088 B costs about 19.4% even with tagged
   overlap, while the full barrier adds only about 1.4% in this synthetic,
   fixed-endpoint workload.

Tagged double buffering may still help real SDPA when interleaved destinations
and CB scheduling increase latency variance, but this A/B test does not support
it as the primary explanation. The remaining difference between the
full-barrier 1088-byte reference (52.87 GB/s) and SDPA (41.12 GB/s) is about
22.2%. Candidate contributors are per-tile interleaved DRAM address changes,
paged tile-ID translation, transposed K writes into L1, CB/compute
backpressure, and reduction/output traffic.

Reproduce the closest A/B pair with:

```bash
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config vanilla-sdpa --pipeline-mode tagged \
  --page-size 1088 --pages-per-block 16

./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config vanilla-sdpa --pipeline-mode full-barrier \
  --page-size 1088 --pages-per-block 16
```

### Real SDPA cross-chunk tagged-prefetch A/B

The experimental paged SDPA reader now uses the existing double-buffered K/V
circular buffers as a cross-chunk pipeline. After publishing `K_i` so compute
can begin QK, it issues `V_i` with transaction ID 2 and prefetches `K_(i+1)`
with transaction ID 1. It waits for and publishes `V_i` first, then publishes
the prefetched K chunk. Unsupported configurations retain the original full
barriers.

The 64K, 6-endpoint A/B below used the same binary and consecutive runs. Each
mean includes the warmup and measured SDPA calls.

| 6-endpoint SDPA | Mean FW duration | Effective K/V bandwidth | Mean kernel duration |
| --- | ---: | ---: | ---: |
| Full barriers | 2.52401 ms | 56.500 GB/s | 2.52136 ms |
| Cross-chunk prefetch | 2.52965 ms | 56.374 GB/s | 2.52216 ms |

Cross-chunk prefetch changed kernel duration by +0.032% and FW duration by
+0.223%. This is neutral within run-to-run noise and does not improve the
remaining SDPA bottleneck. The result suggests that compute/CB consumption
already hides most of the reader gap, or that the limiting latency lies in
per-tile interleaved/paged reads rather than at the K-to-V or chunk boundary.

Enable the experiment with `TT_METAL_SDPA_DECODE_TAGGED_ASYNC=1` together with
the dual-NoC setting. The environment variable retains its original name even
though its implementation is now a cross-chunk prefetch pipeline.

Profiler references:

- ON:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_cross_chunk_2026_07_27_04_45_00`
- OFF:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_dual_noc_6ep_cross_chunk_off_2026_07_27_04_44_00`

### Real SDPA K-chunk size A/B

The 64K, 6-endpoint reader was also measured with larger K/V chunks. Durations
below are the mean of warmup and measured maximum-core kernel spans reconstructed
from the raw device profile.

| K chunk | Mean kernel duration | Effective K/V bandwidth | Latency vs. 128 |
| ---: | ---: | ---: | ---: |
| 128 tokens | 2.51933 ms | 56.605 GB/s | baseline |
| 256 tokens | 2.03641 ms | 70.028 GB/s | -19.17% |
| 512 tokens | 2.00242 ms | 71.217 GB/s | -20.52% |

The 256- and 512-token full-layer outputs were bitwise identical to the
128-token output (max_abs=0). This cur-pos-only test uses the initialized KV
cache, so broader model accuracy and random-page-table coverage are still
required before changing the Blackhole production default.

The result shows that repeated small-chunk compute/softmax setup was a much
larger limitation than K/V cross-chunk prefetch. Moving from 256 to 512 adds
only another 1.67%, so 256 is the safer initial production candidate while 512
is the measured performance winner.

### K-chunk 256 bottleneck follow-up

The uninstrumented 256-token trace shows that the remaining tail is not a pure
matrix-engine limit. Although the program config exposes a 5x4 grid, batch 1
with 8 KV heads computes `floor(20 / 8) = 2` cores per head, so only 16 cores
are active and each head has one reducer plus one worker.

For the measured run, the maximum spans were 2.0375 ms on BRISC, 1.9990 ms on
NCRISC, and 2.0357 ms on TRISC. The eight reducer/worker pairs completed between
about 1.716 ms and 2.036 ms, an approximately 18.6% head-to-head spread. Some
reducers finish their local K/V work early but remain alive until the paired
worker arrives and the cross-core softmax merge completes. This makes the
slowest reader/worker path, followed by reduction synchronization, the tail.

An accumulated `cb_wait_front` probe reported 0.369-0.619 ms per active core in
the measured run. This includes intermediate compute-pipeline CB waits as well
as K/V input waits, so it is not a direct DRAM-stall measurement. A second
QK/QKV call-site probe was intentionally discarded for absolute timing because
its repeated scopes inflated the kernel from about 2.04 ms to about 3.27 ms;
it was used only to confirm the reducer/worker imbalance pattern.

At 512 tokens, the maximum NCRISC and TRISC spans were still about 1.981 ms and
2.004 ms. Their small improvement from 256 matches the overall 1.67% gain and
explains the plateau: larger chunks have already removed most per-chunk
compute/softmax overhead, leaving K/V delivery imbalance and the per-head merge
tail as the next targets. The next useful A/B is balanced endpoint/chunk
assignment across each reducer-worker pair, measured with non-intrusive kernel
spans and NOC counters.

### K-chunk 256 endpoint-balance A/B

Two opt-in endpoint assignments were tested against the same 256-token,
dual-NoC, 6-endpoint baseline. Each result below contains four SDPA calls from
two processes (warmup plus measured call per process). The baseline endpoint
loads are `3/3/2/3/3/2`, with aggregate NoC0/NoC1 loads of `8/8`.

| Assignment | Mean max-core span | Effective K/V bandwidth | Delta vs. baseline |
| --- | ---: | ---: | ---: |
| Existing route-minimum assignment | 2.02951 ms | 70.266 GB/s | baseline |
| Opposite NoC within every reducer/worker pair | 2.03730 ms | 69.998 GB/s | +0.384% latency |
| Nominal physical-bank reader split 5/5/6 | 2.03313 ms | 70.141 GB/s | +0.179% latency |

The pair-balanced variant changes four same-NoC pairs to opposite-NoC pairs
without changing total endpoint loads or route cost. It reduces the eight-head
completion spread from about 305 us to about 136 us, but only by slowing the
previously fast heads; the critical heads are unchanged. Enable it with
`TT_METAL_SDPA_DECODE_PAIR_BALANCED_ENDPOINTS=1`.

The bank-balanced variant changes endpoint loads to `3/2/3/3/3/2`. With BOS
endpoint pairs `{x0,x1}`, `{x2,x5}`, and `{x3,x4}`, this changes the nominal
reader-count split from 6/4/6 to 5/5/6 while keeping NoC0/NoC1 at 8/8. It also
reduces the assignment route cost from 26 to 21, but does not improve the
critical path. Enable it with `TT_METAL_SDPA_DECODE_BANK_BALANCED_ENDPOINTS=1`.

Both variants produced full-layer outputs bitwise identical to the baseline.
Neither is enabled by default. The neutral-to-negative result shows that static
endpoint reader counts and pair-level NoC symmetry do not model the actual
paged, interleaved K/V service tail well enough. A better next measurement is
per-core/per-bank NOC transaction accounting or a chunk-phase rotation A/B,
rather than another assignment based only on endpoint counts.

Raw device profiles:

- Existing assignment:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_off_2026_07_31`
  and
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_off_rep2_2026_07_31`
- Pair-balanced:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_on_2026_07_31`
  and
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_pair_balance_on_rep2_2026_07_31`
- Bank-balanced:
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_bank_balance_on_2026_07_31`
  and
  `/home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_bank_balance_on_rep2_2026_07_31`

The experiment is selected with --sdpa-k-chunk-size {128,256,512} in
models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py.

Raw device profiles:

- 128:
  /home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_128_profile_retry_2026_07_31
- 256:
  /home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_256_device_profile_2026_07_31
- 512:
  /home/iris_hb4/profiler_runs/sdpa_decode_64k_6ep_kchunk_512_device_profile_2026_07_31

Profiler reference:

`/home/iris_hb4/profiler_runs/sdpa_decode_64k_vanilla_curpos_only_npe_2026_07_26_03_31_33`

Other tuning options are `--device-id`, `--reader-config`, `--pipeline-mode`,
`--page-size`,
`--pages-per-core`, `--pages-per-block`, `--iteration-quanta`, and
`--num-tests`.


### Contiguous and DRAM-sharded synthetic KV A/B

The cur-pos-only runner supports three KV layouts through `--kv-layout`:

- `paged`: production paged KV cache and page table
- `contiguous`: non-paged contiguous cache with the existing interleaved DRAM layout
- `contiguous-sharded`: non-paged cache allocated directly as a six-way DRAM height-sharded tensor

Synthetic modes leave the zero-initialized KV cache unchanged and skip the
single-token cache update. They retain the normal QK, softmax, AV, accumulation,
and output path. `--cores-per-kv-head 1` uses a 4x2 grid to assign one core to
each of the eight KV heads; the default value 2 uses the 5x4 grid and 16 active
cores.

The 64K, K-chunk 256 results below use the same effective K/V byte definition
as the earlier experiments.

| Active cores | KV layout | Measured max-core span | Effective K/V bandwidth |
| ---: | --- | ---: | ---: |
| 16 | paged, interleaved | 2.0295 ms | 70.266 GB/s |
| 16 | contiguous, interleaved | 2.0213 ms | 70.551 GB/s |
| 16 | contiguous, six-way DRAM sharded | 5.8520 ms | 24.369 GB/s |
| 8 | contiguous, interleaved | 3.7264 ms | 38.269 GB/s |
| 8 | contiguous, six-way DRAM sharded | 5.8546 ms | 24.358 GB/s |

Removing paging improves the 16-core result by only about 0.4%, so page-table
lookup and paged tile-ID translation are not the primary limitation for the
identity page table used here. Naively replacing the cache memory config with a
generic DRAM-sharded `TensorAccessor` path is substantially worse: it is about
57% slower than interleaved at the same eight-core count.

This is not the specialized DRAM-sharded matmul reader from the bandwidth tech
report. That implementation collocates one reader and compute core with each
bank and performs long bank-local reads. SDPA still issues tile-granular reads
through its generic accessor, has eight KV heads over six exposed DRAM shards,
and does not align worker ownership with shard ownership. Reaching the tech
report behavior therefore requires a dedicated bank-local KV reader plus an
explicit distribution path to SDPA compute cores, rather than only changing the
tensor memory config.

Reproduce the two 16-core synthetic cases with:

```bash
TT_METAL_SDPA_DECODE_DUAL_NOC=1 TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 \
python_env/bin/python models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py \
  --precision-mode accuracy --context-len 65536 --sdpa-k-chunk-size 256 \
  --kv-layout contiguous

TT_METAL_SDPA_DECODE_DUAL_NOC=1 TT_METAL_SDPA_DECODE_ENDPOINT_COUNT=6 \
python_env/bin/python models/bos_model/llama32/run_llama32_3b_curpos_only_single_layer_npe.py \
  --precision-mode accuracy --context-len 65536 --sdpa-k-chunk-size 256 \
  --kv-layout contiguous-sharded
```


### Six endpoint-local readers with 16-core fanout

`--reader-config six-reader-fanout` is a proof of concept for the proposed
SDPA ownership topology. Six producer cores are selected next to the six DRAM
endpoints, with three producers on NOC0 and three on NOC1. The producers are
also members of the 16 active compute-core set; the remaining ten cores are
assigned to producer groups of size `3/3/2/3/3/2` by a minimum-total-Manhattan-
distance search.

Each producer uses two L1 slots. It starts the next tagged DRAM read before
retiring and forwarding the current slot, unicasts the completed block to its
one or two remote helpers, then signals a per-slot ready semaphore. Receivers
acknowledge a slot as soon as its payload is visible, allowing the producer to
reuse it. All six producer L1 results and all ten receiver L1 results are
validated after repeated workload launches.

With 0.604 GB of actual DRAM reads per run, the measured sweep was:

| Pages/block | Block size | Effective DRAM bandwidth |
| ---: | ---: | ---: |
| 8 | 32 KiB | 48.181 GB/s |
| 16 | 64 KiB | 49.704 GB/s |
| 32 | 128 KiB | 51.070 GB/s |
| 64 | 256 KiB | 49.091 GB/s |

At the 128-KiB optimum, the ten remote copies add 1.007 GB of relay payload.
The aggregate DRAM-plus-relay payload rate is 136.187 GB/s. A same-block-size
20-reader direct baseline reaches 62.491 GB/s; the existing 32-KiB direct
saturation configuration reaches about 86.8 GB/s.

This rules out a direct translation of one endpoint owner plus full-K/V fanout
as a bandwidth win for the current SDPA. The relay copies and acknowledgements
consume the producer data-movement RISC and NoC bandwidth before the DRAM
endpoints saturate. The useful part to carry into SDPA is endpoint-local
ownership, but consumers should receive disjoint sequence shards and merge
partial softmax state, rather than receiving duplicate full blocks. That keeps
DRAM traffic balanced without multiplying K/V traffic on-chip.

Reproduce the best fanout point with:

```bash
TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-reader-fanout --pipeline-mode tagged \
  --pages-per-block 32 --num-tests 3
```


### Six-reader disjoint sequence-shard relay

`--reader-config six-reader-sharded` removes the duplication from the full
fanout experiment. The same six endpoint-local producers cover 16 logical
consumer streams, but each DRAM block belongs to exactly one consumer. The
bank-aware group sizes are `3/2/3/3/3/2`: NOC0 and NOC1 each own eight streams,
and the physical-bank stream loads are `5/5/6`.

The six producers perform tagged double-buffered reads for their group streams.
Ten remote streams receive one unicast copy; the six local streams keep their
payload in the producer L1. Receiver acknowledgements protect staging-slot
reuse. Every stream reads a different source range, and all 16 final L1 blocks
are checked against those distinct ranges after repeated launches.

The default workload is normalized to the same 0.604 GB of DRAM traffic as the
other saturation configurations:

| Pages/block | Block size | Effective DRAM bandwidth |
| ---: | ---: | ---: |
| 8 | 32 KiB | 67.818 GB/s |
| 16 | 64 KiB | 74.772 GB/s |
| 32 | 128 KiB | 77.369 GB/s |
| 64 | 256 KiB | 71.873 GB/s |

A separate five-run check at 128 KiB measured 76.915 GB/s average, with a
75.255--78.664 GB/s range. The ten remote streams add 0.377 GB of relay
payload, and the corresponding aggregate DRAM-plus-relay payload rate is
124.987 GB/s. This is about 10% above the 70.266 GB/s effective K/V bandwidth
of the 64K, K-chunk-256 SDPA decode baseline, while full-block fanout reached
only 51.070 GB/s.

SDPA already assigns disjoint sequence chunks to the two cores for each KV head
and merges their partial `(m, l, O)` state in the head reducer. Therefore the
remaining SDPA integration work is specifically a reader split: six owner
readers must fill the existing K/V CBs for ten receiver cores while preserving
the current compute and reducer kernels. The benchmark establishes useful
headroom for that change; it does not count the additional CB handoff required
inside SDPA.

Reproduce the best point with:

```bash
TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-reader-sharded --pipeline-mode tagged \
  --pages-per-block 32 --num-tests 5
```
## 2026-08-02 post-reboot direct-burst sanity run

After the user confirmed that the host/server had been recovered, the mandatory
32x32 BF16 `ttnn.add` smoke test completed with `SMOKE_VALUE 2.0`, exit code 0,
and an explicit `DEVICE_CLOSED` marker. The existing direct DRAM benchmark was
then run once without Tracy or NoC profiling:

```bash
env TT_METAL_HOME=/home/iris_hb4/tt-metal-hb4 \
  timeout --signal=INT --kill-after=15s 90s \
  ./build_home_release/test/tt_metal/perf_microbenchmark/12_dram_20_core_6_noc_read/test_dram_20_core_6_noc \
  --reader-config six-endpoint --pipeline-mode tagged \
  --page-size 4096 --pages-per-block 8 --num-tests 1
```

The configuration used 20 readers, all 6 worker NoC endpoints across 3
physical DRAM banks, 10 readers on each NoC, and balanced traffic of 201.327 MB
per physical bank. Each tagged burst was 32 KiB (`4 KiB x 8`) and the measured
transfer was 0.604 GB.

```text
run 1/1: 7.149 ms, 84.482 GB/s
Test Passed
exit code: 0
```

This single sanity measurement is 2.7% below the earlier 86.83 GB/s best result.
One sample is insufficient to classify that difference as a regression. This
run validates the direct saturation control path only; it does not yet validate
an SDPA helper-to-consumer direct-burst implementation.

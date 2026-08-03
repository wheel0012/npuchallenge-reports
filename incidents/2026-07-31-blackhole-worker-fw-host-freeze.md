# Incident report: Blackhole worker-FW timeout and host freeze

## Summary

On 2026-07-31 UTC, investigation of a stalled Llama 3.2 3B SDPA decode run
progressed from process termination and device open/close checks to UMD reset
experiments. The final experiment directly issued the legacy
`RESET_PCIE_LINK` ioctl for device 0, waited three seconds, and then issued
`RESTORE_STATE`. Immediately afterward, `lspci` reported PCI function
`0000:02:00.0` with revision `ff`. The user subsequently reported that the
server froze.

The host freeze is strongly correlated with the direct PCIe-link reset. It is
not attributed to the SDPA reader or compute kernel itself: before the final
reset, Watcher showed that worker cores never entered any SDPA kernel body.
They were stuck in firmware initialization.

No further device or reset commands should be run on this machine until it has
been recovered out of band and its PCIe/KMD state has been checked.

## Impact

- The interactive server became unresponsive and required external recovery.
- No valid SDPA performance result was produced during this investigation.
- The source worktree remained clean; no SDPA source change from this incident
  remains in the repository.
- One persistent TT-Metal cache directory was moved, not deleted:
  `/home/iris_hb4/.cache/tt-metal-cache/142354481e/11626558668738827881`
  was quarantined as
  `/tmp/tt-metal-cache-stale-11626558668738827881-20260731`.
- Two temporary reset helpers were created under `/tmp`:
  `tt_warm_reset_device0` and `tt_legacy_pcie_reset_device0`.

## Test configuration

- Device: Blackhole/P100, logical device 0
- PCI function: `0000:02:00.0`
- Driver: `bos`
- KMD version reported by UMD: `2.3.0`
- Repository: `/home/iris_hb4/tt-metal-hb4`
- Llama decode target: batch 1, paged KV cache, 24 Q heads, 8 KV heads,
  head dimension 128, block size 32, K chunk 256
- Main grid under investigation: `5x4` with two cores per KV head

## Timeline and evidence

### 1. Initial process stop and device open/close

The long-running Llama runner was terminated. A minimal `ttnn.open_device(0)`
followed by `ttnn.close_device()` succeeded. This demonstrated only that the
PCI device and user-mode driver could be opened; it did not exercise a worker
Tensix program.

### 2. Source/build consistency check

Only the SDPA compute CB indices had remained changed (`c30/c31` had been moved
to `c14/c15`) while no matching host/reader six-reader path was present in the
worktree. The compute CB indices were restored to the vanilla `c30/c31` values.
The resulting worktree was clean.

Because an older custom `_ttnncpp.so` could still have survived after source
restoration, the SDPA program-factory translation units were forced to rebuild.
The build relinked both `_ttnncpp.so` and `_ttnn.so` successfully.

### 3. Vanilla and isolated SDPA runs

The following runs did not complete:

- Full vanilla Llama layer at 64K: timed out after 15 minutes.
- Full vanilla Llama layer at 8K: timed out after 3 minutes.
- Isolated paged SDPA at 8K on `5x4`: timed out after 3 minutes.
- Isolated paged SDPA at 8K on `4x2`: worker-FW timeout.

Reducing the context length and core count did not remove the failure.

### 4. Worker firmware timeout

After relinking, runtime diagnostics reported:

```text
Device 0: Timeout (10000 ms) waiting for physical cores to finish
Device 0 init: failed to initialize FW! Try resetting the board.
```

For the `4x2` isolated SDPA run, the timed-out physical cores were the eight
cores `(x=0..3, y=0..1)`.

### 5. Warm reset and cache isolation

The supported UMD call `WarmReset::warm_reset({0})` reported:

```text
Starting reset for blackhole architecture.
Reset succesfully completed.
```

This did not recover worker-FW launch. The relevant persistent kernel cache was
then quarantined and regenerated from current source. Old and newly generated
reader artifacts had identical checksums, and the failure remained. This
rules out a corrupt or stale SDPA kernel cache as the primary cause.

An important limitation is that KMD `2.3.0` does not support UMD's
architecture-agnostic ASIC reset path. The source requires KMD `2.4.1` or
newer. On this system, the Blackhole warm reset therefore used the legacy
configuration-write path.

### 6. Watcher result

With `TT_METAL_WATCHER=1`, all eight target worker cores remained at:

```text
BRISC waypoint: I
NCRISC/TRISC waypoints: X
run message: GO
```

Watcher identified the intended kernels as `writer_decode_all`,
`reader_decode_all`, and `sdpa_flash_decode`, but none entered its kernel body.
The dispatch cores remained alive. This is evidence of a worker firmware
launch/initialization failure, not an SDPA circular-buffer or reader/compute
barrier deadlock.

The captured log is `generated/watcher/watcher.log` in the repository, subject
to being overwritten by later Watcher runs.

### 7. Non-SDPA smoke test

A minimal 32x32 `ttnn.add` failed with the same worker-FW initialization timeout
on the same eight cores. Therefore the failure had become device-wide for
Tensix program launch and was no longer specific to SDPA.

### 8. Freeze-triggering reset experiment

The final helper directly called the legacy UMD reset ioctls in this order:

```text
RESET_PCIE_LINK(device 0)
wait 3 seconds
RESTORE_STATE(device 0)
```

The helper exited, but the next PCI query showed:

```text
0000:02:00.0 ... [16c3:abcd] (rev ff)
```

Revision `ff` indicates that PCI configuration space was not responding
normally. The server then froze, as reported by the user. This direct
PCIe-link reset is the incident trigger with the highest confidence.

The exact kernel-level cause of the host freeze is not proven because host
`dmesg`, AER, and panic logs were not captured after the machine became
unresponsive. Plausible mechanisms include incomplete link restoration, a
driver/device reset race, or host accesses to an unavailable BAR while the BOS
driver still considered the device active.

## Root-cause assessment

There are two distinct failures:

1. **Pre-existing worker-FW launch failure.** Worker cores accepted a GO state
   but never left firmware initialization. It affected SDPA and a trivial add.
   Its underlying hardware, firmware, or driver cause remains unresolved.
2. **Host-freeze incident.** A direct legacy `RESET_PCIE_LINK`/`RESTORE_STATE`
   sequence was issued on KMD 2.3.0. PCI config subsequently returned revision
   `ff`, followed by the reported server freeze. This reset experiment should
   be treated as the proximate cause of the freeze.

The six-reader SDPA design is not implicated by the available evidence. The
worker cores did not reach reader, compute, or writer kernel code.

## Actions that must not be repeated

- Do not run `/tmp/tt_legacy_pcie_reset_device0`.
- Do not directly issue `RESET_PCIE_LINK`, `RESTORE_STATE`, sysfs FLR, PCI
  remove/rescan, or driver unbind/rebind from an interactive experiment.
- Do not continue retrying SDPA after a trivial Tensix smoke test reports
  worker-FW initialization timeout.
- Do not infer device health from open/close alone; it does not validate worker
  firmware or kernel dispatch.
- Do not combine Watcher, device profiler, and reset experiments in one run.

## Safe recovery and next steps

1. Recover the host and card out of band using the machine owner's standard
   reboot or full power-cycle procedure.
2. After recovery, collect `journalctl -k -b -1`, PCIe AER messages, and BOS KMD
   logs before running any TT workload.
3. Confirm the PCI function no longer reports revision `ff` and verify the BOS
   driver and `/dev/bos/0` are healthy.
4. Run a single minimal Tensix test such as 32x32 add with a short timeout. Stop
   immediately if worker-FW initialization fails.
5. Validate vanilla isolated SDPA at a small sequence length before running the
   Llama layer or any six-reader experiment.
6. Use administrator-controlled reset tooling only. Prefer KMD/UMD versions
   that support the architecture-agnostic reset flow (`2.4.1+`) before further
   automated reset testing.
7. Preserve Watcher, kernel, AER, and host logs in a timestamped run directory
   for any recurrence.

## Recommended test guardrails

- Add a preflight add smoke test before every SDPA experiment.
- Use an external 30--60 second process timeout for first launches.
- On timeout, terminate the test once and collect Watcher state; do not loop
  open/close or reset attempts.
- Require explicit operator approval and out-of-band access before any action
  that changes the PCIe link or driver binding.
- Keep source and `_ttnncpp.so` build provenance together and record the git
  commit, build directory, KMD version, kernel-cache hash, and environment for
  every performance run.

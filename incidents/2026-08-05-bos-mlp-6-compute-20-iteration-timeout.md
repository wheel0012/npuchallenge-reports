# BOS MLP 6-compute 20-iteration timeout

- 날짜: 2026-08-05 UTC
- 장치: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- 영향: isolated MLP completion/close 없음, exit 137, 장치 격리

## 타임라인

- 재부팅 뒤 32×32 BF16 add: value 2.0, close, exit 0.
- 12-compute tagged depth-1 correctness: PCC 0.9996410623, close, exit 0.
- 12-compute depth-2 20회: mean 1.467559 ms, close, exit 0.
- 12-compute depth-1 20회: mean 1.493915 ms, close, exit 0.
- 04:24 UTC: fanout off 6-reader/6-compute 20회 시작.
- factory log: readers 6, compute workers 6, W1/W3 read page 6528 B, W2 8704 B.
- 90초 timeout 뒤 SIGINT cleanup 미완료. `--kill-after=15s` 발동, exit 137.
- `MLP_COMPLETED`, `DEVICE_CLOSED`, PCC와 samples 출력 없음.
- 종료 뒤 관련 Python/Tracy/capture child 없음. 추가 device workload 없음.

## 마지막 실행 구성

```bash
env -u TT_METAL_MLP_FUSED_GATE_UP -u TT_METAL_MLP_FUSED_GATE_UP_EPILOGUE \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH1 \
  -u TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED_DEPTH3 \
  -u TT_METAL_SDPA_DECODE_REDUCE_ONLY_HELPER \
  MLP_AB_ITERATIONS=20 TT_METAL_MLP_DRAM_SHARDED=1 \
  TT_METAL_MLP_W2_IN0_BLOCK_W=16 TT_METAL_MLP_DRAM_SHARDED_16K_READ_PAGE=1 \
  TT_METAL_MLP_DRAM_SHARDED_FANOUT2=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT2_TAGGED=0 \
  TT_METAL_MLP_DRAM_SHARDED_BALANCED_ENDPOINTS=0 \
  TT_METAL_MLP_DRAM_SHARDED_PREFETCH_HELPERS=0 TT_METAL_MLP_DRAM_SHARDED_FANOUT3=0 \
  TT_METAL_TURBOQUANT=0 timeout --signal=INT --kill-after=15s 90s \
  /home/iris_hb4/tt-metal-hb4/python_env/bin/python \
  models/bos_model/llama32/tests/run_mlp_block_width_ab.py
```

## 원인 평가

확정 사실은 6-reader/6-compute 구성과 90초 미완료뿐이다. correctness call, measured loop 또는 close
중 어디에서 정지했는지 marker가 없어 모른다. 6 compute가 단순히 느린지 kernel deadlock인지도
구분하지 못한다. 따라서 성능 수치나 compute 부족 비율을 추론하지 않는다.

## 복구 및 예방

- exit 137 즉시 장치를 격리했다.
- 다음 device workload 전 사용자의 host/server 재시작 확인이 필요하다.
- 동일 20-iteration 6-compute 명령을 재실행하지 않는다.
- 재부팅 뒤 필요하면 1 measured call만 실행하고 host-side correctness/iteration waypoint를 추가한다.
- 첫 1회가 통과한 뒤에만 3회로 늘린다. profiler와 Watcher는 사용하지 않는다.
- endpoint admission 비교의 현행 결론은 성공한 depth-2/depth-1 A/B만 사용한다.

Artifact는 없다. console transcript만 존재한다.

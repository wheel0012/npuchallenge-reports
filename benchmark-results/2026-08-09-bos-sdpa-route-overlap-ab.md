# BOS SDPA directed-link route-overlap A/B

## 결론

6×5 torus의 directed-link 공유량을 직접 최소화하는 exact DFS opt-in을 구현했다. Static model의
`shared-edge pairs`는 33에서 30으로 9.09% 줄었지만, 최대 directed-link load는 4로 그대로였다.

Llama 3.2 3B 64K decode에서 세 쌍을 측정한 결과 candidate는 synchronized SDPA wall mean이
2,329.544 us에서 2,337.976 us로 0.362% 악화됐다. Profiled full-token latency도 0.107% 악화됐다.
성능 이득이 없으므로 stable 기본값에는 적용하지 않고 opt-in을 off로 유지한다.

## 장치와 topology

- board identity: custom 20-core BOS NPU
- runtime/code architecture: Blackhole
- available worker grid: logical 5×4 = 20 cores
- SDPA active compute/readers: 16; idle program cores: 4
- physical NoC grid: 6 columns × 5 rows
- DRAM endpoints: physical row `y=3`, `x=0..5`
- physical DRAM banks: endpoint pairs `{x0,x1}`, `{x2,x5}`, `{x3,x4}`
- current endpoint loads: `3/2/3/3/3/2`; physical-bank loads: `5/5/6`
- reader NoC loads: NoC0/NoC1 `8/8`

## 구현

Opt-in:

```text
TT_METAL_SDPA_DECODE_ROUTE_OVERLAP_OPTIMIZED=1
```

수정 위치:

```text
/home/iris_hb4/tt-metal-hb4/
  ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/
    sdpa_decode_program_factory.cpp
```

Route model은 `tt-npe-bos` Blackhole 계약과 같은 dimension order를 쓴다.

```text
NoC0: east(x+) until destination x, then south(y+) until DRAM row y=3
NoC1: north(y-) until DRAM row y=3, then west(x-) until destination x
wrap: 6×5 torus
```

기존 제약은 유지했다.

1. endpoint loads `3/2/3/3/3/2`
2. physical-bank loads `5/5/6`
3. NoC0/NoC1 `8/8`
4. 같은 KV head의 reducer/worker 두 core는 반대 NoC
5. output core의 기존 writer-swap penalty

Leaf assignment의 목적함수는 아래 lexicographic 순서다.

```text
1. peak directed-link load
2. shared-edge pairs = sum(load * (load - 1) / 2)
3. total route hops
4. legacy Manhattan/output penalty
```

Audit patch:

```text
/home/iris_hb4/tmp/codex-patches/20260809-043000-sdpa-route-overlap-optimizer.patch
SHA-256 16a0dbdd9b93792179536bbb5cf095a3096dca3e195591dd15e53ce55bde4b06
```

`cmake --build build_home_release --target ttnn -j 8`은 성공했다. 실행 전 `ldd`로
`build_home_release/ttnn/_ttnncpp.so` 선택을 확인했다.

## Assignment 변화

Baseline:

```text
[2,4,4,3,1,0,4,2,3,5,0,3,2,5,1,0]
```

Candidate:

```text
[2,4,2,5,1,0,4,2,3,5,0,3,4,3,1,0]
```

Reader id 2/3과 12/13의 endpoint만 바뀐다.

| static metric | baseline | candidate | delta |
|---|---:|---:|---:|
| peak directed-link load | 4 | 4 | 0 |
| shared-edge pairs | 33 | 30 | -9.09% |
| total hops | 67 | 67 | 0 |
| endpoint loads | 3/2/3/3/3/2 | 3/2/3/3/3/2 | 동일 |
| NoC0/NoC1 | 8/8 | 8/8 | 동일 |

## 측정법

- model: `meta-llama/Llama-3.2-3B-Instruct`
- batch 1, paged KV, synthetic zero-initialized 64K context
- decode positions: 65,486..65,535
- K chunk: 256 tokens
- warmup 3, measured 50 tokens
- stable grouped concat과 stable 12-compute MLP 유지
- `TT_PROFILE_SDPA_WALL=1`: 각 SDPA call 전후 device synchronize
- 각 process의 SDPA calls: 1,484
- 순서: baseline→candidate, baseline→candidate, candidate→baseline
- Watcher, Tracy, NoC profiler 사용 안 함

## 결과

| pair | 구성 | SDPA avg us | full ms/token | tok/s | final token |
|---:|---|---:|---:|---:|---:|
| 1 | baseline | 2329.031 | 145.543196 | 6.870812 | 499 |
| 1 | candidate | 2322.431 | 144.571456 | 6.916995 | 499 |
| 2 | baseline | 2323.143 | 145.566014 | 6.869735 | 499 |
| 2 | candidate | 2343.300 | 146.426559 | 6.829362 | 499 |
| 3 | baseline | 2336.457 | 145.316894 | 6.881512 | 499 |
| 3 | candidate | 2348.197 | 145.895821 | 6.854206 | 499 |
| mean | baseline | 2329.544 | 145.475368 | 6.874020 | 499 |
| mean | candidate | 2337.976 | 145.631279 | 6.866854 | 499 |
| delta | candidate vs baseline | +0.362% | +0.107% | -0.104% | 동일 |

모든 여섯 profiled run은 `WARMUP_COMPLETE`, `RESULT_JSON`, `DEVICE_CLOSED`, exit 0을 기록했다.

추가 unprofiled candidate는 128.821304 ms/token, 7.762691 tok/s, final token 499였다. 직전 stable
unprofiled 기록 7.769222 tok/s보다 0.084% 낮다. 같은 즉시 A/B 쌍은 아니므로 보조 근거로만 쓴다.

## 제외한 isolated run

기존 isolated script의 analytic constant-output check는 baseline에서 finite output을 얻었지만
`max_abs_error=0.6875`로 자체 threshold 0.1을 넘었다. Program은 completion 및 `DEVICE_CLOSED` 뒤 exit 1로
종료했다. Timeout, signal, exit 124/137은 아니며 A/B 성능 표본으로 쓰지 않았다.

## 해석

관측 사실:

- endpoint, bank, NoC load와 total hops를 고정하고 shared-edge pairs만 줄였다.
- peak directed-link load 4는 줄지 않았다.
- 세 쌍 평균 SDPA latency는 개선되지 않았다.

추론:

- peak link가 그대로라 pair-count 3 감소는 critical service tail을 바꾸기에 부족하다.
- static request route는 DRAM response/data phase, paged KV transaction timing, VC pressure를 표현하지 않는다.
- reader가 같은 link를 쓰더라도 issue phase가 다르면 static overlap은 실제 congestion이 아닐 수 있다.

다음 route 최적화는 static edge count보다 time-expanded edge load 또는 실제 chunk-phase weight를 써야 한다.
MLP는 weight reader 12개가 모두 NoC1을 쓰므로 이 모델을 적용할 다음 우선 대상이다.

## Artifact

Directory:

```text
/home/iris_hb4/profiler_runs/sdpa_route_overlap_ab_2026_08_09_04_45_00
```

주요 파일 SHA-256:

```text
full_baseline.log    72b88737550a8054022c236ee89289a16b29ffc64e7c8c8a50f737e5482d8a8d
full_candidate.log   c0478d4a66c6e04d77b7661dbc601a4e4b6766b179df2ae30318e87e4844ae9a
full_baseline_2.log  cd06eddb8e48ea77af57635bff7c63481428bd57890c0f9e2092c587aeeec7e8
full_candidate_2.log dc6b28ba4eaac8a71bea85fcb03420476c3bb20118be00ad2c3be8d13030f9ca
full_baseline_3.log  349cec965ec73548394d78b30b3800d7982a93f3f7263fa76973957fae90381b
full_candidate_3.log 5ccac7e54cd9228cfd5546259f25d0c9bcddac23cef7c0721579b5c7079ae24a
unprofiled_candidate.log a0845fffb6ec2b0a14267f74f8a4d1a85c90a4c72032e2979d685183c429dbf1
```

## 결정

- `TT_METAL_SDPA_DECODE_ROUTE_OVERLAP_OPTIMIZED`: 실험용 opt-in으로 유지
- stable 실행: unset 또는 `0`
- 현재 stable endpoint assignment 유지
- MLP 적용 전 host-side time-expanded route model부터 작성

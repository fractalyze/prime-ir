# NTT Benchmark Results

```shell
Run on (14 X 24 MHz CPU s)
CPU Caches:
L1 Data 64 KiB
L1 Instruction 128 KiB
L2 Unified 4096 KiB (x14)
Load Average: 22.54, 38.87, 26.62
```

## Macbook M4 Pro

> Note: `INTT` fails for more than 1 iteration so it seems like it is modifying
> the input. Fix later

```shell
# degree @ 20
bazel run -c opt //benchmark/ntt:ntt_benchmark_test
```

| Benchmark Name                 | Time (s) | CPU (s) | Iterations |
| ------------------------------ | -------- | ------- | ---------- |
| BM_ntt_benchmark               | 0.321    | 0.320   | 2          |
| BM_intt_benchmark/iterations:1 | 0.475    | 0.473   | 1          |

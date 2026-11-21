# CPU Inference Benchmark

## Background and goal

[An earlier experiment](v0.6.md) found that using four asynchronous inference requests improved mean throughput by approximately 11% over sequential execution. I repeated the comparison to see whether the gain would hold across multiple runs and whether asynchronous execution would preserve the classification results.

The benchmark compares one synchronous request with pools of 2, 4, and 8 asynchronous requests. It first checks that every submitted job completes and that the Top-10 output matches the synchronous reference. It then measures throughput, run-to-run variation, and CPU usage.

## Experimental setup

| Setting | Value |
| --- | --- |
| Hardware | AMD Ryzen 7 PRO 7840U |
| Environment | Ubuntu on WSL2, AC power connected, Windows Balanced mode |
| Runtime | OpenVINO 2026.4.0, CPU device |
| Model | SqueezeNet 1.1 (`models/squeezenet1.1-7.onnx`) |
| Input | `banana.bmp` and `dog.bmp` from `data/images`, repeated in a fixed order |
| OpenVINO settings | `INFERENCE_NUM_THREADS=8`, `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1` |
| Build | CMake Release build with OpenCV disabled |

All four modes use the same executable, model, inputs, and OpenVINO settings. The request count controls how many image jobs the application can have in flight. The OpenVINO inference thread count is fixed at eight.

## Build the benchmark

Run the following from the repository root in WSL with the OpenVINO C Runtime and sample headers installed.

```bash
build=benchmark/v0.7-memory-analysis/build/p1

cmake -S benchmark/v0.7-memory-analysis -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON &&
cmake --build "$build" --parallel
```

## Correctness validation

Before timing any runs, the [benchmark runner](../scripts/p1_benchmark.py) compares each asynchronous configuration with the synchronous baseline at three workload sizes.

| Jobs per configuration | What the test covers |
| ---: | --- |
| 1 | Input ends after the first submission. Each asynchronous configuration must collect the single pending result. |
| 33 | Each asynchronous configuration reuses request slots. After the last submission, it must collect all remaining results. |
| 2,000 | Completion counts and Top-10 results are checked at the workload size used for performance measurement. |

For each workload size, `validate()` runs the synchronous configuration first and stores its Top-10 results as the reference. It then runs the configurations with 2, 4, and 8 asynchronous requests.

1. `run()` launches the C executable and saves its standard output and error logs. It checks the exit code, the settings reported by the compiled model, and the expected, submitted, and completed job counts.
2. `read_top10()` reads the CSV produced by the executable. It checks that each job has exactly ten results with unique ranks, that the image matches the job ID, and that scores are finite. It indexes the results by `(job_id, rank)` so the comparison works even when the output rows are recorded in a different order.
3. `validate()` compares each asynchronous result with the matching reference entry. The class ID must match, and the score difference must satisfy `abs(async_score - reference_score) <= 1e-6 + 1e-5 * abs(reference_score)`.
4. The runner records the number of entries checked, the number of exact score matches, and the maximum absolute score difference in `validation.csv`. If a check fails, the runner stops and retains the logs.

### Run the correctness checks

[Build the benchmark](#build-the-benchmark), then run the following from the repository root in WSL. Connect AC power and select Windows Balanced mode before running. The `--power-mode` option records the operator’s confirmation.

```bash
build=benchmark/v0.7-memory-analysis/build/p1
out="$build/verify-$(date +%Y%m%d-%H%M%S)"

python3 scripts/p1_benchmark.py \
  --binary "$build/hello_classification_c" \
  --power-mode ac-balanced \
  --output "$out" \
  --verify-only &&
cat "$out/validation.csv"
```

### Validation results

All **12 validation runs** passed across workloads of 1, 33, and 2,000 jobs. For the 2,000-job workload, each asynchronous configuration matched the synchronous baseline on all **20,000 Top-10 entries**, including class IDs, ranks, and scores. The maximum absolute score difference was **0.0**.

[Correctness results](../benchmark/results/p1/2026-11-21/validation.csv)

## Performance measurement

The main program loads and compiles the OpenVINO model, then passes it and the benchmark settings to the C benchmark. [Main program](../benchmark/v0.7-memory-analysis/main.c) · [C benchmark](../benchmark/v0.7-memory-analysis/p1_benchmark.c)

### Warmup and measurement

The C benchmark creates the inference request pool and completes **100 warmup jobs**. It then starts measuring elapsed time and process CPU time, with separate counters for measured submissions and completions.

The timed phase covers **2,000 jobs**. For each job, the program reads and decodes a BMP image, creates and attaches the input tensor, and submits the inference request. To collect a result, it waits for the request as needed, reads and sorts the output, and releases the resources associated with that image. After the last submission, it collects all remaining results and stops the measurements.

Model loading and compilation, request pool creation, warmup, and final teardown take place outside the timed window. The benchmark disables per-image diagnostic logging during performance measurement.

### Execution order

The Python runner starts a new benchmark process for each run. Within a four-round group, each mode appears once in each execution position. This reduces the chance that running early or late consistently favors one mode. [Python runner](../scripts/p1_benchmark.py)

| Round within each group | Request counts in execution order |
| ---: | --- |
| 1 | 1, 2, 4, 8 |
| 2 | 2, 8, 1, 4 |
| 3 | 4, 1, 8, 2 |
| 4 | 8, 4, 2, 1 |

The runner repeats the group twice for **eight runs per mode and 32 measured runs in total**. After each run, it checks the request count and job totals reported by the executable and writes the measurements to `runs.csv`.

### Metrics and summary

The C benchmark uses `clock_gettime(CLOCK_MONOTONIC)` to measure elapsed time and `getrusage(RUSAGE_SELF)` to measure process CPU time. After each run, it prints the completed job count and measured times. The Python runner uses these values to calculate the following metrics.

- **Throughput** is the completed job count divided by elapsed time, in images per second.
- **CPU usage** is the sum of user and system CPU time divided by elapsed time, expressed in logical CPU equivalents.
- **Run-to-run variation** is reported as the sample standard deviation and the coefficient of variation (CV) across the eight throughput measurements for each mode. The reported CV is `100 × standard deviation / mean throughput`.
- **Change versus sequential** is the percentage change in mean throughput relative to the sequential baseline.

The runner writes the per-mode summary to `summary.csv` and `summary.md`.

### Run the performance benchmark

[Build the benchmark](#build-the-benchmark), then run the following from the repository root in WSL with AC power connected and Windows in Balanced mode. The runner performs the correctness checks before starting the 32 timed runs.

```bash
build=benchmark/v0.7-memory-analysis/build/p1
out="$build/measure-$(date +%Y%m%d-%H%M%S)"

python3 scripts/p1_benchmark.py \
  --binary "$build/hello_classification_c" \
  --power-mode ac-balanced \
  --output "$out"
```

## Results

Each of the **32 measured runs** completed all **2,000 submitted jobs**.

| Mode | Runs | Mean throughput (img/s) | SD (img/s) | CV | Mean CPU (logical CPU equivalents) | vs. sequential |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential | 8 | 620.487 | 38.286 | 6.17% | 7.998 | Baseline |
| Async 2 | 8 | 678.412 | 47.236 | 6.96% | 8.242 | +9.34% |
| **Async 4** | **8** | **694.311** | **23.151** | **3.33%** | **8.242** | **+11.90%** |
| Async 8 | 8 | 668.188 | 53.389 | 7.99% | 8.239 | +7.69% |

Four asynchronous requests outperformed sequential execution in all eight rounds and had the lowest run-to-run variation. Increasing the request count from four to eight reduced mean throughput from **694.3** to **668.2 img/s**, while mean CPU usage stayed near **8.24 logical CPU equivalents** in both modes. The cause of the throughput decline at eight requests remains undetermined.

[Per-run measurements](../benchmark/results/p1/2026-11-21/runs.csv)

## Conclusion

Four asynchronous requests again delivered the highest mean throughput among the tested configurations, consistent with the earlier v0.6 experiment. In this benchmark, the four-request configuration reached **694.3 images/s**, **11.90% above** the sequential baseline. Its Top-10 outputs matched the synchronous reference exactly in the correctness checks. [Earlier v0.6 experiment](v0.6.md)

The result describes the pipeline on this WSL2 host using SqueezeNet 1.1 and two BMP inputs repeated after warmup. The measurement includes image decoding, inference, result processing, and per-image cleanup.
# CPU Inference Benchmark

## Background and goal

An earlier experiment found that using four asynchronous inference requests improved mean throughput by approximately 11% over sequential execution. I reran the comparison to see whether the gain would hold across repeated runs and whether asynchronous execution would preserve the classification results.

The benchmark compares one synchronous request with pools of 2, 4, and 8 asynchronous requests. It first checks that every submitted job completes and that the Top-10 output matches the synchronous reference. Then measures throughput, run-to-run variation, and CPU usage.

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

The benchmark uses the same executable, model, inputs, and OpenVINO settings for all four modes. The request count controls how many image jobs the application can have in flight. The OpenVINO inference thread setting stays at eight.

## Correctness validation

Before measuring performance, the [benchmark runner](../scripts/p1_benchmark.py) compares each asynchronous configuration against the synchronous baseline at three workload sizes.

The runner validates each workload size by following these steps:

1. `validate()` selects a workload of 1, 33, or 2,000 jobs and starts with the synchronous configuration.
2. `run()` launches the C inference executable and saves its standard output and error logs. It checks the exit code, compiled model settings, and expected, submitted, and completed job counts.
3. The executable writes the Top-10 results to a CSV file. `read_top10()` reads the file, checks for invalid or duplicate entries, and stores each result using `(job_id, rank)` as its key.
4. `validate()` keeps the synchronous results as the reference, then runs the asynchronous configurations with 2, 4, and 8 requests.
5. For each asynchronous result, it uses `(job_id, rank)` to find the corresponding reference result. It checks that the class ID matches and the score is within the specified tolerance.
6. The runner writes the number of entries checked, the number of exact score matches, and the maximum absolute score difference to `validation.csv`. If a check fails, it raises an error and stops.

| Jobs per configuration | What the test covers |
| ---: | --- |
| 1 | Input ends after the first submission, leaving one result for the final drain in every asynchronous mode. |
| 33 | Request slots are reused in every asynchronous mode. The final submission starts another pass through the pool, and the runner must collect all remaining results. |
| 2,000 | Completion counts and Top-10 outputs are checked at the same workload size used for performance measurement. |

The runner checks the number of submissions and completions, then matches the output by job ID. Each job must have ten uniquely ranked results. Class IDs must match at each rank. Scores must be finite and satisfy `abs(async_score - reference_score) <= 1e-6 + 1e-5 * abs(reference_score)`.

All 12 validation runs passed across workloads of 1, 33, and 2,000 jobs. For the 2,000-job case, each asynchronous configuration matched the synchronous baseline on all **20,000 Top-10 entries**, including class IDs, ranks, and scores. The maximum absolute score difference was **0.0**.

[Correctness results](../benchmark/results/p1/2026-11-21/validation.csv)

## Performance measurement

The [main program](../benchmark/v0.7-memory-analysis/main.c) loads and compiles the OpenVINO model, then passes the compiled model and benchmark settings to the [C benchmark](../benchmark/v0.7-memory-analysis/p1_benchmark.c).

### Warmup and measurement

The C benchmark creates the inference request pool and processes **100 warmup jobs**. After the warmup, it resets the submission and completion counters and starts measuring elapsed time and process CPU time.

During the timed phase, the program processes **2,000 jobs**. For each job, it reads a BMP image, creates an input tensor, sets the tensor on an inference request, and starts inference. When a request is ready to be collected, the program waits for it, reads and sorts the output, and releases the image-specific resources. After the last submission, it collects any remaining results and stops the measurements.

The timed phase includes image reading and BMP decoding, tensor setup, inference, waiting, score sorting, and per-image cleanup. Model loading and compilation, request pool creation, warmup, and final teardown occur outside the timed phase. Per-image diagnostic logging is disabled during performance measurement.

### Execution order

The [Python runner](../scripts/p1_benchmark.py) launches a new benchmark process for each run. It changes the execution order of the four modes so that each mode appears once in every position within a four-round group.

| Round within each group | Request counts in execution order |
| ---: | --- |
| 1 | 1, 2, 4, 8 |
| 2 | 2, 8, 1, 4 |
| 3 | 4, 1, 8, 2 |
| 4 | 8, 4, 2, 1 |

The runner repeats this group twice for **eight runs per mode and 32 measured runs in total**. For each run, it reads the benchmark output, checks the request count and completion totals, and saves the measurements to `runs.csv`.

### Metrics and summary

The C benchmark measures elapsed time with `clock_gettime(CLOCK_MONOTONIC)` and process user and system CPU time with `getrusage(RUSAGE_SELF)`. It prints the completed job count and the measured times after each run.

The Python runner uses those values to calculate the following metrics:

- **Throughput** — Completed jobs divided by elapsed time, in images per second.
- **CPU usage** — Process user plus system CPU time divided by elapsed time, expressed in logical CPU equivalents.
- **Run-to-run variation** — Sample standard deviation and coefficient of variation (CV) across the eight throughput measurements for each mode. CV is the standard deviation divided by mean throughput.
- **Change versus sequential** — The percentage difference between each mode’s mean throughput and the sequential mean.

The runner saves the per-mode summary to `summary.csv` and `summary.md`.

## Results

All 32 measured runs completed all 2,000 submitted jobs.

| Mode | Runs | Mean throughput (img/s) | SD (img/s) | CV | Mean CPU (logical CPU equivalents) | vs. sequential |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential | 8 | 620.487 | 38.286 | 6.17% | 7.998 | Baseline |
| Async 2 | 8 | 678.412 | 47.236 | 6.96% | 8.242 | +9.34% |
| **Async 4** | **8** | **694.311** | **23.151** | **3.33%** | **8.242** | **+11.90%** |
| Async 8 | 8 | 668.188 | 53.389 | 7.99% | 8.239 | +7.69% |

Four asynchronous requests outperformed sequential execution in all eight rounds and had the lowest variation across runs. Increasing the request count from four to eight reduced mean throughput from **694.3** to **668.2 img/s**, while mean CPU usage stayed near **8.24 logical CPU equivalents** in both modes.

[Per-run measurements](../benchmark/results/p1/2026-11-21/runs.csv)

## Conclusion

The repeat benchmark confirmed the [earlier throughput gain](v0.6.md). On this WSL2 CPU workload, four asynchronous requests delivered the highest mean throughput at **694.3 images/s**, **11.90% above** the sequential baseline, with identical Top-10 outputs in the correctness check. Eight requests used roughly the same CPU time per second and processed fewer images per second than four requests. The measurements show where throughput peaked among the tested settings. They leave the cause of the decline at eight requests open.

The result applies to the measured pipeline on this host, using SqueezeNet 1.1 and two BMP inputs repeated after warmup. It covers image decoding through result processing and cleanup.

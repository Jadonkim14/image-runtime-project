# Image Classification Runtime Optimization

A Linux systems project that profiles and optimizes an OpenVINO image classification pipeline. Based on an OpenVINO C sample, the project adds benchmarking and stage-level timing, evaluates CPU thread settings, and implements asynchronous CPU inference for multiple images.

## 1. Open-Source Contribution

I fixed an allocation/deallocation mismatch in OpenVINO’s BMP image wrapper used. The wrapper released a buffer allocated with `malloc()` using `delete`. I changed the deallocation to `std::free()`, and the patch was merged upstream as OpenVINO PR #38735.

- [Merged OpenVINO PR #38735](https://github.com/openvinotoolkit/openvino/pull/38735)
- [Contribution notes and verification results](https://github.com/Jadonkim14/open-source-contributions/tree/main/openvino/pr-38735)

**Tools and techniques**

- **C/C++ memory ownership** — Traced the BMP buffer from allocation in the C reader through `image_read()` to release in the C++ wrapper.
- **GCC and G++ on Linux (WSL2)** — Compiled the BMP reader and wrapper into separate object files and linked them into a minimal test executable.
- **Python and binary test fixtures** — Used `struct` to generate a 4×4, 24-bit uncompressed BMP for the image load-and-free test.
- **AddressSanitizer (ASan)** — Built the test with `-fsanitize=address` and ran it before and after the patch. 
- **Git and clang-format** — Fixed a CI include-order failure and verified it with `git diff --check` and `clang-format-18`.

## 2. CPU Inference Benchmark

Compared sequential inference with 2, 4, and 8 asynchronous requests on the CPU. Validated that all requests completed and their Top-10 results matched the sequential baseline.

Four asynchronous requests achieved the highest mean throughput, **11.90% higher** than sequential execution.

- [Benchmark notes and reproduction guide](docs/p1-benchmark.md)
- [Per-run measurements](benchmark/results/p1/2026-11-21/runs.csv)
- [Correctness checks](benchmark/results/p1/2026-11-21/validation.csv)

**Tools and techniques**

- **C and the OpenVINO C API** — Ran SqueezeNet 1.1 (ONNX) using synchronous and asynchronous inference requests.
- **Linux timing and CPU accounting** — Measured elapsed time with `clock_gettime(CLOCK_MONOTONIC)` and process CPU time with `getrusage(RUSAGE_SELF)`.
- **Python and CSV** — Automated 32 timed runs, summarized throughput and variability, and recorded completion counts and Top-10 validation results.
- **CMake and Ubuntu on WSL2** — Built and ran the benchmark in Release mode with OpenCV disabled.

## Notes

| Version | Work completed | Details |
| --- | --- | --- |
| v0.1 | Ran and analyzed the OpenVINO sample. | [Notes](docs/v0.1.md) |
| v0.2 | Established baseline performance measurements. | [Notes](docs/v0.2.md) |
| v0.3 | Added stage-level timing to the pipeline. | [Notes](docs/v0.3.md) |
| v0.4 | Profiled CPU execution and thread scaling. | [Notes](docs/v0.4.md) |
| v0.5 | Tested a CPU thread configuration hypothesis. | [Notes](docs/v0.5.md) |
| v0.6 | Implemented multi-image asynchronous inference. | [Notes](docs/v0.6.md) |
| v0.7 | Analyzed memory allocation and data movement. | [Notes](docs/v0.7.md) |

# Image Classification Runtime Optimization

A Linux systems project that profiles and optimizes an OpenVINO image classification pipeline. Based on an OpenVINO C sample, the project adds benchmarking and stage-level timing, evaluates CPU thread settings, and implements asynchronous CPU inference for multiple images.

## Open-Source Contribution

I fixed an allocation/deallocation mismatch in OpenVINO’s BMP image wrapper used when OpenCV is disabled. The wrapper released a buffer allocated with `malloc()` using `delete`. I changed the deallocation to `std::free()`, and the patch was merged upstream as OpenVINO PR #38735.

In a post-merge AddressSanitizer (ASan) test with a 4×4 BMP, the parent commit reported `alloc-dealloc-mismatch`. The merged commit completed the same test with exit code `0` and no ASan diagnostics.

- [Merged OpenVINO PR #38735](https://github.com/openvinotoolkit/openvino/pull/38735)
- [Contribution notes and verification results](https://github.com/Jadonkim14/open-source-contributions/tree/main/openvino/pr-38735)

## Performance Engineering

| Version | Work completed | Details |
| --- | --- | --- |
| v0.1 | Ran and analyzed the OpenVINO sample. | [Notes](docs/v0.1.md) |
| v0.2 | Established baseline performance measurements. | [Notes](docs/v0.2.md) |
| v0.3 | Added stage-level timing to the pipeline. | [Notes](docs/v0.3.md) |
| v0.4 | Profiled CPU execution and thread scaling. | [Notes](docs/v0.4.md) |
| v0.5 | Tested a CPU thread configuration hypothesis. | [Notes](docs/v0.5.md) |
| v0.6 | Implemented multi-image asynchronous inference. | [Notes](docs/v0.6.md) |
| v0.7 | Analyzed memory allocation and data movement. | [Notes](docs/v0.7.md) |

## Tech Stack

- **Languages** — C and C++
- **Environment** — Ubuntu on WSL2
- **Inference** — OpenVINO with SqueezeNet 1.1 (ONNX)
- **Build and measurement** — CMake, `perf`, `/usr/bin/time`, `clock_gettime()`
- **Memory diagnostics** — AddressSanitizer

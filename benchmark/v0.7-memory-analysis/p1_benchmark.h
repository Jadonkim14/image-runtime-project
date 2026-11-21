// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <openvino/c/openvino.h>

enum { P1_IMAGE_COUNT = 2 };

int p1_benchmark(ov_compiled_model_t* model, const char* directory, size_t request_count,
                 size_t warmup_repeats, size_t measured_jobs, const char* output_path);
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <time.h>

#include "infer_result_util.h"

double g_result_malloc_time_ms = 0.0;
double g_result_copy_time_ms = 0.0;
double g_result_sort_time_ms = 0.0;

size_t g_result_malloc_count = 0;
size_t g_result_copy_count = 0;
size_t g_result_sort_count = 0;

int compare(const void* a, const void* b) {
    const struct infer_result* sa = (const struct infer_result*)a;
    const struct infer_result* sb = (const struct infer_result*)b;
    if (sa->probability < sb->probability) {
        return 1;
    } else if ((sa->probability == sb->probability) && (sa->class_id > sb->class_id)) {
        return 1;
    } else if (sa->probability > sb->probability) {
        return -1;
    }
    return 0;
}

void infer_result_sort(struct infer_result* results, size_t result_size) {
    struct timespec start;
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    qsort(
        results,
        result_size,
        sizeof(struct infer_result),
        compare
    );

    clock_gettime(CLOCK_MONOTONIC, &end);

    g_result_sort_time_ms +=
        (end.tv_sec - start.tv_sec) * 1000.0 +
        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    g_result_sort_count++;
}

struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size) {
    ov_shape_t output_shape = {0};
    ov_status_e status = ov_tensor_get_shape(tensor, &output_shape);
    if (status != OK)
        return NULL;

    *result_size = output_shape.dims[1];

    struct timespec start;
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    struct infer_result* results =
        (struct infer_result*)malloc(
            sizeof(struct infer_result) * (*result_size)
        );

    clock_gettime(CLOCK_MONOTONIC, &end);

    g_result_malloc_time_ms +=
        (end.tv_sec - start.tv_sec) * 1000.0 +
        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    g_result_malloc_count++;

    if (!results) {
        ov_shape_free(&output_shape);
        return NULL;
    }

    void* data = NULL;
    status = ov_tensor_data(tensor, &data);
    if (status != OK) {
        ov_shape_free(&output_shape);
        free(results);
        return NULL;
    }
    float* float_data = (float*)(data);

    clock_gettime(CLOCK_MONOTONIC, &start);

    size_t i;
    for (i = 0; i < *result_size; ++i) {
        results[i].class_id = i;
        results[i].probability = float_data[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    g_result_copy_time_ms +=
        (end.tv_sec - start.tv_sec) * 1000.0 +
        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    g_result_copy_count++;

    ov_shape_free(&output_shape);
    return results;
}

void print_infer_result(struct infer_result* results, size_t result_size, const char* img_path) {
    printf("\nImage %s\n", img_path);
    printf("\nclassid probability\n");
    printf("------- -----------\n");
    size_t i;
    for (i = 0; i < result_size; ++i) {
        printf("%zu       %f\n", results[i].class_id, results[i].probability);
    }
}

double get_result_malloc_time_ms(void) {
    return g_result_malloc_time_ms;
}

double get_result_copy_time_ms(void) {
    return g_result_copy_time_ms;
}

double get_result_sort_time_ms(void) {
    return g_result_sort_time_ms;
}

size_t get_result_malloc_count(void) {
    return g_result_malloc_count;
}

size_t get_result_copy_count(void) {
    return g_result_copy_count;
}

size_t get_result_sort_count(void) {
    return g_result_sort_count;
}
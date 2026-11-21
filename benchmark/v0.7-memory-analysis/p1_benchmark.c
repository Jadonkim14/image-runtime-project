// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "infer_result_util.h"
#include "opencv_c_wrapper.h"
#include "p1_benchmark.h"

enum { MAX_REQUESTS = 8, TOP_COUNT = 10 };
static const char* const image_names[P1_IMAGE_COUNT] = {"banana.bmp", "dog.bmp"};
enum { IMAGE_COUNT = sizeof(image_names) / sizeof(image_names[0]) };

struct request_slot {
    ov_infer_request_t* request;
    ov_tensor_t* input;
    ov_shape_t shape;
    c_mat_t image;
    size_t job_id;
    bool pending;
};

struct counts {
    size_t submitted;
    size_t completed;
};

static bool check_status(ov_status_e status, const char* operation) {
    if (status != OK)
        fprintf(stderr, "[P1] %s failed (%d)\n", operation, status);
    return status == OK;
}

static void release_input(struct request_slot* slot) {
    if (slot->input)
        ov_tensor_free(slot->input);
    slot->input = NULL;
    ov_shape_free(&slot->shape);
    slot->shape = (ov_shape_t){0};
    image_free(&slot->image);
    slot->image = (c_mat_t){0};
}

static bool finish(struct request_slot* slot, FILE* results_file) {
    if (slot->pending) {
        if (!check_status(ov_infer_request_wait(slot->request), "wait"))
            return false;
        slot->pending = false;
    }

    ov_tensor_t* output = NULL;
    if (!check_status(ov_infer_request_get_output_tensor_by_index(slot->request, 0, &output), "get output"))
        return false;

    size_t size = 0;
    struct infer_result* results = tensor_to_infer_result(output, &size);
    ov_tensor_free(output);
    if (!results || size < TOP_COUNT) {
        fprintf(stderr, "[P1] invalid output for job %zu\n", slot->job_id);
        free(results);
        return false;
    }

    infer_result_sort(results, size);
    bool written = true;
    if (results_file) {
        const char* image_name = image_names[slot->job_id % IMAGE_COUNT];
        for (size_t rank = 0; rank < TOP_COUNT; ++rank) {
            if (fprintf(results_file, "%zu,%s,%zu,%zu,%.9g\n", slot->job_id, image_name,
                        rank + 1, results[rank].class_id, results[rank].probability) < 0) {
                written = false;
                break;
            }
        }
    }

    free(results);
    release_input(slot);
    return written;
}

static bool process_images(struct request_slot* slots, size_t request_count, size_t jobs,
                           char* const paths[], FILE* results_file, struct counts* counts) {
    for (size_t job = 0; job < jobs; ++job) {
        struct request_slot* slot = &slots[job % request_count];
        if (slot->pending) {
            if (!finish(slot, results_file))
                return false;
            ++counts->completed;
        }

        const char* path = paths[job % IMAGE_COUNT];
        if (image_read(path, &slot->image) != 0 || !slot->image.mat_data) {
            fprintf(stderr, "[P1] failed to read %s\n", path);
            return false;
        }

        const int64_t dims[] = {1, slot->image.mat_height, slot->image.mat_width, 3};
        if (!check_status(ov_shape_create(4, dims, &slot->shape), "create shape") ||
            !check_status(ov_tensor_create_from_host_ptr(U8, slot->shape, slot->image.mat_data,
                                                        &slot->input), "create tensor") ||
            !check_status(ov_infer_request_set_input_tensor_by_index(slot->request, 0, slot->input),
                          "set input"))
            return false;

        slot->job_id = job;
        if (request_count == 1) {
            if (!check_status(ov_infer_request_infer(slot->request), "infer"))
                return false;
            ++counts->submitted;
            if (!finish(slot, results_file))
                return false;
            ++counts->completed;
        } else {
            if (!check_status(ov_infer_request_start_async(slot->request), "start async"))
                return false;
            slot->pending = true;
            ++counts->submitted;
        }
    }

    for (size_t i = 0; i < request_count; ++i) {
        if (slots[i].pending) {
            if (!finish(&slots[i], results_file))
                return false;
            ++counts->completed;
        }
    }

    return counts->submitted == jobs && counts->completed == jobs;
}

static double cpu_seconds(const struct rusage* usage) {
    return (double)usage->ru_utime.tv_sec + usage->ru_stime.tv_sec +
           ((double)usage->ru_utime.tv_usec + usage->ru_stime.tv_usec) / 1000000.0;
}

int p1_benchmark(ov_compiled_model_t* model, const char* directory, size_t request_count,
                 size_t warmup_repeats, size_t measured_jobs, const char* output_path) {
    struct request_slot slots[MAX_REQUESTS] = {0};
    char* paths[IMAGE_COUNT] = {0};
    FILE* output = NULL;
    int result = EXIT_FAILURE;
    struct counts warmup = {0};
    struct counts measured = {0};
    const size_t warmup_jobs = warmup_repeats * IMAGE_COUNT;

    for (size_t i = 0; i < IMAGE_COUNT; ++i) {
        const size_t length = strlen(directory) + strlen(image_names[i]) + 2;
        paths[i] = malloc(length);
        if (!paths[i])
            goto cleanup;
        snprintf(paths[i], length, "%s/%s", directory, image_names[i]);
    }

    for (size_t i = 0; i < request_count; ++i) {
        if (!check_status(ov_compiled_model_create_infer_request(model, &slots[i].request), "create request"))
            goto cleanup;
    }

    if (output_path) {
        output = fopen(output_path, "w");
        if (!output || fputs("job_id,image,rank,class_id,score\n", output) == EOF) {
            perror(output_path);
            goto cleanup;
        }
    }

    if (!process_images(slots, request_count, warmup_jobs, paths, NULL, &warmup))
        goto cleanup;

    struct timespec start, end;
    struct rusage usage_before, usage_after;
    if (getrusage(RUSAGE_SELF, &usage_before) || clock_gettime(CLOCK_MONOTONIC, &start)) {
        perror("start measurement");
        goto cleanup;
    }

    if (!process_images(slots, request_count, measured_jobs, paths, output, &measured))
        goto cleanup;

    if (clock_gettime(CLOCK_MONOTONIC, &end) || getrusage(RUSAGE_SELF, &usage_after)) {
        perror("end measurement");
        goto cleanup;
    }

    const double elapsed = (double)(end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    const double cpu = cpu_seconds(&usage_after) - cpu_seconds(&usage_before);
    if (elapsed <= 0)
        goto cleanup;

    if (output) {
        const int close_status = fclose(output);
        output = NULL;
        if (close_status != 0) {
            perror(output_path);
            goto cleanup;
        }
    }

    printf("P1_RESULT,%zu,%zu,%zu,%zu,%zu,%.9f,%.9f\n",
           request_count, warmup.completed, measured_jobs, measured.submitted, measured.completed, elapsed, cpu);
    result = EXIT_SUCCESS;

cleanup:
    if (output)
        fclose(output);
    for (size_t i = 0; i < request_count; ++i) {
        if (slots[i].pending)
            ov_infer_request_wait(slots[i].request);
        if (slots[i].request)
            ov_infer_request_free(slots[i].request);
        release_input(&slots[i]);
    }
    for (size_t i = 0; i < IMAGE_COUNT; ++i)
        free(paths[i]);
    if (result != EXIT_SUCCESS)
        fprintf(stderr, "[P1] failed: warmup %zu/%zu, submitted %zu/%zu, completed %zu/%zu\n",
                warmup.completed, warmup_jobs, measured.submitted, measured_jobs, measured.completed, measured_jobs);
    return result;
}
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>

#include "samples_util/path_util.h"
// Uses windows.h must be before openvino/c/openvino.h
#include "infer_result_util.h"
#include "opencv_c_wrapper.h"
#include "openvino/c/openvino.h"
#include "bmp_reader.h"

void print_model_input_output_info(ov_model_t* model) {
    char* friendly_name = NULL;
    ov_model_get_friendly_name(model, &friendly_name);
    printf("[INFO] model name: %s \n", friendly_name);
    ov_free(friendly_name);
}

int compare_latency(const void *a, const void *b)
{
    double value_a = *(const double *)a;
    double value_b = *(const double *)b;

    if (value_a < value_b)
        return -1;

    if (value_a > value_b)
        return 1;

    return 0;
}

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char** argv) {
    // -------- Check input parameters --------
    if (argc != 6) {
        printf("Usage : ./hello_classification_c <path_to_model> <path_to_image_directory> "
               "<device_name> <inference_threads> <repeat_count>\n");
        return EXIT_FAILURE;
    }

    #define NUM_REQUESTS 4

    ov_infer_request_t* infer_requests[NUM_REQUESTS] = {NULL};
    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_tensor_t* representative_tensor = NULL;
    ov_tensor_t* tensors[NUM_REQUESTS] = {NULL};
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_model_t* new_model = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_preprocess_input_model_info_t* p_input_model = NULL;
    ov_preprocess_output_info_t* output_info = NULL;
    ov_preprocess_output_tensor_info_t* output_tensor_info = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_tensor_t* output_tensor = NULL;
    struct infer_result* results = NULL;
    ov_layout_t* input_layout = NULL;
    ov_layout_t* model_layout = NULL;
    ov_shape_t representative_input_shape = {.rank = 0, .dims = NULL};
    ov_shape_t input_shapes[NUM_REQUESTS] = {0};
    ov_output_const_port_t* output_port = NULL;
    ov_output_const_port_t* input_port = NULL;
    c_mat_t images[NUM_REQUESTS] = {0};
    c_mat_t representative_img = {0};
    struct timespec start;
    struct timespec end;
    DIR *dir = NULL;
    struct dirent *entry = NULL;

    // -------- Get OpenVINO runtime version --------
    ov_version_t version;
    CHECK_STATUS(ov_get_openvino_version(&version));
    printf("---- OpenVINO INFO----\n");
    printf("Description : %s \n", version.description);
    printf("Build number: %s \n", version.buildNumber);
    ov_version_free(&version);

    // -------- Parsing and validation of input arguments --------
    const char* input_model = argv[1];
    char input_directory_path[PATH_MAX];
    char representative_image_path[PATH_MAX];

    if (!sanitize_path(argv[2], input_directory_path, sizeof(input_directory_path))) {
        goto err;
    }

    snprintf(
        representative_image_path,
        sizeof(representative_image_path),
        "%s/%s",
        input_directory_path,
        "banana.bmp"
    );

    dir = opendir(input_directory_path);
    if (dir == NULL) {
        fprintf(stderr, "[ERROR] Failed to open image directory: %s\n",
                input_directory_path);
        goto err;
    }

    const char* device_name = argv[3];

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    CHECK_STATUS(ov_core_create(&core));

    // Set CPU inference threads
    CHECK_STATUS(
        ov_core_set_property(
            core,
            device_name,
            "INFERENCE_NUM_THREADS",
            argv[4]
        )
    );

    int repeat_count = atoi(argv[5]);
    if (repeat_count <= 0) {
        fprintf(stderr, "repeat_count must be greater than 0\n");
        return EXIT_FAILURE;
    }

    // -------- Step 2. Read a model --------
    printf("[INFO] Loading model files: %s\n", input_model);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_model_input_output_info(model);

    double model_load_time = 
    	(end.tv_sec - start.tv_sec) * 1000.0
    	+ (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("\nModel Load: %.3f ms", model_load_time);

    CHECK_STATUS(ov_model_const_output(model, &output_port));
    if (!output_port) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 output only %d\n", __LINE__);
        goto err;
    }

    CHECK_STATUS(ov_model_const_input(model, &input_port));
    if (!input_port) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 input only %d\n", __LINE__);
        goto err;
    }

    // -------- Step 3. Set up input
    clock_gettime(CLOCK_MONOTONIC, &start);
    image_read(representative_image_path, &representative_img);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double image_load_time = 
    	(end.tv_sec - start.tv_sec) * 1000.0
    	+ (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("\nImage Load: %.3f ms", image_load_time);

    ov_element_type_e input_type = U8;
    int64_t representative_dims[4] = {
        1,
        (size_t)representative_img.mat_height,
        (size_t)representative_img.mat_width,
        3
    };

    clock_gettime(CLOCK_MONOTONIC, &start);
    ov_shape_create(4, representative_dims, &representative_input_shape);
    CHECK_STATUS(
        ov_tensor_create_from_host_ptr(
            input_type,
            representative_input_shape,
            representative_img.mat_data,
            &representative_tensor
        )
    );
    clock_gettime(CLOCK_MONOTONIC, &end);

    double tensor_prep_time = 
    	(end.tv_sec - start.tv_sec) * 1000.0
    	+ (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("\nTensor Preparation: %.3f ms", tensor_prep_time);

    // -------- Step 4. Configure preprocessing --------
    clock_gettime(CLOCK_MONOTONIC, &start);
    CHECK_STATUS(ov_preprocess_prepostprocessor_create(model, &preprocess));
    CHECK_STATUS(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));

    CHECK_STATUS(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_from(input_tensor_info, representative_tensor));

    const char* input_layout_desc = "NHWC";
    CHECK_STATUS(ov_layout_create(input_layout_desc, &input_layout));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, input_layout));

    CHECK_STATUS(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    CHECK_STATUS(ov_preprocess_preprocess_steps_resize(input_process, RESIZE_LINEAR));
    CHECK_STATUS(ov_preprocess_input_info_get_model_info(input_info, &p_input_model));

    const char* model_layout_desc = "NCHW";
    CHECK_STATUS(ov_layout_create(model_layout_desc, &model_layout));
    CHECK_STATUS(ov_preprocess_input_model_info_set_layout(p_input_model, model_layout));

    CHECK_STATUS(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    CHECK_STATUS(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    CHECK_STATUS(ov_preprocess_output_set_element_type(output_tensor_info, F32));

    CHECK_STATUS(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    clock_gettime(CLOCK_MONOTONIC, &end);

    double preproc_config_time =
            (end.tv_sec - start.tv_sec) * 1000.0
            + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("\nPreprocessing Configuration: %.3f ms\n", preproc_config_time);


    // -------- Configure dynamic input shape --------
    ov_dimension_t dynamic_dims[4] = {
        {1, 1},     // N
        {-1, -1},   // H
        {-1, -1},   // W
        {3, 3}      // C
    };

    ov_partial_shape_t dynamic_input_shape = {0};

    CHECK_STATUS(
        ov_partial_shape_create(
            4,
            dynamic_dims,
            &dynamic_input_shape
        )
    );

    CHECK_STATUS(
        ov_model_reshape_single_input(
            new_model,
            dynamic_input_shape
        )
    );


    // -------- Step 5. Loading a model to the device --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CHECK_STATUS(
        ov_core_compile_model(
            core,
            new_model,
            device_name,
            0,
            &compiled_model
        )
    );

    clock_gettime(CLOCK_MONOTONIC, &end);

    double model_compile_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Model Compile: %.3f ms\n", model_compile_time);

    // Check the actual number of inference threads
    char* inference_threads = NULL;

    CHECK_STATUS(
        ov_compiled_model_get_property(
            compiled_model,
            "INFERENCE_NUM_THREADS",
            &inference_threads
        )
    );

    printf("INFERENCE_NUM_THREADS: %s\n", inference_threads);

    ov_free(inference_threads);

    // -------- Check compiled model input shape --------
    ov_output_const_port_t* compiled_input_port = NULL;
    ov_partial_shape_t compiled_input_partial_shape = {0};

    CHECK_STATUS(
        ov_compiled_model_input_by_index(
            compiled_model,
            0,
            &compiled_input_port
        )
    );

    CHECK_STATUS(
        ov_port_get_partial_shape(
            compiled_input_port,
            &compiled_input_partial_shape
        )
    );

    char* compiled_shape_str =
        ov_partial_shape_to_string(compiled_input_partial_shape);

    printf(
        "Compiled Model Input Shape: %s\n",
        compiled_shape_str
    );

    ov_free(compiled_shape_str);


    // -------- Step 6. Create an infer request --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < NUM_REQUESTS; i++) {
        CHECK_STATUS(
            ov_compiled_model_create_infer_request(
                compiled_model,
                &infer_requests[i]
            )
        );
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double infer_request_creation_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Infer Request Creation: %.3f ms\n", infer_request_creation_time);

    rewinddir(dir);

    size_t total_images = 0;
    double total_async_submit_time = 0.0;
    double total_result_processing_time = 0.0;
    double total_input_tensor_set_time = 0.0;
    double total_image_load_time = 0.0;
    double total_tensor_preparation_time = 0.0;
    double total_wait_time = 0.0;

    int slot_busy[NUM_REQUESTS] = {0};
    size_t submitted_images = 0;

    struct timespec sequential_start;
    struct timespec sequential_end;

    clock_gettime(CLOCK_MONOTONIC, &sequential_start);

    for (int repeat = 0; repeat < repeat_count; repeat++) {

        rewinddir(dir);

        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            const char *ext = strrchr(entry->d_name, '.');

            if (ext == NULL || strcmp(ext, ".bmp") != 0) {
                continue;
            }

            int slot = submitted_images % NUM_REQUESTS;

            if (slot_busy[slot]) {
                ov_infer_request_t* request_to_wait = infer_requests[slot];

                clock_gettime(CLOCK_MONOTONIC, &start);
                CHECK_STATUS(ov_infer_request_wait(request_to_wait));
                clock_gettime(CLOCK_MONOTONIC, &end);

                double wait_time =
                    (end.tv_sec - start.tv_sec) * 1000.0
                    + (end.tv_nsec - start.tv_nsec) / 1000000.0;

                total_wait_time += wait_time;

                //  -------- Process output --------
                clock_gettime(CLOCK_MONOTONIC, &start);

                CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(
                    request_to_wait, 0, &output_tensor));

                size_t results_num;
                results = tensor_to_infer_result(output_tensor, &results_num);

                infer_result_sort(results, results_num);

                clock_gettime(CLOCK_MONOTONIC, &end);

                double result_processing_time =
                    (end.tv_sec - start.tv_sec) * 1000.0
                    + (end.tv_nsec - start.tv_nsec) / 1000000.0;
                
                total_result_processing_time += result_processing_time;

                // printf("Result Processing: %.3f ms\n", result_processing_time);


                // -------- Print classification results --------
                size_t top = 10;

                if (top > results_num) {
                    top = results_num;
                }

                // printf("\nTop %zu results:\n", top);
                // print_infer_result(results, top, input_image_path);  
                total_images++;

                free(results);
                results = NULL;

                if (output_tensor) {
                    ov_tensor_free(output_tensor);
                    output_tensor = NULL;
                }

                if (tensors[slot]) {
                    ov_tensor_free(tensors[slot]);
                    tensors[slot] = NULL;
                }

                ov_shape_free(&input_shapes[slot]);
                input_shapes[slot].rank = 0;
                input_shapes[slot].dims = NULL;

                image_free(&images[slot]);
                images[slot] = (c_mat_t){0};

                slot_busy[slot] = 0;
            }

            ov_infer_request_t* current_request = infer_requests[slot]; 

            char input_image_path[PATH_MAX];

            snprintf(
                input_image_path,
                sizeof(input_image_path),
                "%s/%s",
                input_directory_path,
                entry->d_name
            );

            // printf("Processing: %s\n", input_image_path);

            clock_gettime(CLOCK_MONOTONIC, &start);
            image_read(input_image_path, &images[slot]);
            clock_gettime(CLOCK_MONOTONIC, &end);

            double image_load_time =
                (end.tv_sec - start.tv_sec) * 1000.0
                + (end.tv_nsec - start.tv_nsec) / 1000000.0;

            total_image_load_time += image_load_time;

            clock_gettime(CLOCK_MONOTONIC, &start);

            size_t dims[4] = {
                1,
                (size_t)images[slot].mat_height,
                (size_t)images[slot].mat_width,
                3
            };

            ov_shape_create(4, dims, &input_shapes[slot]);

            CHECK_STATUS(
                ov_tensor_create_from_host_ptr(
                    input_type,
                    input_shapes[slot],
                    images[slot].mat_data,
                    &tensors[slot]
                )
            );

            clock_gettime(CLOCK_MONOTONIC, &end);

            double tensor_preparation_time =
                (end.tv_sec - start.tv_sec) * 1000.0
                + (end.tv_nsec - start.tv_nsec) / 1000000.0;

            total_tensor_preparation_time += tensor_preparation_time;

            // -------- Step 7. Prepare input --------
            clock_gettime(CLOCK_MONOTONIC, &start);

            CHECK_STATUS(
                ov_infer_request_set_input_tensor_by_index(
                    current_request,
                    0,
                    tensors[slot]
                )
            );

            clock_gettime(CLOCK_MONOTONIC, &end);

            double input_tensor_set_time =
                (end.tv_sec - start.tv_sec) * 1000.0
                + (end.tv_nsec - start.tv_nsec) / 1000000.0;
            
            total_input_tensor_set_time += input_tensor_set_time;

            // printf("Input Tensor Set: %.3f ms\n", input_tensor_set_time);

            // -------- Step 8. Do inference synchronously --------
            clock_gettime(CLOCK_MONOTONIC, &start);

            CHECK_STATUS(ov_infer_request_start_async(current_request));

            slot_busy[slot] = 1;
            submitted_images++;

            clock_gettime(CLOCK_MONOTONIC, &end);

            double async_submit_time =
                (end.tv_sec - start.tv_sec) * 1000.0
                + (end.tv_nsec - start.tv_nsec) / 1000000.0;

            total_async_submit_time += async_submit_time;

            // printf("Inference Time: %.3f ms\n", inference_time);

            
        }
    }

    // Drain remaining in-flight requests
    for (int slot = 0; slot < NUM_REQUESTS; slot++) {

        if (!slot_busy[slot]) {
            continue;
        }

        ov_infer_request_t* request_to_wait = infer_requests[slot];

        clock_gettime(CLOCK_MONOTONIC, &start);
        CHECK_STATUS(ov_infer_request_wait(request_to_wait));
        clock_gettime(CLOCK_MONOTONIC, &end);

        double wait_time =
            (end.tv_sec - start.tv_sec) * 1000.0
            + (end.tv_nsec - start.tv_nsec) / 1000000.0;

        total_wait_time += wait_time;

        // Process the completed inference result
        clock_gettime(CLOCK_MONOTONIC, &start);

        CHECK_STATUS(
            ov_infer_request_get_output_tensor_by_index(
                request_to_wait,
                0,
                &output_tensor
            )
        );

        size_t results_num;
        results = tensor_to_infer_result(output_tensor, &results_num);

        infer_result_sort(results, results_num);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double result_processing_time =
            (end.tv_sec - start.tv_sec) * 1000.0
            + (end.tv_nsec - start.tv_nsec) / 1000000.0;

        total_result_processing_time += result_processing_time;

        total_images++;

        free(results);
        results = NULL;

        if (output_tensor) {
            ov_tensor_free(output_tensor);
            output_tensor = NULL;
        }

        if (tensors[slot]) {
            ov_tensor_free(tensors[slot]);
            tensors[slot] = NULL;
        }

        ov_shape_free(&input_shapes[slot]);
        input_shapes[slot].rank = 0;
        input_shapes[slot].dims = NULL;

        image_free(&images[slot]);
        images[slot] = (c_mat_t){0};

        slot_busy[slot] = 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &sequential_end);

    double total_processing_time =
        (sequential_end.tv_sec - sequential_start.tv_sec) * 1000.0
        + (sequential_end.tv_nsec - sequential_start.tv_nsec) / 1000000.0;
    
    double average_latency = 0.0;
    double throughput = 0.0;

    if (total_images > 0) {
        average_latency = total_processing_time / total_images;

        throughput =
            total_images / (total_processing_time / 1000.0);
    }

    printf("\n===== Async %d-Request Benchmark =====\n", NUM_REQUESTS);
    printf("Total Images: %zu\n", total_images);
    printf("Total Processing Time: %.3f ms\n", total_processing_time);
    printf("Average Time per Image: %.3f ms/image\n", average_latency);
    printf("Throughput: %.3f images/sec\n", throughput);
    
    printf("\n===== Stage Breakdown =====\n");
    printf("Image Load: %.3f ms total, %.6f ms/image\n",
        total_image_load_time,
        total_image_load_time / total_images);

    printf("Tensor Preparation: %.3f ms total, %.6f ms/image\n",
        total_tensor_preparation_time,
        total_tensor_preparation_time / total_images);

    printf("Input Tensor Set: %.3f ms total, %.6f ms/image\n",
        total_input_tensor_set_time,
        total_input_tensor_set_time / total_images);

    printf(
        "Async Submit: %.3f ms total, %.6f ms/image\n",
        total_async_submit_time,
        total_async_submit_time / total_images
    );

    printf(
        "Wait: %.3f ms total, %.6f ms/image\n",
        total_wait_time,
        total_wait_time / total_images
    );

    printf("Result Processing: %.3f ms total, %.6f ms/image\n",
        total_result_processing_time,
        total_result_processing_time / total_images);

    double measured_stage_time =
        total_image_load_time +
        total_tensor_preparation_time +
        total_input_tensor_set_time +
        total_async_submit_time +
        total_wait_time +
        total_result_processing_time;

    double other_time =
        total_processing_time - measured_stage_time;

    printf("Other: %.3f ms total, %.6f ms/image\n",
        other_time,
        other_time / total_images);

    printf("\n===== Allocation Breakdown =====\n");

    size_t malloc_count = get_malloc_count();
    double malloc_time_ms = get_total_malloc_time_ms();

    printf("Malloc Count: %zu\n", malloc_count);
    printf("Total Malloc Time: %.6f ms\n", malloc_time_ms);

    if (malloc_count > 0) {
        printf(
            "Average Malloc Time: %.9f ms/call\n",
            malloc_time_ms / malloc_count
        );
    }

    printf("\n===== Result Processing Breakdown =====\n");

    printf(
        "Result Malloc: %.6f ms total, %zu calls\n",
        get_result_malloc_time_ms(),
        get_result_malloc_count()
    );

    printf(
        "Result Copy: %.6f ms total, %zu calls\n",
        get_result_copy_time_ms(),
        get_result_copy_count()
    );

    printf(
        "Result Sort: %.6f ms total, %zu calls\n",
        get_result_sort_time_ms(),
        get_result_sort_count()
    );

    // -------- free allocated resources --------
err:
    // -------- Per-image resources --------
    free(results);

    if (output_tensor)
        ov_tensor_free(output_tensor);

    for (int i = 0; i < NUM_REQUESTS; i++) {
        if (tensors[i]) {
            ov_tensor_free(tensors[i]);
            tensors[i] = NULL;
        }

        ov_shape_free(&input_shapes[i]);
        input_shapes[i].rank = 0;
        input_shapes[i].dims = NULL;

        image_free(&images[i]);
        images[i] = (c_mat_t){0};
    }


    // -------- Representative image resources --------
    if (representative_tensor)
        ov_tensor_free(representative_tensor);

    ov_shape_free(&representative_input_shape);
    image_free(&representative_img);


    // -------- Model input/output ports --------
    ov_output_const_port_free(output_port);
    ov_output_const_port_free(input_port);


    // -------- OpenVINO runtime resources --------
    for (int i = 0; i < NUM_REQUESTS; i++) {
        if (infer_requests[i]) {
            ov_infer_request_free(infer_requests[i]);
        }
    }

    // Debug resource: compiled model input shape
    ov_partial_shape_free(&compiled_input_partial_shape);

    if (compiled_input_port)
        ov_output_const_port_free(compiled_input_port);

    if (compiled_model)
        ov_compiled_model_free(compiled_model);

    if (input_layout)
        ov_layout_free(input_layout);

    if (model_layout)
        ov_layout_free(model_layout);

    if (output_tensor_info)
        ov_preprocess_output_tensor_info_free(output_tensor_info);

    if (output_info)
        ov_preprocess_output_info_free(output_info);

    if (p_input_model)
        ov_preprocess_input_model_info_free(p_input_model);

    if (input_process)
        ov_preprocess_preprocess_steps_free(input_process);

    if (input_tensor_info)
        ov_preprocess_input_tensor_info_free(input_tensor_info);

    if (input_info)
        ov_preprocess_input_info_free(input_info);

    if (preprocess)
        ov_preprocess_prepostprocessor_free(preprocess);

    ov_partial_shape_free(&dynamic_input_shape);

    if (new_model)
        ov_model_free(new_model);

    if (model)
        ov_model_free(model);

    if (core)
        ov_core_free(core);


    // -------- Directory resource --------
    if (dir)
        closedir(dir);

    return EXIT_SUCCESS;
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "samples_util/path_util.h"
// Uses windows.h must be before openvino/c/openvino.h
#include "infer_result_util.h"
#include "opencv_c_wrapper.h"
#include "openvino/c/openvino.h"

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
    if (argc != 5) {
        printf("Usage : ./hello_classification_c <path_to_model> <path_to_image> "
               "<device_name> <inference_threads>\n");
        return EXIT_FAILURE;
    }

    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_tensor_t* tensor = NULL;
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_model_t* new_model = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_preprocess_input_model_info_t* p_input_model = NULL;
    ov_preprocess_output_info_t* output_info = NULL;
    ov_preprocess_output_tensor_info_t* output_tensor_info = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_tensor_t* output_tensor = NULL;
    struct infer_result* results = NULL;
    ov_layout_t* input_layout = NULL;
    ov_layout_t* model_layout = NULL;
    ov_shape_t input_shape = {.rank = 0, .dims = NULL};
    ov_output_const_port_t* output_port = NULL;
    ov_output_const_port_t* input_port = NULL;
    c_mat_t img = {0};
    struct timespec start;
    struct timespec end;

    // -------- Get OpenVINO runtime version --------
    ov_version_t version;
    CHECK_STATUS(ov_get_openvino_version(&version));
    printf("---- OpenVINO INFO----\n");
    printf("Description : %s \n", version.description);
    printf("Build number: %s \n", version.buildNumber);
    ov_version_free(&version);

    // -------- Parsing and validation of input arguments --------
    const char* input_model = argv[1];
    char input_image_path[PATH_MAX];
    if (!sanitize_path(argv[2], input_image_path, sizeof(input_image_path))) {
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
    image_read(input_image_path, &img);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double image_load_time = 
    	(end.tv_sec - start.tv_sec) * 1000.0
    	+ (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("\nImage Load: %.3f ms", image_load_time);

    ov_element_type_e input_type = U8;
    int64_t dims[4] = {1, (size_t)img.mat_height, (size_t)img.mat_width, 3};

    clock_gettime(CLOCK_MONOTONIC, &start);
    ov_shape_create(4, dims, &input_shape);
    CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img.mat_data, &tensor));
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
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

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

    // -------- Step 5. Loading a model to the device --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, 0, &compiled_model));

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


    // -------- Step 6. Create an infer request --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

    clock_gettime(CLOCK_MONOTONIC, &end);

    double infer_request_creation_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Infer Request Creation: %.3f ms\n", infer_request_creation_time);


    // -------- Step 7. Prepare input --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CHECK_STATUS(ov_infer_request_set_input_tensor_by_index(infer_request, 0, tensor));

    clock_gettime(CLOCK_MONOTONIC, &end);

    double input_tensor_set_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Input Tensor Set: %.3f ms\n", input_tensor_set_time);


    // -------- Step 8. Do inference synchronously --------
    #define INFERENCE_COUNT 10000

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < INFERENCE_COUNT; i++) {
        CHECK_STATUS(ov_infer_request_infer(infer_request));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_inference_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    double avg_inference_time =
        total_inference_time / INFERENCE_COUNT;

    double throughput =
        INFERENCE_COUNT / (total_inference_time / 1000.0);

    printf("Inference Count: %d\n", INFERENCE_COUNT);
    printf("Total Inference Time: %.3f ms\n", total_inference_time);
    printf("Average Inference Time: %.3f ms\n", avg_inference_time);
    printf("Inference Throughput: %.2f infer/sec\n", throughput); 


    // -------- Step 9. Process output --------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(
        infer_request, 0, &output_tensor));

    size_t results_num;
    results = tensor_to_infer_result(output_tensor, &results_num);

    infer_result_sort(results, results_num);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double result_processing_time =
        (end.tv_sec - start.tv_sec) * 1000.0
        + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Result Processing: %.3f ms\n", result_processing_time);


    // -------- Print classification results --------
    size_t top = 10;

    if (top > results_num) {
        top = results_num;
    }

    printf("\nTop %zu results:\n", top);
    print_infer_result(results, top, input_image_path);   


    // -------- free allocated resources --------
err:
    free(results);
    image_free(&img);
    ov_shape_free(&input_shape);
    ov_output_const_port_free(output_port);
    ov_output_const_port_free(input_port);
    if (output_tensor)
        ov_tensor_free(output_tensor);
    if (infer_request)
        ov_infer_request_free(infer_request);
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
    if (new_model)
        ov_model_free(new_model);
    if (tensor)
        ov_tensor_free(tensor);
    if (model)
        ov_model_free(model);
    if (core)
        ov_core_free(core);
    return EXIT_SUCCESS;
}

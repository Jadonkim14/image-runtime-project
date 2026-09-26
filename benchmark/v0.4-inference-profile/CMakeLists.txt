# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 4.2.3)
project(v02b_benchmark)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(
	/usr/share/openvino/samples/c/common/opencv_c_wrapper
	opencv_c_wrapper_build
)

add_subdirectory(
	/usr/share/openvino/samples/c/common/sample_util
	sample_util_build
)

add_executable(
	hello_classification_c
	main.c
	infer_result_util.c
)

target_include_directories(
    hello_classification_c
    PRIVATE
    /usr/share/openvino/samples/c/common/sample_util/include/
    /usr/share/openvino/samples/c/common/opencv_c_wrapper/include/
)

target_link_libraries(
    hello_classification_c
    PRIVATE
    openvino::runtime::c
    opencv_c_wrapper
    ie_samples_util_c
)


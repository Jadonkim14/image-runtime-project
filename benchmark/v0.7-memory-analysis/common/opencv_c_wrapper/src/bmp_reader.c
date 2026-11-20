// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bmp_reader.h"

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CLEANUP_AND_RETURN(x)                           \
    if (0 != x && NULL != image && NULL != image->data) \
        free(image->data);                              \
    if (input != NULL)                                  \
        fclose(input);                                  \
    return x;

#define OPENVINO_SAMPLE_BMP_DIM_LIMIT (32 * 1024)

double g_total_malloc_time_ms = 0.0;
size_t g_malloc_count = 0;

int readBmpImage(const char* fileName, BitMap* image) {
    size_t cnt;
    FILE* input = 0;

    if (NULL == fileName || NULL == image) {
        printf("[BMP] bad arguments\n");
        CLEANUP_AND_RETURN(-1);
    }

    memset(image, 0, sizeof(BitMap));

    input = fopen(fileName, "rb");
    if (input == NULL) {
        printf("[BMP] file %s is not opened\n", fileName);
        CLEANUP_AND_RETURN(-1);
    }

    cnt = fread(&image->header.type, sizeof(image->header.type), sizeof(unsigned char), input);
    if (cnt != sizeof(unsigned char)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(-2);
    }

    if (image->header.type != 'M' * 256 + 'B') {
        printf("[BMP] file is not bmp type\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.size, sizeof(image->header.size), sizeof(unsigned char), input);
    if (cnt != sizeof(unsigned char)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.reserved, sizeof(image->header.reserved), sizeof(unsigned char), input);
    if (cnt != sizeof(unsigned char)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.offset, sizeof(image->header.offset), sizeof(unsigned char), input);
    if (cnt != sizeof(unsigned char)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->infoHeader, sizeof(BmpInfoHeader), sizeof(unsigned char), input);
    if (cnt != sizeof(unsigned char)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    // Limit BMP image size
    if ((image->infoHeader.width < 0) || (image->infoHeader.width > OPENVINO_SAMPLE_BMP_DIM_LIMIT)) {
        printf("[BMP] wrong width \n");
        CLEANUP_AND_RETURN(2);
    }

    if ((image->infoHeader.height < -OPENVINO_SAMPLE_BMP_DIM_LIMIT) ||
        (image->infoHeader.height > OPENVINO_SAMPLE_BMP_DIM_LIMIT)) {
        printf("[BMP] wrong height \n");
        CLEANUP_AND_RETURN(2);
    }

    image->width = image->infoHeader.width;
    image->height = abs(image->infoHeader.height);

    if (image->infoHeader.bits != 24) {
        printf("[BMP] 24bpp only supported. But input has: %d\n", image->infoHeader.bits);
        CLEANUP_AND_RETURN(3);
    }

    if (image->infoHeader.compression != 0) {
        printf("[BMP] compression not supported\n");
        CLEANUP_AND_RETURN(4);
    }

    const size_t pad_size = ((size_t)image->width) & 3U;
    char pad[3];
    const int row_size = image->width * 3;
    const int size = sizeof(char) * row_size * image->height;

    if (size <= 0) {
        printf("[BMP] Cannot allocate memory for image (can be too big)\n");
        CLEANUP_AND_RETURN(3);
    }

    struct timespec alloc_start;
    struct timespec alloc_end;

    clock_gettime(CLOCK_MONOTONIC, &alloc_start);
    image->data = malloc(size);
    clock_gettime(CLOCK_MONOTONIC, &alloc_end);

    double alloc_time_ms =
        (alloc_end.tv_sec - alloc_start.tv_sec) * 1000.0 +
        (alloc_end.tv_nsec - alloc_start.tv_nsec) / 1000000.0;

    g_total_malloc_time_ms += alloc_time_ms;
    g_malloc_count++;

    if (NULL == image->data) {
        printf("[BMP] memory allocation failed\n");
        CLEANUP_AND_RETURN(5);
    }

    if (0 != fseek(input, image->header.offset, SEEK_SET)) {
        printf("[BMP] file seek error\n");
        CLEANUP_AND_RETURN(2);
    }

    // reading by rows in invert vertically
    int i;
    for (i = 0; i < image->height; ++i) {
        int storeAt = image->infoHeader.height < 0 ? i : image->height - 1 - i;
        cnt = fread(image->data + row_size * storeAt, row_size, sizeof(unsigned char), input);
        if (cnt != sizeof(unsigned char)) {
            printf("[BMP] file read error\n");
            CLEANUP_AND_RETURN(2);
        }
        cnt = fread(pad, pad_size, sizeof(unsigned char), input);
        if ((pad_size != 0 && cnt != 0) && (cnt != sizeof(unsigned char))) {
            printf("[BMP] file read error\n");
            CLEANUP_AND_RETURN(2);
        }
    }
    fclose(input);
    return 0;
}

double get_total_malloc_time_ms(void) {
    return g_total_malloc_time_ms;
}

size_t get_malloc_count(void) {
    return g_malloc_count;
}
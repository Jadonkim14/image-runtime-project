#!/bin/bash

THREADS="8 4 8 4 8 4"
RUN=1

RESULT_FILE="benchmark/v0.4-inference-profile/v0.5_final_validation_results.txt"
> "$RESULT_FILE"

for thread in $THREADS
do
    echo "===== Run: $RUN / Threads: $thread =====" >> "$RESULT_FILE"

    perf stat -- ./benchmark/v0.4-inference-profile/build/hello_classification_c \
        models/squeezenet1.1-7.onnx \
        data/images/banana.bmp \
        CPU \
        "$thread" >> "$RESULT_FILE" 2>&1

    RUN=$((RUN + 1))

    sleep 2
done

#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=8

EXP_NAME=$1
echo "Experiment Name: llava.eval.sg.$EXP_NAME"
CKPT="llava-v1.5-13b-$EXP_NAME"

WITH_SG=1

#Step 1: Scene-Graph Generation:
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${GPULIST[$IDX]}
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.sg.$EXP_NAME \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench-filtered.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/full_sg/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --scene_graph $WITH_SG \
        --conv-mode vicuna_v1 &
    #sleep 2 - only for Sphinx-V2
done

wait

output_file=./playground/data/eval/seed_bench/full_sg/$CKPT/merge_$EXP_NAME.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/full_sg/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


#Step 2: Answer Extraction and Evaluation:
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.sg.$EXP_NAME \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file ./playground/data/eval/seed_bench/full_sg/$CKPT/merge_$EXP_NAME.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    #sleep 2 - Sphinx-V2
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge_$EXP_NAME.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/llava-v1.5-13b-$EXP_NAME.jsonl

# --result-file $output_file \

# --result-file ./playground/data/eval/seed_bench/gt_merge.jsonl \
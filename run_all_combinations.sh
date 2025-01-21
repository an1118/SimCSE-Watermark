#!/bin/bash
set -e

model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
gpu_id=2
batch_size=8
train_epochs=2
LOSS_FUNCTION_IDS=(2 4)  # 2 4 3
NEG_WEIGHTS=(1 32)  # 1 32 64 128

for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
  for neg_weight in "${NEG_WEIGHTS[@]}"; do
    bash run_sup_example_inbatch.sh \
      --gpu_id $gpu_id \
      --batch_size $batch_size \
      --train_epochs $train_epochs \
      --loss_function_id $loss_function_id \
      --neg_weight $neg_weight \
      --model_name $model_name
  done
done

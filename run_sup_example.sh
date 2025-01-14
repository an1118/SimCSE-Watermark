#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# CUDA_VISIBLE_DEVICES=1 python train.py \
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path cardiffnlp/twitter-roberta-base-sentiment \
    --train_file /mnt/data2/lian/projects/watermark/data/lfqa_train_small128.csv \
    --validation_file /mnt/data2/lian/projects/watermark/data/lfqa_test_final_simcse.csv \
    --output_dir result/end2end-simcse_and_adaptive-roberta-sentiment-lambda1100_small128 \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 50 \
    --learning_rate 5e-5 \
    --max_seq_length 320 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --eval_steps 5 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --report_to="wandb" \
    --run_name="sanity_check_wm-simcse_and_adaptive-roberta-sentiment-lambda1100" \
    --logging_steps=1 \
    "$@"
    # --gradient_accumulation_steps 16 \
    # --freeze_roberta

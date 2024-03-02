#!/usr/bin/env bash
OMP_NUM_THREADS=1

OUTPUT_DIR='test_results/vit_b_pd_ft_weights_19e'
DATA_PATH='pd_data'
# MODEL_PATH='vit_b_k710_dl_from_giant.pth'
# MODEL_PATH='train_results/vit_b_pd_ft/checkpoint-best.pth'
MODEL_PATH='train_results/vit_b_pd_ft_weights/checkpoint-19.pth'

python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set PD \
        --nb_classes 3 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 7e-4 \
        --drop_path 0.1 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --epochs 20 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --mixup 0 \
        --cutmix 0 \
        --smoothing 0 \
        --eval

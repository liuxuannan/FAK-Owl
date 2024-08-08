#!/bin/bash
ROOT=../
export PYTHONPATH=$ROOT:$PYTHONPATH
deepspeed --include localhost:0,1 --master_port 28400 train_DGM4.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --save_path  ./ckpt/train_DGM4/\
    --log_path ./ckpt/train_DGM4/log_rest/\
    --config ./fake_config/train_bbc.yaml


#!/bin/bash

cd ../src &&

# Example script for PickCube training (with a good set of hyper-parameters).
CUDA_VISIBLE_DEVICES=6 python train.py \
   --model_name=PIS-0910 \
   --num_traj=500 --n_iters=1_800_000 --weight_decay=1e-3 --lr_schedule=cos_decay_with_warmup \
   --context_length=60 --model_type=s+a+cot \
   --task=PegInsertionSide-v0 --key_state_coeff=0.1 \
   --n_layer=4 --vq_n_e=10 --key_state_loss=0 --key_states=ab \
   --init_lr=5e-4 --num_workers=20 --save_every=20000 \



# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model_name=some_model_name \
#     --num_traj=500 --n_iters=1_600_000 \
#     --context_length=60 --model_type=s+a+cot \
#     --task=TurnFaucet-v0 --key_state_coeff=0.1 \
#     --key_state_loss=0 --key_states=ab \
#     --init_lr=5e-4 --num_workers=20
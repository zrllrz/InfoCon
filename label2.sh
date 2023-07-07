#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=2 python val_label2.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=kae-code_10-smooth_0.5-seq_k-example_1.0 \
   --from_ckpt=5000 \

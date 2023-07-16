#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=7 python val_label2.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=kae-code_5-smooth_0.5-seq_k-example_10.0-reg_diff_0.01_bound_0.9_begin_0.01_bound_0.9  \
   --from_ckpt=5000 \

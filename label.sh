#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=0 python label.py \
   --task=PegInsertionSide-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=10 \
   --model_name=TEST_GPTHN_NN_EMA_GOAL_RECS_TIME++-k4-r4-c10_KT0.1_EMA0.5_temb1.2-egpthn_s2_a1-emb128-key128-e128-cluster0.5-rec1.0 \
   --from_ckpt=2000 \


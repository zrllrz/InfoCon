#!/bin/bash

cd src &&

CUDA_VISIBLE_DEVICES=3 python label.py \
   --task=PickCube-v0 --control_mode=pd_joint_delta_pos --obs_mode=state \
   --seed=0 \
   --n_traj=-1 \
   --model_name=PickCube_GPTHN_NN_EMA_GOAL2_RECS_TIME++_ENCO+_k4-r4-f1-c10_KT0.1_EMA0.95_temb1.2-r_l10.0-egpthn_s3_a1-emb128-key128-e128-cluster0.001-rec1.0 \
   --from_ckpt=1900 \

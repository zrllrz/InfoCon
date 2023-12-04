#!/bin/bash

cd ../src &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=150 \
  --from_ckpt=2670000 \
  --task=PickCube-v0 \
  --model_name=PC-MARG \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=150 \
  --from_ckpt=1800000 \
  --task=PickCube-v0 \
  --model_name=PC-0921-WAYPOINT \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=1232000 \
  --task=StackCube-v0 \
  --model_name=SC-0919-MIDDLE \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=1520000 \
  --task=StackCube-v0 \
  --model_name=SC-0924-waypoint \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=250 \
  --from_ckpt=2540000 \
  --task=TurnFaucet-v0 \
  --model_name=TF-0917-MIDDLE \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=250 \
  --from_ckpt=1480000 \
  --task=TurnFaucet-v0 \
  --model_name=TF-0924-waypoint \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=1720000 \
  --task=PegInsertionSide-v0 \
  --model_name=PegInsertionSide-v0-8-26 \
  --n_env=1

CUDA_VISIBLE_DEVICES=1 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=1440000 \
  --task=PegInsertionSide-v0 \
  --model_name=PIS-0924-waypoint \
  --n_env=1

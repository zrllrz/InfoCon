#!/bin/bash

TASK=PegInsertionSide-v0
# MODEL_NAME=SC-0919-MIDDLE
MODEL_NAME=PIS-1114-MORE-K0909
# I=2196000
I=3240000



cd ../src &&

#CUDA_VISIBLE_DEVICES=6 python eval.py --eval_max_steps=300 \
#    --from_ckpt=1780000 --task=TurnFaucet-v0-v0 \
#    --model_name=TF-0905 \

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --eval_max_steps=200 \
  --from_ckpt=$I \
  --task=$TASK \
  --model_name=$MODEL_NAME \
  --n_env=25
#!/bin/bash

MODEL_DIR="../save_model/"
TASK=PickCube-v0
MODEL_NAME=PC-MARG
I=2700000

cd ../src &&

#CUDA_VISIBLE_DEVICES=6 python eval.py --eval_max_steps=300 \
#    --from_ckpt=1780000 --task=TurnFaucet-v0-v0 \
#    --model_name=TF-0905 \
while [[ $I -ge 2500000 ]];
do
  echo "$MODEL_DIR$MODEL_NAME""/""$I.pth"
  if test -e "$MODEL_DIR$MODEL_NAME""/""$I.pth"; then
    CUDA_VISIBLE_DEVICES=7 python eval.py \
      --eval_max_steps=150 \
      --from_ckpt=$I \
      --task=$TASK \
      --model_name=$MODEL_NAME \
      --n_env=25
  else
    echo "wait"
  fi
  ((I-=2000))
done

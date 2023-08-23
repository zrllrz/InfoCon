#!/bin/bash

cd ../src &&

python eval.py --eval_max_steps=200 \
    --from_ckpt=1_040_000 --task=PegInsertionSide-v0 \
    --model_name=some_model_name

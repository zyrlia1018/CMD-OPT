#!/bin/bash

python generate.py \
    --model-choice transformer \
    --data-path data/chembl_02 \
    --test-file-name ripk1 \
    --model-path experiments/train_transformer/checkpoint \
    --save-directory evaluation_transformer \
    --epoch 60


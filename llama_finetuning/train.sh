#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed

accelerate launch finetune.py \
    --output-dir output/resonance-yarn-7b-32k \
    --model meta-llama/Llama-2-7b-chat-hf \
    --scaling-factor 8 \
    --truncate 32768 \
    --max-train-steps 50 \
    --warmup-steps 2 \
    --architecture llama \
    --deepspeed \
    --resonance-rope

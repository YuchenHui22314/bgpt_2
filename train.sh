#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train-gen.py &>> ./logs/logs-test_text.txt
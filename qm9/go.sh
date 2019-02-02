#!/bin/bash -ex
CUDA_VISIBLE_DEVICES=$1 python3 main.py $2 $3

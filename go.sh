#!/bin/bash -ex

python3 main.py --DS MUTAG  --num-gc-layers 6 --lr 0.0001 --no-node-labels --no-node-attr --log-interval 10 --output-dim 1024
#python3 main.py --DS PROTEINS_full --num-gc-layers 6 --lr 0.0001 --no-node-labels --no-node-attr

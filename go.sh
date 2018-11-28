#!/bin/bash -ex

for gc in 2 3 4 5 6 7 8 9
do
    python3 main.py --DS MUTAG  --no-node-labels --no-node-attr --output-dim 100 --log-interval 10 --lr 0.0001 --num-gc-layers $gc --epochs 100
done

for gc in 2 3 4 5 6 7 8 9
do 
  python3 main.py --DS MUTAG  --log-interval 10 --output-dim 100 -log-interval --lr 0.0001 --num-gc-layers $gc --epochs
done

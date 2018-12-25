#!/bin/bash -ex

DS=$1
for i in 1 2 3 4 5 6 7 8 9 10
do
  for gc in 16
  do 
    python3 main.py --DS $DS  --output-dim 100 --loss-type bce \
      --log-interval 1 --epochs 100 \
      --lr 0.0001 --num-gc-layers $gc 

    python3 main.py --DS $DS  --output-dim 100 --loss-type bce \
      --log-interval 1 --epochs 100 \
      --lr 0.0001 --num-gc-layers $gc --local

      python3 main.py --DS $DS --no-node-labels --no-node-attr --output-dim 100 --loss-type bce \
        --log-interval 1 --epochs 100 \
        --lr 0.0001 --num-gc-layers $gc 

      python3 main.py --DS $DS --no-node-labels --no-node-attr --output-dim 100 --loss-type bce \
        --log-interval 1 --epochs 100 \
        --lr 0.0001 --num-gc-layers $gc --local
  done
done

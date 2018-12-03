#!/bin/bash -ex

DS=$1
for i in 1 2 3 4 5 6 7 8 9 10
do
  for gc in 9 8 7 6 5 4 3 2
  do 
    python3 main.py --DS $DS  --log-interval 1 --output-dim 100 --lr 0.0001 --num-gc-layers $gc --epochs 10 --batch-size 2
  done

  for gc in 9 8 7 6 5 4 3 2
  do
      python3 main.py --DS $DS --no-node-labels --no-node-attr --output-dim 100 --log-interval 1 --lr 0.0001 --num-gc-layers $gc --epochs 10 --batch-size 2
  done
done

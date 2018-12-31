#!/bin/bash -ex

DS=$1
CUDA=$2

for i in 1 2 3 4 5 6 7 8 9 10
do
  for gc in 4 8 12
  do 
    python3 main.py --DS $DS  --output-dim 512 \
      --log-interval 1 --epochs 10 \
      --lr 0.0001 --num-gc-layers $gc --local --cuda $CUDA --local-ds gitgraph-proc-subgraph

    #python3 main.py --DS $DS  --output-dim 512 \
      #--log-interval 1 --epochs 10 \
      #--lr 0.0001 --num-gc-layers $gc --local --cuda $CUDA --local-ds gitgraph-proc-subgraph2

    python3 main.py --DS $DS  --output-dim 512 \
      --log-interval 1 --epochs 10 \
      --lr 0.0001 --num-gc-layers $gc --cuda $CUDA
  done
done

#!/bin/bash -ex

DS=$1
CUDA=$2
lr=.01
workers=0
for i in 1 2 3 4 5 6 7 8 9 10
do
  for gc in 4 8 12 16
  do 

    python3 main.py --DS $DS  --embedding-dim 512 \
      --log-interval 1 --epochs 10  \
      --lr $lr --num-gc-layers $gc --cuda $CUDA --no-node-attr --concat --batch-size 64 --glob --num-workers $workers

    python3 main.py --DS $DS  --embedding-dim 512 \
      --log-interval 1 --epochs 10 \
      --lr $lr --num-gc-layers $gc --cuda $CUDA --no-node-attr --concat --batch-size 64 --glob --prior --num-workers $workers

    python3 main.py --DS $DS  --embedding-dim 512 \
      --log-interval 1 --epochs 10  \
      --lr $lr --num-gc-layers $gc --cuda $CUDA --no-node-attr --concat --batch-size 64 --local --prior --glob --num-workers $workers

    python3 main.py --DS $DS  --embedding-dim 512 \
      --log-interval 1 --epochs 10 \
      --lr $lr --num-gc-layers $gc --cuda $CUDA --no-node-attr --concat --batch-size 64 --local --prior --num-workers $workers

    python3 main.py --DS $DS  --embedding-dim 512 \
      --log-interval 1 --epochs 10 \
      --lr $lr --num-gc-layers $gc --cuda $CUDA --no-node-attr --concat --batch-size 64 --local --num-workers $workers

  done
done

#!/bin/bash -ex

. ~/ENV2/bin/activate

# run preprocess before running all following command
#python3 preprocess.py

python main.py -c ../data/DD -l ../data/DD.Labels -d 1024 --wlk_h 3 -e 2000 -lr 0.01
python main.py -c ../data/COLLAB -l ../data/COLLAB.Labels -d 1024 --wlk_h 3 -e 2000 -lr 0.5
python main.py -c ../data/IMDB-BINARY -l ../data/IMDB-BINARY.Labels -d 1024 --wlk_h 3 -e 2000 -lr 0.5
python main.py -c ../data/IMDB-MULTI-5K -l ../data/IMDB-MULTI-5K.Labels -d 1024 --wlk_h 3 -e 2000 -lr 0.5
#python main.py -c ../data/kdd_datasets/mutag -l ../data/kdd_datasets/mutag.Labels -b 256 -d 1024 --wlk_h 3 -e 1000 -lr 0.5

#python main.py -c ../data/kdd_datasets/ptc -l ../data/kdd_datasets/ptc.Labels -b 256 -d 1024 --wlk_h 3 -e 1000 -lr 0.5

#python main.py -c ../data/kdd_datasets/proteins -l ../data/kdd_datasets/proteins.Labels -b 1024 -d 1024 --wlk_h 3 -e 1000 -lr 0.5

#python main.py -c ../data/kdd_datasets/nci1 -l ../data/kdd_datasets/nci1.Labels -b 1024 -d 512 --wlk_h 3 -e 1000 -lr 0.5
#python main.py -c ../data/kdd_datasets/nci109 -l ../data/kdd_datasets/nci109.Labels -b 1204 -d 512 --wlk_h 3 -e 1000 -lr 0.5

#python main.py                   \
  #-c ../../../gexf-stars/   \
  #-l ../../../esc/star-Labels  \
  #-lf op \
  #-b 512 -d 512 --wlk_h 3 -e 2 -lr 0.5


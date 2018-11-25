#!/bin/bash -ex
.  ../TEST/bin/activate
python3 main.py --dataset $@

## Usage
This subdirectory utilizes the code from paper [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://openreview.net/forum?id=r1lfF2NYvH).
What I've been trying to do is to run the unsupervised method on my customed dataset (e.g. in `dataset.py`) but failed due to the usage of `data.y` in the original code. I created an issue under the original repo and would update here once respnded.

My testing environment: `pytorch 1.13.0` and `pytorch_geometric 2.2.0` on MacBook Pro with m1 silicon.

##### sample command
```
$ python main.py --DS KKI --lr 0.001 --num-gc-layers 2  --hidden-dim 10
```

## Dataset
Dataset should be automatically downloaded when you have [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) installed properly.

You can also downloaded datasets manually at [https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)

## Baselines
For other baseline implementations, refer to my other repo: https://github.com/fanyun-sun/graph-classification.git

---
Bellow are instructions from the original repo:

Tested on pytorch 1.6.0 and [pytorch\_geometric](https://github.com/rusty1s/pytorch_geometric) 1.6.1. Experiments reported on the paper are conducted in 2019 with `pytorch_geometric==1.3.1`. 
Note that the code regarding of QM9 dataset in pytorch\_geometric has been changed since then. Thus, if you run this repo with `pytorch_geometric>=1.6.1`, you may obtain results differ from those reported on the paper.

Code regarding mutual information maximization is partially referenced from: [https://github.com/rdevon/DIM](https://github.com/rdevon/DIM)
#### Cite

Please cite [our paper](https://openreview.net/pdf?id=r1lfF2NYvH) if you use this code in your own work:

```
@inproceedings{sun2019infograph,
  title={InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization},
  author={Sun, Fan-Yun and Hoffman, Jordan and Verma, Vikas and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```



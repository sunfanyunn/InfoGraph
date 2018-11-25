## Kernel Graph Convolutional Neural Networks
Code for the paper [Kernel Graph Convolutional Neural Networks](https://arxiv.org/pdf/1710.10689.pdf).

### Requirements
Code is written in Python 3.6 and requires:
* PyTorch 0.3
* NetworkX 1.11
* igraph 0.7
* scikit-learn 0.18

### Datasets
Use the following link to download datasets: 
```
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
```
Extract the datasets into the `datasets` folder.

### Run the model
First, specify the dataset and the hyperparameters in the `main.py` file. Then, use the following command:

```
$ python main.py
```

### Cite
Please cite our paper if you use this code:
```
@article{nikolentzos2017kernel,
  title={Kernel Graph Convolutional Neural Networks},
  author={Nikolentzos, Giannis and Meladianos, Polykarpos and Tixier, Antoine Jean-Pierre and Skianis, Konstantinos and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:1710.10689},
  year={2017}
}
```

-----------

Provided for academic use only

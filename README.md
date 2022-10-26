# DAGAD: Data Augmentation for Graph Anomaly Detection (ICDM 2022)

This repository is the official PyTorch implementation of DAGAD in the following paper:

DAGAD: Data Augmentation for Graph Anomaly Detection, ICDM 2022 [[arXiv](https://arxiv.org/pdf/2210.09766.pdf)]

**Brief Intro.** DAGAD allieviates `anomalous sample scarcity` and `class imbalance` issues of anomalous node detection with THREE strengths:

- DAGAD employs a data augmentation module to derive additional training samples in the embedding space, which is also extendable to other graph learning tasks that rely on learning features from a very limited number of labeled instances.
- Augmented samples together with original ones are leveraged by two classifiers in a complementary manner, to learn discriminative
representations for the anomalous and normal classes.
- DAGAD develops class-wise losses to alleviate the suffering from class imbalance, which is can be easily integrated into semi-supervised anomaly detectors.

If you find this work helpful for your research, please cite this paper:

        @inproceedings{liu2022DAGAD, 
    	    author = {Fanzhen Liu and Xiaoxiao Ma and Jia Wu and Jian Yang and Shan Xue and Amin Behesht and 
                          Chuan Zhou and Hao Peng and Quan Z. Sheng and Charu C. Aggarwal},
    	    title = {DAGAD: Data Augmentation for Graph Anomaly Detection},
    	    booktitle = {ICDM},
    	    year = {2022},
        }

## Requirements:
- Python: 3.7.11
- [Pytorch](https://pytorch.org/): '1.10.0+cu102'
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/): 2.0.2
- numpy: 1.21.4

## Parameter Setting:
-   ('--seed', type=int, default=7, help='Random seed.')
-    ('--dataset', type=str, default='BlogCatalog', help="['BlogCatalog', 'ACM', 'Flickr')")
-    ('--gnn_layer', type=str, default='GCN', help="['GCN','GAT']")
-    ('--epoch_num', type=int, default=200, help='Number of epochs to train.')
-    ('--learning_rate', default=0.005, help='Learning rate of the optimiser.')
-    ('--weight_decay', default=5e-4, help='Weight decay of the optimiser.')
    
-    ('--gnn_dim', type=int, default=64)
-    ('--fcn_dim', type=int, default=32)
-    ('--gce_q', default=0.7, help='gce q')
-    ('--alpha', type=float, default=1.5)
-    ('--beta', type=float, default=0.5)
-    ('--gat_heads', default=8, help='GAT heads')

## How to Use
Example on BlogCatalog dataset
- testing DAGAD_GCN: `python main.py --dataset BlogCatalog --gnn_layer GCN`
- testing DAGAD_GAT: `python main.py --dataset BlogCatalog --gnn_layer GAT` 



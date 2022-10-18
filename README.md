# DAGAD: Data Augmentation for Graph Anomaly Detection

## Parameters to run our model and their default values:
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

## Example (run test on BlogCatalog dataset):
- testing DAGAD_GCN: python main.py --dataset BlogCatalog --gnn_layer GCN 
- testing DAGAD_GAT: python main.py --dataset BlogCatalog --gnn_layer GAT 

## Python and related packages:
- Python: 3.7.11
- torch: '1.10.0+cu102'
- PyG: 2.0.2
- numpy: 1.21.4
    
        @inproceedings{liu2022DAGAD, 
    	    author = {Fanzhen Liu and Xiaoxiao Ma and Jia Wu and Jian Yang and Shan Xue and Amin Behesht and Chuan Zhou and
                      Hao Peng and Quan Z. Sheng and Charu C. Aggarwal},
    	    booktitle = {ICDM},
    	    title = {DAGAD: Data Augmentation for Graph Anomaly Detection},
    	    year = {2022},
        }

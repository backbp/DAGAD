'''
Author: Xiaoxiao Ma, Fanzhen Liu
Description: DAGAD Entrence.
'''
import argparse
import torch
from utils import load_data
from DAGAD import AUG_AD_swap, AUG_AD_swap_GAT, train, test

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='BlogCatalog', help="['BlogCatalog', 'ACM', 'Flickr'].")
    parser.add_argument('--gnn_layer', type=str, default='GCN', help="['GCN','GAT']")
    parser.add_argument('--epoch_num', type=int, default=200, help='The number of epochs for training.')
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate of the optimizer.')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimizer.')
    
    # Model hyperparameters
    parser.add_argument('--gnn_dim', type=int, default=64)
    parser.add_argument('--fcn_dim', type=int, default=32)
    parser.add_argument('--gce_q', default=0.7, help='gce q')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=0.5)

    # For GAT only, num of attention heads
    parser.add_argument('--gat_heads', default=8, help='GAT heads')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse args
    args = arg_parser()
    epochs = args.epoch_num
    lr = args.learning_rate
    wd = args.weight_decay
    gnn_dim = args.gnn_dim
    fcn_dim = args.fcn_dim
    q = args.gce_q
    alpha = args.alpha
    beta = args.beta
    heads = args.gat_heads
    gnn_layer = args.gnn_layer

    # Load data
    data = load_data(args.dataset, args.seed)
    data = data.to(device)

    # Model Training
    input_dim = data.x.shape[1]
    num_classes = len(set(data.y.tolist()))
    if args.gnn_layer == 'GCN':
        print("Training model: DAGAD_{} on dataset {}".format(args.gnn_layer, args.dataset))
        model = AUG_AD_swap(input_dim, gnn_dim, fcn_dim, num_classes, device)
        model = model.to(device)
    if args.gnn_layer == 'GAT':
        print("Training model: DAGAD_{} on dataset {}".format(args.gnn_layer, args.dataset))
        model = AUG_AD_swap_GAT(input_dim, gnn_dim, fcn_dim, heads, num_classes, device)
        model = model.to(device)

    train(model, data, epochs, lr, wd, alpha, beta)
    test(model, data)





from utils import GeneralizedCELoss1
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report

def sys_config():
    os.system("module load cuda/10.1")
    os.system("module load cudnn/7.6.5-cuda10.1")
    torch.cuda.is_available()

class AUG_AD_swap(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, fcn_dim, num_classes, device):

        """
        Using GCN layers to learn node embeddings.
        """
        super(AUG_AD_swap, self).__init__()
        self.device = device
        self.hid = hidden_dim
        self.fcn_dim = 32
        self.name = 'DAGAD-GCN'

        self.GNN_a_conv1 = GCNConv(input_dim, hidden_dim)
        self.GNN_a_conv2 = GCNConv(hidden_dim*2, num_classes)

        self.GNN_b_conv1 = GCNConv(input_dim, hidden_dim)
        self.GNN_b_conv2 = GCNConv(hidden_dim*2, num_classes)

        self.fc1_a = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_a = nn.Linear(fcn_dim, num_classes)

        self.fc1_b = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, data):
        h_a = self.GNN_a_conv1(data.x, data.edge_index)
        h_b = self.GNN_b_conv1(data.x, data.edge_index)

        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)

        h_aug_back_a, h_aug_back_b, data = self.swap_operation(data, h_b, h_a)

        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)

        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)

        h_back_a = F.dropout(h_back_a, training=self.training)
        h_back_b = F.dropout(h_back_b, training=self.training)

        h_aug_back_a = F.dropout(h_aug_back_a, training=self.training)
        h_aug_back_b = F.dropout(h_aug_back_b, training=self.training)

        h_back_a = self.fc1_a(h_back_a)
        h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        h_back_b = h_back_b.relu()

        h_aug_back_a = self.fc1_a(h_aug_back_a)
        h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        h_aug_back_b = h_aug_back_b.relu()

        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)

        pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        pred_aug_back_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)

        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_back_b, data

    def swap_operation(self, data, h_b, h_a, reclass=1):
        indices = np.random.permutation(h_b.shape[0])
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        sorter = np.argsort(indices)
        indices_org = np.array(range(data.y.shape[0]))

        aug_id = sorter[np.searchsorted(indices, indices_org, sorter=sorter)]
        aug_id = torch.LongTensor(aug_id).to(self.device)
        data.aug_id = aug_id

        data.aug_train_mask = aug_id[data.train_mask]
        data.aug_val_mask = aug_id[data.val_mask]
        data.aug_test_mask = aug_id[data.test_mask]

        if reclass == 0:
            data.aug_train_anm = aug_id[data.train_anm]
            data.aug_train_norm = aug_id[data.train_norm]
        else:
            temp1 = data.y[data.aug_train_mask] == 1
            data.aug_train_anm = data.aug_train_mask[temp1]
            temp2 = data.y[data.aug_train_mask] == 0
            data.aug_train_norm = data.aug_train_mask[temp2]

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)

        return h_aug_back_a, h_aug_back_b, data

class AUG_AD_swap_GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, fcn_dim, heads_num, num_classes, device):
        super(AUG_AD_swap_GAT, self).__init__()

        self.device =device
        self.hid = hidden_dim
        self.fcn_dim=32
        self.heads = heads_num
        self.name = 'DAGAD-GAT'

        self.GNN_a_conv1 = GATConv(input_dim, hidden_dim, self.heads)
        self.GNN_a_conv2 = GATConv(hidden_dim*2*self.heads, num_classes, self.heads)

        self.GNN_b_conv1 = GATConv(input_dim, hidden_dim, self.heads)
        self.GNN_b_conv2 = GATConv(hidden_dim*2*self.heads, num_classes)

        self.fc1_a = nn.Linear(hidden_dim*2*self.heads, fcn_dim)
        self.fc2_a = nn.Linear(fcn_dim, num_classes)

        self.fc1_b = nn.Linear(hidden_dim*2*self.heads, fcn_dim)
        self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, data):
        h_a = self.GNN_a_conv1(data.x, data.edge_index)
        h_b = self.GNN_b_conv1(data.x, data.edge_index)

        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)

        h_aug_back_a, h_aug_back_b, data = self.swap_operation(data, h_b, h_a)

        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)

        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)

        h_back_a = F.dropout(h_back_a, training=self.training)
        h_back_b = F.dropout(h_back_b, training=self.training)

        h_aug_back_a = F.dropout(h_aug_back_a, training=self.training)
        h_aug_back_b = F.dropout(h_aug_back_b, training=self.training)

        h_back_a = self.fc1_a(h_back_a)
        h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        h_back_b = h_back_b.relu()

        h_aug_back_a = self.fc1_a(h_aug_back_a)
        h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        h_aug_back_b = h_aug_back_b.relu()

        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)

        pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        pred_aug_back_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)

        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_back_b, data

    def swap_operation(self, data, h_b, h_a, reclass=1):
        indices = np.random.permutation(h_b.shape[0])
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        """update the indices of augmented data """
        sorter = np.argsort(indices)
        indices_org = np.array(range(data.y.shape[0]))

        aug_id = sorter[np.searchsorted(indices, indices_org, sorter=sorter)]
        aug_id = torch.LongTensor(aug_id).to(self.device)
        data.aug_id = aug_id

        data.aug_train_mask = aug_id[data.train_mask]
        data.aug_val_mask = aug_id[data.val_mask]
        data.aug_test_mask = aug_id[data.test_mask]

        if reclass == 0:
            data.aug_train_anm = aug_id[data.train_anm]
            data.aug_train_norm = aug_id[data.train_norm]
        else:
            temp1 = data.y[data.aug_train_mask] == 1
            data.aug_train_anm = data.aug_train_mask[temp1]
            temp2 = data.y[data.aug_train_mask] == 0
            data.aug_train_norm = data.aug_train_mask[temp2]

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)

        return h_aug_back_a, h_aug_back_b, data

def train(model_ad, data, epochs, lr, wd, alpha, beta):

    """
    Train the anomaly detection model:
    model, data, epochs, lr, wd, alpha, beta
    """
    labels = data.y
    optimizer_ad = torch.optim.Adam(model_ad.parameters(), lr=lr, weight_decay=wd)
    criterion_gce = GeneralizedCELoss1(q=0.7)
    criterion = torch.nn.CrossEntropyLoss()

    loss_results = []
    loss_ce_results = []
    loss_gce_results = []
    loss_gce_aug_results = []

    loss_ce_a_results = []
    loss_ce_b_results = []

    #classifier a
    test_prec_a = []
    test_rec_a = []
    test_f1_a = []
    auc_sc_a = []

    #classfier b
    test_prec_b = []
    test_rec_b = []
    test_f1_b = []
    auc_sc_b = []

    model_ad.train()

    for epoch in range(0, epochs):
        t_start = t.time()
        optimizer_ad.zero_grad()

        pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_back_b, data = model_ad(data)

        loss_ce_a = criterion(pred_org_back_a[data.train_mask], labels[data.train_mask])
        loss_ce_b = criterion(pred_org_back_b[data.train_mask], labels[data.train_mask])
        loss_ce_weight = loss_ce_b / (loss_ce_b + loss_ce_a + 1e-8)
        loss_ce_a_results.append(loss_ce_a.item())
        loss_ce_b_results.append(loss_ce_b.item())

        loss_ce_anm = criterion(pred_org_back_a[data.train_anm], labels[data.train_anm])
        loss_ce_norm = criterion(pred_org_back_a[data.train_norm], labels[data.train_norm])
        loss_ce = loss_ce_weight * (loss_ce_anm + loss_ce_norm)/2

        loss_gce = 0.5 * criterion_gce(pred_org_back_b[data.train_anm], labels[data.train_anm]) \
                   + 0.5 * criterion_gce(pred_org_back_b[data.train_norm], labels[data.train_norm])

        loss_gce_aug = 0.5 * criterion_gce(pred_org_back_b[data.aug_train_anm], labels[data.aug_train_anm]) \
                        + 0.5 * criterion_gce(pred_org_back_b[data.aug_train_norm], labels[data.aug_train_norm])

        loss = alpha * loss_ce + loss_gce + beta * loss_gce_aug

        loss.backward()
        optimizer_ad.step()

        with torch.no_grad():
            pred_a = pred_org_back_a.argmax(dim=1)
            pred_b = pred_org_back_b.argmax(dim=1)

            loss_results.append(loss.item())
            loss_ce_results.append(loss_ce.item())
            loss_gce_results.append(loss_gce.item())
            loss_gce_aug_results.append(loss_gce_aug.item())

            test_precision_a = precision_score(labels[data.test_mask].cpu(), pred_a[data.test_mask].cpu(), average='macro', labels=np.unique(pred_a.cpu()))
            test_prec_a.append(test_precision_a)
            test_recall_a = recall_score(labels[data.test_mask].cpu(), pred_a[data.test_mask].cpu(), average='macro')
            test_rec_a.append(test_recall_a)
            test_fscore_a = f1_score(labels[data.test_mask].cpu(), pred_a[data.test_mask].cpu(), average='macro')
            test_f1_a.append(test_fscore_a)
            auc_score_a = roc_auc_score(labels[data.test_mask].cpu(), pred_a[data.test_mask].cpu())
            auc_sc_a.append(auc_score_a)

            test_precision_b = precision_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro', labels=np.unique(pred_b.cpu()))
            test_prec_b.append(test_precision_b)
            test_recall_b = recall_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
            test_rec_b.append(test_recall_b)
            test_fscore_b = f1_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
            test_f1_b.append(test_fscore_b)
            auc_score_b = roc_auc_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu())
            auc_sc_b.append(auc_score_b)

        print(f'Epoch: {epoch+1:03d}, Train_Loss: {loss:.4f}, test_prec_b: {test_precision_b:.4f}, T_Rec_b: {test_recall_b:.4f}, '
              f'T_F1_b: {test_fscore_b:.4f}, AUC_score_b: {auc_score_b:.4f}, Time: {t.time()-t_start:.4f}')

def test(model, data):
    model.eval()
    with torch.no_grad():
        labels = data.y
        pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_back_b, data = model(data)

        pred_a = pred_org_back_a.argmax(dim=1)
        pred_b = pred_org_back_b.argmax(dim=1)

        test_precision_b = precision_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro', labels=np.unique(pred_b.cpu()))
        test_recall_b = recall_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
        test_fscore_b = f1_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
        auc_score_b = roc_auc_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu())

        print(f'test_prec_b: {test_precision_b:.4f}, T_Rec_b: {test_recall_b:.4f}, T_F1_b: {test_fscore_b:.4f}, AUC_score_b: {auc_score_b:.4f}')

        print(classification_report(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), target_names=['normal_b', 'anomaly_b']))

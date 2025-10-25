from sklearn.metrics import matthews_corrcoef
import numpy as np
import copy
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import average_precision_score
from numpy.core import multiarray
from torch.nn.parameter import Parameter
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import DataEnhancement
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model import *
from kl_loss import kl_loss
from utils import f1_score_binary, precision_binary, recall_binary, accuracy_binary
import scipy.sparse as sp
from hypergraph_utils import *
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import os  # 必须导入 os

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 设置 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子
seed = 48
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])  # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)
    return norm_A


def cross_validation_5fold(k_folds):
    fold = int(total_associations / k_folds)  # e.g., total positive samples

    auc_sum = 0
    aupr_sum = 0
    rec_sum = 0
    pre_sum = 0
    f1_sum = 0
    acc_sum = 0
    mcc_sum = 0
    tprs = []
    fprs = []
    aucs = []
    precisions = []
    recalls = []
    auprs = []
    loss_lists = []
    accuracy_lists = []
    mcc_lists = []

    # 五折交叉验证
    for f in range(1, k_folds + 1):
        print(f'{f} fold:')
        if f == k_folds:
            testset = shuffle_data[((f - 1) * fold): total_associations]
        else:
            testset = shuffle_data[((f - 1) * fold): f * fold]

        (auc1, aupr1, recall1, precision1, f11, acc1, mcc1,
         loss_list, accuracy_list, mcc_list, all_fpr, all_tpr, all_auc, fpr, tpr) = train(testset, epochs)

        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision1)
        recalls.append(recall1)
        aucs.append(auc1)
        auprs.append(aupr1)
        loss_lists.append(loss_list)
        accuracy_lists.append(accuracy_list)
        mcc_lists.append(mcc_list)

        auc_sum += auc1
        aupr_sum += aupr1
        rec_sum += recall1
        pre_sum += precision1
        f1_sum += f11
        acc_sum += acc1
        mcc_sum += mcc1

    auc_mean = auc_sum / k_folds
    aupr_mean = aupr_sum / k_folds
    pre_mean = pre_sum / k_folds
    rec_mean = rec_sum / k_folds
    f1_mean = f1_sum / k_folds
    acc_mean = acc_sum / k_folds
    mcc_mean = mcc_sum / k_folds

    print("cv_mean:")
    print('auc: {:.4f}, aupr: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}, acc: {:.4f}, mcc: {:.4f}'
          .format(auc_mean, aupr_mean, pre_mean, rec_mean, f1_mean, acc_mean, mcc_mean))

    metric = ["{:.4f}".format(v) for v in [auc_mean, aupr_mean, pre_mean, rec_mean, f1_mean, acc_mean, mcc_mean]]
    return metric, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_fpr, all_tpr, all_auc, fpr, tpr


def train(testset, epochs):
    all_neg_indices = np.random.permutation(len(Index_zeroRow))
    test_pos = list(testset)
    test_neg = all_neg_indices[:len(test_pos)]
    train_neg = list(set(all_neg_indices) - set(test_neg))

    X = copy.deepcopy(RD)
    Xn = copy.deepcopy(X)
    Xn2 = copy.deepcopy(RD2)

    true_list = np.zeros((len(test_pos) + len(test_neg), 1))
    for ii in range(len(test_pos)):
        Xn[pos_rna_idx[testset[ii]], pos_drug_idx[testset[ii]]] = 0
        Xn2[pos_rna_idx[testset[ii]], pos_drug_idx[testset[ii]]] = 0
        true_list[ii, 0] = 1

    train_mask = np.ones_like(Xn)
    for ii in range(len(test_pos)):
        train_mask[pos_rna_idx[testset[ii]], pos_drug_idx[testset[ii]]] = 0
        train_mask[Index_zeroRow[test_neg[ii]], Index_zeroCol[test_neg[ii]]] = 0
    train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool).to(device)
    label = true_list

    model = Gai_HGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_hypergraphs():
        # 高斯核相似性构建超图
        HHMG = construct_H_with_KNN(MG)
        HMG = generate_G_from_H(HHMG).to(torch.float32)

        HHDG = construct_H_with_KNN(DG)
        HDG = generate_G_from_H(HHDG).to(torch.float32)

        A = copy.deepcopy(Xn)
        AT = A.T
        L = A.copy()

        # 基于关联矩阵
        HHMD = construct_H_with_KNN(A)
        HMD = generate_G_from_H(HHMD).to(torch.float32)

        HHDM = construct_H_with_KNN(AT)
        HDM = generate_G_from_H(HHDM).to(torch.float32)

        # 序列相似性
        HHMM = construct_H_with_KNN(RNA_sim)
        HMM = generate_G_from_H(HHMM).to(torch.float32)

        HHDD = construct_H_with_KNN(Drug_sim)
        HDD = generate_G_from_H(HHDD).to(torch.float32)

        A_tensor = torch.from_numpy(A).to(device)
        AT_tensor = torch.from_numpy(AT).to(device)
        L_tensor = torch.from_numpy(L).to(device)

        threshold = 0.1

        # ncRNA 超图 (3 types)
        NC_1 = HMM.to(device)
        edge_indexNC1 = torch.tensor(np.array(np.where(HMM.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        NC_2 = HMG.to(device)
        edge_indexNC2 = torch.tensor(np.array(np.where(HMG.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        NC_3 = HMD.to(device)
        edge_indexNC3 = torch.tensor(np.array(np.where(HMD.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        # Drug 超图 (3 types)
        D_1 = HDD.to(device)
        edge_indexD1 = torch.tensor(np.array(np.where(HDD.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        D_2 = HDG.to(device)
        edge_indexD2 = torch.tensor(np.array(np.where(HDG.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        D_3 = HDM.to(device)
        edge_indexD3 = torch.tensor(np.array(np.where(HDM.cpu().numpy() > threshold)), dtype=torch.long).to(device)

        # One-hot features
        ncrna_identity = torch.eye(RD.shape[0], device=device)  # (249, 249)
        drug_identity = torch.eye(RD.shape[1], device=device)   # (62, 62)

        return (L_tensor, AT_tensor, A_tensor, HMG, HDG,
                ncrna_identity, drug_identity, HMD, HDM, HMM, HDD,
                NC_1, NC_2, NC_3, D_1, D_2, D_3,
                edge_indexNC1, edge_indexNC2, edge_indexNC3,
                edge_indexD1, edge_indexD2, edge_indexD3)

    (L, AT, A, HMG, HDG, ncrna_feat, drug_feat,
     HMD, HDM, HMM, HDD,
     NC_1, NC_2, NC_3, D_1, D_2, D_3,
     edge_indexNC1, edge_indexNC2, edge_indexNC3,
     edge_indexD1, edge_indexD2, edge_indexD3) = build_hypergraphs()

    pos_weight = float(A.shape[0] * A.shape[1] - A.sum()) / A.sum()

    accuracy_list = []
    mcc_list = []
    loss_list = []

    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        optimizer.zero_grad()

        recover = model(
            NC_1, NC_2, NC_3, D_1, D_2, D_3,
            edge_indexNC1, edge_indexNC2, edge_indexNC3,
            edge_indexD1, edge_indexD2, edge_indexD3,
            ncrna_feat, drug_feat, A, L
        )

        outputs = recover.t().cpu().detach().numpy()
        test_predict = create_resultlist(
            outputs, testset,
            pos_rna_idx, pos_drug_idx,
            Index_zeroRow, Index_zeroCol,
            len(test_pos), zero_length, test_neg
        )

        MA = torch.masked_select(A, train_mask_tensor)
        rec = torch.masked_select(recover.t(), train_mask_tensor)

        loss = F.binary_cross_entropy_with_logits(rec, MA, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)

        print('Epoch: {:04d}, loss: {:.5f}, auc_val: {:.5f}, aupr_val: {:.5f}'
              .format(epoch + 1, loss.item(), auc_val, aupr_val))
        loss_list.append(loss.item())

        max_f1_score, threshold = f1_score_binary(
            torch.from_numpy(label).float(),
            torch.from_numpy(test_predict).float()
        )
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)

        test_pred_tensor = torch.from_numpy(test_predict)
        binary_pred = (test_pred_tensor >= threshold).float()
        accuracy = (binary_pred == torch.from_numpy(label).float()).float().mean().item()
        mcc = matthews_corrcoef(label.ravel(), binary_pred.numpy())

        accuracy_list.append(accuracy)
        mcc_list.append(mcc)

        print(f"max_f1_score: {max_f1_score:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}")

    # Final metrics
    auc1, aupr1 = auc_val, aupr_val
    fpr, tpr, _ = roc_curve(label, test_predict)
    roc_auc = auc(fpr, tpr)

    all_fpr = [fpr]
    all_tpr = [tpr]
    all_auc = [roc_auc]

    return auc1, aupr1, recall, precision, max_f1_score, accuracy, mcc, loss_list, accuracy_list, mcc_list, all_fpr, all_tpr, all_auc, fpr, tpr


# 主函数
if __name__ == '__main__':
    # 加载数据（ncRNA-Drug）
    RD = np.loadtxt("ncRNADRUG(249,62)/association.txt")          # (249, 62)
    RNA_sim = np.loadtxt("ncRNADRUG(249,62)/ncRNA.txt")           # ncRNA similarity
    Drug_sim = np.loadtxt("ncRNADRUG(249,62)/drug.txt")           # Drug similarity
    DG = np.loadtxt("ncRNADRUG(249,62)/GKGIP_drug.txt")           # Drug-Gene
    MG = np.loadtxt("ncRNADRUG(249,62)/GKGIP_ncRNA.txt")          # ncRNA-Gene

    # 数据增强（SVD）
    RNA_sim = DataEnhancement.SVD(RNA_sim)
    Drug_sim = DataEnhancement.SVD(Drug_sim)
    DG = DataEnhancement.SVD(DG)
    MG = DataEnhancement.SVD(MG)

    # 正样本索引
    pos_pairs = np.argwhere(RD == 1)
    pos_rna_idx = pos_pairs[:, 0]    # ncRNA indices
    pos_drug_idx = pos_pairs[:, 1]   # Drug indices

    # 负样本索引
    neg_pairs = np.argwhere(RD == 0)
    Index_zeroRow = neg_pairs[:, 0]
    Index_zeroCol = neg_pairs[:, 1]

    zero_length = len(Index_zeroRow)
    total_associations = len(pos_rna_idx)

    shuffle_data = np.random.permutation(total_associations)

    # 矩阵补全（假设 MF 模块存在）
    try:
        RD2 = MF.run_MC_4(RD, MG, DG)
    except NameError:
        print("Warning: MF not defined. Using original RD as RD2.")
        RD2 = RD.copy()

    # 超参数
    lr = 0.002
    weight_decay = 0.02
    epochs = 120
    k_folds = 5

    # 执行五折交叉验证
    result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_fpr, all_tpr, all_auc, fpr, tpr = cross_validation_5fold(k_folds)

    print("Final CV Result:", result)
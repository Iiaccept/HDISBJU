import numpy as np
import copy
import math
import torch
from torch import nn, optim
import torch.nn.init as init
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpy.core import multiarray
from torch.nn.parameter import Parameter
#以下两句用来忽略版本错误信息
import warnings
# 设置随机数种子
seed = 48
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
torch.backends.cudnn.benchmark = False  #禁用卷积算法选择，确保结果可重复

# 防止 Python hash 随机化影响字典/集合遍历顺序
import os
os.environ['PYTHONHASHSEED'] = str(seed)
warnings.filterwarnings("ignore")
#设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#从超图关联矩阵H计算G,param H: 超图关联矩阵H,param variable_weight: 超边的权重是否可变
def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G
#从超图关联矩阵H计算G,param H: 超图关联矩阵H,param variable_weight: 超边的权重是否可变
def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge) # 超边的权重
    DV = np.sum(H * W, axis=1) # 节点的度
    DE = np.sum(H, axis=0) # 超边的度

    a = []
    for i in DE:
        if i == 0:
            a.append(0)
        else:
            a.append(np.power(i, -1))
    invDE = np.mat(np.diag(a))

    b = []
    for i in DV:
        if i == 0:
            b.append(0)
        else:
            b.append(np.power(i, -0.5))
    DV2 = np.mat(np.diag(b))

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2

    G = torch.Tensor(G)
    return G 

#将H_list中的超边组合并,param H_list: 包含两个或两个以上超图关联矩阵的超边组
#return: 融合后的超图关联矩阵
def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64, dropout=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),  # 比 Tanh 更好
            nn.Linear(hidden_size, 1, bias=False)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.project:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, z):
        # 自动处理输入类型，避免强制 float()

        w = self.project(z)
        beta = torch.softmax(w, dim=1) + 1e-6  # 防止除零或 nan
        beta = beta / beta.sum(dim=1, keepdim=True)  # 再归一化一次保证数值稳定
        beta = self.dropout(beta)
        return (beta * z).sum(1), beta




class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = lambda x: x  # 直接在这里指定恒等激活函数
    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return self.act(z) 

class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = lambda x: x  # 直接在这里指定恒等激活函数

    def forward(self, z_node, z_hyperedge):
        z_node_ = z_node
        z_hyperedge_ = z_hyperedge
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return self.act(z)



#--------------解纠缠图神经网络----------------------------------
#in_dim是每个节点特征向量的长度
class DisenEncoder(nn.Module):
   def __init__(self, in_dim, x_dim=32): #可选;让x_dim可配置
       super(DisenEncoder, self).__init__()
       self.k = 2  #解耦子空间的数量
       self.routit = 3 #动态路由迭代次数
       self.x_dim = x_dim  #每个子空间维度
       self.linear = nn.Linear(in_dim, self.k * self.x_dim)

   def forward(self, x, src_trg):
       # x = x.to(torch.float32).to(device)
       # src_trg = src_trg.to(torch.long).to(device)
       x = self.linear(x)
       m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
       n, d = x.shape
       k, delta_d = self.k, d // self.k

       x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
       z = x[src].view(m, k, delta_d)  # neighbors' feature
       c = x  # node-neighbor attention aspect factor
       for t in range(self.routit):
           p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)  # update node-neighbor attention aspect factor
           p = F.softmax(p, dim=1)
           p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)
           weight_sum = (p * z).view(m, d)  # weight sum (node attention * neighbors feature)
           c = c.index_add_(0, trg, weight_sum)  # update output embedding

           c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)  # embedding normalize aspect factor
       return c


#---------------------------分而治之-联合更新--------------------------------
def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    """
    带 mask 的 softmax 函数
    :param vec: tensor of shape (N, M)
    :param mask: tensor of shape (N, M), dtype=bool or 0/1
    :param dim: softmax 维度
    :return: softmax 后的结果
    """
    mask = mask.float()
    masked_vec = vec * mask + (mask == 0).float() * (-1e9)  # 把不参与计算的位置设为极小值
    return F.softmax(masked_vec, dim=dim)

class BidirectionalJointUpdater(nn.Module):
    def __init__(self, embed_dim):
        super(BidirectionalJointUpdater, self).__init__()
        # 共享线性变换层
        self.shared_linear = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, user_emb, item_emb, adj_matrix=None):
        """
        :param user_emb: [N, D]
        :param item_emb: [M, D]
        :param adj_matrix: [N, M] 用户-物品交互矩阵（0或1），可选
        :return: 更新后的 user_emb 和 item_emb，都融合了对方的信息
        """
        # 1. 线性变换 + 激活
        transformed_user = self.act(self.shared_linear(user_emb))  # [N, D]
        transformed_item = self.act(self.shared_linear(item_emb))  # [M, D]

        # 如果没有传入 adj_matrix，则默认全连接
        if adj_matrix is None:
            adj_matrix = torch.ones((user_emb.size(0), item_emb.size(0)), device=user_emb.device)
        #2. 用户聚合物品信息
        user_item_attn = torch.matmul(user_emb, item_emb.t())  # [N, M]
        user_mask = adj_matrix > 0  # bool mask
        user_weights = masked_softmax(user_item_attn, user_mask, dim=-1)  # [N, M]
        user_context = torch.matmul(user_weights, item_emb)  # [N, D]
        updated_user = transformed_user + user_context  # 融合原始+上下文
        # 3. 物品聚合用户信息
        item_user_attn = torch.matmul(item_emb, user_emb.t())  # [M, N]
        item_mask = adj_matrix.t() > 0  # transpose for item -> user mask
        item_weights = masked_softmax(item_user_attn, item_mask, dim=-1)  # [M, N]
        item_context = torch.matmul(item_weights, user_emb)  # [M, D]
        updated_item = transformed_item + item_context  # 融合原始+上下文
        return updated_user, updated_item







# 修改后的模型，包含多头注意力
class Gai_HGNN(nn.Module):
    def __init__(self, num_in_node = 129, num_in_edge = 48, num_hidden1 = 128, num_out=128):
        super(Gai_HGNN, self).__init__()

        #解纠缠编码器
        self.hgnn_encoder1 = DisenEncoder(599)  # 只传前三个参数
        self.hgnn_encoder2 = DisenEncoder(91)  # 只传前三个参数
        # 注意力机制
        self.attention = Attention(64)
        #联合更新JointMessageUpdating（里面的参数是输入特征维度）
        self.Joint= BidirectionalJointUpdater(64)  # 只传前三个参数
        self.decoder1 = decoder1()

    def forward(self, NC_1, NC_2, NC_3, D_1,D_2, D_3, edge_indexNC1, edge_indexNC2, edge_indexNC3, edge_indexD1, edge_indexD2, edge_indexD3, heterogeneous, heterogeneous1,A, L):

        # 基于ncRNA和drug序列信息
        nc_feature_1 = self.hgnn_encoder1(NC_1, edge_indexNC1)#599*256
        d_feature_1 = self.hgnn_encoder2(D_1, edge_indexD1) #91*256

        #基于ncRNA和drug高斯核相似性信息
        nc_feature_2 = self.hgnn_encoder1(NC_2, edge_indexNC2)#599*256
        d_feature_2 = self.hgnn_encoder2(D_2, edge_indexD2) #91*256

        #基于ncRNA和drug关联矩阵信息
        nc_feature_3 = self.hgnn_encoder1(NC_3, edge_indexNC3)#599*256
        d_feature_3 = self.hgnn_encoder2(D_3, edge_indexD3) #91*256

        #注意力机制融合ncRNA信息
        emb1 = torch.stack([nc_feature_1, nc_feature_2, nc_feature_3 ], dim=1)
        emb1, att = self.attention(emb1)
        nc = emb1

        #注意力机制融合drug信息
        emb2 = torch.stack([ d_feature_1, d_feature_2, d_feature_3], dim=1)
        emb2, att = self.attention(emb2)
        d = emb2

        #联合更新
        nc, d = self.Joint(nc, d, A)
        reconstructionG = self.decoder1(d, nc)
        #对图使用注意力机制
        recover = reconstructionG
        return recover


def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):
    result_list = np.zeros((test_length_p+len(test_f), 1))
    for i in range(test_length_p):
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    return result_list

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
#要计算两组向量之间的余弦相似度矩阵
def cosine_similarity_matrix(h1, h2):
    h1_norm = F.normalize(h1, p=2, dim=1)
    h2_norm = F.normalize(h2, p=2, dim=1)
    return torch.mm(h1_norm, h2_norm.t())
#使用余弦相似度作为相似度测量来计算对比损失。
# temperature=0.7
def compute_contrastive_loss(h1, h2, temperature=0.7):
    sim_mat = cosine_similarity_matrix(h1, h2)
    # Scale the similarity by temperature
    sim_mat_scaled = torch.exp(sim_mat / temperature)
    # Compute the loss
    positive_pairs = torch.diag(sim_mat_scaled)
    all_pairs_sum = torch.sum(sim_mat_scaled, dim=1)
    # Avoid division by zero
    all_pairs_sum = torch.clamp(all_pairs_sum, min=1e-9)
    contrastive_loss = -torch.log(positive_pairs / all_pairs_sum).mean()
    return contrastive_loss
'''
这个版本的代码将相似度计算和对比损失计算拆分成了两个不同的函数，这有助于提高代码的可读性和复用性。cosine_similarity_matrix函数负责计算两组向量之间的余弦相似度矩阵，而compute_contrastive_loss函数则使用这个相似度矩阵来计算对比损失。
此外，我在计算对比损失时添加了一个小的数值（1e-9）到分母中，以避免可能的除以零错误。 这是处理浮点数计算中常见的数值稳定性问题的一种常见做法。
'''
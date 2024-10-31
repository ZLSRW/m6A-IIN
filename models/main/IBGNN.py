"""""
(1) 输入序列的两种表示（初始化表示、序列二级结构图），构建序列的两种图,以及序列的初始表示。在生物序列中，则一个为原始序列对应的图，一个为二级结构序列对应的图。在其他领域，可以类似对比学习，将两种图认为是两种增广形式。（在另一个文件完成，需要满足对称，并尽可能实现归一化）
(2) 设计两个权重矩阵，以哈达玛乘积的形式对生成的图进行池化。（Confidence Pooling）。权重矩阵需要保留，一方面用于motif或pattern的提取（保留高权重），另一方面用于图卷积的可逆，进行图的还原。
(3) 基于图获得两种拉普拉斯算子（D-W），不进行归一化，归一化操作尝试在图化的环节完成。
(4) 图结构和序列表示的对齐（或者称为双图对齐，切比雪夫近似，使用三阶）。接收一种序列表示，作为节点表示（one-hot或cgr或transformer），进行谱图卷积。+可逆
(5) 表示的孪生耦合。+可逆

部分类缺乏super函数，可能会报错
"""""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from .GraphInteraction import *
from .configure import *
from .Mamba import *

from sklearn.preprocessing import normalize
import numpy as np
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from .Graph_Wavelet import *
from scipy.integrate import quad
import time
from .ELMom6A import *


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# class 1: 拉普拉斯矩阵生成（池化+（D-W））+可逆
# class 2: 空间对齐（切比雪夫近似） +可逆
# class 3: 孪生-耦合 +可逆

class PoolingLayer(nn.Module):
    # def __init__(self, size, batch, device="cuda:0"):
    def __init__(self, size, device="cuda:0"):
        super(PoolingLayer, self).__init__()
        self.size = size
        # self.batch = batch
        # self.weight = nn.Parameter(torch.rand(size=(self.batch, self.size, self.size)))
        self.weight = nn.Parameter(torch.rand(size=(1, self.size, self.size)))
        self.to(device)

        return

    def forward(self, A):
        A = torch.mul(self.weight, A)
        A = 0.5 * (A + A.permute(0, 2, 1))  # 可逆部分需要写出这部分的计算
        # print(self.weight)
        return A


class TransposeHook:  # 保证在更新梯度时，U和L的权重互为转置，这样的操作可能会影响模型的训练和收敛性
    def __init__(self, source_layer):
        self.source_layer = source_layer

    def __call__(self, grad):
        self.source_layer.weight.data.copy_(self.source_layer.weight.data.permute(0, 2, 1))
        return grad



"""
在正向过程中，输入的参数分别为：1）二级结构构成的图【b,41,41】;2)RNA序列的类型+理化性质表示【b,41,N+D】;3) RNA序列的ELMo表示【b,41,256】。
在逆向过程中，输入的参数分别为：1）归纳偏置项；2）类型+表征的潜在空间

在该网络中需要实现的结构：1）图级别的交互（结构图对序列表示的加权；序列表示构图对二级结构的加权），输出一个归纳偏置项和表征空间；2）图小波神经网络（实际是基于归纳偏置项计算小波基），输出小波基矩阵；
3）Mamba（对RNA的顺序关系进行建模，加入一些任务的引导信息）：输出形状不变的序列表示；4）小波基和表示进行乘积，得到序列表示；添加缩放层（池化），识别。

逆向过程：主要是要引入可选择的缩放系数。

"""


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class ScalingLayer(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(41, data_dim, requires_grad=True))

    def forward(self, x):  # x为【b,41,256】
        scaling_coff = torch.exp(self.log_scale_vector)
        # print(type(scaling_coff))
        x = scaling_coff.unsqueeze(0) * x
        return x, scaling_coff


class InvertibleBigraphNeuralNetwork(nn.Module):  # 当前为一层的卷积过程,默认均为三维张量
    def __init__(self, num, size, batch, batch1, device="cuda:0"):
        super(InvertibleBigraphNeuralNetwork, self).__init__()

        self.size = size
        self.batch = batch
        self.batch1 = batch1
        self.num = num
        self.device = device
        self.args = ModelArgs(d_model=271, n_layer=4, vocab_size=271)

        self.graph_Interaction = Graph_Interaction(num, batch, size)

        self.mamba = Mamba(self.args)

        self.bn = nn.BatchNorm1d(41)  # 表征的标准化
        self.bnSR = nn.BatchNorm1d(15)  # 表征的标准化
        self.bn1 = nn.BatchNorm1d(271)  # 表征的标准化
        self.scalingLayer = ScalingLayer(271)

        self.graph_wavelet_layer=GraphWaveletNeuralNetwork(node_cnt=41,feature_dims=271,hidden_dims=271,dropout_rate=0.2)

        self.shape1 = nn.Sequential(
            nn.Linear(271, 271),
            # nn.Tanh(),
            nn.ReLU(),  # 权重
            nn.Linear(271, 271),
        )

        self.scalingWeight = nn.Sequential(
            nn.Linear(41, 41),
            # nn.Softmax(),
            nn.Sigmoid(),
            # nn.PReLU(),  # 权重
        )

        self.scalingWeight_SR = nn.Sequential(
            nn.Linear(15, 15),
            # nn.Softmax(),
            nn.Sigmoid(),
            # nn.PReLU(),  # 权重
        )

        self.prob = nn.Sequential(
            nn.Linear(271, 1),
            # nn.Softmax(),  # 权重
            nn.Sigmoid(),  # 权重
        )

        self.gru = nn.GRU(271, 271)

        self.semantics=Semantic_network()

        self.to(device)

    def laplacian_multi(self, W):  # 输入的是三维张量

        # hard_attention = max_min(W)
        hard_attention = W
        degree = torch.sum(hard_attention, dim=-1)
        hard_attention = 0.5 * (hard_attention + hard_attention.permute(0, 2, 1))  # 局部矩阵的
        degree_l = tensor_diag(degree)  # 度的对角矩阵 32x12x512x512
        diagonal_degree_hat = tensor_diag(1 / (torch.sqrt(degree) + 1e-6))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - hard_attention, diagonal_degree_hat))
        return laplacian

    def fourier(self, L, algo='eigh', k=100):
        """Return the Fourier basis, i.e. the EVD of the Laplacian."""

        L_array = L.cpu().detach().numpy()

        def sort(lamb, U):
            idx = lamb.argsort()
            return lamb[idx], U[:, idx]

        all_lamb = np.array([])
        all_U = np.array([])

        if algo is 'eig':
            for i in range(L_array.shape[0]):
                lamb, U = np.linalg.eig(L_array[i])
                lamb, U = sort(lamb, U)
                np.append(all_lamb, lamb)
                np.append(all_U, U)
        elif algo is 'eigh':
            temp_list_lamb = []
            temp_list_U = []
            for i in range(L_array.shape[0]):
                lamb, U = np.linalg.eigh(L_array[i])
                lamb, U = sort(lamb, U)
                temp_list_lamb.append(lamb)
                temp_list_U.append(U)
                all_lamb = np.stack(temp_list_lamb, axis=0)
                all_U = np.stack(temp_list_U, axis=0)
        elif algo is 'eigs':
            lamb, U = sp.linalg.eigs(L, k=k, which='SM')
            lamb, U = sort(lamb, U)
        elif algo is 'eigsh':
            lamb, U = sp.linalg.eigsh(L, k=k, which='SM')

        return all_lamb, all_U

    def largest_lamb(self, L, k=1):
        """Return the Fourier basis, i.e. the EVD of the Laplacian."""

        L_array = L.cpu().detach().numpy()

        temp_list_lamb = []
        for i in range(L_array.shape[0]):
            lamb, U = sp.linalg.eigsh(L_array[i], k=k, which='LM')
            temp_list_lamb.append(lamb[0])
        all_lamb=np.array(temp_list_lamb)

        return all_lamb

    def weight_wavelet(self, s, lamb, U):

        s = s
        for i in range(len(lamb)):
            for j in range(len(lamb[0])):
                lamb[i][j] = math.exp(-lamb[i][j] * s)

        lamb = torch.exp(-lamb * s)

        lamb_diag = torch.diag_embed(lamb)

        Weight = torch.matmul(torch.matmul(U, lamb_diag), U.permute(0, 2, 1))

        return Weight

    def weight_wavelet_inverse(self, s, lamb, U):
        s = s
        for i in range(len(lamb)):
            for j in range(len(lamb[0])):
                lamb[i][j] = math.exp(-lamb[i][j] * s)

        lamb = torch.exp(lamb * s)

        lamb_diag = torch.diag_embed(lamb)

        Weight = torch.matmul(torch.matmul(U, lamb_diag), U.permute(0, 2, 1))

        return Weight

    def wavelet_basis(self, s, lamb, U, threshold):

        lamb, U = (torch.from_numpy(lamb)).float().to(self.device), (
            torch.from_numpy(U)).float().to(self.device)

        Weight = self.weight_wavelet(s, lamb, U)
        inverse_Weight = self.weight_wavelet_inverse(s, lamb, U)

        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

        Weight = F.normalize(Weight, p=1, dim=2)
        inverse_Weight = F.normalize(inverse_Weight, p=1, dim=2)

        return Weight, inverse_Weight


    def fast_wavelet_basis(self, adj, s, threshold, m):

        L=self.laplacian_multi(adj)
        lamb = self.largest_lamb(L)

        a = lamb / 2
        c = []
        inverse_c = []
        for j in range(len(a)):
            Temp_c = []
            Temp_inverse_c = []
            for i in range(m + 1): #
                f_res = 2 * np.exp(s * a[j]) * iv(i, s * a[j])
                inverse_f_res = 2 * np.exp(-s * a[j]) * iv(i, -s * a[j])
                Temp_c.append(f_res)
                Temp_inverse_c.append(inverse_f_res)
            c.append(Temp_c)
            inverse_c.append(Temp_inverse_c)

        c=torch.tensor(c).unsqueeze(-1).unsqueeze(-1).to(self.device)
        inverse_c=torch.tensor(inverse_c).unsqueeze(-1).unsqueeze(-1).to(self.device)

        L=L.to(self.device)

        L_cheb=self.cheb_polynomial_multi(L).to(self.device) # 切比雪夫四阶近似 (706，4,41,41)

        Weight = torch.sum(c * L_cheb, dim=1)

        # 计算 inverse_Weight
        inverse_Weight = torch.sum(inverse_c * L_cheb, dim=1)

        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

        Weight = F.normalize(Weight, p=1, dim=2)
        inverse_Weight = F.normalize(inverse_Weight, p=1, dim=2)

        return Weight.float(),inverse_Weight.float()

    def cheb_polynomial_multi(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子
        # print('laplacian.shape '+str(laplacian.shape)) #torch.Size([145, 41, 41])
        bat, N, N = laplacian.size()  # [N, N] 512
        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=1)

        return multi_order_laplacian  # bx4x41x41

    def reconstruction_loss(self, rev_input, input):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(rev_input, input)
        return loss.to(self.device)

    def forward(self, G, Loop_G, RNA_OP, RNA_ELMo, invertible):
        if not invertible:
            G_output, SR = self.graph_Interaction(G, Loop_G, RNA_OP, invertible)  # 图级别交互

            # 小波基计算（不涉及权重，写在本类里就行）, 小波基计算需要改改
            # laplacian = self.laplacian_multi(G_output)
            # all_lamb, all_U = self.fourier(laplacian)
            # wavelets, wavelet_inv = self.wavelet_basis(0.8, all_lamb, all_U, 1e-4) #计算一次需要6秒

            wavelets, wavelet_inv = self.fast_wavelet_basis(G_output,0.3,1e-4,3) #计算一次需要6秒

            # multi_order_laplacian = self.cheb_polynomial_multi(laplacian)
            # multi_order_laplacian = torch.sum(multi_order_laplacian, dim=1)

            # SR+RNA_ELMo,输入Mamba
            type_structure_tensor = torch.cat([SR, RNA_ELMo], dim=2)  # 有没有可能在ssm模型中为每一个SSM
            # type_structure_tensor = RNA_ELMo # 有没有可能在ssm模型中为每一个SSM
            type_structure_tensor = type_structure_tensor.float()

            type_structure_tensor=self.semantics(type_structure_tensor)

            # type_structure_tensor, scaling_coff = self.mamba(type_structure_tensor)  # 带缩放层的mamba

            # type_structure_tensor, scaling_coff = self.mamba(type_structure_tensor)

            # 小波图卷积
            # final_representation = torch.matmul(multi_order_laplacian, type_structure_tensor)  # 最终表示(b,41,271)
            final_representation = self.graph_wavelet_layer(type_structure_tensor,wavelets,wavelet_inv)

            # GRU
            # final_representation, _ = self.gru(final_representation)
            # final_representation= self.bn1(final_representation.permute(0,2,1)).permute(0,2,1)
            final_representation, scaling_coff = self.scalingLayer(final_representation)


            # final_representation, scaling_coff=self.mamba(final_representation)

            # 引入归纳偏置，执行逆向过程，以计算重建损失
            scaling_weight = torch.matmul(scaling_coff, scaling_coff.permute(1, 0))
            scaling_weight=self.scalingWeight(self.bn(scaling_weight))
            scaling_weight=(scaling_weight+scaling_weight.permute(1,0))/2.0
            scaling_weight[scaling_weight<0.5]=0.0
            # print(scaling_weight)

            scaling_coff_SR = self.scalingWeight_SR(self.bnSR(scaling_coff[:, :15]))
            # scaling_coff_SR[scaling_coff_SR<0.5]=0.0

            Inductive_graph = scaling_weight * G_output  # 这部分要改改，找一个合适的方式
            G_hat, SR_hat = self.graph_Interaction(Inductive_graph, Loop_G, scaling_coff_SR*SR, True)  # 图级别交互的逆过程

            # 分别为重建损失和缩放损失
            r_loss = (self.reconstruction_loss(G_hat, G) + self.reconstruction_loss(SR_hat, RNA_OP)) / (
                        2 * G_output.shape[0])
            scaling_Loss = torch.sum(scaling_weight) / G_output.shape[0]  # 缩放损失

            final_representation = self.shape1(final_representation)

            sequence_feature = self.bn1(torch.mean(final_representation, dim=1))
            # 识别过程（标准化过个全连接层，再接个分类器）
            scores = self.prob(sequence_feature.unsqueeze(1))
            # scores = self.prob(torch.sum(final_representation, dim=1).unsqueeze(1))

            return sequence_feature, scores, r_loss, scaling_Loss

        return G, G

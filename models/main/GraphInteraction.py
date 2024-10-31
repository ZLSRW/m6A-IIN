import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from .configure import *


class Cross_Structure_Perception(nn.Module):  # 对序列的进一步处理，因为是前向传播过程所以并不需要可逆
    def __init__(self, batch, unit, device="cuda:0"):
        super(Cross_Structure_Perception, self).__init__()
        self.batch = batch
        self.unit = unit

        self.relu = nn.LeakyReLU()

        self.bn = nn.BatchNorm1d(self.unit)

        self.bn1 = nn.BatchNorm1d(15)  # 表征的标准化

        # 对输入的所有批次，均采用相同的注意力机制
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_value = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))

        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_value.data, gain=1.414)
        self.to(device)

    def Representation_mapping(self, R):  # 由序列表示为图传递信息

        # print(R.shape)

        R = self.bn1(R.permute(0,2,1)).permute(0,2,1)

        A = torch.matmul(R, R.permute(0, 2, 1))

        key = torch.matmul(A, self.weight_key)
        query = torch.matmul(A, self.weight_query)
        value = torch.matmul(A, self.weight_value)

        data = torch.matmul(query, key.permute(0, 2, 1))
        data = self.bn(data)
        data = self.relu(data)  # 暂时去掉
        coefficient = F.softmax(data, dim=2)
        Perception = coefficient @ A @ value
        # print('Perception[0] '+str(Perception[0]))

        # 标准化、mask、对称归一化 (去掉)
        Perception = normalize_and_symmetrize_tensor(Perception, 0.5)
        # print('Perception[0] '+str(Perception[0]))

        return Perception

    def forward(self, A):  # 希望A1,A2分别为对称归一化矩阵
        A=A.float()
        # print(type(A))
        Perception = self.Representation_mapping(A)
        A = 0.5 * (Perception + Perception.permute(0, 2, 1))  # 严格保持对称归一化
        return A


class loop_learning(nn.Module):  # 基于二级结构图捕获环等有用的结构，并将其传递给RNA序列
    def __init__(self, batch, unit, device="cuda:0"):
        super(loop_learning, self).__init__()
        self.batch = batch
        self.unit = unit

        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(self.unit)
        self.bn1 = nn.BatchNorm1d(15)  # 表征的标准化

        # 定义一个可学习权重
        self.weight_loop = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        nn.init.xavier_uniform_(self.weight_loop.data, gain=1.414)

        # 将核函数的loop球面映射为表示
        self.loop_representation = nn.Sequential(
            nn.Linear(41, 41),
            # nn.LeakyReLU(),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(41, 15),
        )

        self.to(device)

    '''
    1. 依据输入的矩阵获得潜在的环矩阵
    2. 环矩阵和可学习权重进行哈达玛乘积
    3. 引入径向核函数进行重映射
    4. 对映射矩阵按行相加进行降维，得到映射权重
    '''

    def Structure_mapping(self, LoopA):  # 由序列表示为图传递信息
        sigma = 0.5

        LoopA=LoopA+LoopA.permute(0,2,1)

        filtered_matrix = self.weight_loop * LoopA

        loop_matrix = torch.exp(-0.5 * (filtered_matrix / sigma) ** 2)  # 期待是【b,41,41】

        loop_representation=self.bn1(self.loop_representation(loop_matrix).permute(0,2,1)).permute(0,2,1)  # 表示获取

        # loop_weight = torch.mean(loop_matrix, dim=2)

        return loop_representation  # 期待是【b，41,256】 使用（onehot加理化性质则为4+d）

    def forward(self, LoopA):  # 希望A1,A2分别为对称归一化矩阵

        loop_representation=self.Structure_mapping(LoopA)

        return loop_representation


class GraphCouplingLayer(nn.Module):
    def __init__(self, batch, size, device="cuda:0"):
        super(GraphCouplingLayer, self).__init__()
        self.batch = batch
        self.size = size

        self.informationLoop = loop_learning(self.batch, self.size)
        self.informationTrans = Cross_Structure_Perception(self.batch, self.size)


    def forward(self, A1, LoopA1, A2, invertible):
        if not invertible:
            H1 = A1
            H2 = A2 + self.informationLoop(LoopA1)  # 为RNA序列引入结构知识

            B1 = H1 + self.informationTrans(H2)  # 为图结构引入潜在关系
            B2 = H2

            return torch.tensor(B1), torch.tensor(B2)

        H2 = A2
        H1 = A1 - self.informationTrans(H2)


        B2 = H2 - self.informationLoop(LoopA1)
        B1 = H1

        return torch.tensor(B1), torch.tensor(B2)


class Graph_Interaction(nn.Module):
    def __init__(self, num, batch, size, device="cuda:0"):
        super(Graph_Interaction, self).__init__()
        self.batch = batch
        self.size = size
        self.coupling = GraphCouplingLayer(self.batch, self.size)
        self.Graph_Coupling_Layers = nn.ModuleList()
        for _ in range(num):
            self.Graph_Coupling_Layers.append(self.coupling)
        self.to(device)

    def forward(self, A1, LoopA1, A2, invertible):  # A1为二级结构构图，A2为RNA序列对应的类型和理化性质表示

        for i, layer in enumerate(self.Graph_Coupling_Layers):
            B1, B2 = layer(A1, LoopA1, A2, invertible)

        return B1, B2

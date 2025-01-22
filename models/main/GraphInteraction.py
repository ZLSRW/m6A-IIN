import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from .configure import *


class Cross_Structure_Perception(nn.Module):
    def __init__(self, batch, unit, device="cuda:0"):
        super(Cross_Structure_Perception, self).__init__()
        self.batch = batch
        self.unit = unit

        self.relu = nn.LeakyReLU()

        self.bn = nn.BatchNorm1d(self.unit)

        self.bn1 = nn.BatchNorm1d(15)  # 表征的标准化

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_value = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))

        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_value.data, gain=1.414)
        self.to(device)

    def Representation_mapping(self, R):

        R = self.bn1(R.permute(0,2,1)).permute(0,2,1)

        A = torch.matmul(R, R.permute(0, 2, 1))

        key = torch.matmul(A, self.weight_key)
        query = torch.matmul(A, self.weight_query)
        value = torch.matmul(A, self.weight_value)

        data = torch.matmul(query, key.permute(0, 2, 1))
        data = self.bn(data)
        data = self.relu(data)
        coefficient = F.softmax(data, dim=2)
        Perception = coefficient @ A @ value

        Perception = normalize_and_symmetrize_tensor(Perception, 0.5)

        return Perception

    def forward(self, A):
        A=A.float()
        Perception = self.Representation_mapping(A)
        A = 0.5 * (Perception + Perception.permute(0, 2, 1))
        return A


class loop_learning(nn.Module):
    def __init__(self, batch, unit, device="cuda:0"):
        super(loop_learning, self).__init__()
        self.batch = batch
        self.unit = unit

        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(self.unit)
        self.bn1 = nn.BatchNorm1d(15)

        self.weight_loop = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        nn.init.xavier_uniform_(self.weight_loop.data, gain=1.414)

        self.loop_representation = nn.Sequential(
            nn.Linear(41, 41),
            nn.Tanh(),
            nn.Linear(41, 15),
        )

        self.to(device)

    def Structure_mapping(self, LoopA):
        sigma = 0.5

        LoopA=LoopA+LoopA.permute(0,2,1)

        filtered_matrix = self.weight_loop * LoopA

        loop_matrix = torch.exp(-0.5 * (filtered_matrix / sigma) ** 2)

        loop_representation=self.bn1(self.loop_representation(loop_matrix).permute(0,2,1)).permute(0,2,1)  # 表示获取

        return loop_representation

    def forward(self, LoopA):

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
            H2 = A2 + self.informationLoop(LoopA1)

            B1 = H1 + self.informationTrans(H2)
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

    def forward(self, A1, LoopA1, A2, invertible):

        for i, layer in enumerate(self.Graph_Coupling_Layers):
            B1, B2 = layer(A1, LoopA1, A2, invertible)

        return B1, B2

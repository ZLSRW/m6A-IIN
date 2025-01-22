import torch
import torch.nn as nn
import torch.nn.functional as F

from .GraphInteraction import *
from .configure import *
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from .Graph_Wavelet import *
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

class PoolingLayer(nn.Module):
    def __init__(self, size, device="cuda:0"):
        super(PoolingLayer, self).__init__()
        self.size = size
        self.weight = nn.Parameter(torch.rand(size=(1, self.size, self.size)))
        self.to(device)

        return

    def forward(self, A):
        A = torch.mul(self.weight, A)
        A = 0.5 * (A + A.permute(0, 2, 1))
        return A


class TransposeHook:
    def __init__(self, source_layer):
        self.source_layer = source_layer

    def __call__(self, grad):
        self.source_layer.weight.data.copy_(self.source_layer.weight.data.permute(0, 2, 1))
        return grad


class ScalingLayer(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(41, data_dim, requires_grad=True))

    def forward(self, x):
        scaling_coff = torch.exp(self.log_scale_vector)
        x = scaling_coff.unsqueeze(0) * x
        return x, scaling_coff


class InvertibleBigraphNeuralNetwork(nn.Module):
    def __init__(self, num, size, batch, batch1, device="cuda:0"):
        super(InvertibleBigraphNeuralNetwork, self).__init__()

        self.size = size
        self.batch = batch
        self.batch1 = batch1
        self.num = num
        self.device = device

        self.graph_Interaction = Graph_Interaction(num, batch, size)

        self.bn = nn.BatchNorm1d(41)
        self.bnSR = nn.BatchNorm1d(15)
        self.bn1 = nn.BatchNorm1d(271)
        self.scalingLayer = ScalingLayer(271)

        self.graph_wavelet_layer=GraphWaveletNeuralNetwork(node_cnt=41,feature_dims=271,hidden_dims=271,dropout_rate=0.2)

        self.shape1 = nn.Sequential(
            nn.Linear(271, 271),
            nn.ReLU(),
            nn.Linear(271, 271),
        )

        self.scalingWeight = nn.Sequential(
            nn.Linear(41, 41),
            nn.Sigmoid(),
        )

        self.scalingWeight_SR = nn.Sequential(
            nn.Linear(15, 15),
            nn.Sigmoid(),
        )

        self.prob = nn.Sequential(
            nn.Linear(271, 1),
            nn.Sigmoid(),
        )

        self.gru = nn.GRU(271, 271)

        self.semantics=Semantic_network()

        self.to(device)

    def laplacian_multi(self, W):

        hard_attention = W
        degree = torch.sum(hard_attention, dim=-1)
        hard_attention = 0.5 * (hard_attention + hard_attention.permute(0, 2, 1))
        degree_l = tensor_diag(degree)
        diagonal_degree_hat = tensor_diag(1 / (torch.sqrt(degree) + 1e-6))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - hard_attention, diagonal_degree_hat))
        return laplacian

    def fourier(self, L, algo='eigh', k=100):

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

        L_array = L.cpu().detach().numpy()
        temp_list_lamb = []
        for i in range(L_array.shape[0]):
            lamb, U = sp.linalg.eigsh(L_array[i], k=k, which='LM')
            temp_list_lamb.append(lamb[0])
        all_lamb=np.array(temp_list_lamb)

        return all_lamb

    def weight_wavelet(self, s, lamb, U):
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
            for i in range(m + 1):
                f_res = 2 * np.exp(s * a[j]) * iv(i, s * a[j])
                inverse_f_res = 2 * np.exp(-s * a[j]) * iv(i, -s * a[j])
                Temp_c.append(f_res)
                Temp_inverse_c.append(inverse_f_res)
            c.append(Temp_c)
            inverse_c.append(Temp_inverse_c)

        c=torch.tensor(c).unsqueeze(-1).unsqueeze(-1).to(self.device)
        inverse_c=torch.tensor(inverse_c).unsqueeze(-1).unsqueeze(-1).to(self.device)

        L=L.to(self.device)

        L_cheb=self.cheb_polynomial_multi(L).to(self.device)

        Weight = torch.sum(c * L_cheb, dim=1)
        inverse_Weight = torch.sum(inverse_c * L_cheb, dim=1)

        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

        Weight = F.normalize(Weight, p=1, dim=2)
        inverse_Weight = F.normalize(inverse_Weight, p=1, dim=2)

        return Weight.float(),inverse_Weight.float()

    def cheb_polynomial_multi(self, laplacian):
        bat, N, N = laplacian.size()
        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=1)

        return multi_order_laplacian

    def reconstruction_loss(self, rev_input, input):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(rev_input, input)
        return loss.to(self.device)

    def forward(self, G, Loop_G, RNA_OP, RNA_ELMo, invertible):
        if not invertible:
            G_output, SR = self.graph_Interaction(G, Loop_G, RNA_OP, invertible)
            wavelets, wavelet_inv = self.fast_wavelet_basis(G_output,0.3,1e-4,3)

            type_structure_tensor = torch.cat([SR, RNA_ELMo], dim=2)
            type_structure_tensor = type_structure_tensor.float()
            type_structure_tensor=self.semantics(type_structure_tensor)

            final_representation = self.graph_wavelet_layer(type_structure_tensor,wavelets,wavelet_inv)

            final_representation, scaling_coff = self.scalingLayer(final_representation)

            scaling_weight = torch.matmul(scaling_coff, scaling_coff.permute(1, 0))
            scaling_weight=self.scalingWeight(self.bn(scaling_weight))
            scaling_weight=(scaling_weight+scaling_weight.permute(1,0))/2.0
            scaling_weight[scaling_weight<0.5]=0.0

            scaling_coff_SR = self.scalingWeight_SR(self.bnSR(scaling_coff[:, :15]))

            Inductive_graph = scaling_weight * G_output
            G_hat, SR_hat = self.graph_Interaction(Inductive_graph, Loop_G, scaling_coff_SR*SR, True)

            r_loss = (self.reconstruction_loss(G_hat, G) + self.reconstruction_loss(SR_hat, RNA_OP)) / (
                        2 * G_output.shape[0])
            scaling_Loss = torch.sum(scaling_weight) / G_output.shape[0]

            final_representation = self.shape1(final_representation)
            sequence_feature = self.bn1(torch.mean(final_representation, dim=1))
            scores = self.prob(sequence_feature.unsqueeze(1))

            return sequence_feature, scores, r_loss, scaling_Loss

        return G, G


from sklearn.metrics import roc_curve, auc, precision_recall_curve

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def safe_elementwise_divide(mat1, mat2):

    mat2_nonzero = mat2.clone()
    mat2_nonzero[mat2_nonzero == 0] = 1.0
    result = mat1 / mat2_nonzero
    mat2=mat2.repeat(mat1.size(0),1,1)
    result[mat2 == 0] = 0.0

    return result


def solve_for_A(B, x):

    batch_size = B.size(0)
    A = torch.empty(batch_size, x.size(-2), x.size(-2))
    for i in range(batch_size):
        A[i] = torch.abs(torch.matmul(B[i],torch.pinverse(x[i])))

    return A


def remove_last_element(input_list):
    new_list = []
    for i in range(len(input_list)):
        new_2d_list = []
        for j in range(len(input_list[i])):
            new_2d_list.append(input_list[i][j][:-1])
        new_list.append(new_2d_list)

    return new_list


def get_zero_grad_hook(mask, device="cuda:0"):
    def hook(grad):
        return grad * mask.to(device)
    return hook


def Prediction_label(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    predlabels=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    predlabels[i][j][k]=1.0
                else:
                    predlabels[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(predlabels)


def Zero_One_Mask(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    pred_arry[i][j][k]=1.0
                else:
                    pred_arry[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(pred_arry)

def Index_Mask(pred,inputs_site,scale=0.5):
    a, b, c = pred.size()
    d=math.ceil(c*scale)

    a1,b1,c1 = inputs_site.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    pred_mask = np.zeros((a, b, d))
    pred_mask_index = np.zeros((a, b, d))
    inputs_site_mask = np.zeros((a1, b1, d))

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            counter=0
            index=0
            while k < c:
                if inputs_site[i][j][k] == 1.0 and counter<=3:
                    counter+=1
                    pred_mask_index[i][j][index] = k
                    pred_mask[i][j][index]=pred_arry[i][j][k]
                    inputs_site_mask[i][j][index]=inputs_site[i][j][k]
                    index+=1
                k += 1
            while index < d:
                n=0
                while n < len(inputs_site[i][j]):
                    if inputs_site[i][j][n]==0.0:
                        pred_mask_index[i][j][index] = n
                        pred_mask[i][j][index] = pred_arry[i][j][n]
                        inputs_site_mask[i][j][index] = inputs_site[i][j][n]
                        index += 1
                        break
                    n+=1



            j += 1
        i += 1

    return torch.from_numpy(pred_mask_index).float(),torch.from_numpy(pred_mask).float(),torch.from_numpy(inputs_site_mask) .float()#下标，概率，位点与否

def Sequence_reduction(pred,inputs_site,ID_len):
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()
    prediction=[]
    label=[]
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0

            while k < int(ID_len_array[i][j][1]):
                temp_id_pre = []
                temp_id_lab = []
                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label

def Sequence_reduction1(pred,inputs_site,ID_len): #将窗口中的转态全部转换为二维列表，左边为状态，右边为标签
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()

    prediction=[]
    label=[]
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            while k < c:
                temp_id_pre = []
                temp_id_lab = []

                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label


def tensor_diag(A):

    device = torch.device("cuda:0")
    rows, cols = A.size()
    B = torch.zeros((rows, cols, cols))
    for i in range(rows):
        B[i] = torch.diag(A[i])

    return B.to(device)

def normalize_and_symmetrize_tensor(input_tensor, threshold=0.5):
    min_values, _ = input_tensor.min(dim=-1, keepdim=True)
    max_values, _ = input_tensor.max(dim=-1, keepdim=True)

    normalized_tensor = (input_tensor - min_values) / (max_values - min_values)

    row_sums = normalized_tensor.sum(dim=-1, keepdim=True)

    epsilon = 1e-8
    row_sums = row_sums + epsilon

    final_normalized_tensor = normalized_tensor / row_sums

    final_normalized_tensor = (final_normalized_tensor + final_normalized_tensor.transpose(-1, -2)) / 2

    return final_normalized_tensor

def max_min(input):
    a, b, c = input.size()
    input_array = input.cpu().detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            k=0
            max_x=max(input_array[i][j])
            min_x=min(input_array[i][j])
            diff=max_x-min_x
            while k<c:
                input_array[i][j][k]=((input_array[i][j][k])-min_x)/diff
                k+=1
            j += 1

        i += 1
    return torch.from_numpy(input_array).float()

def max_min_2D(input):
    a=len(input)
    b = len(input[0])
    threahold=0.5
    input_array = input
    i = 0
    while i < a:
        j = 0
        max_x=max(input_array[i])
        min_x=min(input_array[i])
        diff=max_x-min_x
        while j<b:
            input_array[i][j]=((input_array[i][j])-min_x)/diff
            if input_array[i][j]<threahold:
                input_array[i][j]=0.0
            j += 1
        i += 1
    return input_array

def pos_neg_mask(input,pos_mask,neg_mask):
    a, b = input.size()
    input_array = input.detach().numpy()
    pos_mask = pos_mask.detach().numpy()
    neg_mask = neg_mask.detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            if input_array[i][j]<0:
                neg_mask[i][j]=0
            else:
                pos_mask[i][j]=0
            j += 1
        i += 1
    return torch.from_numpy(pos_mask).float(),torch.from_numpy(neg_mask).float()

def find_duplicates(lst):
    indices = {}
    for i, x in enumerate(lst):
        if x not in indices:
            indices[x] = [i]
        else:
            indices[x].append(i)
    return [v for v in indices.values() if len(v) > 1]

def normalize_motif(ids, tensor):
    for row in ids:
        idxs = row
        slices = [tensor[:, idx, :] for idx in idxs]
        mean_ = torch.stack(slices, axis=0).mean()
        for slice_ in slices:
            if mean_ != 0:
                slice_ /= mean_
            else:
                slice_.fill_(0)

    return tensor

def auroc(prob, label):
    y_true = label.data.cpu().numpy().flatten()
    y_scores = prob.data.cpu().numpy().flatten()
    y_scores=np.nan_to_num(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score, fpr, tpr


def AUC(prob, label):
    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc_score = auc(fpr, tpr)
    return auroc_score

def auprc(prob,label):
    y_true=label.data.cpu().numpy().flatten()
    y_scores=prob.data.cpu().numpy().flatten()
    y_scores = np.nan_to_num(y_scores)
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall

def AUPR(prob,label):
    precision,recall,thresholds=precision_recall_curve(prob,label)
    auprc_score=auc(recall,precision)
    return auprc_score

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction
def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def Confuse_Indicator(CM,y_real,y_predict):
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    TN = CM[0]
    FP = CM[1]
    FN = CM[2]
    TP = CM[3]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN)
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    f1 = (2*Prec*Sen)/(Prec+Sen)
    Auc,_,_= auroc(torch.tensor(y_predict),torch.tensor(x))
    Aupr,_,_=auprc(torch.tensor(y_predict),torch.tensor(x))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    print('MCC:%.4f, Acc:%.4f, Sen:%.4f, Sp:%.4f' % (MCC, Acc, Sen, Spec))

    return Result, RealAndPrediction,RealAndPredictionProb

def Confuse(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=list(y_real.data.cpu().numpy().flatten())
    y_predict=list(y_predict.data.cpu().numpy().flatten())
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    CM = confusion_matrix(x, y)

    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    return TN,FP,FN,TP,y_real,y_predict

def Indicator(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=y_real.data.cpu().numpy().flatten()
    y_predict=y_predict.data.cpu().numpy().flatten()
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    CM = confusion_matrix(x, y)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print(CM)
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN)
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    f1 = f1_score(y, x)
    Auc,_,_= auroc(torch.tensor(y_predict),torch.tensor(x))
    Aupr,_,_=auprc(torch.tensor(y_predict),torch.tensor(x))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    print('MCC:%.4f, Acc:%.4f, Sen:%.4f, Sp:%.4f' % (MCC, Acc, Sen, Spec))

    return Result,RealAndPrediction,RealAndPredictionProb

def normalized_input(input):
    a,b,c=input.size()
    input=input.detach().numpy()
    new_input=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                new_input[i][j][k]=(input[i][j][k]-min(input[i][j]))/(max(input[i][j])-min(input[i][j]))
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(new_input).float()

def mahalanobis_distance(X, regularization_param=1e-5):
    mean_vector = torch.mean(X, dim=1, keepdim=True)
    X_c = X - mean_vector
    cov_matrix = torch.matmul(X_c.transpose(1, 2), X_c) / (X.size(1) - 1)
    cov_matrix += regularization_param * torch.eye(cov_matrix.size(-1), device=cov_matrix.device)
    inv_cov_matrix = torch.inverse(cov_matrix)
    mahalanobis_dist = torch.bmm(torch.bmm(X_c, inv_cov_matrix), X_c.transpose(1, 2))

    return mahalanobis_dist.squeeze()

def gaussian_kernel_similarity(mahalanobis_dist, sigma=1.0):
    similarity = torch.exp(-0.5 * (mahalanobis_dist / sigma)**2)
    return similarity

import torch
def row_normalize_softmax(input_tensor):

    b, n, _ = input_tensor.shape
    for i in range(b):
        for j in range(n):
            row_values = input_tensor[i, j, :]
            input_tensor[i, j, :] = torch.nn.functional.softmax(row_values, dim=-1)

    return input_tensor

batch_size = 2
num_nodes = 3

def threshold_filter_motif(input_tensor, k):
    b, n, _ = input_tensor.shape
    for i in range(b):
        for j in range(n):
            for l in range(n):
                input_tensor[i, j, j] = 1.0
                if abs(j - l) > k:
                    input_tensor[i, j, l] = 0.0
                elif input_tensor[i, j, l] < 0.5:
                    input_tensor[i, j, :] = 0.0

    return input_tensor

def threshold_filter(adj_tensor):
    row_means = adj_tensor.mean(dim=2, keepdim=True)
    adj_tensor = torch.where(torch.lt(adj_tensor, row_means), torch.tensor(0.0).to('cuda:0'), adj_tensor)

    return adj_tensor

def keep_top_k_values(adj_tensor, k):
    top_k_values, top_k_indices = adj_tensor.topk(k, dim=2)
    mask = torch.zeros_like(adj_tensor)
    mask.scatter_(2, top_k_indices, 1)
    adj_tensor = adj_tensor * mask

    return adj_tensor

import numpy as np
import scipy.sparse as sp
import math
def threshold_to_zero(mx, threshold):
    high_values_indexes = set(zip(*((np.abs(mx) >= threshold).nonzero())))
    nonzero_indexes = zip(*(mx.nonzero()))

    if not sp.isspmatrix_lil(mx):
        mx = mx.tolil()

    for s in nonzero_indexes:
        if s not in high_values_indexes:
            mx[s] = 0.0
    mx = mx.tocoo()
    mx.eliminate_zeros()
    return mx



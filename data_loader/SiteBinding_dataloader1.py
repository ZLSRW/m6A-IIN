import torch.utils.data as torch_data
import random
import csv
import torch
import numpy as np
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 注意表头
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
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


def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            # print(counter)
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df):
        self.data = df
        self.df_length = len(df[0])
        self.x_idx = self.get_idx()

    def __getitem__(self, index):

        hi = self.x_idx[index]
        lo = hi - 1
        graph = self.data[0][lo: hi][0]
        Loop_graph = self.data[1][lo: hi][0]
        onelabels = self.data[2][lo: hi][0]
        oneFeature = self.data[3][lo: hi][0]

        graph, Loop_graph, labels, onehots, onehot_feature = self.get_data(graph, Loop_graph, onelabels,
                                                                           oneFeature)
        graph = torch.from_numpy(graph).type(torch.float)
        Loop_graph = torch.from_numpy(Loop_graph).type(torch.float)
        labels = torch.from_numpy(labels).type(torch.float)
        onehot_feature = torch.from_numpy(onehot_feature).type(torch.float)

        return graph, Loop_graph, labels, onehots, onehot_feature

    def __len__(self):
        return len(self.x_idx)

    def get_idx(self):
        x_index_set = range(1, self.df_length)
        x_end_idx = [x_index_set[j] for j in range((len(x_index_set)))]
        return x_end_idx

    def get_data(self, graph, Loop_graph, onelabels, oneFeature):
        graphx = graph
        Loop_graphx = Loop_graph

        onehot_labelx = onelabels
        onehot_featurex = oneFeature

        labels = onehot_labelx[-1]
        onehots = onehot_labelx[:-1]


        return np.array(graphx, dtype='float64'), np.array(Loop_graphx, dtype='float64'), np.array(labels), np.array(
            onehots, dtype='float64'), np.array(onehot_featurex, dtype='float64')

if __name__ == '__main__':
    print("done!")


import math
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
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])  # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def sliding_window(lst):
    window_size = 4
    step = 16
    num_windows = 16
    B = []

    node_num = 0
    while node_num < (len(lst) / 4):
        i = 0
        seq = []
        Flag = True
        while i < num_windows:
            start = int((node_num * window_size + i * step))
            if Flag and start < len(lst):  # 未超出上界
                start = int((node_num * window_size + i * step))
            else:
                Flag = False
                nex = 0
                start = int((node_num * window_size + nex * step) % len(lst))
                nex += 1

            seq.extend(lst[start:start + window_size])
            i += 1
        B.append(seq)
        node_num += 1
    return B

seq_types = ['rat_liver']

File_name = '../data/'
file_tail = '/Train_Test/Test_graph'
file_tail_loop = '/Train_Test/Test_Loop_graph'
file_tail1 = '/Train_Test/Test_OnehotLabel'
file_tail3 = '/Train_Test/Test_Feature'

file_tail4 = '/Train_Test/Train_graph'
file_tail4_loop = '/Train_Test/Train_Loop_graph'
file_tail5 = '/Train_Test/Train_OnehotLabel'
file_tail7 = '/Train_Test/Train_Feature'

i = 0
while i < len(seq_types):
    print(str(seq_types[i])+' begin')
    counter = 0
    while counter < 5:
        Positive_sample = []
        Negative_sample = []

        input_file_name = File_name + seq_types[i] + file_tail + str(counter) + '.npy'
        input_file_name_Loop = File_name + seq_types[i] + file_tail_loop + str(counter) + '.npy'

        input_file_name1 = File_name + seq_types[i] + file_tail1 + str(counter) + '.npy'
        input_file_name3 = File_name + seq_types[i] + file_tail3 + str(counter) + '.npy'

        input_file_name4 = File_name + seq_types[i] + file_tail4 + str(counter) + '.npy'
        input_file_name4_Loop = File_name + seq_types[i] + file_tail4_loop + str(counter) + '.npy'

        input_file_name5 = File_name + seq_types[i] + file_tail5 + str(counter) + '.npy'
        input_file_name7 = File_name + seq_types[i] + file_tail7 + str(counter) + '.npy'

        Test_graph = np.load(input_file_name).tolist()
        Test_Loop_graph = np.load(input_file_name_Loop).tolist()

        Test_onehotLabel = np.load(input_file_name1).tolist()
        Test_onehotFeature = np.load(input_file_name3,allow_pickle=True).tolist()

        Test_list=[Test_graph,Test_Loop_graph,Test_onehotLabel,Test_onehotFeature]

        Train_graph = np.load(input_file_name4).tolist()
        Train_Loop_graph = np.load(input_file_name4_Loop).tolist()

        Train_onehotLabel = np.load(input_file_name5).tolist()
        Train_onehotFeature = np.load(input_file_name7,allow_pickle=True).tolist()

        Train_list = [Train_graph,Train_Loop_graph, Train_onehotLabel,Train_onehotFeature]

        save_name1 = File_name + seq_types[i] + '/Train_Test/all/TestData' + str(counter) + '.npy'
        save_name2 = File_name + seq_types[i] + '/Train_Test/all/TrainData' + str(counter) + '.npy'

        np.save(save_name1, np.array(Test_list))
        np.save(save_name2, np.array(Train_list))

        counter += 1
    print(str(seq_types[i]) + ' done')
    i += 1

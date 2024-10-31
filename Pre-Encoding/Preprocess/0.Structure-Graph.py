"""""
(1) 读取正样本和负样本的fasta文件
(2) 序列处理成onehot向量的形式，点括号表示处理成二维矩阵。
(3) 分别将二维矩阵和one-hot表征存储为三维的张量。
"""""
import numpy as np

fix_seed = 2023

np.random.seed(fix_seed)


# one-hot编码
def convert_seq_to_bicoding(seq):  # 序列-->编码矩阵
    # return bicoding for a sequence
    # seq=seq.replace('U','T') #turn rna seq to dna seq if have(将RNA替换成DNA) # 拓展为15维的初始特征向量
    feat_bicoding = []
    bicoding_dict = {'A': [1, 0, 0, 0, 1, 1, 1, 0.2837, 0.3183, 0.1873, 0.3299, 0.2652, 0.3696, 0.1292, 0.2618],
                     'C': [0, 1, 0, 0, 0, 1, 0, 0.2162, 0.2098, 0.2602, 0.2031, 0.2180, 0.1901, 0.2902, 0.2319],
                     'G': [0, 0, 1, 0, 1, 0, 0, 0.2972, 0.2643, 0.2882, 0.2668, 0.2966, 0.2661, 0.2572, 0.2810],
                     'U': [0, 0, 0, 1, 0, 0, 1, 0.2027, 0.2073, 0.2642, 0.2000, 0.2200, 0.1741, 0.3231, 0.2251], }
    # if len(seq)<41:
    #     seq=seq+'N'*(41-len(seq)) #补全，N表示未测序的核苷酸
    for each_nt in seq:
        feat_bicoding += bicoding_dict[each_nt]  # 164

    # print(feat_bicoding)
    return feat_bicoding


# 将二级结构的点括号表示转换为图
def match_brackets(s):
    n = len(s)
    A = [[0] * n for _ in range(n)]
    stack = []

    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                A[i][j] = A[j][i] = 1
            else:
                raise ValueError("Unmatched right parenthesis at index", i)
        A[i][i] = 1

    if stack:
        raise ValueError("Unmatched left parenthesis at index", stack.pop())

    return A

    # loop区域学习


def filter_elements(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    filtered_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for xs in range(rows):
        flag = False
        i = xs
        j = xs + 1

        while i > 1:
            while j < cols - 2:
                # 检查条件，保留符合条件的元素
                if matrix[i][j] != 0 and matrix[i - 1][j + 1] != 0:  # 这个语句确保找到次对角线方向上的符合茎结构的区域，即说明该区域符合要求
                    flag = True
                    i = 0
                    j = cols - 2
                j += 1
            i -= 1

        if flag:  # 对整条次对角区域进行遍历并求反

            h = xs
            l = xs + 1

            while h > 0 and l < cols - 1:
                if matrix[h][l] == 0:  filtered_matrix[h][l] = 1
                h = h - 1
                l = l + 1

    # 构造mask矩阵（以次对角线及其平行的元素为基准，构建正方形局部mask） 这样定义是因为环不会重叠，所以如果可视化的话，应该是表现出一个个的方块
    for xs in range(rows):
        h = xs
        l = xs + 1
        counter = 0
        while h > 1 and l <= cols - 2:
            if counter == 0 and filtered_matrix[h][l] != 0:
                start = h
                counter += 1
            elif counter > 0 and filtered_matrix[h][l] != 0:
                counter += 1
            elif counter > 0 and filtered_matrix[h][l] == 0:
                for i in range(start, start + counter):
                    for j in range(start, start + counter):
                        matrix[i][j] = 1
            h = h - 1
            l = l + 1

    return filtered_matrix


def OneHotGraph(fasta_file, Positive):
    # 用SeqIO.parse打开fasta文件
    records = SeqIO.parse(fasta_file, "fasta")

    All_one_hot_codes = []
    All_Graphs = []
    All_Loop_Graphs = []
    counter = 0
    for record in records:
        # print("Header:", record.id)
        # 获取序列和二级结构信息
        all_str = str(record.seq)

        lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]

        seq_str = lines[0]
        onehotcode = convert_seq_to_bicoding(seq_str)

        if Positive:
            onehotcode.append(1)
            # counter+=1
        else:
            onehotcode.append(0)
            # counter += 1
        # print(onehotcode)
        All_one_hot_codes.append(onehotcode)

        structure_str = lines[1]

        A = match_brackets(structure_str)  # 对角线赋1

        LoopA = filter_elements(A)

        All_Graphs.append(A)
        All_Loop_Graphs.append(LoopA)
    # print(counter)
    # print(seq_str)
    # print(structure_str)
    return np.array(All_one_hot_codes), np.array(All_Graphs), np.array(All_Loop_Graphs)


# seq_types = ['Human_Brain', 'Human_Kidney', 'Human_Liver', 'Mouse_brain', 'Mouse_heart',
#              'Mouse_kidney', 'Mouse_liver', 'Mouse_test']
# seq_types = ['Human_Brain', 'Human_Kidney', 'Human_Liver', 'Mouse_brain', 'Mouse_heart',
#              'Mouse_kidney', 'Mouse_liver', 'Mouse_test', 'rat_brain', 'rat_kidney','rat_liver']
from Bio import SeqIO

seq_types = ['rat_liver']

P_file_name1 = '../Dataset/Tissue Specific/'
P_file_tail = '/Positivey.fasta'
N_file_name1 = '../Dataset/Tissue Specific/'
N_file_tail = '/Negativex.fasta'

i = 0
while i < len(seq_types):
    Positive_sample = []
    Negative_sample = []
    All_sample = []

    input_P_file_name = P_file_name1 + seq_types[i] + P_file_tail
    input_N_file_name = N_file_name1 + seq_types[i] + N_file_tail

    All_one_hot_codes_P, All_Graphs_P, All_Loop_Graphs_P = OneHotGraph(input_P_file_name, True)
    All_one_hot_codes_N, All_Graphs_N, All_Loop_Graphs_N = OneHotGraph(input_N_file_name, False)

    np.save(P_file_name1 + seq_types[i] + '/PositiveGraphs.npy', All_Graphs_P)
    np.save(P_file_name1 + seq_types[i] + '/PositiveOnehot.npy', All_one_hot_codes_P)
    np.save(P_file_name1 + seq_types[i] + '/PositiveLoopGraphs.npy', All_Loop_Graphs_P)

    np.save(N_file_name1 + seq_types[i] + '/NegativeGraphs.npy', All_Graphs_N)
    np.save(N_file_name1 + seq_types[i] + '/NegativeOnehot.npy', All_one_hot_codes_N)
    np.save(N_file_name1 + seq_types[i] + '/NegativeLoopGraphs.npy', All_Loop_Graphs_N)

    i += 1

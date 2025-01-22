import numpy as np

fix_seed = 2023

np.random.seed(fix_seed)

def convert_seq_to_bicoding(seq):
    feat_bicoding = []
    bicoding_dict = {'A': [1, 0, 0, 0, 1, 1, 1, 0.2837, 0.3183, 0.1873, 0.3299, 0.2652, 0.3696, 0.1292, 0.2618],
                     'C': [0, 1, 0, 0, 0, 1, 0, 0.2162, 0.2098, 0.2602, 0.2031, 0.2180, 0.1901, 0.2902, 0.2319],
                     'G': [0, 0, 1, 0, 1, 0, 0, 0.2972, 0.2643, 0.2882, 0.2668, 0.2966, 0.2661, 0.2572, 0.2810],
                     'U': [0, 0, 0, 1, 0, 0, 1, 0.2027, 0.2073, 0.2642, 0.2000, 0.2200, 0.1741, 0.3231, 0.2251], }

    for each_nt in seq:
        feat_bicoding += bicoding_dict[each_nt]

    return feat_bicoding

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
                if matrix[i][j] != 0 and matrix[i - 1][j + 1] != 0:
                    flag = True
                    i = 0
                    j = cols - 2
                j += 1
            i -= 1

        if flag:

            h = xs
            l = xs + 1

            while h > 0 and l < cols - 1:
                if matrix[h][l] == 0:  filtered_matrix[h][l] = 1
                h = h - 1
                l = l + 1

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
    records = SeqIO.parse(fasta_file, "fasta")

    All_one_hot_codes = []
    All_Graphs = []
    All_Loop_Graphs = []
    counter = 0
    for record in records:
        all_str = str(record.seq)

        lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]

        seq_str = lines[0]
        onehotcode = convert_seq_to_bicoding(seq_str)

        if Positive:
            onehotcode.append(1)
        else:
            onehotcode.append(0)
        All_one_hot_codes.append(onehotcode)

        structure_str = lines[1]

        A = match_brackets(structure_str)

        LoopA = filter_elements(A)

        All_Graphs.append(A)
        All_Loop_Graphs.append(LoopA)

    return np.array(All_one_hot_codes), np.array(All_Graphs), np.array(All_Loop_Graphs)

from Bio import SeqIO

seq_types = ['rat_liver']

P_file_name1 = '../Dataset/Tissue Specific/'
P_file_tail = '/Positive.fasta'
N_file_name1 = '../Dataset/Tissue Specific/'
N_file_tail = '/Negative.fasta'

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

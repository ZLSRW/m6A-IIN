"""""
(1) Read the fasta files for positive and negative samples.
(2) Process the sequences into one-hot vectors, with brackets indicating conversion into a two-dimensional matrix.
(3) Store the two-dimensional matrix and one-hot representations as three-dimensional tensors.
(4) Disruption of primary structure: only disrupt the motif sequence, while preserving the graph and loop information. This requires rerunning the one-hot and feature embedding steps.
(5) Disruption of secondary structure: only disrupt the secondary structure. This step involves mapping to the potential secondary structure, then disrupting the corresponding stem-loop structures, which in turn affects the transmission of secondary structure and loop information.
(6) Separate considerations: for common parts, simultaneously disrupt both types of information.
"""""
import numpy as np
import csv
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
from flair.embeddings import WordEmbeddings, ELMoEmbeddings
from flair.data import Sentence

fix_seed = 2023
np.random.seed(fix_seed)

fix_seed = 2023

np.random.seed(fix_seed)


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = row[i]
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def convert_seq_to_bicoding(seq, start_index, end_index):
    feat_bicoding = []
    modified_seq = list(seq)

    bicoding_dict = {'A': [1, 0, 0, 0, 1, 1, 1, 0.2837, 0.3183, 0.1873, 0.3299, 0.2652, 0.3696, 0.1292, 0.2618],
                     'C': [0, 1, 0, 0, 0, 1, 0, 0.2162, 0.2098, 0.2602, 0.2031, 0.2180, 0.1901, 0.2902, 0.2319],
                     'G': [0, 0, 1, 0, 1, 0, 0, 0.2972, 0.2643, 0.2882, 0.2668, 0.2966, 0.2661, 0.2572, 0.2810],
                     'U': [0, 0, 0, 1, 0, 0, 1, 0.2027, 0.2073, 0.2642, 0.2000, 0.2200, 0.1741, 0.3231, 0.2251],
                     'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    for i, each_nt in enumerate(seq):
        if start_index <= i <= end_index:
            feat_bicoding += bicoding_dict['N']
            modified_seq[i] = 'N'
        else:
            feat_bicoding += bicoding_dict[each_nt]

    return feat_bicoding, ''.join(modified_seq)


def convert_seq_to_bicoding_LoopOnly(seq):
    feat_bicoding = []
    bicoding_dict = {'A': [1, 0, 0, 0, 1, 1, 1, 0.2837, 0.3183, 0.1873, 0.3299, 0.2652, 0.3696, 0.1292, 0.2618],
                     'C': [0, 1, 0, 0, 0, 1, 0, 0.2162, 0.2098, 0.2602, 0.2031, 0.2180, 0.1901, 0.2902, 0.2319],
                     'G': [0, 0, 1, 0, 1, 0, 0, 0.2972, 0.2643, 0.2882, 0.2668, 0.2966, 0.2661, 0.2572, 0.2810],
                     'U': [0, 0, 0, 1, 0, 0, 1, 0.2027, 0.2073, 0.2642, 0.2000, 0.2200, 0.1741, 0.3231, 0.2251]}
    for i, each_nt in enumerate(seq):
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

def OneHotGraphLoop_only(records, info_file, Positive):
    All_one_hot_codes = []
    All_Graphs = []
    All_Loop_Graphs = []
    All_seq=[]
    counter=0
    for ele in info_file:
        seq_str_inverse = ele[1]
        All_seq.append(seq_str_inverse)
        start = int(ele[3])
        end = int(ele[4])

        onehotcode = convert_seq_to_bicoding_LoopOnly(seq_str_inverse)

        if Positive:
            onehotcode.append(1)
        else:
            onehotcode.append(0)

        All_one_hot_codes.append(onehotcode)
        flag=False
        xxx=0
        for record in records:
            all_str = str(record.seq)

            lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]
            seq_str = lines[0]
            xxx+=1
            if seq_str_inverse == seq_str:
                counter+=1
                structure_str = lines[1]
                structure_start = int(ele[3])
                structure_end = int(ele[4])
                structure_str_new = replace_parentheses_with_dots(structure_str, structure_start, structure_end)
                A = match_brackets(structure_str_new)  # 对角线赋1
                LoopA = filter_elements(A)
                All_Graphs.append(A)
                All_Loop_Graphs.append(LoopA)
                flag=True
                break
    return np.array(All_one_hot_codes), np.array(All_Graphs), np.array(All_Loop_Graphs),All_seq


def OneHotGraphMotif_only(records, info_file, Positive):
    All_one_hot_codes = []
    All_Graphs = []
    All_Loop_Graphs = []
    All_Seq=[]

    for ele in info_file:
        seq_str_inverse = ele[1]
        start = int(ele[3])
        end = int(ele[4])

        onehotcode,strN = convert_seq_to_bicoding(seq_str_inverse, start, end)
        All_Seq.append(strN)

        if Positive:
            onehotcode.append(1)
        else:
            onehotcode.append(0)

        All_one_hot_codes.append(onehotcode)

        for record in records:
            all_str = str(record.seq)
            lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]
            seq_str = lines[0]
            if seq_str_inverse == seq_str:
                structure_str = lines[1]
                A = match_brackets(structure_str)
                LoopA = filter_elements(A)

                All_Graphs.append(A)
                All_Loop_Graphs.append(LoopA)
                break


    return np.array(All_one_hot_codes), np.array(All_Graphs), np.array(All_Loop_Graphs),All_Seq


def replace_parentheses_with_dots(dot_bracket: str, start: int, end: int) -> str:
    dot_bracket_list = list(dot_bracket)

    if start >= end or start < 0 or end >= len(dot_bracket_list):
        raise ValueError("Invalid loop positions.")

    stack = []
    pairings = {}

    for i, char in enumerate(dot_bracket_list):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                opening_index = stack.pop()
                pairings[opening_index] = i
                pairings[i] = opening_index


    for i in range(start, end + 1):
        if i in pairings:
            matching_index = pairings[i]
            dot_bracket_list[i] = '.'
            dot_bracket_list[matching_index] = '.'

    i, j = start - 1, end + 1
    while i >= 0 and j < len(dot_bracket_list):
        if i in pairings and pairings[i] == j:
            dot_bracket_list[i] = '.'
            dot_bracket_list[j] = '.'
            i -= 1
            j += 1
        else:
            break

    return "".join(dot_bracket_list)


def OneHotGraphLoopRegion(records, info_file, Positive):

    All_one_hot_codes = []
    All_Graphs = []
    All_Loop_Graphs = []
    All_seq=[]

    for ele in info_file:
        seq_str_inverse = ele[1]
        start = int(ele[3])
        end = int(ele[4])

        onehotcode,strN = convert_seq_to_bicoding(seq_str_inverse, start, end)
        All_seq.append(strN)

        if Positive:
            onehotcode.append(1)
        else:
            onehotcode.append(0)

        All_one_hot_codes.append(onehotcode)

        for record in records:
            all_str = str(record.seq)
            lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]
            seq_str = lines[0]
            if seq_str_inverse == seq_str:
                structure_str = lines[1]
                structure_start = int(ele[6])
                structure_end = int(ele[7])
                structure_str_new = replace_parentheses_with_dots(structure_str, structure_start, structure_end)

                A = match_brackets(structure_str_new)
                LoopA = filter_elements(A)
                All_Graphs.append(A)
                All_Loop_Graphs.append(LoopA)
                break

    return np.array(All_one_hot_codes), np.array(All_Graphs), np.array(All_Loop_Graphs),All_seq

def save_sequences_to_fasta(sequences, file_name, descriptions=None):
    with open(file_name, 'w') as fasta_file:
        for i, sequence in enumerate(sequences):
            # 如果没有提供描述信息，自动生成
            if descriptions is None:
                description = f"Sequence_{i+1}"
            else:
                description = descriptions[i]

            # 写入描述行和序列
            fasta_file.write(f">{description}\n")
            fasta_file.write(f"{sequence}\n")

from Bio import SeqIO

# seq_types = ['Human_Brain', 'Human_Kidney', 'Human_Liver']
# MotifLoopTypes = ['H-b', 'H-k', 'H-l']

seq_types = ['rat_liver']
MotifLoopTypes = ['R-l']

P_file_name1 = '../../../Dataset/Tissue Specific/'
P_file_tail = '/Positivex.fasta'


i = 0
while i < len(seq_types):
    input_P_file_name = P_file_name1 + seq_types[i] + P_file_tail
    records = list(SeqIO.parse(input_P_file_name, "fasta"))
    j=0
    while j<5:
        loop_only = []
        loopRegion_common_regions = []
        motif_only = []

        ReadMyCsv(loop_only, MotifLoopTypes[i] + '/loop_only'+str(j)+'.csv')
        ReadMyCsv(motif_only, MotifLoopTypes[i] + '/motif_only' + str(j) + '.csv')
        ReadMyCsv(loopRegion_common_regions, MotifLoopTypes[i] + '/loopRegion_common_regions'+str(j)+'.csv')

        print("*"*99)
        print('fold '+str(j))
        print(np.array(loop_only).shape)
        print(np.array(motif_only).shape)
        print(np.array(loopRegion_common_regions).shape)
        print("*" * 99)

        print("begin Loop only")
        # loop_only
        All_one_hot_codes_P_loop_only, All_Graphs_P_loop_only, All_Loop_Graphs_P_loop_only,All_seq_loop_only = OneHotGraphLoop_only(records, loop_only, True)
        print("end Loop only")

        # motif_only
        print('begin motif only')
        All_one_hot_codes_P_motif_only, All_Graphs_P_motif_only, All_Loop_Graphs_P_motif_only,All_seq_P_motif_only = OneHotGraphMotif_only(
            records, motif_only, True)
        print('end motif only')

        print("begin Loop_region")
        # loopRegion_common_regions
        All_one_hot_codes_P_loopRegion_common_regions, All_Graphs_P_loopRegion_common_regions, All_Loop_Graphs_P_loopRegion_common_regions,All_seq_P_loopRegion_common_regions \
            = OneHotGraphLoopRegion(records, loopRegion_common_regions, True)
        print('end Loop_region')



        print("-" * 99)
        print(All_Graphs_P_loop_only.shape)
        print(All_Graphs_P_motif_only.shape)
        print(All_Loop_Graphs_P_loopRegion_common_regions.shape)
        print("-" * 99)

        # Loop_only
        np.save(MotifLoopTypes[i] + '/All_one_hot_codes_P_loop_only'+str(j)+'.npy', All_one_hot_codes_P_loop_only)
        np.save(MotifLoopTypes[i] + '/All_Graphs_P_loop_only' + str(j) + '.npy', All_Graphs_P_loop_only)
        np.save(MotifLoopTypes[i] + '/All_Loop_Graphs_P_loop_only'+str(j)+'.npy', All_Loop_Graphs_P_loop_only)
        save_sequences_to_fasta(All_seq_loop_only,MotifLoopTypes[i]+'/All_seq_loop_only'+str(j)+'.fasta')

        # motif_only
        np.save(MotifLoopTypes[i] + '/All_one_hot_codes_P_motif_only' + str(j) + '.npy', All_one_hot_codes_P_motif_only)
        np.save(MotifLoopTypes[i] + '/All_Graphs_P_motif_only' + str(j) + '.npy', All_Graphs_P_motif_only)
        np.save(MotifLoopTypes[i] + '/All_Loop_Graphs_P_motif_only' + str(j) + '.npy', All_Loop_Graphs_P_motif_only)
        save_sequences_to_fasta(All_seq_P_motif_only, MotifLoopTypes[i] + '/All_seq_P_motif_only' + str(j) + '.fasta')

        # Loop_region
        np.save(MotifLoopTypes[i] + '/All_one_hot_codes_P_loopRegion_common_regions'+str(j)+'.npy', All_one_hot_codes_P_loopRegion_common_regions)
        np.save(MotifLoopTypes[i] + '/All_Graphs_P_loopRegion_common_regions' + str(j) + '.npy',All_Graphs_P_loopRegion_common_regions)
        np.save(MotifLoopTypes[i] + '/All_Loop_Graphs_P_loopRegion_common_regions'+str(j)+'.npy', All_Loop_Graphs_P_loopRegion_common_regions)
        save_sequences_to_fasta(All_seq_P_loopRegion_common_regions, MotifLoopTypes[i]+'/All_seq_P_loopRegion_common_regions'+str(j)+'.fasta')

        j+=1

    i += 1

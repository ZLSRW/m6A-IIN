
import numpy as np
from flair.embeddings import WordEmbeddings, ELMoEmbeddings
from flair.data import Sentence

fix_seed = 2023
np.random.seed(fix_seed)

def fa2ELMo(fasta_file1,fasta_file2):
    records1 = SeqIO.parse(fasta_file1, "fasta")
    records2 = SeqIO.parse(fasta_file2, "fasta")
    Temp_Positive_sequence=[]
    Temp_Negative_sequence=[]
    for record in records1:
        all_str = str(record.seq)
        lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]
        seq_str = lines[0]
        Temp_Positive_sequence.append(seq_str)
    for record in records2:
        all_str = str(record.seq)
        lines = [all_str[i:i + 41] for i in range(0, len(all_str), 41)]
        seq_str = lines[0]

        Temp_Negative_sequence.append(seq_str)

    p1=[]
    for j in range(len(Temp_Positive_sequence)):
        k = ""
        for i in range(40):
            k = k + Temp_Positive_sequence[j][i] + " "
        k = k + Temp_Positive_sequence[j][40]
        p1 = p1 + [k]

    p2=[]
    for j in range(len(Temp_Negative_sequence)):
        k = ""
        for i in range(40):
            k = k + Temp_Negative_sequence[j][i] + " "
        k = k + Temp_Negative_sequence[j][40]
        p2 = p2 + [k]


    p1=np.array(p1)
    p2=np.array(p2)
    p1.reshape(-1, 1)
    p2.reshape(-1, 1)

    print('p1[0] '+str(p1[0]))
    print('p2[0] '+str(p2[0]))

    embedding = ELMoEmbeddings('small')
    P = []
    for i in range(p1.shape[0]):
        sentence = Sentence(p1[i])
        embedding.embed(sentence)
        temp = []
        for token in sentence:
            swap1 = token.embedding.cpu().numpy()
            t1=swap1[:256]
            t2=swap1[256:512]
            t3=swap1[512:]
            t=[(x + y + z) for x, y, z in zip(t1, t2, t3)]
            temp.append(t)
        P.append(temp)
        if i % 200 == 0:
            print("Coding----")

    print(P[0])
    N = []
    for i in range(p2.shape[0]):
        sentence = Sentence(p2[i])
        embedding.embed(sentence)
        temp = []
        for token in sentence:
            swap1 = token.embedding.cpu().numpy()

            t1 = swap1[:256]
            t2 = swap1[256:512]
            t3 = swap1[512:]
            t = [(x + y + z)  for x, y, z in zip(t1, t2, t3)]

            temp.append(t)

        N.append(temp)
        if i % 200 == 0:
            print("Coding1----")

    print(N[0])
    return np.array(P),np.array(N)

from Bio import SeqIO

seq_types = ['rat_liver']

P_file_name1 = '../Dataset/Tissue Specific/'
P_file_tail = '/Positivex.fasta'
N_file_name1 = '../Dataset/Tissue Specific/'
N_file_tail = '/Negativex.fasta'

i = 0
while i < len(seq_types):
    Positive_sample = []
    Negative_sample = []
    All_sample = []

    input_P_file_name = P_file_name1 + seq_types[i] + P_file_tail
    input_N_file_name = N_file_name1 + seq_types[i] + N_file_tail

    Positive_feature,Negative_feature=fa2ELMo(input_P_file_name,input_N_file_name)

    np.save(P_file_name1 + seq_types[i]+'/PositiveFeaturex.npy',Positive_feature)
    np.save(P_file_name1 + seq_types[i]+'/NegativeFeaturex.npy',Negative_feature)
    print(seq_types[i])
    i += 1






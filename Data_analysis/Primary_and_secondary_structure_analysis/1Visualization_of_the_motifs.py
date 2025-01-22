import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = int(row[i])
        SaveList.append(row)
    return


def plot_heatmap_CC(data, save_path, seq):
    label_font_axis = {
        'weight': 'bold',
        'size': 50,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'bold',
        'size': 60,
        'family': 'Arial'
    }


    data = np.array(data)


    fig, ax = plt.subplots(figsize=(30, 34))

    cmap = 'coolwarm'
    cmap_modified = plt.cm.get_cmap(cmap)
    min_val = data.min()

    heatmap = ax.imshow(data, cmap=cmap_modified, aspect='auto', vmin=min_val)

    x_labels = seq
    y_labels = seq

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels, fontdict=label_font_axis)
    ax.set_yticklabels(y_labels, fontdict=label_font_axis)


    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='gray', facecolor='none',
                             linestyle='dashed')
            ax.add_patch(rect)
            ax.text(j, i, data[i, j], ha='center', va='center', color='black', fontdict=label_font_axis)

    ax.set_xlabel('Cell lines', fontdict=label_font)
    ax.set_ylabel('Cell lines', fontdict=label_font)

    cbar = plt.colorbar(heatmap, shrink=1.0)
    cbar.set_label('Values', fontdict=label_font_axis)
    cbar.ax.tick_params(axis='y', labelsize=label_font_axis['size'], width=2)

    ax.set_xticklabels(x_labels, fontdict=label_font_axis, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, fontdict=label_font_axis, rotation=45, va='top')

    plt.tight_layout()

    plt.savefig(save_path, dpi=350)
    plt.show()

    return


def plot_heatmap_G(data, save_path, seq):
    label_font_axis = {
        'weight': 'normal',
        'size': 50,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'bold',
        'size': 60,
        'family': 'Arial'
    }

    fig, ax = plt.subplots(figsize=(30, 30))

    from matplotlib.colors import Normalize, ListedColormap

    cmap = 'Blues'
    data = np.array(data)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    heatmap = ax.imshow(data, cmap, norm=norm, aspect='auto')  # 设置aspect='auto'

    y_labels = seq
    x_labels = seq

    selected_y_indices = list(range(len(y_labels)))
    selected_y_labels = [y_labels[i] for i in selected_y_indices]
    selected_x_indices = list(range(len(x_labels)))
    selected_x_labels = [x_labels[i] for i in selected_x_indices]

    ax.set_xticks(selected_x_indices)
    ax.set_yticks(selected_y_indices)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='gray', facecolor='none')
            ax.add_patch(rect)

    ax.set_xticklabels(selected_x_labels, fontdict=label_font_axis)
    ax.set_yticklabels(selected_y_labels, fontdict=label_font_axis)

    ax.set_xlabel('Nucletides', fontdict=label_font)
    ax.set_ylabel('Nucletides', fontdict=label_font)

    cbar = plt.colorbar(heatmap, shrink=1.0)
    cbar.set_label('Weight', fontdict=label_font)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(50)
        label.set_fontname('Times New Roman')

    plt.tight_layout()

    plt.savefig(save_path, dpi=350)
    plt.show()

    return

def plot_heatmap_position(data, save_path):
    label_font_axis = {
        'weight': 'normal',
        'size': 50,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'bold',
        'size': 60,
        'family': 'Arial'
    }

    label_font1 = {
        'weight': 'bold',
        'size': 50,
        'family': 'Arial'
    }

    fig, ax = plt.subplots(figsize=(40, 8))

    from matplotlib.colors import Normalize, ListedColormap

    cmap = 'Blues'
    data = np.array(data)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    heatmap = ax.imshow(data, cmap, norm=norm, aspect='auto')  # 设置aspect='auto'

    y_labels = range(1, 21)
    x_labels = range(1, 42)

    selected_y_indices = [0, len(y_labels) // 2, len(y_labels) - 1]
    selected_y_labels = [y_labels[i] for i in selected_y_indices]
    selected_x_indices = [0, len(x_labels) // 2, len(x_labels) - 1]
    selected_x_labels = [x_labels[i] for i in selected_x_indices]
    ax.set_xticks(selected_x_indices)
    ax.set_yticks(selected_y_indices)

    ax.set_xticklabels(selected_x_labels, fontdict=label_font_axis)
    ax.set_yticklabels(selected_y_labels, fontdict=label_font_axis)

    cbar = plt.colorbar(heatmap, shrink=1.0)  # 调整shrink参数

    cbar.ax.tick_params(labelsize=label_font1['size'])

    plt.tight_layout()  # 自动调整布局

    plt.savefig(save_path, dpi=350)
    plt.show()

    return


def plot_heatmap_physicochemical(data, save_path):
    label_font_axis = {
        'weight': 'normal',
        'size': 50,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'bold',
        'size': 60,
        'family': 'Arial'
    }

    label_font1 = {
        'weight': 'bold',
        'size': 50,
        'family': 'Arial'
    }

    fig, ax = plt.subplots(figsize=(15, 20))

    from matplotlib.colors import Normalize, ListedColormap

    cmap = 'Blues'
    data = np.array(data)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    heatmap = ax.imshow(data, cmap, norm=norm, aspect='auto')  # 设置aspect='auto'

    # 设置横坐标和纵坐标标签
    y_labels = range(1, 21)
    x_labels = range(1, 12)

    selected_y_indices = [0, len(y_labels) // 2, len(y_labels) - 1]
    selected_y_labels = [y_labels[i] for i in selected_y_indices]
    selected_x_indices = [0, len(x_labels) // 2, len(x_labels) - 1]
    selected_x_labels = [x_labels[i] for i in selected_x_indices]
    ax.set_xticks(selected_x_indices)
    ax.set_yticks(selected_y_indices)


    ax.set_xticklabels(selected_x_labels, fontdict=label_font_axis)
    ax.set_yticklabels(selected_y_labels, fontdict=label_font_axis)

    cbar = plt.colorbar(heatmap, shrink=1.0)

    cbar.ax.tick_params(labelsize=label_font1['size'])

    plt.tight_layout()

    plt.savefig(save_path, dpi=350)
    plt.show()

    return


def plot_heatmap_GP(data, save_path, seq):
    label_font_axis = {
        'weight': 'normal',
        'size': 50,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'bold',
        'size': 60,
        'family': 'Arial'
    }

    fig, ax = plt.subplots(figsize=(15, 30))

    from matplotlib.colors import Normalize, ListedColormap

    cmap = 'Blues'
    data = np.array(data)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    heatmap = ax.imshow(data, cmap, norm=norm, aspect='auto')


    y_labels = seq
    x_labels = range(1, 12)

    selected_y_indices = list(range(len(y_labels)))
    selected_y_labels = [y_labels[i] for i in selected_y_indices]
    selected_x_indices = list(range(len(x_labels)))
    selected_x_labels = [x_labels[i] for i in selected_x_indices]

    ax.set_xticks(selected_x_indices)
    ax.set_yticks(selected_y_indices)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='gray', facecolor='none')
            ax.add_patch(rect)

    ax.set_xticklabels(selected_x_labels, fontdict=label_font_axis)
    ax.set_yticklabels(selected_y_labels, fontdict=label_font_axis)



    ax.set_xlabel('Nucletides', fontdict=label_font)
    ax.set_ylabel('Nucletides', fontdict=label_font)

    cbar = plt.colorbar(heatmap, shrink=1.0)
    cbar.set_label('Weight', fontdict=label_font)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(50)
        label.set_fontname('Times New Roman')

    plt.tight_layout()

    plt.savefig(save_path, dpi=350)
    plt.show()

    return


def plot_heatmap_motif(data, seq, save_path):
    label_font_axis = {
        'weight': 'normal',
        'size': 30,
        'family': 'Arial'
    }
    label_font = {
        'weight': 'normal',
        'size': 60,
        'family': 'Arial'
    }

    fig, ax = plt.subplots(figsize=(20, 25))

    from matplotlib.colors import Normalize, ListedColormap

    cmap = 'Blues'
    data = np.array(data)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    heatmap = ax.imshow(data, cmap, norm=norm, aspect='auto')

    y_labels = range(1, 42)
    x_labels = seq

    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))


    ax.set_xticklabels(x_labels, fontdict=label_font_axis, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, fontdict=label_font_axis)



    cbar = plt.colorbar(heatmap, shrink=1.0)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(50)
        label.set_fontname('Arial')

    plt.tight_layout()

    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.show()

    return


def process_array(B):
    result = []
    for i in range(B.shape[0]):
        if np.all(B[i, :4] == [1, 0, 0, 0]):
            result.append('A')
        elif np.all(B[i, :4] == [0, 1, 0, 0]):
            result.append('C')
        elif np.all(B[i, :4] == [0, 0, 1, 0]):
            result.append('G')
        elif np.all(B[i, :4] == [0, 0, 0, 1]):
            result.append('U')
        else:
            result.append('Unknown')
    return result


def filter_rows(arr, k=0.5, e=2):
    selected_rows = []
    selected_indices = []
    for i, row in enumerate(arr):
        avg = np.mean(row)
        if row[20] < avg - k and all(row[j] >= row[20] + e for j in [18, 19, 21, 22]):
            selected_rows.append(row)
            selected_indices.append(i)
    return np.array(selected_rows), np.array(selected_indices)


SR_hat0 = np.load('best_SR_hat0.npy', allow_pickle=True).tolist()
SR0 = np.load('best_SR0.npy', allow_pickle=True).tolist()


SR_nucletide_weights0 = np.sum(SR_hat0, axis=2)
SR_nucletide_weights_real0 = np.sum(SR0, axis=2)


row_max = np.max(SR_nucletide_weights0, axis=1, keepdims=True)
row_min = np.min(SR_nucletide_weights0, axis=1, keepdims=True)

normalized_arr = (SR_nucletide_weights0 - row_min) / (row_max - row_min)  # 计算核苷酸的可能性权重

row_max = np.max(SR_nucletide_weights_real0, axis=1, keepdims=True)
row_min = np.min(SR_nucletide_weights_real0, axis=1, keepdims=True)


normalized_arr_real = (SR_nucletide_weights_real0 - row_min) / (row_max - row_min)  # 计算核苷酸的可能性权重


All_top = []

ReadMyCsv2(All_top, '正负样本下标文件0.csv')

Positive_Top = All_top[0]
Negative_Top = All_top[1]

P_top_all = []
N_top_all = []

i = 0
while i < len(Positive_Top):
    P_top_all.append(SR_nucletide_weights0[Positive_Top[i]])
    N_top_all.append(SR_nucletide_weights0[Negative_Top[i]])
    i += 1


P_top20, P_top20_index = filter_rows(P_top_all)
N_top20, N_top20_index = filter_rows(N_top_all)

all_index_P = []
all_index_N = []

i = 0
while i < len(P_top20_index):
    all_index_P.append(P_top20_index[i])
    i += 1

i = 0
while i < len(N_top20_index):
    all_index_N.append(N_top20_index[i])
    i += 1

print(len(P_top20))
print(len(N_top20))
print(all_index_P)
print(all_index_N)

prob = []
ReadMyCsv(prob, 'Real_Predition_prob0.csv')

SRmapP = np.array(SR0[all_index_P[0]])
SRmapNameP = ''.join(process_array(SRmapP))
print('positive (' + str(prob[all_index_P[0]]) + '): ' + SRmapNameP)

SRmapN = np.array(SR0[all_index_N[0]])
SRmapNameN = ''.join(process_array(SRmapN))
print('Negative (' + str(prob[all_index_N[0]]) + '): ' + SRmapNameN)

def find_most_important_subregions(A, B):
    result = []
    for i in range(len(B)):
        row_idx = B[i]
        row = A[row_idx]

        # Calculate the average of the entire row
        row_avg = np.mean(row)

        # Initialize variables to track the most important subregion
        max_avg = 0
        start_idx = None
        end_idx = None

        # Iterate over all possible subregions of length greater than 3
        for j in range(len(row) - 2):
            for k in range(j + 3, len(row)):
                subregion = row[j:k]
                subregion_avg = np.mean(subregion)

                # Check if the subregion's average is greater than any other subregion
                if subregion_avg > max_avg and subregion_avg > row_avg:
                    max_avg = subregion_avg
                    start_idx = j
                    end_idx = k - 1

        # If a most important subregion is found, record its details
        if start_idx is not None and end_idx is not None:
            temp = []
            temp.append(row_idx)
            temp.append(start_idx)
            temp.append(end_idx)
            result.append(temp)
    return result


result = find_most_important_subregions(normalized_arr, Positive_Top)

All_seq_motif_info = []
All_PPMs_motif = []

i = 0
while i < len(result):
    temp = []
    index = result[i][0]
    start_idx = result[i][1]
    end_idx = result[i][2]

    RNA_seq = process_array(np.array(SR0[index]))
    motif = ''.join(RNA_seq[start_idx:end_idx + 1])

    seq = ''.join(RNA_seq)

    temp.append(index)
    temp.append(seq)
    temp.append(motif)
    temp.append(start_idx)
    temp.append(end_idx)  # 包含该值

    # 把标签和评分也加上去
    label = prob[index][0]
    pre_prob = round(prob[index][1], 3)

    temp.append(label)
    temp.append(pre_prob)

    All_seq_motif_info.append(temp)

    # 存储相应的PPM矩阵
    PPM_motif = np.array(SR_hat0[index])[:, :4]
    # print('PPM_motif.shape: '+str(PPM_motif.shape))

    row_max = np.max(PPM_motif, axis=1, keepdims=True)
    row_min = np.min(PPM_motif, axis=1, keepdims=True)
    normalized_PPM_motif = (PPM_motif - row_min) / (row_max - row_min)  # 计算核苷酸的可能性权重
    row_sums = normalized_PPM_motif.sum(axis=1, keepdims=True)
    normalized_PPM_motif = normalized_PPM_motif / row_sums

    All_PPMs_motif.append(normalized_PPM_motif[start_idx:end_idx+1])

    i += 1

storFile(All_PPMs_motif, 'All_PPMs_motif.csv')  # 所有潜在motif对应的位置概率矩阵
storFile(All_seq_motif_info, 'All_seq_motif_info.csv')  # 存储所有潜在motif的基本信息，包括序列信息等

def save_3d_list_to_txt(A, filename):
    with open(filename, 'w') as f:
        # Write MEME version
        f.write("MEME version 4\n\n")
        # Write ALPHABET
        f.write("ALPHABET= ACGU\n\n")

        # Iterate over each element in A
        for i, motif in enumerate(A):
            # Write MOTIF i
            f.write("MOTIF {}\n".format(i))
            # Write letter-probability matrix
            alength = 4
            w = len(motif)
            f.write("letter-probability matrix: alength= {} w= {}\n".format(alength, w))

            # Write motif matrix
            for row in motif:
                f.write(" ".join(str(round(x, 6)) for x in row) + "\n")
            f.write("\n")


save_3d_list_to_txt(All_PPMs_motif, 'output0.txt')

index_list=range(0,len(P_top_all))


i = 0
All_visible_motifs = []
seq_name = []
seq_info = []
PPM_selected=[]
counter1=0
counter2=0
counter3=0
while i < len(index_list):
    seq_weight = P_top_all[index_list[i]]
    motif = All_seq_motif_info[i][2]
    start_idx = All_seq_motif_info[i][3]
    end_idx = All_seq_motif_info[i][4]

    if (end_idx-start_idx==2 and counter1<10) or (end_idx-start_idx==2 and start_idx>28 and counter3<4) or (end_idx-start_idx>=3 and counter2<8):

        if end_idx-start_idx==2 and start_idx<28: counter1+=1
        if end_idx-start_idx>=3 and start_idx>=17: counter2+=1
        if end_idx - start_idx == 2 and start_idx > 28: counter3 += 1

        j = 0
        while j < len(seq_weight):
            if start_idx <= j <= end_idx:
                seq_weight[j] = seq_weight[j]
            else:
                seq_weight[j] = 0.0
            j += 1

        All_visible_motifs.append(seq_weight)
        seq_name.append(str(motif))
        seq_info.append(All_seq_motif_info[i])
        PPM_selected.append(All_PPMs_motif[i])

    i += 1

transposed_matrix = list(zip(*All_visible_motifs))

storFile(seq_info,'seq_info_for_PPM.csv')



plot_heatmap_motif(transposed_matrix, seq_name, 'All_visible_motifs0.png')




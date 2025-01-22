import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition, manifold
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

try:
    import plotly.express as px
except ImportError:
    px = None


def scatter(color_map, size_map, data, centers, dimension="2D", point_size=2, sty='default',
            label=None, title=None, alpha=None, aes_label=None, fig_size=(12, 8),
            **kwargs):
    # Error messages.
    if dimension not in ["2D", "3D"]:
        raise ValueError('Dimension must be "2D" or "3D".')
    if (dimension == "2D" and len(data[0]) != 2) or (dimension == "3D" and len(data[0]) != 3):
        raise ValueError('Data shape must match dimension!')
    if (label is not None) and len(data) != len(label):
        raise ValueError('Number of rows in data must equal to length of label!')

    plt.style.use(sty)

    label_font = {
        'weight': 'bold',
        'size': 14,
        'family': 'Times New Roman'
    }

    plt.figure(figsize=fig_size)

    # 2D scatter plot
    if dimension == "2D":
        # Plot with label
        if label is not None:
            label = np.array(label)
            lab = list(set(label))
            for index, l in enumerate(lab):
                color = color_map[index % len(color_map)] if color_map is not None else 'C{!r}'.format(index)
                size = size_map[index % len(size_map)] if size_map is not None else point_size
                if l == 0:
                    xx = 'R-R-Non-modification'
                elif l == 1:
                    xx = "R-R"
                elif l == 2:
                    xx = "M-R"
                elif l == 3:
                    xx = "H-R"
                elif l == 4:
                    xx = "M-R-Non-modification"
                elif l == 5:
                    xx = "M-R-modification"
                ax = plt.scatter(data[label == l, 0], data[label == l, 1],
                                 c=color,
                                 s=size,
                                 label=xx,
                                 alpha=alpha)

            plt.legend(prop={'weight': 'normal', 'size': 20, 'family': 'Arial'}, **kwargs)

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.tight_layout()

            plt.rc('axes', linewidth=2.0)

            for i, center in enumerate(centers):
                plt.scatter(center[0], center[1], c='black', s=100, marker='x', label=f'Center {i + 1}')

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    if not (i == 1 and j == 2):
                        plt.plot([centers[i][0], centers[j][0]], [centers[i][1], centers[j][1]], 'k--', lw=1)

        else:
            plt.scatter(data[:, 0], data[:, 1], s=point_size, alpha=alpha)

        if aes_label is not None:
            plt.xlabel(aes_label[0], fontsize=14, fontweight='bold')
            plt.ylabel(aes_label[1], fontsize=14, fontweight='bold')

    # 3D scatter plot
    if dimension == "3D":
        splot = plt.subplot(111, projection='3d')

        # Plot with label
        if label is not None:
            label = np.array(label)
            lab = list(set(label))

            for index, l in enumerate(lab):
                splot.scatter(data[label == l, 0], data[label == l, 1], data[label == l, 2],
                              s=point_size,
                              color=color_map[index % len(color_map)],
                              label=l)
            plt.legend(**kwargs)
        # Plot without label
        else:
            splot.scatter(data[:, 0], data[:, 1], data[:, 2], s=point_size)

        if aes_label is not None:
            splot.set_xlabel(aes_label[0], fontsize=14, fontweight='bold')
            splot.set_ylabel(aes_label[1], fontsize=14, fontweight='bold')
            splot.set_zlabel(aes_label[2], fontsize=14, fontweight='bold')

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.savefig('Rat_as_test.tif', dpi=350)
    plt.show()


def draw(tsne, emb, fig_size=(12, 8), random_state=42):
    color_map = ['#9370DB', '#4682B4', '#CD5C5C']
    size_map = [10, 10, 10, 10, 10, 10]  # Customize the size map

    if tsne:
        tsne_model = manifold.TSNE(2, random_state=random_state)
        rawdata = emb.iloc[:, 1:]
        pc = tsne_model.fit_transform(rawdata)
        centers = calculate_centers(pc, emb.label)
        scatter(color_map, size_map, pc, centers, dimension="2D", label=emb.label, title='H-R-Model', fig_size=fig_size)
        calculate_distances(pc, emb.label)
    else:
        pca = decomposition.PCA(2)
        rawdata = emb.iloc[:, 1:]
        pc = pca.fit_transform(rawdata)
        centers = calculate_centers(pc, emb.label)
        scatter(color_map, size_map, pc, centers, dimension="2D", label=emb.label, title='pca',
                aes_label=['PCA-1', 'PCA-2'],
                fig_size=fig_size)


def calculate_distances(pc, labels):
    unique_labels = np.unique(labels)
    distances = {}

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                points_i = pc[labels == label_i]
                points_j = pc[labels == label_j]

                dist = cdist(points_i, points_j, metric='euclidean').mean()
                distances[f"{label_i}-{label_j}"] = dist

    print("Class Pairwise Distances:")
    for pair, dist in distances.items():
        print(f"{pair}: {dist}")


def calculate_centers(pc, labels):
    unique_labels = np.unique(labels)
    centers = []

    for label in unique_labels:
        center = pc[labels == label].mean(axis=0)
        centers.append(center)

    return np.array(centers)


if __name__ == "__main__":
    emb = pd.read_csv('Rat_as_test.csv', header=None)
    emb = emb.rename(columns={0: "label"})
    draw(True, emb, fig_size=(8, 6.5), random_state=42)

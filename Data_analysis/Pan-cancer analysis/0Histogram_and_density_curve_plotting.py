import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Candidate-pan-cancer.csv"
overlap_counts = pd.read_csv(file_path, header=None).iloc[:, 0]

overlap_counts = overlap_counts[overlap_counts > 0]

bins = np.arange(0, 120, 10)


fig, ax1 = plt.subplots(figsize=(10, 6))


sns.set(style="white")


hist_color = '#4682B4'
kde_color = 'red'


plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

ax1 = sns.distplot(overlap_counts, bins=bins, kde=False, color=hist_color,
                   hist_kws={'edgecolor':'black', 'linewidth': 2})


ax2 = ax1.twinx()
sns.kdeplot(overlap_counts, color=kde_color, linewidth=2, ax=ax2,legend=False)

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width() / 2., height + 0.5, int(height), ha="center")


ax1.set_xlabel('Overlap Regions Count Interval')
ax1.set_ylabel('Number of Genes')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.savefig('Candidate-pan-cancer.tif',dpi=350)
plt.show()



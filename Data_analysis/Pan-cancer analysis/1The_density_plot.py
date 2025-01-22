import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

file_candidate = "Candidate-pan-cancer.csv"
file_validated = "Clinical-pan-cancer.csv"
file_non_pancancer = "Non-pan-cancer.csv"


candidate_counts = pd.read_csv(file_candidate, header=None).iloc[:, 0]
validated_counts = pd.read_csv(file_validated, header=None).iloc[:, 0]
non_pancancer_counts = pd.read_csv(file_non_pancancer, header=None).iloc[:, 0]


candidate_counts = candidate_counts[(candidate_counts >= 5) & (candidate_counts <= 108)]
validated_counts = validated_counts[(validated_counts >= 5) & (validated_counts <= 108)]
non_pancancer_counts = non_pancancer_counts[(non_pancancer_counts >= 5) & (non_pancancer_counts <= 108)]


mean_candidate = candidate_counts.mean()
mean_validated = validated_counts.mean()
mean_non_pancancer = non_pancancer_counts.mean()

print(f'Mean of Candidate Regions: {mean_candidate:.2f}')
print(f'Mean of Validated Regions: {mean_validated:.2f}')
print(f'Mean of Non-PanCancer Regions: {mean_non_pancancer:.2f}')


fig, ax = plt.subplots(figsize=(10, 10))


main_ax_border_width = 3
for spine in ax.spines.values():
    spine.set_linewidth(main_ax_border_width)


density_linewidth = 3
sns.kdeplot(candidate_counts, color='#5F4690', linewidth=density_linewidth, label='Candidate Regions', shade=True, ax=ax)
sns.kdeplot(validated_counts, color='#1D6996', linewidth=density_linewidth, label='Validated Regions', shade=True, ax=ax)
sns.kdeplot(non_pancancer_counts, color='#38A6A5', linewidth=density_linewidth, label='Non-PanCancer Regions', shade=True, ax=ax)


x_candidate = np.linspace(5, 108, 100)
density_candidate = stats.gaussian_kde(candidate_counts).evaluate(x_candidate)

x_validated = np.linspace(5, 108, 100)
density_validated = stats.gaussian_kde(validated_counts).evaluate(x_validated)

x_non_pancancer = np.linspace(5, 108, 100)
density_non_pancancer = stats.gaussian_kde(non_pancancer_counts).evaluate(x_non_pancancer)

peak_candidate = x_candidate[np.argmax(density_candidate)]
peak_validated = x_validated[np.argmax(density_validated)]
peak_non_pancancer = x_non_pancancer[np.argmax(density_non_pancancer)]


print(f'Peak of Candidate Regions: {peak_candidate:.2f}')
print(f'Peak of Validated Regions: {peak_validated:.2f}')
print(f'Peak of Non-PanCancer Regions: {peak_non_pancancer:.2f}')


dashed_line_width = 2
ax.axvline(x=peak_candidate, color='#5F4690', linestyle='--', linewidth=dashed_line_width)
ax.axvline(x=peak_validated, color='#1D6996', linestyle='--', linewidth=dashed_line_width)
ax.axvline(x=peak_non_pancancer, color='#38A6A5', linestyle='--', linewidth=dashed_line_width)


ax.set_xlabel('Overlap Regions Count Interval')
ax.set_ylabel('Density')
ax.set_title('Density Plot with Violin Plot Overlay')


ax.legend()

inset_width = "50%"
inset_height = "30%"

inset_x = -0.22
inset_y = -0.27

ax_inset = inset_axes(ax, width=inset_width, height=inset_height,
                     bbox_to_anchor=(inset_x, inset_y, 1.2, 1.1),  # 调整子图位置（左, 下, 宽, 高）
                     bbox_transform=ax.transAxes)

inset_ax_border_width = 3
for spine in ax_inset.spines.values():
    spine.set_linewidth(inset_ax_border_width)

violin_linewidth = 3
sns.violinplot(data=[candidate_counts, validated_counts, non_pancancer_counts],
               palette=['#5F4690', '#1D6996', '#38A6A5'], linewidth=violin_linewidth, ax=ax_inset)

ax_inset.set_xticks([0, 1, 2])
ax_inset.set_xticklabels(['Candidate', 'Validated', 'Non-PanCancer'])
ax_inset.set_ylabel('Counts')


t_stat_candidate, p_value_candidate = stats.ttest_ind(validated_counts, candidate_counts)

t_stat_non_pancancer, p_value_non_pancancer = stats.ttest_ind(validated_counts, non_pancancer_counts)


print(f'T-test p-value between Validated and Candidate Regions: {p_value_candidate:.5f}')
print(f'T-test p-value between Validated and Non-PanCancer Regions: {p_value_non_pancancer:.15f}')

t_stat_density, p_value_density = stats.ttest_ind(candidate_counts, non_pancancer_counts)
print(f'T-test p-value between Candidate and Non-PanCancer Regions (Density): {p_value_density:.15f}')


plt.tight_layout()
plt.savefig('density_with_violin_plot_custom_position.tif', dpi=350)
plt.show()

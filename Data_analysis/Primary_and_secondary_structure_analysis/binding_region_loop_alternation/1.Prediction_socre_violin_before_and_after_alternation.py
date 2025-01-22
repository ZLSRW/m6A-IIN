import matplotlib.pyplot as plt
import numpy as np
import csv

def ReadMyCsv(SaveList, fileName):
    with open(fileName, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row = [float(i) for i in row]
            SaveList.append(row)

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def calculate_statistics(data_list):
    results = []
    for data in data_list:
        a = sum(x > 0.5 for x in data)
        b = sum(x <= 0.5 for x in data)
        ratio = a / (a + b) if (a + b) > 0 else 0
        mean_value = np.mean(data)
        results.append((a, b, round(ratio, 3), round(mean_value, 3)))
    return results

# 生成示例数据
loop_beforex = []
binding_region_beforex = []
binding_and_loop_beforex = []
loop_afterx = []
binding_region_afterx = []
binding_and_loop_afterx = []

datasets = 'H-l/all_results/'

ReadMyCsv(loop_beforex, datasets+'before_loop_only.csv')
ReadMyCsv(binding_region_beforex, datasets+'before_motif_only.csv')
ReadMyCsv(binding_and_loop_beforex, datasets+'before_loopRegion_common_regions.csv')

ReadMyCsv(loop_afterx, datasets+'after_loop_only.csv')
ReadMyCsv(binding_region_afterx, datasets+'after_motif_only.csv')
ReadMyCsv(binding_and_loop_afterx, datasets+'after_loopRegion_common_regions.csv')

loop_before = [float(ele[0]) for ele in loop_beforex]
binding_region_before = [float(ele[0]) for ele in binding_region_beforex]
binding_and_loop_before = [float(ele[0]) for ele in binding_and_loop_beforex]

loop_after = [float(ele[0]) for ele in loop_afterx]
binding_region_after = [float(ele[0]) for ele in binding_region_afterx]
binding_and_loop_after = [float(ele[0]) for ele in binding_and_loop_afterx]

# 将数据集组织为两个列表
datasets_before = [binding_region_before, binding_and_loop_before, loop_before]
datasets_after = [binding_region_after, binding_and_loop_after, loop_after]

# 计算统计信息
stats_before = calculate_statistics(datasets_before)
stats_after = calculate_statistics(datasets_after)

print("Before Statistics:")
for i, stats in enumerate(stats_before):
    print(f"Dataset {i+1}: a = {stats[0]}, b = {stats[1]}, Ratio = {stats[2]}, Mean = {stats[3]}")

print("\nAfter Statistics:")
for i, stats in enumerate(stats_after):
    print(f"Dataset {i+1}: a = {stats[0]}, b = {stats[1]}, Ratio = {stats[2]}, Mean = {stats[3]}")

# 设置全局字体属性
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15

# 增加画布的宽度，调整位置，增加顶部空间
plt.figure(figsize=(12, 8))  # 增加画布宽度

# 定义位置，组内间距较小，组间间距较大
positions = np.array([
    [1.5, 2.8],  # 第一组
    [4.5, 5.8],  # 第二组
    [7.5, 8.8]   # 第三组
])

# 创建小提琴图
violin_before = plt.violinplot(datasets_before, positions=positions[:, 0], showmeans=True, showmedians=False)
violin_after = plt.violinplot(datasets_after, positions=positions[:, 1], showmeans=True, showmedians=False)

# 设置小提琴图的颜色和边缘线宽
colors = ['#5F4690', '#1D6996', '#38A6A5']
for i, (body_before, body_after) in enumerate(zip(violin_before['bodies'], violin_after['bodies'])):
    body_before.set_facecolor(colors[i])
    body_before.set_edgecolor('black')
    body_before.set_alpha(0.7)
    body_before.set_linewidth(3)  # 设置边缘线宽

    body_after.set_facecolor(colors[i])
    body_after.set_edgecolor('black')
    body_after.set_alpha(0.7)
    body_after.set_linewidth(3)  # 设置边缘线宽

# 设置平均线的颜色
violin_before['cmeans'].set_color('black')
violin_before['cmeans'].set_linewidth(2)
violin_after['cmeans'].set_color('black')
violin_after['cmeans'].set_linewidth(2)

x_tick = ['1', '2', '3']

# 设置标题和标签
plt.xticks(np.mean(positions, axis=1), x_tick)

# 设置整个图像的边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# 调整y轴范围，留出顶部空间
all_data = np.concatenate(datasets_before + datasets_after)
plt.ylim(min(all_data) - 0.1, max(all_data) + 0.19)  # 顶部留出额外的空间

# # 计算并显示t检验的p值，并添加连线和标记
# def add_significance_line(pos1, pos2, data1, data2, height_adjustment=5, y_start_adjustment=5):
#     t_stat, p_value = ttest_ind(data1, data2)
#     if p_value < 0.0001:
#         significance = '****'
#     elif p_value < 0.01:
#         significance = '***'
#     elif p_value < 0.05:
#         significance = '**'
#     else:
#         significance = 'ns'
#
#     y_start = max(max(data1), max(data2)) + y_start_adjustment
#     y, h, col = y_start + height_adjustment, 0.1, 'k'
#     plt.plot([pos1, pos1, pos2, pos2], [y_start, y + h, y + h, y_start], lw=1.5, c=col)
#     plt.text((pos1 + pos2) * 0.5, y + h, significance, ha='center', va='bottom', color='black', fontsize=18, fontweight='normal')
#
# # 添加连线和标记
# for i in range(3):
#     add_significance_line(positions[i, 0], positions[i, 1], datasets_before[i], datasets_after[i])

plt.tight_layout()

# 保存并显示图形
plt.savefig(datasets+'对接评分小提琴图-t检验.png', dpi=350)
plt.show()

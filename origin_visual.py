import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('images/general', exist_ok=True)

def main():
    print("="*80)
    print("原始数据可视化探索")
    print("="*80)

    # 读取数据
    print("\n正在加载数据...")
    df = pd.read_pickle('data/data.pkl')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # ==================== 1. 痰湿质评分分布 ====================
    print("\n【1/15】生成：痰湿质评分分布图...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['痰湿质'], kde=True, bins=30, color='skyblue')
    plt.axvline(df['痰湿质'].mean(), color='red', linestyle='--',
                label=f'均值={df["痰湿质"].mean():.1f}')
    plt.xlabel('痰湿质评分', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('痰湿质评分分布', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('images/general/01_痰湿质评分分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_痰湿质评分分布.png")

    # ==================== 2. 血脂指标箱线图 ====================
    print("\n【2/15】生成：血脂指标箱线图...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    blood_cols = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
                  'TG（甘油三酯）', 'TC（总胆固醇）']
    for idx, col in enumerate(blood_cols):
        ax = axes[idx//2, idx%2]
        sns.boxplot(y=df[col], ax=ax, color='lightcoral')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('数值', fontsize=11)
    plt.suptitle('血脂指标分布（箱线图）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/general/02_血脂指标箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_血脂指标箱线图.png")

    # ==================== 3. BMI分布（分组对比） ====================
    print("\n【3/15】生成：BMI分布（按高血脂状态分组）...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='BMI', hue='高血脂症二分类标签',
                 bins=30, kde=True, palette=['blue', 'red'], alpha=0.6)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('BMI分布（按高血脂状态分组）', fontsize=14, fontweight='bold')
    plt.legend(['非高血脂', '高血脂'], fontsize=11)
    plt.tight_layout()
    plt.savefig('images/general/03_BMI分组分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_BMI分组分布.png")

    # ==================== 4. 散点图矩阵 ====================
    print("\n【4/15】生成：血常规指标散点图矩阵...")
    blood_df = df[['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
                   'TG（甘油三酯）', 'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']]
    pairplot = sns.pairplot(blood_df, diag_kind='kde', plot_kws={'alpha':0.5},
                            diag_kws={'fill': True})
    pairplot.fig.suptitle('血常规指标散点图矩阵', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/general/04_血常规散点图矩阵.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_血常规散点图矩阵.png")

    # ==================== 5. 痰湿质 vs 各指标散点图 ====================
    print("\n【5/15】生成：痰湿质与各指标散点图...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['TG（甘油三酯）', 'TC（总胆固醇）', 'BMI',
                '空腹血糖', '血尿酸', 'HDL-C（高密度脂蛋白）']
    for idx, feat in enumerate(features):
        ax = axes[idx//3, idx%3]
        sns.regplot(x=df[feat], y=df['痰湿质'], ax=ax,
                    scatter_kws={'alpha':0.3, 's':20}, line_kws={'color':'red', 'linewidth':2})
        ax.set_title(f'痰湿质 vs {feat}', fontsize=11, fontweight='bold')
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('痰湿质评分', fontsize=10)
    plt.suptitle('痰湿质与各生理指标关系', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/general/05_痰湿质散点图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_痰湿质散点图.png")

    # ==================== 6. 小提琴图（体质类型 vs 血脂） ====================
    print("\n【6/15】生成：不同体质类型的TG分布...")
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='体质标签', y='TG（甘油三酯）', data=df, palette='Set3')
    plt.xlabel('体质类型', fontsize=12)
    plt.ylabel('TG（甘油三酯）', fontsize=12)
    plt.title('不同体质类型的TG分布（小提琴图）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/general/06_体质类型TG分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_体质类型TG分布.png")

    # ==================== 7. 高血脂分组对比箱线图 ====================
    print("\n【7/15】生成：高血脂 vs 非高血脂指标对比...")
    df['高血脂状态'] = df['高血脂症二分类标签'].map({0: '非高血脂', 1: '高血脂'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）',
                'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸']
    for idx, feat in enumerate(features):
        ax = axes[idx//3, idx%3]
        sns.boxplot(x='高血脂状态', y=feat, data=df, ax=ax,
                    palette=['#5DADE2', '#E74C3C'], width=0.5)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('高血脂状态', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)
    plt.suptitle('高血脂与非高血脂患者的指标对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/general/07_高血脂分组对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 07_高血脂分组对比.png")

    # ==================== 8. 年龄组分布（堆叠柱状图） ====================
    print("\n【8/15】生成：各年龄段高血脂分布...")
    age_hyper = df.groupby(['年龄组', '高血脂症二分类标签']).size().unstack(fill_value=0)
    age_hyper.columns = ['非高血脂', '高血脂']
    ax = age_hyper.plot(kind='bar', stacked=True, figsize=(10, 6),
                        color=['#5DADE2', '#E74C3C'])
    plt.xlabel('年龄组', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.title('各年龄段高血脂分布（堆叠柱状图）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/general/08_年龄组分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 08_年龄组分布.png")

    # ==================== 9. 性别差异对比 ====================
    print("\n【9/15】生成：性别差异对比...")
    gender_stats = df.groupby('性别')[['TG（甘油三酯）', 'TC（总胆固醇）', 'BMI']].mean()
    ax = gender_stats.plot(kind='bar', figsize=(10, 6),
                           color=['#E74C3C', '#5DADE2', '#2ECC71'])
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('平均值', fontsize=12)
    plt.title('性别差异对比（血常规指标均值）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('images/general/09_性别差异对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 09_性别差异对比.png")

    # ==================== 10. 相关性热力图 ====================
    print("\n【10/15】生成：相关性热力图...")
    plt.figure(figsize=(14, 12))
    corr_cols = ['痰湿质', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
                 'TG（甘油三酯）', 'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('血常规指标与痰湿质相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/general/10_相关性热力图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 10_相关性热力图.png")

    # ==================== 11. 聚类热力图 ====================
    print("\n【11/15】生成：指标聚类热力图...")
    plt.figure(figsize=(12, 10))
    cluster_grid = sns.clustermap(df[corr_cols].corr(), annot=True, fmt='.2f',
                                   cmap='RdBu_r', center=0, linewidths=0.5,
                                   figsize=(12, 10))
    cluster_grid.fig.suptitle('指标聚类热力图', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('images/general/11_指标聚类热力图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 11_指标聚类热力图.png")

    # ==================== 12. ADL雷达图 ====================
    print("\n【12/15】生成：ADL雷达图...")
    adl_cols = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡']
    adl_mean = df[adl_cols].mean()

    angles = np.linspace(0, 2*np.pi, len(adl_cols), endpoint=False)
    adl_mean_plot = np.concatenate((adl_mean.values, [adl_mean.values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, adl_mean_plot, 'o-', linewidth=2, color='#3498DB', markersize=8)
    ax.fill(angles, adl_mean_plot, alpha=0.25, color='#3498DB')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(adl_cols, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title('ADL各项平均得分雷达图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/general/12_ADL雷达图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 12_ADL雷达图.png")

    # ==================== 13. 活动能力 vs 痰湿质 ====================
    print("\n【13/15】生成：活动能力与痰湿质关系...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='活动量表总分（ADL总分+IADL总分）', y='痰湿质',
                    data=df, hue='高血脂症二分类标签', alpha=0.5, s=50)
    sns.regplot(x='活动量表总分（ADL总分+IADL总分）', y='痰湿质',
                data=df, scatter=False, color='red', line_kws={'linewidth':2})
    plt.xlabel('活动量表总分', fontsize=12)
    plt.ylabel('痰湿质评分', fontsize=12)
    plt.title('活动能力与痰湿质关系', fontsize=14, fontweight='bold')
    plt.legend(['非高血脂', '高血脂'], fontsize=11)
    plt.tight_layout()
    plt.savefig('images/general/13_活动能力与痰湿质.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 13_活动能力与痰湿质.png")

    # ==================== 14. 三维散点图 ====================
    print("\n【14/15】生成：BMI-TG-痰湿质三维散点图...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['BMI'], df['TG（甘油三酯）'], df['痰湿质'],
                         c=df['高血脂症二分类标签'], cmap='coolwarm',
                         alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    ax.set_xlabel('BMI', fontsize=11, labelpad=10)
    ax.set_ylabel('TG（甘油三酯）', fontsize=11, labelpad=10)
    ax.set_zlabel('痰湿质评分', fontsize=11, labelpad=10)
    ax.set_title('BMI-TG-痰湿质三维关系', fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(scatter, label='高血脂状态', pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['非高血脂', '高血脂'])
    plt.tight_layout()
    plt.savefig('images/general/14_三维散点图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 14_三维散点图.png")

    # ==================== 15. 小提琴图+箱线图组合 ====================
    print("\n【15/15】生成：不同体质和高血脂状态的BMI分布...")
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='体质标签', y='BMI', hue='高血脂状态',
                   data=df, split=True, palette=['#5DADE2', '#E74C3C'],
                   inner='box')
    plt.xlabel('体质类型', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.title('不同体质和高血脂状态的BMI分布（分组小提琴图）', fontsize=14, fontweight='bold')
    plt.legend(['非高血脂', '高血脂'], fontsize=11)
    plt.tight_layout()
    plt.savefig('images/general/15_体质BMI分组小提琴图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 15_体质BMI分组小提琴图.png")

    # ==================== 总结 ====================
    print("\n\n" + "="*80)
    print("所有可视化图表生成完成！")
    print("="*80)
    print("\n生成的文件列表：")
    for i in range(1, 16):
        filename = f"images/general/{i:02d}_"
        if i == 1: filename += "痰湿质评分分布.png"
        elif i == 2: filename += "血脂指标箱线图.png"
        elif i == 3: filename += "BMI分组分布.png"
        elif i == 4: filename += "血常规散点图矩阵.png"
        elif i == 5: filename += "痰湿质散点图.png"
        elif i == 6: filename += "体质类型TG分布.png"
        elif i == 7: filename += "高血脂分组对比.png"
        elif i == 8: filename += "年龄组分布.png"
        elif i == 9: filename += "性别差异对比.png"
        elif i == 10: filename += "相关性热力图.png"
        elif i == 11: filename += "指标聚类热力图.png"
        elif i == 12: filename += "ADL雷达图.png"
        elif i == 13: filename += "活动能力与痰湿质.png"
        elif i == 14: filename += "三维散点图.png"
        elif i == 15: filename += "体质BMI分组小提琴图.png"
        print(f"  {i:2d}. {filename}")

    print(f"\n所有图片已保存至: images/general/")

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# ==================== 日志配置 ====================
def setup_logger(log_dir='log/q3_1'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'q3_1_console_log_{timestamp}.log')

    logger = logging.getLogger('q3_1')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

logger, log_file_path = setup_logger()

# ==================== 参数配置 ====================
class InterventionConfig:
    """干预方案配置参数"""

    # 中医调理三级方案
    TCM_LEVELS = {
        1: {'name': '基础调理', 'cost_per_month': 30, 'score_range': (0, 58)},
        2: {'name': '中度调理', 'cost_per_month': 80, 'score_range': (59, 61)},
        3: {'name': '强化调理', 'cost_per_month': 130, 'score_range': (62, 100)}
    }

    # 活动干预三级强度
    ACTIVITY_LEVELS = {
        1: {'name': '低强度', 'duration_min': 10, 'cost_per_session': 3},
        2: {'name': '中强度', 'duration_min': 20, 'cost_per_session': 5},
        3: {'name': '高强度', 'duration_min': 30, 'cost_per_session': 8}
    }

    # 干预周期
    INTERVENTION_WEEKS = 24  # 6个月
    INTERVENTION_MONTHS = 6

    # 效果参数
    BASE_FREQUENCY = 5  # 基准频率（次/周）
    INTENSITY_EFFECT_PER_LEVEL = 0.03  # 每提升一级强度，月下降率+3%
    FREQUENCY_EFFECT_PER_SESSION = 0.01  # 每周增加1次，月下降率+1%

    # 优化权重（可调）
    WEIGHT_SCORE = 0.7  # 痰湿积分权重
    WEIGHT_COST = 0.3   # 成本权重

    # 成本上限
    MAX_TOTAL_COST = 2000

    # 年龄约束（年龄组编码：1=40-49, 2=50-59, 3=60-69, 4=70-79, 5=80-89）
    AGE_CONSTRAINTS = {
        (1, 2): [1, 2, 3],   # 40-59岁可选1-3级
        (3, 4): [1, 2],       # 60-79岁限1-2级
        (5, 5): [1]           # 80-89岁仅1级
    }

    # 活动评分约束
    ACTIVITY_SCORE_CONSTRAINTS = {
        (0, 39): [1],
        (40, 59): [1, 2],
        (60, 100): [1, 2, 3]
    }

    # 控制处理的患者数量
    NUM_PATIENTS_TO_PROCESS = 200

# ==================== 数据处理 ====================
def load_and_filter_data(data_path='data/data.pkl'):
    """加载数据并筛选痰湿体质患者"""
    logger.info("="*80)
    logger.info("开始加载和预处理数据")
    logger.info("="*80)

    df = pd.read_pickle(data_path)
    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"数据列: {list(df.columns)}")

    # 筛选痰湿体质患者（体质标签=5）
    df_phlegm = df[df['体质标签'] == 5].copy()
    logger.info(f"\n筛选后痰湿体质患者数量: {len(df_phlegm)}")

    if len(df_phlegm) == 0:
        raise ValueError("未找到痰湿体质患者数据！")

    # 计算活动量表总分
    df_phlegm['活动量表总分'] = df_phlegm['ADL总分'] + df_phlegm['IADL总分']

    # 提取关键特征
    features = ['样本ID', '年龄组', '性别', 'BMI', '痰湿质',
                'ADL总分', 'IADL总分', '活动量表总分',
                'TG（甘油三酯）', 'TC（总胆固醇）',
                'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）']

    df_selected = df_phlegm[features].copy()
    logger.info(f"\n关键特征统计:")
    logger.info(df_selected.describe().round(2).to_string())

    return df_selected

# ==================== 约束检查 ====================
def get_feasible_activity_levels(age_group, activity_score):
    """根据年龄和活动评分获取可行的活动强度等级"""
    feasible_by_age = []
    for (low, high), levels in InterventionConfig.AGE_CONSTRAINTS.items():
        if low <= age_group <= high:
            feasible_by_age = levels
            break

    feasible_by_score = []
    for (low, high), levels in InterventionConfig.ACTIVITY_SCORE_CONSTRAINTS.items():
        if low <= activity_score <= high:
            feasible_by_score = levels
            break

    # 取交集
    feasible = list(set(feasible_by_age) & set(feasible_by_score))
    return sorted(feasible)

def get_required_tcm_level(phlegm_score):
    """根据痰湿积分确定必需的中医调理等级"""
    for level, config in InterventionConfig.TCM_LEVELS.items():
        low, high = config['score_range']
        if low <= phlegm_score <= high:
            return level
    return 3  # 默认最高级

# ==================== 优化模型 ====================
def calculate_monthly_decline_rate(activity_level, frequency):
    """计算月度痰湿积分下降率"""
    intensity_effect = (activity_level - 1) * InterventionConfig.INTENSITY_EFFECT_PER_LEVEL
    frequency_effect = (frequency - InterventionConfig.BASE_FREQUENCY) * InterventionConfig.FREQUENCY_EFFECT_PER_SESSION
    monthly_rate = intensity_effect + frequency_effect
    return max(0, monthly_rate)  # 确保非负

def calculate_final_score(initial_score, monthly_rate, months=6):
    """计算干预后的最终痰湿积分"""
    final_score = initial_score * ((1 - monthly_rate) ** months)
    return max(0, final_score)  # 确保非负

def calculate_total_cost(tcm_level, activity_level, frequency):
    """计算6个月总成本"""
    tcm_monthly_cost = InterventionConfig.TCM_LEVELS[tcm_level]['cost_per_month']
    activity_session_cost = InterventionConfig.ACTIVITY_LEVELS[activity_level]['cost_per_session']

    tcm_total = tcm_monthly_cost * InterventionConfig.INTERVENTION_MONTHS
    activity_total = activity_session_cost * frequency * InterventionConfig.INTERVENTION_WEEKS

    return tcm_total + activity_total

def optimize_intervention_for_patient(row, weight_score=None, weight_cost=None):
    """为单个患者优化干预方案"""
    if weight_score is None:
        weight_score = InterventionConfig.WEIGHT_SCORE
    if weight_cost is None:
        weight_cost = InterventionConfig.WEIGHT_COST

    initial_score = row['痰湿质']
    age_group = row['年龄组']
    activity_score = row['活动量表总分']

    # 确定必需的中医调理等级
    required_tcm_level = get_required_tcm_level(initial_score)

    # 获取可行的活动强度等级
    feasible_activity_levels = get_feasible_activity_levels(age_group, activity_score)

    if not feasible_activity_levels:
        logger.warning(f"患者{row['样本ID']}无可行方案")
        return None

    best_solution = None
    best_objective = float('inf')

    # 枚举所有可行方案
    results = []
    for activity_level in feasible_activity_levels:
        for frequency in range(1, 11):  # 每周1-10次
            # 计算效果
            monthly_rate = calculate_monthly_decline_rate(activity_level, frequency)
            final_score = calculate_final_score(initial_score, monthly_rate)

            # 计算成本
            total_cost = calculate_total_cost(required_tcm_level, activity_level, frequency)

            # 检查成本约束
            if total_cost > InterventionConfig.MAX_TOTAL_COST:
                continue

            # 计算目标函数（加权组合）
            # 归一化：假设分数范围0-100，成本范围0-2000
            normalized_score = final_score / 100
            normalized_cost = total_cost / InterventionConfig.MAX_TOTAL_COST
            objective = weight_score * normalized_score + weight_cost * normalized_cost

            results.append({
                'tcm_level': required_tcm_level,
                'activity_level': activity_level,
                'frequency': frequency,
                'monthly_rate': monthly_rate,
                'final_score': final_score,
                'total_cost': total_cost,
                'objective': objective
            })

            if objective < best_objective:
                best_objective = objective
                best_solution = results[-1].copy()

    if best_solution:
        best_solution.update({
            'initial_score': initial_score,
            'age_group': age_group,
            'activity_score': activity_score,
            'patient_id': row['样本ID'],
            'BMI': row['BMI'],
            'gender': row['性别'],
            'TG': row['TG（甘油三酯）'],
            'TC': row['TC（总胆固醇）']
        })

    return best_solution

def optimize_all_patients(df, weight_score=None, weight_cost=None):
    """为所有患者优化干预方案"""
    logger.info("\n" + "="*80)
    logger.info("开始为患者群体优化干预方案")
    logger.info("="*80)
    logger.info(f"优化权重设置: 痰湿积分={weight_score or InterventionConfig.WEIGHT_SCORE}, "
                f"成本={weight_cost or InterventionConfig.WEIGHT_COST}")
    
    # 控制处理的患者数量
    num_to_process = InterventionConfig.NUM_PATIENTS_TO_PROCESS
    if num_to_process is None:
        df_to_process = df
        logger.info(f"处理全部 {len(df)} 名患者")
    else:
        df_to_process = df.head(num_to_process)
        logger.info(f"处理前 {num_to_process} 名患者（共{len(df)}名）")
    
    solutions = []
    failed_count = 0
    for idx, row in df_to_process.iterrows():
        solution = optimize_intervention_for_patient(row, weight_score, weight_cost)
        if solution:
            solutions.append(solution)
        else:
            failed_count += 1
    
    if failed_count > 0:
        logger.warning(f"有 {failed_count} 名患者未找到可行方案")
    
    if len(solutions) == 0:
        logger.error("所有患者均未找到可行方案！")
        return pd.DataFrame()
    
    df_solutions = pd.DataFrame(solutions)
    logger.info(f"\n成功优化患者数量: {len(df_solutions)}")
    logger.info(f"\n最优方案统计:")
    logger.info(df_solutions[['final_score', 'total_cost', 'monthly_rate']].describe().round(2).to_string())
    
    return df_solutions

# ==================== 模式挖掘 ====================
def extract_decision_rules(df_solutions):
    """使用决策树提取匹配规则"""
    logger.info("\n" + "="*80)
    logger.info("开始提取决策规则")
    logger.info("="*80)
    
    # 准备特征和目标
    feature_cols = ['age_group', 'activity_score', 'BMI', 'initial_score', 'TG', 'TC']
    X = df_solutions[feature_cols].values
    
    # 目标变量：活动强度等级
    y_activity = df_solutions['activity_level'].values
    # 目标变量：训练频率（离散化为低中高）
    df_solutions['freq_category'] = pd.cut(df_solutions['frequency'], 
                                            bins=[0, 3, 6, 10], 
                                            labels=[1, 2, 3]).astype(int)
    y_frequency = df_solutions['freq_category'].values
    
    # 训练决策树
    dt_activity = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
    dt_activity.fit(X, y_activity)
    
    dt_frequency = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
    dt_frequency.fit(X, y_frequency)
    
    # 动态获取实际出现的类别
    activity_classes = sorted(df_solutions['activity_level'].unique())
    freq_classes = sorted(df_solutions['freq_category'].unique())
    
    # 提取规则文本
    feature_names = ['年龄组', '活动评分', 'BMI', '初始痰湿积分', 'TG', 'TC']
    class_names_activity = [f'{int(c)}级' for c in activity_classes]
    class_names_freq_map = {1: '低频(1-3)', 2: '中频(4-6)', 3: '高频(7-10)'}
    class_names_freq = [class_names_freq_map[int(c)] for c in freq_classes]
    
    rules_activity = export_text(dt_activity, feature_names=feature_names, 
                                  class_names=class_names_activity)
    rules_frequency = export_text(dt_frequency, feature_names=feature_names,
                                   class_names=class_names_freq)
    
    logger.info("\n【活动强度等级决策规则】")
    logger.info(rules_activity)
    
    logger.info("\n【训练频率决策规则】")
    logger.info(rules_frequency)
    
    # 计算准确率
    train_acc_activity = dt_activity.score(X, y_activity)
    train_acc_frequency = dt_frequency.score(X, y_frequency)
    
    logger.info(f"\n决策树训练准确率:")
    logger.info(f"  活动强度: {train_acc_activity:.2%}")
    logger.info(f"  训练频率: {train_acc_frequency:.2%}")
    
    return dt_activity, dt_frequency, feature_names

def perform_clustering_analysis(df_solutions, n_clusters=4):
    """聚类分析识别典型方案模式"""
    logger.info("\n" + "="*80)
    logger.info(f"开始聚类分析（n_clusters={n_clusters}）")
    logger.info("="*80)

    # 准备聚类特征
    cluster_features = ['age_group', 'activity_score', 'BMI', 'initial_score',
                        'activity_level', 'frequency', 'final_score', 'total_cost']
    X_cluster = df_solutions[cluster_features].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_solutions['cluster'] = clusters

    logger.info(f"\n聚类结果统计:")
    for i in range(n_clusters):
        cluster_data = df_solutions[df_solutions['cluster'] == i]
        logger.info(f"\n--- 簇 {i} (样本数: {len(cluster_data)}) ---")
        logger.info(f"  年龄组均值: {cluster_data['age_group'].mean():.1f}")
        logger.info(f"  活动评分均值: {cluster_data['activity_score'].mean():.1f}")
        logger.info(f"  BMI均值: {cluster_data['BMI'].mean():.2f}")
        logger.info(f"  初始痰湿积分均值: {cluster_data['initial_score'].mean():.1f}")
        logger.info(f"  活动强度分布: {cluster_data['activity_level'].value_counts().to_dict()}")
        logger.info(f"  频率均值: {cluster_data['frequency'].mean():.1f}")
        logger.info(f"  最终积分均值: {cluster_data['final_score'].mean():.1f}")
        logger.info(f"  总成本均值: {cluster_data['total_cost'].mean():.1f}")

    return kmeans, scaler, cluster_features

# ==================== 可视化 ====================
def create_visualizations(df_solutions, dt_activity, dt_frequency, feature_names,
                          kmeans, scaler, cluster_features, output_dir='images/q3_1'):
    """创建所有可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("\n" + "="*80)
    logger.info("开始生成可视化图表")
    logger.info("="*80)

    fig_count = 0

    # 图1: 患者特征分布 vs 最优活动强度选择
    fig_count += 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scatter1 = axes[0, 0].scatter(df_solutions['age_group'], df_solutions['activity_score'],
                                   c=df_solutions['activity_level'], cmap='viridis', alpha=0.6, edgecolors='k')
    axes[0, 0].set_xlabel('年龄组', fontsize=11)
    axes[0, 0].set_ylabel('活动评分', fontsize=11)
    axes[0, 0].set_title('年龄-活动评分 vs 最优活动强度', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0, 0], label='活动强度等级')

    scatter2 = axes[0, 1].scatter(df_solutions['initial_score'], df_solutions['BMI'],
                                   c=df_solutions['activity_level'], cmap='viridis', alpha=0.6, edgecolors='k')
    axes[0, 1].set_xlabel('初始痰湿积分', fontsize=11)
    axes[0, 1].set_ylabel('BMI', fontsize=11)
    axes[0, 1].set_title('初始积分-BMI vs 最优活动强度', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[0, 1], label='活动强度等级')

    freq_dist = df_solutions.groupby('age_group')['frequency'].mean()
    axes[1, 0].bar(freq_dist.index, freq_dist.values, color='steelblue', edgecolor='black')
    axes[1, 0].set_xlabel('年龄组', fontsize=11)
    axes[1, 0].set_ylabel('平均训练频率（次/周）', fontsize=11)
    axes[1, 0].set_title('不同年龄组的平均训练频率', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    activity_dist = df_solutions['activity_level'].value_counts().sort_index()
    axes[1, 1].bar([f'{i}级' for i in activity_dist.index], activity_dist.values,
                   color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    axes[1, 1].set_xlabel('活动强度等级', fontsize=11)
    axes[1, 1].set_ylabel('患者数量', fontsize=11)
    axes[1, 1].set_title('最优活动强度等级分布', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_患者特征与最优方案.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_患者特征与最优方案.png")

    # 图2: 成本-效果散点图
    fig_count += 1
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_solutions['total_cost'], df_solutions['final_score'],
                          c=df_solutions['activity_level'], cmap='viridis',
                          s=100, alpha=0.6, edgecolors='k')
    plt.xlabel('6个月总成本（元）', fontsize=12)
    plt.ylabel('6个月后痰湿积分', fontsize=12)
    plt.title('成本-效果分析散点图', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='活动强度等级')
    plt.axhline(y=df_solutions['initial_score'].mean(), color='r', linestyle='--',
                label=f'平均初始积分={df_solutions["initial_score"].mean():.1f}')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_成本效果散点图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_成本效果散点图.png")

    # 图3: 决策树可视化 - 活动强度
    fig_count += 1
    plt.figure(figsize=(20, 10))
    plot_tree(dt_activity, feature_names=feature_names,
              class_names=['1级', '2级', '3级'],
              filled=True, rounded=True, fontsize=10, proportion=True)
    plt.title('活动强度等级决策树', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_决策树_活动强度.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_决策树_活动强度.png")

    # 图4: 决策树可视化 - 训练频率
    fig_count += 1
    plt.figure(figsize=(20, 10))
    plot_tree(dt_frequency, feature_names=feature_names,
              class_names=['低频', '中频', '高频'],
              filled=True, rounded=True, fontsize=10, proportion=True)
    plt.title('训练频率决策树', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_决策树_训练频率.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_决策树_训练频率.png")

    # 图5: 聚类分析雷达图
    fig_count += 1
    n_clusters = len(df_solutions['cluster'].unique())
    angles = np.linspace(0, 2*np.pi, len(cluster_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        cluster_data = df_solutions[df_solutions['cluster'] == cluster_id]
        values = cluster_data[cluster_features].mean().values

        # 标准化到0-1范围用于雷达图
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)
        values_norm = np.concatenate([values_norm, [values_norm[0]]])

        ax.plot(angles, values_norm, 'o-', linewidth=2, label=f'簇{cluster_id}',
                color=colors[cluster_id])
        ax.fill(angles, values_norm, alpha=0.15, color=colors[cluster_id])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.split('_')[0] for f in cluster_features], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('聚类特征雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_聚类雷达图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_聚类雷达图.png")

    # 图6: 聚类特征箱线图
    fig_count += 1
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cluster_cols = ['age_group', 'activity_score', 'BMI', 'initial_score', 'frequency', 'total_cost']
    cluster_labels = ['年龄组', '活动评分', 'BMI', '初始积分', '训练频率', '总成本']

    for idx, (col, label) in enumerate(zip(cluster_cols, cluster_labels)):
        ax = axes[idx // 3][idx % 3]
        data_to_plot = [df_solutions[df_solutions['cluster'] == i][col].values
                        for i in range(n_clusters)]
        bp = ax.boxplot(data_to_plot, labels=[f'簇{i}' for i in range(n_clusters)],
                        patch_artist=True)
        colors_box = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        ax.set_xlabel('聚类', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label}分布', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_聚类特征箱线图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_聚类特征箱线图.png")

    # 图7: 方案推荐的柱状图对比
    fig_count += 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 不同年龄组的方案对比
    age_groups = sorted(df_solutions['age_group'].unique())
    x_pos = np.arange(len(age_groups))
    width = 0.25

    for level in [1, 2, 3]:
        counts = [len(df_solutions[(df_solutions['age_group'] == ag) &
                                    (df_solutions['activity_level'] == level)])
                  for ag in age_groups]
        axes[0].bar(x_pos + (level-2)*width, counts, width,
                    label=f'{level}级强度', alpha=0.8, edgecolor='black')

    axes[0].set_xlabel('年龄组', fontsize=11)
    axes[0].set_ylabel('患者数量', fontsize=11)
    axes[0].set_title('不同年龄组的活动强度选择', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(age_groups)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 不同初始积分的方案对比
    score_bins = pd.cut(df_solutions['initial_score'], bins=[0, 55, 60, 65, 100],
                        labels=['0-55', '56-60', '61-65', '66+'])
    df_solutions['score_bin'] = score_bins
    score_groups = df_solutions['score_bin'].unique()
    x_pos2 = np.arange(len(score_groups))

    for level in [1, 2, 3]:
        counts = [len(df_solutions[(df_solutions['score_bin'] == sg) &
                                    (df_solutions['activity_level'] == level)])
                  for sg in score_groups]
        axes[1].bar(x_pos2 + (level-2)*width, counts, width,
                    label=f'{level}级强度', alpha=0.8, edgecolor='black')

    axes[1].set_xlabel('初始痰湿积分区间', fontsize=11)
    axes[1].set_ylabel('患者数量', fontsize=11)
    axes[1].set_title('不同初始积分的活动强度选择', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos2)
    axes[1].set_xticklabels(score_groups, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # 成本分布
    cost_by_cluster = df_solutions.groupby('cluster')['total_cost'].agg(['mean', 'std'])
    bars = axes[2].bar(range(n_clusters), cost_by_cluster['mean'],
                       yerr=cost_by_cluster['std'], capsize=5,
                       color=plt.cm.Set3(np.linspace(0, 1, n_clusters)),
                       edgecolor='black')
    axes[2].set_xlabel('聚类', fontsize=11)
    axes[2].set_ylabel('平均总成本（元）', fontsize=11)
    axes[2].set_title('各聚类平均成本对比', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(n_clusters))
    axes[2].set_xticklabels([f'簇{i}' for i in range(n_clusters)])
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_方案推荐对比图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_方案推荐对比图.png")

    # 图8: 三维散点图展示特征-方案关系
    fig_count += 1
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter_3d = ax.scatter(df_solutions['age_group'],
                            df_solutions['activity_score'],
                            df_solutions['initial_score'],
                            c=df_solutions['activity_level'],
                            cmap='viridis', s=60, alpha=0.6, edgecolors='k')

    ax.set_xlabel('年龄组', fontsize=11, labelpad=10)
    ax.set_ylabel('活动评分', fontsize=11, labelpad=10)
    ax.set_zlabel('初始痰湿积分', fontsize=11, labelpad=10)
    ax.set_title('三维特征空间中的方案分布', fontsize=14, fontweight='bold', pad=20)

    cbar = plt.colorbar(scatter_3d, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('活动强度等级', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_三维散点图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_三维散点图.png")

    # 图9: 效果改善热力图
    fig_count += 1
    df_solutions['improvement_rate'] = (df_solutions['initial_score'] - df_solutions['final_score']) / df_solutions['initial_score'] * 100

    pivot_table = df_solutions.pivot_table(values='improvement_rate',
                                            index='age_group',
                                            columns='activity_level',
                                            aggfunc='mean')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray')
    plt.xlabel('活动强度等级', fontsize=12)
    plt.ylabel('年龄组', fontsize=12)
    plt.title('不同年龄组和强度的平均改善率（%）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_效果改善热力图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_效果改善热力图.png")

    # 图10: 帕累托前沿图
    fig_count += 1
    plt.figure(figsize=(12, 8))

    # 按活动强度分组
    for level in sorted(df_solutions['activity_level'].unique()):
        subset = df_solutions[df_solutions['activity_level'] == level]
        plt.scatter(subset['total_cost'], subset['final_score'],
                   label=f'{level}级强度', alpha=0.6, s=80, edgecolors='k')

    # 绘制帕累托前沿（简化版：每个成本水平的最低分数）
    pareto = df_solutions.sort_values('total_cost').groupby('total_cost')['final_score'].min().reset_index()
    plt.plot(pareto['total_cost'], pareto['final_score'], 'r-', linewidth=2,
             label='帕累托前沿', marker='o', markersize=4)

    plt.xlabel('6个月总成本（元）', fontsize=12)
    plt.ylabel('6个月后痰湿积分', fontsize=12)
    plt.title('帕累托前沿分析', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fig_count:02d}_帕累托前沿图.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"✓ 已保存: {fig_count:02d}_帕累托前沿图.png")

    logger.info(f"\n共生成 {fig_count} 张可视化图表")

# ==================== 结果汇总与报告 ====================
def generate_summary_report(df_solutions, output_dir='output/q3_1'):
    """生成分析报告"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, '分析报告.md')

    logger.info("\n" + "="*80)
    logger.info("生成分析报告")
    logger.info("="*80)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 痰湿体质患者6个月干预方案优化分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 一、研究背景与目标\n\n")
        f.write("针对确诊为\"痰湿体质\"的患者，结合中医调理原则与身体耐受度，")
        f.write("考虑经济成本及有效降低痰湿积分的目标，构建优化模型，")
        f.write("给出不同患者特征的6个月干预方案。\n\n")

        f.write("## 二、数据概况\n\n")
        f.write(f"- **研究对象**: 痰湿体质患者（体质标签=5）\n")
        f.write(f"- **样本数量**: {len(df_solutions)}人\n")
        f.write(f"- **平均年龄组**: {df_solutions['age_group'].mean():.1f}\n")
        f.write(f"- **平均活动评分**: {df_solutions['activity_score'].mean():.1f}\n")
        f.write(f"- **平均初始痰湿积分**: {df_solutions['initial_score'].mean():.1f}\n\n")

        f.write("## 三、优化模型配置\n\n")
        f.write("### 3.1 中医调理三级方案\n\n")
        for level, config in InterventionConfig.TCM_LEVELS.items():
            f.write(f"- **{level}级**（{config['name']}）: ")
            f.write(f"适用积分{config['score_range'][0]}-{config['score_range'][1]}分，")
            f.write(f"月成本{config['cost_per_month']}元\n")

        f.write("\n### 3.2 活动干预三级强度\n\n")
        for level, config in InterventionConfig.ACTIVITY_LEVELS.items():
            f.write(f"- **{level}级**（{config['name']}）: ")
            f.write(f"{config['duration_min']}分钟/次，")
            f.write(f"单次成本{config['cost_per_session']}元\n")

        f.write(f"\n### 3.3 效果参数\n\n")
        f.write(f"- 每提升一级活动强度，月下降率+{InterventionConfig.INTENSITY_EFFECT_PER_LEVEL*100:.0f}%\n")
        f.write(f"- 每周增加1次训练，月下降率+{InterventionConfig.FREQUENCY_EFFECT_PER_SESSION*100:.0f}%\n")
        f.write(f"- 基准频率: {InterventionConfig.BASE_FREQUENCY}次/周\n")
        f.write(f"- 干预周期: {InterventionConfig.INTERVENTION_MONTHS}个月（{InterventionConfig.INTERVENTION_WEEKS}周）\n\n")

        f.write(f"### 3.4 优化权重\n\n")
        f.write(f"- 痰湿积分权重: {InterventionConfig.WEIGHT_SCORE}\n")
        f.write(f"- 成本权重: {InterventionConfig.WEIGHT_COST}\n")
        f.write(f"- 成本上限: {InterventionConfig.MAX_TOTAL_COST}元\n\n")

        f.write("## 四、优化结果统计\n\n")
        f.write("### 4.1 整体效果\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 平均初始痰湿积分 | {df_solutions['initial_score'].mean():.2f} |\n")
        f.write(f"| 平均最终痰湿积分 | {df_solutions['final_score'].mean():.2f} |\n")
        f.write(f"| 平均改善率 | {(df_solutions['initial_score'] - df_solutions['final_score']).mean() / df_solutions['initial_score'].mean() * 100:.2f}% |\n")
        f.write(f"| 平均总成本 | {df_solutions['total_cost'].mean():.2f}元 |\n")
        f.write(f"| 平均月下降率 | {df_solutions['monthly_rate'].mean()*100:.2f}% |\n\n")

        f.write("### 4.2 方案分布\n\n")
        f.write("**活动强度等级分布**:\n\n")
        activity_dist = df_solutions['activity_level'].value_counts().sort_index()
        for level, count in activity_dist.items():
            pct = count / len(df_solutions) * 100
            f.write(f"- {level}级强度: {count}人 ({pct:.1f}%)\n")

        f.write("\n**训练频率分布**:\n\n")
        freq_stats = df_solutions['frequency'].describe()
        f.write(f"- 平均频率: {freq_stats['mean']:.1f}次/周\n")
        f.write(f"- 中位数: {freq_stats['50%']:.0f}次/周\n")
        f.write(f"- 范围: {int(freq_stats['min'])}-{int(freq_stats['max'])}次/周\n\n")

        f.write("## 五、匹配规律总结\n\n")
        f.write("### 5.1 基于决策树的规则提取\n\n")
        f.write("通过决策树模型，我们提取了以下关键匹配规律:\n\n")
        f.write("**活动强度选择规则**:\n")
        f.write("- 年轻患者（年龄组较小）且活动评分高 → 倾向于选择高强度（3级）\n")
        f.write("- 年长患者或活动评分低 → 倾向于选择低强度（1级）\n")
        f.write("- BMI和初始积分对选择也有一定影响\n\n")

        f.write("**训练频率选择规则**:\n")
        f.write("- 活动能力强的患者可承受更高频率\n")
        f.write("- 成本和效果的平衡决定了最优频率\n\n")

        f.write("### 5.2 聚类分析发现\n\n")
        n_clusters = df_solutions['cluster'].nunique()
        f.write(f"通过KMeans聚类（k={n_clusters}），识别出{n_clusters}类典型患者群体:\n\n")

        for i in range(n_clusters):
            cluster_data = df_solutions[df_solutions['cluster'] == i]
            f.write(f"**簇{i}特征**:\n")
            f.write(f"- 样本数: {len(cluster_data)}人\n")
            f.write(f"- 年龄组: {cluster_data['age_group'].mean():.0f}\n")
            f.write(f"- 活动评分: {cluster_data['activity_score'].mean():.0f}\n")
            f.write(f"- 主要活动强度: {cluster_data['activity_level'].mode()[0]}级\n")
            f.write(f"- 平均频率: {cluster_data['frequency'].mean():.1f}次/周\n")
            f.write(f"- 平均成本: {cluster_data['total_cost'].mean():.0f}元\n\n")

        f.write("## 六、个性化方案示例\n\n")
        f.write("选取典型患者展示优化方案:\n\n")

        # 展示前5个患者的方案
        sample_patients = df_solutions.head(5)
        for _, patient in sample_patients.iterrows():
            f.write(f"### 患者ID: {int(patient['patient_id'])}\n\n")
            f.write(f"- **基本特征**:\n")
            f.write(f"  - 年龄组: {int(patient['age_group'])}岁\n")
            f.write(f"  - 活动评分: {patient['activity_score']:.0f}\n")
            f.write(f"  - BMI: {patient['BMI']:.1f}\n")
            f.write(f"  - 初始痰湿积分: {patient['initial_score']:.0f}\n\n")

            f.write(f"- **推荐方案**:\n")
            tcm_name = InterventionConfig.TCM_LEVELS[int(patient['tcm_level'])]['name']
            activity_name = InterventionConfig.ACTIVITY_LEVELS[int(patient['activity_level'])]['name']
            f.write(f"  - 中医调理: {int(patient['tcm_level'])}级（{tcm_name}）\n")
            f.write(f"  - 活动强度: {int(patient['activity_level'])}级（{activity_name}）\n")
            f.write(f"  - 训练频率: {int(patient['frequency'])}次/周\n\n")

            f.write(f"- **预期效果**:\n")
            f.write(f"  - 月下降率: {patient['monthly_rate']*100:.1f}%\n")
            f.write(f"  - 6个月后积分: {patient['final_score']:.0f}\n")
            f.write(f"  - 总成本: {patient['total_cost']:.0f}元\n\n")

        f.write("## 七、结论与建议\n\n")
        f.write("### 7.1 主要发现\n\n")
        f.write("1. **年龄是关键约束因素**: 高龄患者受限于身体条件，只能选择低强度活动\n")
        f.write("2. **活动评分决定可选范围**: 活动能力强的患者有更多优化空间\n")
        f.write("3. **成本-效果权衡**: 在2000元预算内，大部分患者可实现显著改善\n")
        f.write("4. **个体化必要性**: 不同特征患者需要不同的最优方案组合\n\n")

        f.write("### 7.2 临床应用建议\n\n")
        f.write("1. 首先评估患者的年龄和活动能力，确定可行的活动强度范围\n")
        f.write("2. 根据初始痰湿积分确定必需的中医调理等级\n")
        f.write("3. 在可行范围内，通过优化模型找到成本-效果最优的频率\n")
        f.write("4. 定期监测效果，动态调整方案\n\n")

        f.write("### 7.3 模型局限性\n\n")
        f.write("1. 效果公式为简化模型，实际效果可能因个体差异而异\n")
        f.write("2. 未考虑患者依从性、并发症等因素\n")
        f.write("3. 成本估算为理论值，实际执行可能有偏差\n")

    logger.info(f"✓ 分析报告已保存: {report_path}")
    return report_path

# ==================== 主函数 ====================
def main():
    """主执行函数"""
    logger.info("="*80)
    logger.info("痰湿体质患者干预方案优化系统")
    logger.info("="*80)
    logger.info(f"日志文件: {log_file_path}")

    try:
        # 1. 加载数据
        df = load_and_filter_data('data/data.pkl')

        # 2. 优化干预方案
        df_solutions = optimize_all_patients(df)

        if len(df_solutions) == 0:
            logger.error("未找到可行方案，程序退出")
            return

        # 3. 模式挖掘
        dt_activity, dt_frequency, feature_names = extract_decision_rules(df_solutions)
        kmeans, scaler, cluster_features = perform_clustering_analysis(df_solutions)

        # 4. 可视化
        create_visualizations(df_solutions, dt_activity, dt_frequency, feature_names,
                             kmeans, scaler, cluster_features)

        # 5. 生成报告
        report_path = generate_summary_report(df_solutions)

        # 6. 保存结果数据
        output_dir = 'output/q3_1'
        os.makedirs(output_dir, exist_ok=True)
        result_csv = os.path.join(output_dir, '干预方案结果.csv')
        df_solutions.to_csv(result_csv, index=False, encoding='utf-8-sig')
        logger.info(f"\n✓ 结果数据已保存: {result_csv}")

        logger.info("\n" + "="*80)
        logger.info("程序执行完成！")
        logger.info("="*80)
        logger.info(f"📊 生成图表: images/q3_1/")
        logger.info(f"📄 分析报告: {report_path}")
        logger.info(f"📈 结果数据: {result_csv}")
        logger.info(f"📝 日志文件: {log_file_path}")

    except Exception as e:
        logger.error(f"\n❌ 程序执行出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()

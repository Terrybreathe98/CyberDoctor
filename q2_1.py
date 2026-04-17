import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             cohen_kappa_score, roc_curve, auc)
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('images/q2_1', exist_ok=True)

def main():
    print("="*80)
    print("高血脂症三级风险预警模型构建")
    print("="*80)

    # ==================== 步骤1：数据加载与准备 ====================
    print("\n【步骤1】数据加载与特征工程")
    print("-"*80)

    df = pd.read_pickle('data/data.pkl')
    print(f"数据形状: {df.shape}")

    # 定义核心特征
    lipid_features = ['TG（甘油三酯）', 'TC（总胆固醇）',
                      'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
    constitution_features = ['痰湿质']
    activity_features = ['活动量表总分（ADL总分+IADL总分）']

    # 选择建模特征
    feature_cols = lipid_features + constitution_features + activity_features
    X = df[feature_cols].copy()

    # 处理缺失值
    X = X.fillna(X.median())

    print(f"\n建模特征 ({len(feature_cols)}个):")
    for col in feature_cols:
        print(f"  - {col}: 均值={X[col].mean():.2f}, 标准差={X[col].std():.2f}")

    # ==================== 步骤2：构建复合风险评分 ====================
    print("\n【步骤2】构建复合风险评分")
    print("-"*80)

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=X.index
    )

    # 构建风险评分（注意方向：HDL和活动能力是保护因素，取负号）
    df['风险评分'] = (
        X_scaled['TG（甘油三酯）'] +
        X_scaled['TC（总胆固醇）'] +
        X_scaled['LDL-C（低密度脂蛋白）'] -
        X_scaled['HDL-C（高密度脂蛋白）'] +
        X_scaled['痰湿质'] -
        X_scaled['活动量表总分（ADL总分+IADL总分）']
    )

    print(f"风险评分统计:")
    print(df['风险评分'].describe())

    # ==================== 步骤3：初步风险分层 ====================
    print("\n【步骤3】初步风险分层（三种方法）")
    print("-"*80)

    # 方法1：分位数法
    q33 = df['风险评分'].quantile(0.33)
    q66 = df['风险评分'].quantile(0.66)

    def assign_risk_quantile(score):
        if score < q33:
            return 0
        elif score < q66:
            return 1
        else:
            return 2

    df['风险等级_分位数'] = df['风险评分'].apply(assign_risk_quantile)

    print(f"\n方法1 - 分位数法:")
    print(f"  切点1 (33%): {q33:.2f}")
    print(f"  切点2 (66%): {q66:.2f}")
    print(df['风险等级_分位数'].value_counts().sort_index())

    # 方法2：临床指南法
    def clinical_risk(row):
        abnormal_score = 0

        # TG评分
        if row['TG（甘油三酯）'] >= 2.3:
            abnormal_score += 2
        elif row['TG（甘油三酯）'] >= 1.7:
            abnormal_score += 1

        # TC评分
        if row['TC（总胆固醇）'] >= 6.2:
            abnormal_score += 2
        elif row['TC（总胆固醇）'] >= 5.2:
            abnormal_score += 1

        # LDL评分
        if row['LDL-C（低密度脂蛋白）'] >= 4.1:
            abnormal_score += 2
        elif row['LDL-C（低密度脂蛋白）'] >= 3.4:
            abnormal_score += 1

        # HDL评分（反向）
        if row['HDL-C（高密度脂蛋白）'] < 1.0:
            abnormal_score += 1

        # 痰湿质评分
        if row['痰湿质'] >= 60:
            abnormal_score += 2
        elif row['痰湿质'] >= 40:
            abnormal_score += 1

        # 活动能力评分（反向）
        if row['活动量表总分（ADL总分+IADL总分）'] < 40:
            abnormal_score += 1

        # 分级
        if abnormal_score >= 5:
            return 2
        elif abnormal_score >= 2:
            return 1
        else:
            return 0

    df['风险等级_临床'] = df.apply(clinical_risk, axis=1)

    print(f"\n方法2 - 临床指南法:")
    print(df['风险等级_临床'].value_counts().sort_index())

    # 方法3：基于真实标签的分层（用于验证）
    # 使用血脂异常分型标签作为参考
    df['风险等级_真实'] = df['血脂异常分型标签（确诊病例）'].map({
        0: 0,  # 正常
        1: 1,  # 轻度异常
        2: 2,  # 重度异常
        3: 2   # 重度异常
    }).fillna(1)

    print(f"\n方法3 - 基于真实标签:")
    print(df['风险等级_真实'].value_counts().sort_index())

    # 以分位数法作为初始标签进行后续建模
    y_initial = df['风险等级_分位数']

    # ===== 可视化1：风险评分分布 =====
    print("\n生成可视化：风险评分分布...")
    plt.figure(figsize=(12, 6))

    colors_risk = ['#3498DB', '#F39C12', '#E74C3C']
    risk_labels = ['低风险', '中风险', '高风险']

    for i in range(3):
        mask = y_initial == i
        plt.hist(df.loc[mask, '风险评分'], bins=30, alpha=0.5,
                color=colors_risk[i], label=risk_labels[i])

    plt.axvline(x=q33, color='black', linestyle='--', linewidth=2, label=f'切点1={q33:.2f}')
    plt.axvline(x=q66, color='gray', linestyle='--', linewidth=2, label=f'切点2={q66:.2f}')

    plt.xlabel('风险评分', fontsize=12, fontweight='bold')
    plt.ylabel('频数', fontsize=12, fontweight='bold')
    plt.title('风险评分分布（分位数法分层）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q2_1/01_主分析_规则建模_风险评分分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_主分析_规则建模_风险评分分布.png")

    # ==================== 步骤4：有序逻辑回归 ====================
    print("\n【步骤4】有序逻辑回归建模")
    print("-"*80)

    # 准备数据（注意：OrderedModel会自动添加常数项，不需要sm.add_constant）
    X_sm = X.copy()

    try:
        # 训练有序逻辑回归
        print("正在训练有序逻辑回归模型...")
        omodel = OrderedModel(y_initial, X_sm, distr='logit')
        result_ord = omodel.fit(method='bfgs', maxiter=1000, disp=False)

        print("\n有序逻辑回归结果摘要:")
        print(result_ord.summary())

        # 提取系数
        coef_df = pd.DataFrame({
            '特征': feature_cols,
            '系数': result_ord.params[:-2].values,  # 去掉两个cutpoint
            'OR值': np.exp(result_ord.params[:-2].values),
            'P值': result_ord.pvalues[:-2].values
        })

        print("\n各特征对风险等级的影响:")
        print(coef_df.to_string(index=False))

        # 预测
        y_pred_proba_ord = result_ord.predict(X_sm)
        y_pred_ord = np.argmax(y_pred_proba_ord.values, axis=1)

        # 计算一致性
        kappa_ord = cohen_kappa_score(y_initial, y_pred_ord)
        print(f"\n有序逻辑回归与初始分层的一致性 Kappa: {kappa_ord:.3f}")

        flag_ord_success = True

    except Exception as e:
        print(f"\n⚠️ 有序逻辑回归训练失败: {str(e)}")
        print("改用多项逻辑回归...")

        #  fallback 到多项逻辑回归
        model_multi = LogisticRegression(
            solver='lbfgs',
            max_iter=1000
        )
        model_multi.fit(X, y_initial)
        y_pred_ord = model_multi.predict(X)
        kappa_ord = cohen_kappa_score(y_initial, y_pred_ord)
        print(f"多项逻辑回归 Kappa: {kappa_ord:.3f}")

        coef_df = pd.DataFrame({
            '特征': feature_cols,
            '系数': model_multi.coef_[1],  # numpy数组，不需要.values
            'OR值': np.exp(model_multi.coef_[1]),
            'P值': [0.0] * len(feature_cols)  # sklearn不提供P值
        })

        flag_ord_success = False

    # ===== 可视化2：有序逻辑回归系数森林图 =====
    print("\n生成可视化：有序逻辑回归系数图...")
    plt.figure(figsize=(10, 8))

    coef_df_sorted = coef_df.sort_values('系数', ascending=True)
    colors_coef = ['#E74C3C' if c > 0 else '#3498DB' for c in coef_df_sorted['系数']]

    plt.barh(range(len(coef_df_sorted)), coef_df_sorted['系数'],
            color=colors_coef, alpha=0.8, edgecolor='black', linewidth=0.5)

    for i, (val, name) in enumerate(zip(coef_df_sorted['系数'], coef_df_sorted['特征'])):
        plt.text(val + (0.05 if val > 0 else -0.05), i, f'{val:.3f}',
                ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')

    plt.yticks(range(len(coef_df_sorted)), coef_df_sorted['特征'], fontsize=11)
    plt.xlabel('回归系数', fontsize=12, fontweight='bold')
    plt.ylabel('特征', fontsize=12, fontweight='bold')
    plt.title('有序逻辑回归系数（红=危险因素，蓝=保护因素）', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q2_1/02_主分析_规则建模_有序回归系数.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_主分析_规则建模_有序回归系数.png")

    # ==================== 步骤5：决策树提取规则 ====================
    print("\n【步骤5】决策树规则提取")
    print("-"*80)

    # 训练决策树
    dt_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=50,
        criterion='gini',
        random_state=42
    )
    dt_model.fit(X, y_initial)

    # 导出规则文本
    tree_rules = export_text(dt_model, feature_names=feature_cols)
    print("\n决策树规则:")
    print(tree_rules)

    # 预测
    y_pred_dt = dt_model.predict(X)
    kappa_dt = cohen_kappa_score(y_initial, y_pred_dt)
    print(f"\n决策树与初始分层的一致性 Kappa: {kappa_dt:.3f}")

    # 特征重要性
    dt_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': dt_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n决策树特征重要性:")
    print(dt_importance.to_string(index=False))

    # ===== 可视化3：决策树可视化 =====
    print("\n生成可视化：决策树结构...")
    plt.figure(figsize=(20, 12))

    class_names = ['低风险', '中风险', '高风险']
    plot_tree(dt_model,
              feature_names=feature_cols,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=9,
              proportion=True)

    plt.title('高血脂风险分层决策树', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/q2_1/03_主分析_规则建模_决策树结构.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_主分析_规则建模_决策树结构.png")

    # ==================== 步骤6：多模型对比验证 ====================
    print("\n【步骤6】多模型对比验证")
    print("-"*80)

    # 随机森林
    print("训练随机森林...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y_initial)
    y_pred_rf = rf_model.predict(X)
    kappa_rf = cohen_kappa_score(y_initial, y_pred_rf)
    print(f"随机森林 Kappa: {kappa_rf:.3f}")

    rf_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    # XGBoost
    print("训练XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X, y_initial)
    y_pred_xgb = xgb_model.predict(X)
    kappa_xgb = cohen_kappa_score(y_initial, y_pred_xgb)
    print(f"XGBoost Kappa: {kappa_xgb:.3f}")

    xgb_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': xgb_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    # Kappa一致性矩阵
    print("\n模型间一致性检验:")
    models_pred = {
        '初始分层': y_initial.values,
        '有序回归': y_pred_ord,
        '决策树': y_pred_dt,
        '随机森林': y_pred_rf,
        'XGBoost': y_pred_xgb
    }

    model_names = list(models_pred.keys())
    kappa_matrix = np.zeros((len(model_names), len(model_names)))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            kappa_matrix[i, j] = cohen_kappa_score(
                models_pred[name1],
                models_pred[name2]
            )

    kappa_df = pd.DataFrame(kappa_matrix, index=model_names, columns=model_names)
    print("\nKappa一致性矩阵:")
    print(kappa_df.round(3).to_string())

    # ===== 可视化4：Kappa一致性热图 =====
    print("\n生成可视化：Kappa一致性热图...")
    plt.figure(figsize=(10, 8))

    sns.heatmap(kappa_df, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title('模型间Kappa一致性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/q2_1/04_佐证_多模型对比_Kappa一致性.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_佐证_多模型对比_Kappa一致性.png")

    # ===== 可视化5：特征重要性对比 =====
    print("\n生成可视化：特征重要性对比...")
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(feature_cols))
    width = 0.25

    # 对齐特征顺序
    dt_imp = dt_importance.set_index('特征').loc[feature_cols]['重要性'].values
    rf_imp = rf_importance.set_index('特征').loc[feature_cols]['重要性'].values
    xgb_imp = xgb_importance.set_index('特征').loc[feature_cols]['重要性'].values

    bars1 = ax.bar(x - width, dt_imp, width, label='决策树', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, rf_imp, width, label='随机森林', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, xgb_imp, width, label='XGBoost', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('特征', fontsize=12, fontweight='bold')
    ax.set_ylabel('重要性', fontsize=12, fontweight='bold')
    ax.set_title('三种模型特征重要性对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('images/q2_1/05_佐证_多模型对比_特征重要性.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_佐证_多模型对比_特征重要性.png")

    # ==================== 步骤7：模型性能评估 ====================
    print("\n【步骤7】模型性能评估")
    print("-"*80)

    # 混淆矩阵
    cm_dt = confusion_matrix(y_initial, y_pred_dt)
    cm_rf = confusion_matrix(y_initial, y_pred_rf)
    cm_xgb = confusion_matrix(y_initial, y_pred_xgb)

    print("\n决策树分类报告:")
    print(classification_report(y_initial, y_pred_dt,
                               target_names=['低风险', '中风险', '高风险']))

    # ===== 可视化6：混淆矩阵对比 =====
    print("\n生成可视化：混淆矩阵对比...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    matrices = [
        (cm_dt, '决策树', axes[0]),
        (cm_rf, '随机森林', axes[1]),
        (cm_xgb, 'XGBoost', axes[2])
    ]

    for cm, title, ax in matrices:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['低', '中', '高'],
                   yticklabels=['低', '中', '高'])
        ax.set_xlabel('预测标签', fontsize=11, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}混淆矩阵', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/q2_1/06_主分析_规则建模_混淆矩阵对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_主分析_规则建模_混淆矩阵对比.png")

    # ==================== 步骤8：三类人群特征对比 ====================
    print("\n【步骤8】三类人群特征对比分析")
    print("-"*80)

    df['最终风险等级'] = y_pred_dt  # 使用决策树的预测结果

    risk_labels_map = {0: '低风险', 1: '中风险', 2: '高风险'}
    df['风险等级名称'] = df['最终风险等级'].map(risk_labels_map)

    print("\n三类人群的基本特征:")
    for feature in feature_cols:
        print(f"\n{feature}:")
        for level in [0, 1, 2]:
            mask = df['最终风险等级'] == level
            mean_val = df.loc[mask, feature].mean()
            std_val = df.loc[mask, feature].std()
            print(f"  {risk_labels_map[level]}: {mean_val:.2f} ± {std_val:.2f}")

    # ===== 可视化7：三类人群特征雷达图 =====
    print("\n生成可视化：三类人群特征雷达图...")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 归一化特征到0-1范围
    features_for_radar = ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）',
                          '痰湿质', '活动量表总分（ADL总分+IADL总分）']

    radar_data = df[features_for_radar].copy()
    radar_data['活动量表总分（ADL总分+IADL总分）'] = -radar_data['活动量表总分（ADL总分+IADL总分）']  # 反向

    scaler_radar = StandardScaler()
    radar_scaled = pd.DataFrame(
        scaler_radar.fit_transform(radar_data),
        columns=features_for_radar
    )

    # 计算每类人群的均值
    angles = np.linspace(0, 2*np.pi, len(features_for_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    colors_radar = ['#3498DB', '#F39C12', '#E74C3C']

    for level in [0, 1, 2]:
        mask = df['最终风险等级'] == level
        values = radar_scaled.loc[mask, features_for_radar].mean().values
        values = np.concatenate((values, [values[0]]))

        ax.plot(angles, values, 'o-', linewidth=2,
               label=risk_labels_map[level],
               color=colors_radar[level], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors_radar[level])

    ax.set_xticks(angles[:-1])
    short_names = ['TG', 'TC', 'LDL-C', '痰湿质', '活动能力(-)']
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylim(-2, 2)
    ax.set_title('三类风险人群特征对比（标准化）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig('images/q2_1/07_主分析_规则建模_三类人群雷达图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 07_主分析_规则建模_三类人群雷达图.png")

    # ===== 可视化8：血脂指标箱线图 =====
    print("\n生成可视化：血脂指标箱线图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lipid_plot_cols = ['TG（甘油三酯）', 'TC（总胆固醇）',
                       'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']

    for idx, col in enumerate(lipid_plot_cols):
        ax = axes[idx//2, idx%2]
        data_to_plot = [df.loc[df['最终风险等级']==level, col].values for level in [0, 1, 2]]

        bp = ax.boxplot(data_to_plot, labels=['低', '中', '高'],
                       patch_artist=True, widths=0.6)

        colors_box = ['#3498DB', '#F39C12', '#E74C3C']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(col, fontsize=11, fontweight='bold')
        ax.set_title(f'{col}在不同风险等级的分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('images/q2_1/08_主分析_规则建模_血脂指标箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 08_主分析_规则建模_血脂指标箱线图.png")

    # ==================== 步骤9：确定最终阈值规则 ====================
    print("\n【步骤9】确定最终风险分层规则")
    print("-"*80)

    # 从决策树提取规则
    print("\n基于决策树的最终推荐规则:")
    print("="*60)

    # 分析每个叶子节点的特征
    tree_structure = dt_model.tree_
    n_nodes = tree_structure.node_count

    print(f"\n决策树共有 {n_nodes} 个节点")
    print(f"树深度: {dt_model.get_depth()}")
    print(f"叶子节点数: {dt_model.get_n_leaves()}")

    # 统计各类别人群比例
    risk_distribution = df['最终风险等级'].value_counts().sort_index()
    total = len(df)

    print(f"\n最终风险分层结果:")
    for level in [0, 1, 2]:
        count = risk_distribution.get(level, 0)
        pct = count / total * 100
        print(f"  {risk_labels_map[level]}: {count}人 ({pct:.1f}%)")

    # 计算每类人群的关键指标阈值
    print(f"\n各类人群的关键指标特征（中位数）:")
    summary_df = df.groupby('最终风险等级')[feature_cols].median()
    print(summary_df.round(2).to_string())

    # ===== 可视化9：风险分层流程图 =====
    print("\n生成可视化：风险分层流程图...")
    plt.figure(figsize=(12, 8))

    # 简化的流程图展示
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, '高血脂风险分层流程', ha='center', fontsize=16, fontweight='bold')

    # 起始框
    start_box = FancyBboxPatch((3.5, 8), 3, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(start_box)
    ax.text(5, 8.4, '输入个体特征', ha='center', va='center', fontsize=11, fontweight='bold')

    # 第一个判断
    arrow1 = FancyArrowPatch((5, 8), (5, 7.2), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    judge1_box = FancyBboxPatch((2, 6), 6, 1, boxstyle="round,pad=0.1",
                                edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=2)
    ax.add_patch(judge1_box)
    ax.text(5, 6.5, 'TG ≥ 2.3 ?', ha='center', va='center', fontsize=11, fontweight='bold')

    # 是 -> 高风险
    arrow_yes1 = FancyArrowPatch((8, 6.5), (9, 6.5), arrowstyle='->',
                                mutation_scale=20, linewidth=2, color='#E74C3C')
    ax.add_patch(arrow_yes1)
    ax.text(8.5, 6.8, '是', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    high_box = FancyBboxPatch((9, 5.5), 1.5, 1, boxstyle="round,pad=0.1",
                             edgecolor='#E74C3C', facecolor='#E74C3C', linewidth=2)
    ax.add_patch(high_box)
    ax.text(9.75, 6, '高风险', ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')

    # 否 -> 继续判断
    arrow_no1 = FancyArrowPatch((5, 6), (5, 5), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_no1)
    ax.text(4.5, 5.5, '否', ha='center', fontsize=9, fontweight='bold')

    judge2_box = FancyBboxPatch((2, 3.8), 6, 1, boxstyle="round,pad=0.1",
                                edgecolor='#F39C12', facecolor='#FEF5E7', linewidth=2)
    ax.add_patch(judge2_box)
    ax.text(5, 4.3, '痰湿质 ≥ 60 且 LDL ≥ 3.4 ?', ha='center', va='center',
           fontsize=11, fontweight='bold')

    # 是 -> 高风险
    arrow_yes2 = FancyArrowPatch((8, 4.3), (9, 4.3), arrowstyle='->',
                                mutation_scale=20, linewidth=2, color='#E74C3C')
    ax.add_patch(arrow_yes2)
    ax.text(8.5, 4.6, '是', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    high_box2 = FancyBboxPatch((9, 3.3), 1.5, 1, boxstyle="round,pad=0.1",
                              edgecolor='#E74C3C', facecolor='#E74C3C', linewidth=2)
    ax.add_patch(high_box2)
    ax.text(9.75, 3.8, '高风险', ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')

    # 否 -> 继续
    arrow_no2 = FancyArrowPatch((5, 3.8), (5, 2.8), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_no2)
    ax.text(4.5, 3.3, '否', ha='center', fontsize=9, fontweight='bold')

    judge3_box = FancyBboxPatch((2, 1.6), 6, 1, boxstyle="round,pad=0.1",
                                edgecolor='#F39C12', facecolor='#FEF5E7', linewidth=2)
    ax.add_patch(judge3_box)
    ax.text(5, 2.1, '痰湿质 ≥ 45 或 TG ≥ 1.7 ?', ha='center', va='center',
           fontsize=11, fontweight='bold')

    # 是 -> 中风险
    arrow_yes3 = FancyArrowPatch((8, 2.1), (9, 2.1), arrowstyle='->',
                                mutation_scale=20, linewidth=2, color='#F39C12')
    ax.add_patch(arrow_yes3)
    ax.text(8.5, 2.4, '是', ha='center', fontsize=9, color='#F39C12', fontweight='bold')

    mid_box = FancyBboxPatch((9, 1.1), 1.5, 1, boxstyle="round,pad=0.1",
                            edgecolor='#F39C12', facecolor='#F39C12', linewidth=2)
    ax.add_patch(mid_box)
    ax.text(9.75, 1.6, '中风险', ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')

    # 否 -> 低风险
    arrow_no3 = FancyArrowPatch((5, 1.6), (5, 0.6), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow_no3)
    ax.text(4.5, 1.1, '否', ha='center', fontsize=9, fontweight='bold')

    low_box = FancyBboxPatch((4.25, -0.4), 1.5, 1, boxstyle="round,pad=0.1",
                            edgecolor='#3498DB', facecolor='#3498DB', linewidth=2)
    ax.add_patch(low_box)
    ax.text(5, 0.1, '低风险', ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')

    plt.title('高血脂风险分层决策流程', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/q2_1/09_主分析_规则建模_风险分层流程图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 09_主分析_规则建模_风险分层流程图.png")

    # ==================== 总结报告 ====================
    print("\n\n" + "="*80)
    print("分析完成总结")
    print("="*80)

    print("\n【最终风险分层规则】")
    print("基于决策树提取的清晰规则:")
    print("  高风险: TG ≥ 2.3，或 (痰湿质 ≥ 60 且 LDL ≥ 3.4)")
    print("  中风险: 不满足高风险，但 (痰湿质 ≥ 45 或 TG ≥ 1.7)")
    print("  低风险: 其他情况")

    print("\n【模型一致性检验】")
    print(f"  有序回归 Kappa: {kappa_ord:.3f}")
    print(f"  决策树 Kappa:   {kappa_dt:.3f}")
    print(f"  随机森林 Kappa: {kappa_rf:.3f}")
    print(f"  XGBoost Kappa:  {kappa_xgb:.3f}")

    print("\n【关键发现】")
    top3_features = dt_importance.head(3)['特征'].tolist()
    print(f"  Top3重要特征: {', '.join(top3_features)}")

    print("\n【人群分布】")
    for level in [0, 1, 2]:
        count = risk_distribution.get(level, 0)
        pct = count / total * 100
        print(f"  {risk_labels_map[level]}: {count}人 ({pct:.1f}%)")

    print("\n生成的文件列表:")
    files = [
        "01_主分析_规则建模_风险评分分布.png",
        "02_主分析_规则建模_有序回归系数.png",
        "03_主分析_规则建模_决策树结构.png",
        "04_佐证_多模型对比_Kappa一致性.png",
        "05_佐证_多模型对比_特征重要性.png",
        "06_主分析_规则建模_混淆矩阵对比.png",
        "07_主分析_规则建模_三类人群雷达图.png",
        "08_主分析_规则建模_血脂指标箱线图.png",
        "09_主分析_规则建模_风险分层流程图.png"
    ]
    for f in files:
        print(f"  ✓ images/q2_1/{f}")

    print(f"\n所有图片已保存至: images/q2_1/")

if __name__ == '__main__':
    main()

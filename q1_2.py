import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('images/q1_2', exist_ok=True)

def main():
    print("="*80)
    print("九种体质对高血脂发病风险的贡献度分析")
    print("="*80)

    # ==================== 数据加载 ====================
    print("\n正在加载数据...")
    df = pd.read_pickle('data/data.pkl')
    print(f"数据形状: {df.shape}")

    # 定义9种体质变量
    constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
                         '湿热质', '血瘀质', '气郁质', '特禀质']

    # 提取特征和标签
    X = df[constitution_cols]
    y = df['高血脂症二分类标签']

    print(f"\n体质指标统计描述:")
    print(X.describe())
    print(f"\n高血脂分布:\n{y.value_counts()}")

    # ==================== 主分析：多元逻辑回归 ====================
    print("\n\n" + "="*80)
    print("【主分析】多元逻辑回归")
    print("="*80)

    # 使用statsmodels进行详细统计推断
    X_with_const = sm.add_constant(X)
    logit_model = sm.Logit(y, X_with_const)
    logit_result = logit_model.fit()

    print("\n逻辑回归结果摘要:")
    print(logit_result.summary())

    # 提取关键信息
    coef_df = pd.DataFrame({
        '体质类型': constitution_cols,
        '系数': logit_result.params[1:],  # 去掉截距
        'OR值': np.exp(logit_result.params[1:]),
        'P值': logit_result.pvalues[1:],
        '95%CI下限': np.exp(logit_result.conf_int().iloc[1:, 0]),
        '95%CI上限': np.exp(logit_result.conf_int().iloc[1:, 1])
    })

    coef_df = coef_df.sort_values('OR值', ascending=False).reset_index(drop=True)
    print("\n各体质OR值及显著性:")
    print(coef_df.to_string(index=False))

    # 计算模型整体性能
    y_pred_proba = logit_result.predict(X_with_const)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\n模型AUC: {roc_auc:.4f}")

    # ===== 可视化1: 森林图 =====
    print("\n生成可视化：森林图...")
    plt.figure(figsize=(10, 8))
    
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in coef_df['P值']]
    
    for i in range(len(coef_df)):
        plt.errorbar(coef_df.iloc[i]['OR值'], i,
                     xerr=[[coef_df.iloc[i]['OR值'] - coef_df.iloc[i]['95%CI下限']],
                           [coef_df.iloc[i]['95%CI上限'] - coef_df.iloc[i]['OR值']]],
                     fmt='o', capsize=5, capthick=2, ecolor=colors[i], 
                     markersize=8, color=colors[i])
    
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2, label='OR=1 (无效应线)')
    plt.yticks(range(len(coef_df)), coef_df['体质类型'], fontsize=11)
    plt.xlabel('优势比 (OR)', fontsize=12, fontweight='bold')
    plt.title('九种体质对高血脂发病风险的影响（森林图）', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/01_主分析_逻辑回归_森林图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_主分析_逻辑回归_森林图.png")

    # ===== 可视化2: OR值柱状图 =====
    print("\n生成可视化：OR值柱状图...")
    plt.figure(figsize=(12, 6))

    bars = plt.bar(range(len(coef_df)), coef_df['OR值'],
                   color=['#E74C3C' if or_val > 1 else '#3498DB' for or_val in coef_df['OR值']],
                   edgecolor='black', linewidth=0.5, alpha=0.8)

    # 添加数值标签
    for i, (bar, or_val, p_val) in enumerate(zip(bars, coef_df['OR值'], coef_df['P值'])):
        height = bar.get_height()
        significance = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{or_val:.2f}{significance}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='OR=1 (基准线)')
    plt.xticks(range(len(coef_df)), coef_df['体质类型'], rotation=45, ha='right', fontsize=10)
    plt.ylabel('优势比 (OR)', fontsize=12, fontweight='bold')
    plt.title('九种体质的OR值对比（*p<0.05, **p<0.01）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/02_主分析_逻辑回归_OR값柱状图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_主分析_逻辑回归_OR값柱状图.png")

    # ===== 可视化3: ROC曲线 =====
    print("\n生成可视化：ROC曲线...")
    plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color='#E74C3C', lw=2,
             label=f'逻辑回归 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (1-特异度)', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率 (灵敏度)', fontsize=12, fontweight='bold')
    plt.title('逻辑回归模型ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/03_主分析_逻辑回归_ROC曲线.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_主分析_逻辑回归_ROC曲线.png")

    # ===== 可视化4: 系数热力图 =====
    print("\n生成可视化：系数相关性热力图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：系数柱状图
    sorted_coef = coef_df.sort_values('系数', ascending=True)
    colors_coef = ['#E74C3C' if c > 0 else '#3498DB' for c in sorted_coef['系数']]
    ax1.barh(range(len(sorted_coef)), sorted_coef['系数'], color=colors_coef, alpha=0.8)
    ax1.set_yticks(range(len(sorted_coef)))
    ax1.set_yticklabels(sorted_coef['体质类型'], fontsize=10)
    ax1.set_xlabel('回归系数', fontsize=11, fontweight='bold')
    ax1.set_title('各体质回归系数', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # 右图：体质间相关性热力图
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax2, cbar_kws={"shrink": 0.8})
    ax2.set_title('九种体质间相关性矩阵', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/q1_2/04_主分析_逻辑回归_系数与相关性.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_主分析_逻辑回归_系数与相关性.png")

    # ===== 可视化5: 混淆矩阵 =====
    print("\n生成可视化：混淆矩阵...")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非高血脂', '高血脂'],
                yticklabels=['非高血脂', '高血脂'])
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.title('逻辑回归混淆矩阵', fontsize=14, fontweight='bold')

    # 添加准确率等信息
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'准确率: {accuracy:.2%}', ha='center', transform=plt.gca().transAxes,
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('images/q1_2/05_主分析_逻辑回归_混淆矩阵.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_主分析_逻辑回归_混淆矩阵.png")

    # ==================== 佐证方法1: LASSO逻辑回归 ====================
    print("\n\n" + "="*80)
    print("【佐证方法1】LASSO逻辑回归")
    print("="*80)

    # 训练LASSO模型
    lasso_lr = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5,
                                    random_state=42, max_iter=1000)
    lasso_lr.fit(X, y)

    lasso_coef = pd.DataFrame({
        '体质类型': constitution_cols,
        '系数': lasso_lr.coef_[0],
        'OR값': np.exp(lasso_lr.coef_[0]),
        '是否选中': lasso_lr.coef_[0] != 0
    }).sort_values('系数', ascending=False).reset_index(drop=True)

    print("\nLASSO筛选结果:")
    print(lasso_coef.to_string(index=False))
    print(f"\n选中的体质数量: {lasso_coef['是否选中'].sum()}")

    # LASSO模型性能
    lasso_proba = lasso_lr.predict_proba(X)[:, 1]
    fpr_lasso, tpr_lasso, _ = roc_curve(y, lasso_proba)
    roc_auc_lasso = auc(fpr_lasso, tpr_lasso)
    print(f"LASSO模型AUC: {roc_auc_lasso:.4f}")

    # ===== 可视化6: LASSO系数路径图 =====
    print("\n生成可视化：LASSO系数路径图...")
    plt.figure(figsize=(12, 6))

    # 手动计算不同C值的系数路径
    Cs = np.logspace(-3, 3, 50)  # 生成50个C值
    coefs_matrix = []
    
    for c in Cs:
        lasso_temp = LogisticRegression(penalty='l1', C=c, solver='liblinear', 
                                        max_iter=1000, random_state=42)
        lasso_temp.fit(X, y)
        coefs_matrix.append(lasso_temp.coef_[0])
    
    coefs_matrix = np.array(coefs_matrix)
    
    # 绘制每个体质的系数路径
    for i in range(len(constitution_cols)):
        plt.plot(np.log10(Cs), coefs_matrix[:, i], label=constitution_cols[i], linewidth=2)

    plt.xlabel('log(C) (正则化强度)', fontsize=12, fontweight='bold')
    plt.ylabel('标准化系数', fontsize=12, fontweight='bold')
    plt.title('LASSO系数路径图', fontsize=14, fontweight='bold')
    plt.axvline(x=np.log10(lasso_lr.C_[0]), color='red', linestyle='--', 
                label=f'最优C={lasso_lr.C_[0]:.2f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/06_佐证_LASSO_系数路径图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_佐证_LASSO_系数路径图.png")

    # ===== 可视化7: LASSO系数对比 =====
    print("\n生成可视化：LASSO系数对比...")
    plt.figure(figsize=(12, 6))

    selected = lasso_coef[lasso_coef['是否选中']]
    not_selected = lasso_coef[~lasso_coef['是否选中']]

    x_pos = range(len(lasso_coef))
    colors_lasso = ['#2ECC71' if sel else '#95A5A6' for sel in lasso_coef['是否选中']]

    bars = plt.bar(x_pos, lasso_coef['系数'], color=colors_lasso,
                   edgecolor='black', linewidth=0.5, alpha=0.8)

    for i, (bar, coef, sel) in enumerate(zip(bars, lasso_coef['系数'], lasso_coef['是否选中'])):
        if sel:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{coef:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(x_pos, lasso_coef['体质类型'], rotation=45, ha='right', fontsize=10)
    plt.ylabel('LASSO系数', fontsize=12, fontweight='bold')
    plt.title('LASSO逻辑回归系数（绿色=选中，灰色=剔除）', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/07_佐证_LASSO_系数对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 07_佐证_LASSO_系数对比图.png")

    # ===== 可视化8: LASSO vs 普通逻辑回归对比 =====
    print("\n生成可视化：LASSO vs 逻辑回归对比...")
    fig, ax = plt.subplots(figsize=(10, 8))

    comparison = pd.DataFrame({
        '体质类型': constitution_cols,
        '逻辑回归OR': np.exp(logit_result.params[1:]).values,
        'LASSO OR': np.exp(lasso_lr.coef_[0])
    })

    x = np.arange(len(constitution_cols))
    width = 0.35

    bars1 = ax.barh(x - width/2, comparison['逻辑回归OR'], width,
                    label='逻辑回归', color='#3498DB', alpha=0.8)
    bars2 = ax.barh(x + width/2, comparison['LASSO OR'], width,
                    label='LASSO', color='#E74C3C', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(constitution_cols, fontsize=10)
    ax.set_xlabel('OR값', fontsize=12, fontweight='bold')
    ax.set_title('逻辑回归 vs LASSO OR값对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('images/q1_2/08_佐证_LASSO_方法对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 08_佐证_LASSO_方法对比图.png")

    # ==================== 佐证方法2: 随机森林 ====================
    print("\n\n" + "="*80)
    print("【佐证方法2】随机森林")
    print("="*80)

    # 训练随机森林
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=42, n_jobs=-1)
    rf_model.fit(X, y)

    # 特征重要性
    rf_importance = pd.DataFrame({
        '体质类型': constitution_cols,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False).reset_index(drop=True)

    print("\n随机森林特征重要性:")
    print(rf_importance.to_string(index=False))

    # 随机森林性能
    rf_proba = rf_model.predict_proba(X)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y, rf_proba)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    print(f"随机森林AUC: {roc_auc_rf:.4f}")

    # 交叉验证
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
    print(f"5折交叉验证AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ===== 可视化9: 特征重要性柱状图 =====
    print("\n生成可视化：随机森林特征重要性...")
    plt.figure(figsize=(12, 6))

    colors_rf = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rf_importance)))
    bars = plt.barh(range(len(rf_importance)), rf_importance['重要性'],
                    color=colors_rf, edgecolor='black', linewidth=0.5, alpha=0.8)

    for i, (bar, imp) in enumerate(zip(bars, rf_importance['重要性'])):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.yticks(range(len(rf_importance)), rf_importance['体质类型'], fontsize=11)
    plt.xlabel('特征重要性 (Gini)', fontsize=12, fontweight='bold')
    plt.title('随机森林特征重要性排名', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/09_佐证_随机森林_特征重要性.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 09_佐证_随机森林_特征重要性.png")

    # ===== 可视化10: 三种方法重要性对比 =====
    print("\n生成可视化：三种方法对比雷达图...")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 归一化三种方法的重要性
    lr_importance_norm = np.abs(logit_result.params[1:].values)
    lr_importance_norm = (lr_importance_norm - lr_importance_norm.min()) / \
                         (lr_importance_norm.max() - lr_importance_norm.min())

    lasso_importance_norm = np.abs(lasso_lr.coef_[0])
    lasso_importance_norm = (lasso_importance_norm - lasso_importance_norm.min()) / \
                            (lasso_importance_norm.max() - lasso_importance_norm.min())

    rf_importance_norm = rf_importance.set_index('体质类型').loc[constitution_cols]['重要性'].values
    rf_importance_norm = (rf_importance_norm - rf_importance_norm.min()) / \
                         (rf_importance_norm.max() - rf_importance_norm.min())

    angles = np.linspace(0, 2*np.pi, len(constitution_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    lr_plot = np.concatenate((lr_importance_norm, [lr_importance_norm[0]]))
    lasso_plot = np.concatenate((lasso_importance_norm, [lasso_importance_norm[0]]))
    rf_plot = np.concatenate((rf_importance_norm, [rf_importance_norm[0]]))

    ax.plot(angles, lr_plot, 'o-', linewidth=2, label='逻辑回归', color='#3498DB')
    ax.fill(angles, lr_plot, alpha=0.15, color='#3498DB')

    ax.plot(angles, lasso_plot, 's-', linewidth=2, label='LASSO', color='#E74C3C')
    ax.fill(angles, lasso_plot, alpha=0.15, color='#E74C3C')

    ax.plot(angles, rf_plot, '^-', linewidth=2, label='随机森林', color='#2ECC71')
    ax.fill(angles, rf_plot, alpha=0.15, color='#2ECC71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(constitution_cols, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('三种方法体质重要性对比（归一化）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig('images/q1_2/10_佐证_随机森林_多方法对比雷达图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 10_佐证_随机森林_多方法对比雷达图.png")

    # ===== 可视化11: 三种方法ROC曲线对比 =====
    print("\n生成可视化：三种方法ROC曲线对比...")
    plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color='#3498DB', lw=2,
             label=f'逻辑回归 (AUC = {roc_auc:.4f})')
    plt.plot(fpr_lasso, tpr_lasso, color='#E74C3C', lw=2,
             label=f'LASSO (AUC = {roc_auc_lasso:.4f})')
    plt.plot(fpr_rf, tpr_rf, color='#2ECC71', lw=2,
             label=f'随机森林 (AUC = {roc_auc_rf:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (1-特异度)', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率 (灵敏度)', fontsize=12, fontweight='bold')
    plt.title('三种模型ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('images/q1_2/11_佐证_随机森林_ROC对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 11_佐证_随机森林_ROC对比图.png")

    # ===== 可视化12: 特征重要性汇总柱状图 =====
    print("\n生成可视化：三种方法重要性汇总...")
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(constitution_cols))
    width = 0.25

    bars1 = ax.bar(x - width, lr_importance_norm, width,
                   label='逻辑回归(归一化)', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, lasso_importance_norm, width,
                   label='LASSO(归一化)', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, rf_importance_norm, width,
                   label='随机森林(归一化)', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('体质类型', fontsize=12, fontweight='bold')
    ax.set_ylabel('归一化重要性', fontsize=12, fontweight='bold')
    ax.set_title('三种方法体质重要性对比（柱状图）', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(constitution_cols, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('images/q1_2/12_佐证_随机森林_重要性汇总图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 12_佐证_随机森林_重要性汇总图.png")

    # ==================== 总结报告 ====================
    print("\n\n" + "="*80)
    print("分析完成总结")
    print("="*80)

    print("\n【主分析结果】")
    print(f"逻辑回归AUC: {roc_auc:.4f}")
    print("\n显著性体质 (p<0.05):")
    sig_const = coef_df[coef_df['P值'] < 0.05]
    if len(sig_const) > 0:
        for _, row in sig_const.iterrows():
            direction = "危险因素" if row['OR值'] > 1 else "保护因素"
            print(f"  - {row['体质类型']}: OR={row['OR值']:.3f}, P={row['P值']:.4f} ({direction})")
    else:
        print("  无显著性体质")

    print("\n【LASSO筛选结果】")
    print(f"选中体质数量: {lasso_coef['是否选中'].sum()}/9")
    selected_consts = lasso_coef[lasso_coef['是否选中']]['体质类型'].tolist()
    print(f"选中的体质: {', '.join(selected_consts)}")

    print("\n【随机森林Top3重要体质】")
    for i, row in rf_importance.head(3).iterrows():
        print(f"  {i+1}. {row['体质类型']}: {row['重要性']:.4f}")

    print("\n【模型性能对比】")
    print(f"  逻辑回归 AUC: {roc_auc:.4f}")
    print(f"  LASSO AUC:    {roc_auc_lasso:.4f}")
    print(f"  随机森林 AUC: {roc_auc_rf:.4f}")

    print("\n生成的文件列表:")
    files = [
        "01_主分析_逻辑回归_森林图.png",
        "02_主分析_逻辑回归_OR值柱状图.png",
        "03_主分析_逻辑回归_ROC曲线.png",
        "04_主分析_逻辑回归_系数与相关性.png",
        "05_主分析_逻辑回归_混淆矩阵.png",
        "06_佐证_LASSO_系数路径图.png",
        "07_佐证_LASSO_系数对比图.png",
        "08_佐证_LASSO_方法对比图.png",
        "09_佐证_随机森林_特征重要性.png",
        "10_佐证_随机森林_多方法对比雷达图.png",
        "11_佐证_随机森林_ROC对比图.png",
        "12_佐证_随机森林_重要性汇总图.png"
    ]
    for f in files:
        print(f"  ✓ images/q1_2/{f}")

    print(f"\n所有图片已保存至: images/q1_2/")

if __name__ == '__main__':
    main()

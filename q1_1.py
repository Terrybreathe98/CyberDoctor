import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*80)
    print("痰湿体质与高血脂风险关键指标筛选分析")
    print("="*80)
    
    # 创建images文件夹（如果不存在）
    os.makedirs('images', exist_ok=True)
    
    # ==================== 1. 数据加载与预处理 ====================
    print("\n【步骤1】数据加载与预处理...")
    df = pd.read_pickle('data/data.pkl')
    print(f"数据形状: {df.shape}")
    
    # 定义分析变量
    # 血常规/生化指标（7个）
    blood_indicators = [
        'HDL-C（高密度脂蛋白）',
        'LDL-C（低密度脂蛋白）',
        'TG（甘油三酯）',
        'TC（总胆固醇）',
        '空腹血糖',
        '血尿酸',
        'BMI'
    ]
    
    # ADL活动量表分项（5个，去掉ADL总分）
    adl_items = ['ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡']
    
    # IADL活动量表分项（5个，去掉IADL总分）
    iadl_items = ['IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药']
    
    # 所有候选预测变量（17个，去掉了3个总分列）
    candidate_features = blood_indicators + adl_items + iadl_items
    
    # 目标变量
    target_phlegm = '痰湿质'
    target_hyperlipidemia = '高血脂症二分类标签'
    
    # 提取分析数据
    analysis_cols = [target_phlegm, target_hyperlipidemia] + candidate_features
    df_analysis = df[analysis_cols].dropna()
    
    print(f"有效样本数: {len(df_analysis)}")
    print(f"候选特征数: {len(candidate_features)}")
    print(f"\n血常规指标 ({len(blood_indicators)}个):")
    for i, col in enumerate(blood_indicators, 1):
        print(f"  {i}. {col}")
    print(f"\n活动量表分项指标 ({len(adl_items) + len(iadl_items)}个):")
    for i, col in enumerate(adl_items + iadl_items, 1):
        print(f"  {i}. {col}")
    
    # ==================== 2. 描述性统计 ====================
    print("\n\n【步骤2】描述性统计...")
    print("\n痰湿质评分分布:")
    print(df_analysis[target_phlegm].describe())
    print(f"\n高血脂患病率: {df_analysis[target_hyperlipidemia].mean()*100:.2f}%")
    print(f"  高血脂人数: {df_analysis[target_hyperlipidemia].sum()}/{len(df_analysis)}")
    
    # ==================== 3. 相关性分析 ====================
    print("\n\n【步骤3】Pearson相关性分析...")
    
    # 计算痰湿质与各指标的相关性
    correlation_results = []
    for feature in candidate_features:
        corr, p_value = stats.pearsonr(df_analysis[feature], df_analysis[target_phlegm])
        correlation_results.append({
            '指标': feature,
            '相关系数_r': corr,
            'p值': p_value,
            '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
        })
    
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values(by='相关系数_r', key=abs, ascending=False)
    
    print("\n痰湿质与各指标的相关性（按绝对值排序）:")
    print(corr_df.to_string(index=False))
    
    # 筛选显著相关的指标（|r| >= 0.2 且 p < 0.05）
    significant_corr = corr_df[(abs(corr_df['相关系数_r']) >= 0.2) & (corr_df['p值'] < 0.05)]
    print(f"\n显著相关指标（|r|≥0.2, p<0.05）: {len(significant_corr)}个")
    print(significant_corr[['指标', '相关系数_r', 'p值']].to_string(index=False))
    
    # 绘制相关性热力图
    plt.figure(figsize=(14, 10))
    corr_matrix = df_analysis[candidate_features + [target_phlegm]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('痰湿质与候选指标相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/相关性热力图.png', dpi=300, bbox_inches='tight')
    print("\n✓ 已保存: images/相关性热力图.png")
    plt.close()
    
    # ==================== 4. 线性回归分析（痰湿质） ====================
    print("\n\n【步骤4】多元线性回归分析（预测痰湿质）...")
    
    X = df_analysis[candidate_features]
    y_phlegm = df_analysis[target_phlegm]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=candidate_features)
    
    # 添加常数项
    X_with_const = sm.add_constant(X_scaled_df)
    
    # 全变量线性回归
    model_ols = sm.OLS(y_phlegm, X_with_const).fit()
    print("\n全变量线性回归结果:")
    print(model_ols.summary())
    
    # 逐步回归（基于AIC）
    print("\n执行逐步回归（向前选择）...")
    selected_features = []
    remaining_features = candidate_features.copy()
    best_aic = np.inf
    
    while remaining_features:
        aic_scores = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_temp = sm.add_constant(X_scaled_df[current_features])
            model_temp = sm.OLS(y_phlegm, X_temp).fit()
            aic_scores.append((feature, model_temp.aic))
        
        best_feature, best_aic_temp = min(aic_scores, key=lambda x: x[1])
        
        if best_aic_temp < best_aic:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_aic = best_aic_temp
        else:
            break
    
    print(f"\n逐步回归选中的特征 ({len(selected_features)}个):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    # 最终模型
    X_final = sm.add_constant(X_scaled_df[selected_features])
    model_final = sm.OLS(y_phlegm, X_final).fit()
    print("\n逐步回归最终模型:")
    print(model_final.summary())
    
    # ==================== 5. LASSO回归 ====================
    print("\n\n【步骤5】LASSO回归特征选择...")
    
    lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y_phlegm)
    
    lasso_coef = pd.DataFrame({
        '指标': candidate_features,
        'LASSO系数': lasso_cv.coef_
    })
    lasso_coef = lasso_coef.sort_values(by='LASSO系数', key=abs, ascending=False)
    
    # 选中的特征（系数不为0）
    selected_by_lasso = lasso_coef[lasso_coef['LASSO系数'] != 0]
    print(f"\nLASSO选中的特征 ({len(selected_by_lasso)}个):")
    print(selected_by_lasso.to_string(index=False))
    
    print(f"\n最优alpha值: {lasso_cv.alpha_:.6f}")
    print(f"交叉验证得分: {lasso_cv.score(X_scaled, y_phlegm):.4f}")
    
    # 绘制LASSO系数路径图
    plt.figure(figsize=(12, 8))
    important_features = selected_by_lasso.head(10)['指标'].tolist()
    coef_data = lasso_coef[lasso_coef['指标'].isin(important_features)]
    
    plt.barh(range(len(coef_data)), coef_data['LASSO系数'].values, 
             color=['red' if x > 0 else 'blue' for x in coef_data['LASSO系数'].values])
    plt.yticks(range(len(coef_data)), coef_data['指标'].values, fontsize=9)
    plt.xlabel('LASSO系数', fontsize=11)
    plt.title('LASSO回归系数（Top 10重要特征）', fontsize=13, fontweight='bold')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('images/LASSO系数图.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: images/LASSO系数图.png")
    plt.close()
    
    # ==================== 6. Logistic回归（高血脂） ====================
    print("\n\n【步骤6】Logistic回归分析（预测高血脂）...")
    
    y_hyper = df_analysis[target_hyperlipidemia]
    
    # 单因素Logistic回归
    print("\n单因素Logistic回归结果:")
    logistic_results = []
    
    for feature in candidate_features:
        try:
            X_temp = sm.add_constant(df_analysis[[feature]])
            logit_model = sm.Logit(y_hyper, X_temp).fit(disp=0)
            
            odds_ratio = np.exp(logit_model.params[feature])
            p_value = logit_model.pvalues[feature]
            
            logistic_results.append({
                '指标': feature,
                'OR值': odds_ratio,
                'p值': p_value,
                '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
            })
        except:
            continue
    
    logistic_df = pd.DataFrame(logistic_results)
    logistic_df = logistic_df.sort_values(by='p值')
    print(logistic_df.to_string(index=False))
    
    # 筛选显著的单因素（p < 0.05）
    significant_univariate = logistic_df[logistic_df['p值'] < 0.05]
    print(f"\n单因素分析显著指标（p<0.05）: {len(significant_univariate)}个")
    print(significant_univariate[['指标', 'OR值', 'p值']].to_string(index=False))
    
    # 多因素Logistic回归（将显著的单因素纳入）
    if len(significant_univariate) > 0:
        multi_features = significant_univariate['指标'].tolist()
        X_multi = sm.add_constant(df_analysis[multi_features])
        logit_multi = sm.Logit(y_hyper, X_multi).fit(disp=0)
        
        print("\n多因素Logistic回归结果:")
        print(logit_multi.summary())
        
        # 提取多因素中仍然显著的指标
        multi_significant = logit_multi.pvalues[logit_multi.pvalues < 0.05].index.tolist()
        if 'const' in multi_significant:
            multi_significant.remove('const')
        print(f"\n多因素分析独立预测因子: {multi_significant}")
    
    # ==================== 7. ROC曲线分析 ====================
    print("\n\n【步骤7】ROC曲线分析...")
    
    roc_results = []
    plt.figure(figsize=(10, 8))
    
    for feature in candidate_features:
        try:
            auc_score = roc_auc_score(y_hyper, df_analysis[feature])
            fpr, tpr, _ = roc_curve(y_hyper, df_analysis[feature])
            
            roc_results.append({
                '指标': feature,
                'AUC': auc_score
            })
            
            plt.plot(fpr, tpr, linewidth=1.5, label=f'{feature[:15]} (AUC={auc_score:.3f})')
        except:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测 (AUC=0.5)')
    plt.xlabel('假阳性率 (1-特异度)', fontsize=12)
    plt.ylabel('真阳性率 (灵敏度)', fontsize=12)
    plt.title('各指标预测高血脂的ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/ROC曲线.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: images/ROC曲线.png")
    plt.close()
    
    roc_df = pd.DataFrame(roc_results)
    roc_df = roc_df.sort_values(by='AUC', ascending=False)
    print("\n各指标AUC值排序:")
    print(roc_df.to_string(index=False))
    
    # AUC >= 0.6的指标
    good_auc = roc_df[roc_df['AUC'] >= 0.6]
    print(f"\n有预测价值的指标（AUC≥0.6）: {len(good_auc)}个")
    print(good_auc.to_string(index=False))
    
    # ==================== 8. 随机森林特征重要性 ====================
    print("\n\n【步骤8】随机森林特征重要性分析...")
    
    # 预测痰湿质
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reg.fit(X_scaled, y_phlegm)
    
    importance_reg = pd.DataFrame({
        '指标': candidate_features,
        'RF重要性_痰湿质': rf_reg.feature_importances_
    })
    importance_reg = importance_reg.sort_values(by='RF重要性_痰湿质', ascending=False)
    
    print("\n预测痰湿质的特征重要性:")
    print(importance_reg.to_string(index=False))
    
    # 预测高血脂
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_scaled, y_hyper)
    
    importance_clf = pd.DataFrame({
        '指标': candidate_features,
        'RF重要性_高血脂': rf_clf.feature_importances_
    })
    importance_clf = importance_clf.sort_values(by='RF重要性_高血脂', ascending=False)
    
    print("\n预测高血脂的特征重要性:")
    print(importance_clf.to_string(index=False))
    
    # 绘制特征重要性图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    top_n = 10
    imp_reg_top = importance_reg.head(top_n)
    ax1.barh(range(top_n), imp_reg_top['RF重要性_痰湿质'].values, color='steelblue')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(imp_reg_top['指标'].values, fontsize=9)
    ax1.set_xlabel('重要性', fontsize=11)
    ax1.set_title('随机森林-痰湿质预测\n(Top 10)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    imp_clf_top = importance_clf.head(top_n)
    ax2.barh(range(top_n), imp_clf_top['RF重要性_高血脂'].values, color='coral')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(imp_clf_top['指标'].values, fontsize=9)
    ax2.set_xlabel('重要性', fontsize=11)
    ax2.set_title('随机森林-高血脂预测\n(Top 10)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('images/随机森林特征重要性.png', dpi=300, bbox_inches='tight')
    print("\n✓ 已保存: images/随机森林特征重要性.png")
    plt.close()
    
    # ==================== 9. 综合筛选关键指标 ====================
    print("\n\n" + "="*80)
    print("【最终结果】关键指标综合筛选")
    print("="*80)
    
    # 整合所有结果
    final_summary = []
    
    for feature in candidate_features:
        # 相关性
        corr_row = corr_df[corr_df['指标'] == feature]
        corr_val = corr_row['相关系数_r'].values[0] if len(corr_row) > 0 else 0
        corr_p = corr_row['p值'].values[0] if len(corr_row) > 0 else 1
        sig_corr = abs(corr_val) >= 0.2 and corr_p < 0.05
        
        # LASSO
        lasso_row = selected_by_lasso[selected_by_lasso['指标'] == feature]
        in_lasso = len(lasso_row) > 0
        
        # Logistic
        log_row = logistic_df[logistic_df['指标'] == feature]
        log_p = log_row['p值'].values[0] if len(log_row) > 0 else 1
        sig_logistic = log_p < 0.05
        
        # ROC
        roc_row = roc_df[roc_df['指标'] == feature]
        auc_val = roc_row['AUC'].values[0] if len(roc_row) > 0 else 0.5
        good_roc = auc_val >= 0.6
        
        # RF重要性排名
        reg_rank = importance_reg[importance_reg['指标'] == feature].index[0] + 1 if feature in importance_reg['指标'].values else 999
        clf_rank = importance_clf[importance_clf['指标'] == feature].index[0] + 1 if feature in importance_clf['指标'].values else 999
        top_rf = reg_rank <= 10 or clf_rank <= 10
        
        # 满足条件数
        criteria_met = sum([sig_corr, in_lasso, sig_logistic, good_roc, top_rf])
        
        final_summary.append({
            '指标': feature,
            '类型': '血常规' if feature in blood_indicators else '活动量表',
            '相关系数_r': round(corr_val, 4),
            '相关性p值': f"{corr_p:.4f}",
            '显著相关': sig_corr,
            'LASSO选中': in_lasso,
            'Logistic_p值': f"{log_p:.4f}",
            'Logistic显著': sig_logistic,
            'AUC': round(auc_val, 4),
            'ROC达标': good_roc,
            'RF排名前10': top_rf,
            '满足条件数': criteria_met
        })
    
    final_df = pd.DataFrame(final_summary)
    final_df = final_df.sort_values(by='满足条件数', ascending=False)
    
    print("\n所有指标综合评估:")
    print(final_df.to_string(index=False))
    
    # 关键指标（满足至少2个条件）
    key_indicators = final_df[final_df['满足条件数'] >= 2]
    
    print("\n\n" + "🔑"*40)
    print("关键指标列表（满足≥2个筛选条件）")
    print("🔑"*40)
    
    if len(key_indicators) > 0:
        print(f"\n共筛选出 {len(key_indicators)} 个关键指标:\n")
        
        # 按类型分组
        blood_key = key_indicators[key_indicators['类型'] == '血常规']
        activity_key = key_indicators[key_indicators['类型'] == '活动量表']
        
        if len(blood_key) > 0:
            print("【血常规指标】")
            for _, row in blood_key.iterrows():
                print(f"  • {row['指标']}")
                print(f"    - 相关系数: r={row['相关系数_r']}, p={row['相关性p值']}")
                print(f"    - AUC: {row['AUC']}")
                print(f"    - 满足条件: {row['满足条件数']}/5")
                print()
        
        if len(activity_key) > 0:
            print("【活动量表指标】")
            for _, row in activity_key.iterrows():
                print(f"  • {row['指标']}")
                print(f"    - 相关系数: r={row['相关系数_r']}, p={row['相关性p值']}")
                print(f"    - AUC: {row['AUC']}")
                print(f"    - 满足条件: {row['满足条件数']}/5")
                print()
        
        # 保存结果
        key_indicators.to_csv('data/关键指标筛选结果.csv', index=False, encoding='utf-8-sig')
        print("\n✓ 已保存: data/关键指标筛选结果.csv")
        
    else:
        print("\n⚠ 未找到满足条件的关键指标，建议降低筛选标准")
    
    # Top 5最重要指标
    top5 = final_df.head(5)
    print("\n\n⭐ Top 5 最重要指标 ⭐")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['指标']} ({row['类型']})")
        print(f"   满足条件数: {row['满足条件数']}/5")
    
    print("\n\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  1. images/相关性热力图.png")
    print("  2. images/LASSO系数图.png")
    print("  3. images/ROC曲线.png")
    print("  4. images/随机森林特征重要性.png")
    print("  5. data/关键指标筛选结果.csv")

if __name__ == '__main__':
    main()

"""
痰湿体质高风险人群核心特征组合识别分析 - 模块化版本
核心思路：按生理模块分析特征重要性，挖掘特征组合模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
import itertools
import sys
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 注意力机制模型（作为佐证）====================
class TabularAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1), 
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return self.classifier(x * attn_weights), attn_weights.squeeze(-1)


def train_attention_model(X_train, y_train, X_val, y_val, epochs=300, lr=0.0005, batch_size=32, hidden_dim=128, dropout=0.3):
    """训练注意力模型（仅作为佐证方法）"""
    model = TabularAttentionModel(X_train.shape[1], hidden_dim, dropout)
    X_tr, y_tr = torch.FloatTensor(X_train.values), torch.LongTensor(y_train.values)
    X_va, y_va = torch.FloatTensor(X_val.values), torch.LongTensor(y_val.values)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_state = 0, None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx)[0], by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_va)[0].argmax(1) == y_va).float().mean().item()
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 50 == 0: 
            print(f"    Epoch [{epoch+1}/{epochs}], Val Acc: {acc:.4f}")
    
    if best_state: 
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        attn = model(X_va)[1].numpy()
    
    # 如果注意力权重过于均匀，使用梯度-based特征重要性
    if attn.std(axis=0).mean() < 0.01:
        print("    ⚠️ 使用梯度-based特征重要性")
        X_va.requires_grad = True
        model(X_va)[0].sum().backward()
        attn = X_va.grad.abs().mean(0).numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn = np.tile(attn, (X_va.shape[0], 1))
    
    return model, attn, best_acc


# ==================== 模块化特征分析 ====================
def analyze_module_importance(rf_model, shap_values, all_features, feature_modules):
    """
    按模块分析特征重要性
    
    参数:
        rf_model: 随机森林模型
        shap_values: SHAP值数组
        all_features: 所有特征名称
        feature_modules: 字典，定义各模块包含的特征
    
    返回:
        module_analysis: 各模块的分析结果
    """
    print("\n【模块化特征重要性分析】")
    print("="*80)
    
    module_analysis = {}
    
    for module_name, features in feature_modules.items():
        # 提取该模块的特征索引（只保留在all_features中存在的特征）
        valid_features = [f for f in features if f in all_features]
        feat_indices = [all_features.index(f) for f in valid_features]
        
        if not feat_indices:
            print(f"  ⚠️ 模块 '{module_name}' 无匹配特征")
            continue
        
        # 计算模块内特征的SHAP贡献
        module_shap = shap_values[:, feat_indices]
        
        # 处理3维SHAP值（二分类问题）
        if module_shap.ndim == 3:
            # 选择正类（类别1）的SHAP值
            module_shap = module_shap[:, :, 1]
        
        avg_abs_shap = np.abs(module_shap).mean(axis=0)
        
        # Top特征 - 在模块内有效特征中排序（不超过实际特征数）
        n_top = min(5, len(valid_features))
        top_local_indices = np.argsort(avg_abs_shap.flatten())[::-1][:n_top]
        top_local_indices_list = [int(i) for i in top_local_indices]
        
        # 使用局部索引访问valid_features（而非原始的features）
        top_feat_names = [valid_features[i] for i in top_local_indices_list]
        top_feat_vals = [float(avg_abs_shap.flatten()[i]) for i in top_local_indices_list]
        top_features = list(zip(top_feat_names, top_feat_vals))
        
        # 模块总贡献
        total_contribution = avg_abs_shap.sum()
        
        module_analysis[module_name] = {
            'features': features,
            'top_features': top_features,
            'total_contribution': total_contribution,
            'avg_shap_per_feature': avg_abs_shap
        }
        
        print(f"\n  【{module_name}】")
        print(f"    总贡献度: {total_contribution:.4f}")
        print(f"    Top 3 关键特征:")
        for rank, (feat, val) in enumerate(top_features[:3], 1):
            print(f"      {rank}. {feat}: {val:.4f}")
    
    return module_analysis


# ==================== 特征交互效应分析 ====================
def analyze_feature_interactions(X, y_binary, shap_values, all_features, top_n=5):
    """
    分析Top特征之间的交互效应
    
    参数:
        X: 特征矩阵
        y_binary: 目标变量
        shap_values: SHAP值数组
        all_features: 特征名称列表
        top_n: 分析的Top特征数量
    
    返回:
        interaction_results: 交互效应分析结果
    """
    print(f"\n【特征交互效应分析】Top {top_n}特征两两组合")
    print("-"*80)
    
    # 处理3维SHAP值
    if shap_values.ndim == 3:
        shap_values_2d = shap_values[:, :, 1]
    else:
        shap_values_2d = shap_values
    
    # 选择Top N重要特征
    avg_abs_shap = np.abs(shap_values_2d).mean(axis=0)
    top_indices = np.argsort(avg_abs_shap)[::-1][:top_n]
    top_indices_list = [int(i) for i in top_indices]
    top_features = [all_features[i] for i in top_indices_list]
    
    print(f"分析特征: {', '.join(top_features)}\n")
    
    interactions = []
    
    # 对所有特征对进行分析
    for i, j in itertools.combinations(range(len(top_features)), 2):
        feat1, feat2 = top_features[i], top_features[j]
        idx1, idx2 = top_indices_list[i], top_indices_list[j]
        
        # 提取高分组样本
        high_mask = y_binary == 1
        X_high = X[high_mask]
        shap_high = shap_values_2d[high_mask]
        
        # 计算两个特征的联合SHAP贡献
        combined_shap = shap_high[:, idx1] + shap_high[:, idx2]
        
        # 四分位数分组
        q1_feat1 = X_high[feat1].quantile(0.25)
        q3_feat1 = X_high[feat1].quantile(0.75)
        q1_feat2 = X_high[feat2].quantile(0.25)
        q3_feat2 = X_high[feat2].quantile(0.75)
        
        # 高低分组
        high_high = (X_high[feat1] > q3_feat1) & (X_high[feat2] > q3_feat2)
        low_low = (X_high[feat1] < q1_feat1) & (X_high[feat2] < q1_feat2)
        
        if high_high.sum() > 0 and low_low.sum() > 0:
            shap_hh = combined_shap[high_high].mean()
            shap_ll = combined_shap[low_low].mean()
            interaction_strength = abs(shap_hh - shap_ll)
            
            interactions.append({
                'feature_pair': f'{feat1} × {feat2}',
                'feat1': feat1,
                'feat2': feat2,
                'shap_high_high': shap_hh,
                'shap_low_low': shap_ll,
                'interaction_strength': interaction_strength,
                'sample_hh': int(high_high.sum()),
                'sample_ll': int(low_low.sum())
            })
    
    # 按交互强度排序
    interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
    
    print("Top 5 强交互特征对:")
    for idx, inter in enumerate(interactions[:5], 1):
        print(f"  {idx}. {inter['feature_pair']}")
        print(f"     交互强度: {inter['interaction_strength']:.4f}")
        print(f"     双高组SHAP: {inter['shap_high_high']:+.3f} (n={inter['sample_hh']})")
        print(f"     双低组SHAP: {inter['shap_low_low']:+.3f} (n={inter['sample_ll']})")
    
    return interactions[:10]  # 返回Top 10


# ==================== 高风险模式挖掘 ====================
def analyze_high_risk_patterns(X, y_binary, rf_model, all_features, img_dir, mode_subdir, n_clusters=4):
    """
    使用SHAP聚类挖掘高风险人群的不同特征组合模式
    """
    from sklearn.cluster import KMeans
    
    print(f"\n【高风险模式挖掘】SHAP聚类分析 (k={n_clusters})...")
    print("-"*80)
    
    # 计算SHAP值
    shap_vals = shap.TreeExplainer(rf_model).shap_values(X)
    if isinstance(shap_vals, list):
        shap_pos = shap_vals[1]
    else:
        shap_pos = shap_vals
    
    if shap_pos.ndim == 3:
        shap_pos = shap_pos[:, :, 1]
    
    # 提取高分组样本
    high_mask = y_binary == 1
    shap_high = shap_pos[high_mask]
    print(f"高分组样本: {high_mask.sum()}人, SHAP形状: {shap_high.shape}")
    
    # 动态调整聚类数
    if high_mask.sum() < n_clusters * 15:
        n_clusters = max(2, int(high_mask.sum() / 15))
        print(f"⚠️ 样本数不足，调整聚类数为: {n_clusters}")
    
    # K-means聚类
    clusters = KMeans(n_clusters, random_state=42, n_init=10).fit_predict(shap_high)
    
    # 定义模块
    lipid_f = ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
    metab_f = ['BMI', '空腹血糖', '血尿酸']
    const_f = ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质']
    activ_f = [f for f in all_features if 'ADL' in f or 'IADL' in f or '活动量表' in f]
    lifestyle_f = ['年龄组', '性别', '吸烟史', '饮酒史']
    
    cluster_info = []
    for cid in range(n_clusters):
        mask = clusters == cid
        avg = shap_high[mask].mean(0)
        top5 = [(all_features[i], avg[i]) for i in np.argsort(np.abs(avg))[-5:][::-1]]
        
        # 计算各模块贡献
        contribs = {
            '血脂指标': sum(abs(avg[all_features.index(f)]) for f in lipid_f if f in all_features),
            '代谢指标': sum(abs(avg[all_features.index(f)]) for f in metab_f if f in all_features),
            '体质特征': sum(abs(avg[all_features.index(f)]) for f in const_f if f in all_features),
            '活动能力': sum(abs(avg[all_features.index(f)]) for f in activ_f if f in all_features),
            '生活方式': sum(abs(avg[all_features.index(f)]) for f in lifestyle_f if f in all_features)
        }
        
        dom = max(contribs, key=contribs.get)
        pname_map = {
            '血脂指标': '血脂主导型',
            '代谢指标': '代谢主导型', 
            '体质特征': '体质主导型',
            '活动能力': '活动能力相关型',
            '生活方式': '生活方式相关型'
        }
        pname = pname_map.get(dom, '混合型')
        
        cluster_info.append({
            'cluster_id': cid, 
            'pattern_name': pname, 
            'sample_count': int(mask.sum()),
            'top5_features': top5, 
            'avg_shap': avg, 
            'contributions': contribs, 
            'dominant_type': dom
        })
        
        print(f"\n  模式{cid+1}({pname}): {mask.sum()}人")
        print(f"    核心驱动特征:")
        for fn, sv in top5: 
            print(f"      - {fn}: {'↑' if sv>0 else '↓'} (SHAP={sv:+.3f})")
    
    # ==================== 可视化 ====================
    
    # 可视化1: 雷达图（5维）
    print(f"\n生成可视化: 高风险模式雷达图...")
    cats = ['血脂指标', '代谢指标', '体质特征', '活动能力', '生活方式']
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5))
    if n_clusters == 1: 
        axes = [axes]
    
    for i, c in enumerate(cluster_info):
        vals = [c['contributions'][cat] for cat in cats] + [c['contributions'][cats[0]]]
        mx = max(vals[:-1]) or 1
        axes[i].plot(angles, [v/mx for v in vals], 'o-', linewidth=2)
        axes[i].fill(angles, [v/mx for v in vals], alpha=0.25)
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(cats, fontsize=9)
        axes[i].set_title(f'模式{i+1}: {c["pattern_name"]}\n({c["sample_count"]}人)', 
                         fontsize=11, fontweight='bold')
        axes[i].grid(True)
    
    plt.suptitle(f'高风险人群特征组合模式 ({mode_subdir})', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '01_高风险模式挖掘_特征组合雷达图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化2: 特征贡献对比
    print(f"生成可视化: 高风险模式特征贡献对比...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_tops = []
    for c in cluster_info:
        for fn, _ in c['top5_features']:
            if fn not in all_tops: 
                all_tops.append(fn)
    all_tops = all_tops[:10]
    
    x, w = np.arange(len(all_tops)), 0.25
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for i, c in enumerate(cluster_info):
        vals = [c['avg_shap'][all_features.index(f)] if f in all_features else 0 for f in all_tops]
        ax.barh(x + i*w, vals, w, label=c['pattern_name'], color=colors[i%5], alpha=0.8)
    
    ax.set_yticks(x + w*(n_clusters-1)/2)
    ax.set_yticklabels(all_tops, fontsize=10)
    ax.set_xlabel('平均SHAP值（正值增加风险）', fontsize=12, fontweight='bold')
    ax.set_title(f'高风险模式特征贡献对比 ({mode_subdir})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '02_高风险模式挖掘_特征贡献对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化3: 特征共现热力图
    print(f"生成可视化: 高风险特征共现热力图...")
    n_feat = len(all_features)
    cooc = np.zeros((n_feat, n_feat))
    X_high = X[high_mask]
    
    for i in range(len(X_high)):
        t3 = np.argsort(np.abs(shap_high[i]))[-3:]
        for i1 in t3:
            for i2 in t3: 
                cooc[i1, i2] += 1
    
    t15 = np.argsort(cooc.sum(1))[-15:]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cooc[t15][:, t15], annot=False, cmap='YlOrRd', 
               xticklabels=[all_features[i] for i in t15], 
               yticklabels=[all_features[i] for i in t15], ax=ax)
    ax.set_title(f'高风险样本特征共现热力图 ({mode_subdir})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '03_高风险模式挖掘_特征共现热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_info


# ==================== 主分析函数 ====================
def run_analysis(df, feature_cols, target_col, output_dir, activity_total_col, **attn_kwargs):
    """
    运行完整分析流程（移除决策树，强化模块化分析）
    
    参数:
        df: 数据框
        feature_cols: 基础特征列（不含活动量表）
        target_col: 目标变量列名
        output_dir: 输出目录
        activity_total_col: 活动量表总分列名
    """
    print(f"\n{'='*80}")
    print(f"痰湿体质高风险人群特征组合识别分析")
    print(f"{'='*80}")
    
    img_dir = output_dir
    os.makedirs(img_dir, exist_ok=True)
    
    # ==================== 构建特征矩阵 ====================
    print("\n【步骤1】构建特征矩阵")
    print("-"*80)
    
    # 只使用活动量表总分
    all_feats = feature_cols + [activity_total_col]
    X = df[all_feats].fillna(df[all_feats].median())
    y = (df[target_col] >= 60).astype(int)
    
    print(f"总特征数: {len(all_feats)}")
    print(f"目标变量分布:")
    print(f"  痰湿质高分组 (≥60): {y.sum()}人 ({y.mean()*100:.1f}%)")
    print(f"  痰湿质低分组 (<60):  {(1-y).sum()}人 ({(1-y).mean()*100:.1f}%)")
    
    # ==================== 定义特征模块 ====================
    feature_modules = {
        '体质模块': ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质'],
        '血脂模块': ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）'],
        '代谢模块': ['BMI', '空腹血糖', '血尿酸'],
        '生活方式模块': ['年龄组', '性别', '吸烟史', '饮酒史'],
        '活动能力模块': [activity_total_col]
    }
    
    # ==================== 主分析：随机森林 + SHAP ====================
    print("\n【步骤2】主分析：随机森林 + SHAP")
    print("-"*80)
    
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X, y)
    
    y_pred_rf = rf_model.predict(X)
    rf_acc = (y_pred_rf == y).mean()
    print(f"随机森林准确率: {rf_acc:.4f}")
    
    # SHAP分析
    print("\n计算SHAP值...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    else:
        shap_pos = shap_values
    
    # 模块化特征重要性分析
    module_analysis = analyze_module_importance(rf_model, shap_pos, all_feats, feature_modules)
    
    # ===== 可视化1: SHAP摘要图 =====
    print(f"\n生成可视化: SHAP摘要图...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_pos, X, show=False, plot_type='dot', max_display=15)
    plt.title(f'SHAP特征重要性摘要图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '04_主分析_SHAP摘要图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 可视化2: 模块化特征贡献堆叠图 =====
    print(f"生成可视化: 模块化特征贡献堆叠图...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    module_names = list(module_analysis.keys())
    module_contribs = [module_analysis[m]['total_contribution'] for m in module_names]
    colors_module = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    bars = ax.barh(module_names, module_contribs, color=colors_module[:len(module_names)], alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, module_contribs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
               va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('总SHAP贡献度', fontsize=12, fontweight='bold')
    ax.set_ylabel('特征模块', fontsize=12, fontweight='bold')
    ax.set_title('各模块对痰湿质高风险的总贡献度', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '05_主分析_模块化特征贡献堆叠图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 可视化3: 各模块Top特征详细图 =====
    print(f"生成可视化: 各模块Top特征详细图...")
    n_modules = len(module_analysis)
    fig, axes = plt.subplots(1, n_modules, figsize=(5*n_modules, 6))
    if n_modules == 1:
        axes = [axes]
    
    colors_feat = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for idx, (module_name, analysis) in enumerate(module_analysis.items()):
        ax = axes[idx]
        top_feats = analysis['top_features'][:5]
        feat_names = [f[0] for f in top_feats]
        feat_vals = [f[1] for f in top_feats]
        
        bars = ax.barh(range(len(feat_names)), feat_vals, 
                      color=colors_feat[idx % len(colors_feat)], alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars, feat_vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names, fontsize=10)
        ax.set_xlabel('平均|SHAP|', fontsize=11, fontweight='bold')
        ax.set_title(f'{module_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.suptitle('各模块Top 5关键特征', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '06_主分析_各模块Top特征详细图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==================== 佐证：注意力机制 ====================
    print("\n【步骤3】佐证方法：注意力机制验证")
    print("-"*80)
    
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=all_feats)
    X_va_s = pd.DataFrame(scaler.transform(X_va), columns=all_feats)
    
    print("训练注意力模型...")
    attn_model, attn_weights, val_acc = train_attention_model(X_tr_s, y_tr, X_va_s, y_va, **attn_kwargs)
    print(f"最佳验证准确率: {val_acc:.4f}")
    
    attn_imp = pd.DataFrame({
        '特征': all_feats, 
        '注意力权重': attn_weights.mean(0)
    }).sort_values('注意力权重', ascending=False)
    
    print("\n注意力权重 Top 10:")
    print(attn_imp.head(10).to_string(index=False))
    
    # ===== 可视化4: 注意力权重图 =====
    print(f"\n生成可视化: 注意力权重分布...")
    plt.figure(figsize=(12, 8))
    top10 = attn_imp.head(10)
    plt.barh(range(10), top10['注意力权重'], color='#9B59B6', alpha=0.8, edgecolor='black')
    for i, v in enumerate(top10['注意力权重']): 
        plt.text(v+0.005, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    plt.yticks(range(10), top10['特征'], fontsize=11)
    plt.xlabel('注意力权重', fontsize=12, fontweight='bold')
    plt.title('注意力机制特征权重 Top 10', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '07_佐证_注意力权重分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==================== 特征交互效应分析 ====================
    print("\n【步骤4】特征交互效应分析")
    print("-"*80)
    
    interactions = analyze_feature_interactions(X, y, shap_pos, all_feats, top_n=6)
    
    # ===== 可视化5: 交互效应热力图 =====
    if interactions:
        print(f"\n生成可视化: 特征交互效应热力图...")
        top_interactions = interactions[:9]  # Top 9用于3x3网格
        
        n_pairs = len(top_interactions)
        n_rows = int(np.ceil(n_pairs / 3))
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # 处理3维SHAP值 - 确保使用2维数组
        if shap_pos.ndim == 3:
            shap_pos_2d = shap_pos[:, :, 1]
        else:
            shap_pos_2d = shap_pos
        
        for idx, inter in enumerate(top_interactions):
            ax = axes[idx]
            feat1, feat2 = inter['feat1'], inter['feat2']
            
            high_mask = y == 1
            X_high = X[high_mask]
            
            # 获取特征索引
            idx1 = all_feats.index(feat1)
            idx2 = all_feats.index(feat2)
            
            # 从2维SHAP数组中提取数据（确保是1维）
            shap_feat1 = shap_pos_2d[high_mask][:, idx1]
            shap_feat2 = shap_pos_2d[high_mask][:, idx2]
            combined_shap_color = np.abs(shap_feat1 + shap_feat2)
            
            # 调试信息
            print(f"    散点图 {idx+1}: X={X_high[feat1].shape}, Y={X_high[feat2].shape}, C={combined_shap_color.shape}")
            
            scatter = ax.scatter(X_high[feat1], X_high[feat2], 
                               c=combined_shap_color,
                               cmap='RdYlBu_r', alpha=0.6, s=30)
            
            ax.set_xlabel(feat1, fontsize=9, fontweight='bold')
            ax.set_ylabel(feat2, fontsize=9, fontweight='bold')
            ax.set_title(f'{feat1} × {feat2}\n强度:{inter["interaction_strength"]:.3f}', 
                        fontsize=10, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='联合SHAP贡献')
        
        # 隐藏多余的子图
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Top特征交互效应分析', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, '08_交互效应_特征交互热力图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # ==================== 高风险模式挖掘 ====================
    print("\n【步骤5】高风险模式挖掘")
    print("-"*80)
    
    cluster_patterns = analyze_high_risk_patterns(X, y, rf_model, all_feats, img_dir, "整体分析", n_clusters=4)
    
    # ==================== 返回结果 ====================
    results = {
        'features': all_feats,
        'rf_accuracy': rf_acc,
        'attn_accuracy': val_acc,
        'rf_importance': pd.DataFrame({'特征': all_feats, '重要性': rf_model.feature_importances_}).sort_values('重要性', ascending=False),
        'attn_importance': attn_imp,
        'module_analysis': module_analysis,
        'interactions': interactions,
        'cluster_patterns': cluster_patterns
    }
    
    print(f"\n✓ 分析完成！")
    print(f"  发现 {len(cluster_patterns)} 种高风险模式")
    print(f"  识别 {len(interactions)} 个强交互特征对")
    
    return results


# ==================== Markdown报告 ====================
def generate_report(results, output_dir):
    """生成Markdown分析报告"""
    md_path = os.path.join(output_dir, 'q2_2_3', '分析报告.md')
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 痰湿体质高风险人群核心特征组合识别分析报告\n\n")
        f.write("**分析日期**: 2026-04-18\n\n")
        f.write("**核心思路**: 按生理模块分析特征重要性，挖掘特征组合模式，不使用决策树阈值判断\n\n---\n\n")
        
        # 分析概述
        f.write("## 1. 分析概述\n\n")
        f.write("### 1.1 研究目标\n")
        f.write("识别痰湿体质高风险人群（痰湿质评分≥60分）的核心特征组合模式，重点关注：\n")
        f.write("- 各生理模块（体质、血脂、代谢、生活方式）的关键特征\n")
        f.write("- 特征之间的交互效应和组合模式\n")
        f.write("- 不同高风险人群亚型的特征差异\n\n")
        
        f.write("### 1.2 分析方法\n")
        f.write("- **主分析**: 随机森林 + SHAP（提供特征重要性和贡献方向）\n")
        f.write("- **模块化分析**: 按生理意义分组展示特征重要性\n")
        f.write("- **交互效应分析**: 识别强协同作用的特征对\n")
        f.write("- **高风险模式挖掘**: SHAP值聚类分析（识别不同亚型）\n")
        f.write("- **佐证方法**: 注意力机制（独立验证特征重要性）\n\n")
        
        f.write("### 1.3 特征模块划分\n")
        f.write("| 模块 | 包含特征 |\n")
        f.write("|------|----------|\n")
        f.write("| 体质模块 | 平和质、气虚质、阳虚质、阴虚质、湿热质、血瘀质、气郁质、特禀质 |\n")
        f.write("| 血脂模块 | TG、TC、LDL-C、HDL-C |\n")
        f.write("| 代谢模块 | BMI、空腹血糖、血尿酸 |\n")
        f.write("| 生活方式模块 | 年龄组、性别、吸烟史、饮酒史 |\n")
        f.write("| 活动能力模块 | 活动量表总分 |\n\n")
        f.write("---\n\n")
        
        # 模型性能
        f.write("## 2. 模型性能\n\n")
        f.write("| 模型 | 准确率 |\n")
        f.write("|------|--------|\n")
        f.write(f"| 随机森林 | {results['rf_accuracy']:.4f} |\n")
        f.write(f"| 注意力机制 | {results['attn_accuracy']:.4f} |\n\n")
        f.write("---\n\n")
        
        # 模块化分析
        f.write("## 3. 模块化特征重要性分析\n\n")
        f.write("### 3.1 各模块总贡献度对比\n\n")
        f.write("| 模块 | 总贡献度 | 排名 |\n")
        f.write("|------|---------|------|\n")
        
        sorted_modules = sorted(results['module_analysis'].items(), 
                               key=lambda x: x[1]['total_contribution'], reverse=True)
        for rank, (module_name, analysis) in enumerate(sorted_modules, 1):
            f.write(f"| {module_name} | {analysis['total_contribution']:.4f} | {rank} |\n")
        f.write("\n")
        
        # 各模块详细分析
        for module_name, analysis in results['module_analysis'].items():
            f.write(f"### 3.2 {module_name}\n\n")
            f.write(f"**总贡献度**: {analysis['total_contribution']:.4f}\n\n")
            f.write("| 排名 | 特征 | 平均|SHAP| |\n")
            f.write("|------|------|-----------|\n")
            for rank, (feat, val) in enumerate(analysis['top_features'][:5], 1):
                f.write(f"| {rank} | {feat} | {val:.4f} |\n")
            f.write("\n")
        f.write("---\n\n")
        
        # 特征交互效应
        f.write("## 4. 特征交互效应分析\n\n")
        f.write("以下特征对表现出强烈的协同作用（双高/双低时SHAP贡献差异显著）:\n\n")
        f.write("| 排名 | 特征对 | 交互强度 | 双高组SHAP | 双低组SHAP |\n")
        f.write("|------|--------|---------|-----------|-----------|\n")
        for rank, inter in enumerate(results['interactions'][:10], 1):
            f.write(f"| {rank} | {inter['feature_pair']} | {inter['interaction_strength']:.4f} | ")
            f.write(f"{inter['shap_high_high']:+.3f} (n={inter['sample_hh']}) | ")
            f.write(f"{inter['shap_low_low']:+.3f} (n={inter['sample_ll']}) |\n")
        f.write("\n")
        f.write("**解读**: 交互强度越大，说明两个特征同时处于极端值时对痰湿质风险的协同影响越强。\n\n")
        f.write("---\n\n")
        
        # 高风险模式挖掘
        f.write("## 5. 高风险模式挖掘\n\n")
        f.write("通过SHAP聚类识别出高分组人群的**不同特征组合亚型**:\n\n")
        
        for p in results['cluster_patterns']:
            f.write(f"### 5.{p['cluster_id']+1} 模式{p['cluster_id']+1}：{p['pattern_name']}\n\n")
            f.write(f"**样本数**: {p['sample_count']}人\n\n")
            f.write("**核心驱动特征**:\n\n")
            f.write("| 排名 | 特征 | SHAP值 | 影响方向 |\n")
            f.write("|------|------|--------|---------|\n")
            for rk, (fn, sv) in enumerate(p['top5_features'], 1):
                direction = "↑ 增加风险" if sv > 0 else "↓ 降低风险"
                f.write(f"| {rk} | {fn} | {sv:+.3f} | {direction} |\n")
            
            f.write(f"\n**模块贡献分布**:\n")
            for mod_type, contribution in p['contributions'].items():
                f.write(f"- {mod_type}: {contribution:.3f}\n")
            f.write(f"\n**主导类型**: {p['dominant_type']}\n\n")
            f.write("---\n\n")
        
        # 综合结论
        f.write("## 6. 综合结论与建议\n\n")
        f.write("### 6.1 核心发现\n\n")
        
        # 找出最重要的模块
        top_module = sorted_modules[0]
        f.write(f"1. **最重要模块**: {top_module[0]}（贡献度{top_module[1]['total_contribution']:.4f}）\n")
        f.write(f"   - 这是痰湿质高风险的最主要驱动因素\n\n")
        
        # 最强交互
        if results['interactions']:
            top_inter = results['interactions'][0]
            f.write(f"2. **最强交互效应**: {top_inter['feature_pair']}\n")
            f.write(f"   - 交互强度: {top_inter['interaction_strength']:.4f}\n")
            f.write(f"   - 说明这两个特征需要**组合考虑**，单独分析会低估其影响\n\n")
        
        # 高风险模式多样性
        pattern_types = set([p['pattern_name'] for p in results['cluster_patterns']])
        f.write(f"3. **高风险人群异质性**: 识别出 {len(pattern_types)} 种不同类型的高风险模式\n")
        f.write(f"   - 包括: {', '.join(pattern_types)}\n")
        f.write(f"   - 提示临床应采取**差异化干预策略**\n\n")
        
        f.write("### 6.2 临床建议\n\n")
        f.write("基于特征组合模式的发现，建议:\n\n")
        f.write("1. **分层筛查**: 优先关注{top_module[0]}异常的人群\n".format(top_module=top_module))
        f.write("2. **组合干预**: 对于存在强交互效应的特征对，应同时干预\n")
        f.write("3. **个性化方案**: 根据不同高风险模式制定针对性干预措施\n\n")
        f.write("---\n\n")
        f.write("*本报告由自动化分析流程生成*\n")
    
    print(f"\n✓ Markdown报告: {md_path}")


# ==================== 主函数 ====================
def main():
    # 创建日志目录和文件
    log_dir = 'log/q2_2_3'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'q2_2_3_console_log.log')
    
    # 同时输出到控制台和文件
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    logger = Logger(log_file)
    sys.stdout = logger
    
    print("="*80)
    print("痰湿体质高风险人群核心特征组合识别 - 模块化版本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print("="*80)
    
    # 配置
    ATTN_PARAMS = {
        'epochs': 300, 
        'lr': 0.0005, 
        'batch_size': 32, 
        'hidden_dim': 128, 
        'dropout': 0.3
    }
    
    # 读取数据
    df = pd.read_pickle('data/data.pkl')
    print(f"数据规模: {df.shape}")
    
    # 基础特征（不含活动量表分项）
    base_feats = [
        'TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
        '空腹血糖', '血尿酸', 'BMI', 
        '平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质', 
        '年龄组', '性别', '吸烟史', '饮酒史',
        '高血脂症二分类标签', '血脂异常分型标签（确诊病例）'
    ]
    
    # 运行分析（只使用活动量表总分）
    results = run_analysis(
        df=df,
        feature_cols=base_feats,
        target_col='痰湿质',
        output_dir='images/q2_2_3',
        activity_total_col='活动量表总分（ADL总分+IADL总分）',
        **ATTN_PARAMS
    )
    
    # 生成报告
    generate_report(results, 'output')
    
    print("\n" + "="*80)
    print("✓ 全部分析完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志已保存至: {log_file}")
    print("="*80)
    
    # 关闭日志文件
    logger.log.close()
    sys.stdout = logger.terminal


if __name__ == '__main__':
    main()

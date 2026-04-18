"""
痰湿体质高风险人群核心特征组合识别分析 - SHAP聚类模式挖掘版本
新增功能：高风险模式挖掘、SHAP聚类分析、特征组合模式识别
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 注意力机制模型 ====================
class TabularAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim*2), nn.ReLU(), nn.Dropout(dropout), 
                                       nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        attn_weights = self.attention(x)
        return self.classifier(x * attn_weights), attn_weights.squeeze(-1)


def train_attention_model(X_train, y_train, X_val, y_val, epochs=300, lr=0.0005, batch_size=32, hidden_dim=128, dropout=0.3):
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
        if (epoch+1) % 20 == 0: print(f"    Epoch [{epoch+1}/{epochs}], Val Acc: {acc:.4f}")
    
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        attn = model(X_va)[1].numpy()
    if attn.std(axis=0).mean() < 0.01:
        print("    ⚠️ 使用梯度-based特征重要性")
        X_va.requires_grad = True
        model(X_va)[0].sum().backward()
        attn = X_va.grad.abs().mean(0).numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn = np.tile(attn, (X_va.shape[0], 1))
    return model, attn, best_acc


# ==================== 高风险模式挖掘 ====================
def analyze_high_risk_patterns(X, y_binary, rf_model, all_features, img_dir, mode_subdir, n_clusters=4):
    print(f"\n【高风险模式挖掘】SHAP聚类分析 (k={n_clusters})...")
    shap_vals = shap.TreeExplainer(rf_model).shap_values(X)
    
    # 处理SHAP值：如果是列表（二分类），取正类；如果是数组，直接使用
    if isinstance(shap_vals, list):
        shap_pos = shap_vals[1]  # 正类的SHAP值
    else:
        shap_pos = shap_vals
    
    # 确保是2维数组（样本数 × 特征数）
    if shap_pos.ndim == 3:
        shap_pos = shap_pos[:, :, 1]  # 如果是3维，取正类
    
    high_mask = y_binary == 1
    shap_high = shap_pos[high_mask]
    print(f"高分组样本: {high_mask.sum()}人, SHAP形状: {shap_high.shape}")
    
    # 动态调整聚类数
    if high_mask.sum() < n_clusters * 15:
        n_clusters = max(2, int(high_mask.sum() / 15))
        print(f"⚠️ 样本数不足，调整聚类数为: {n_clusters}")
    clusters = KMeans(n_clusters, random_state=42, n_init=10).fit_predict(shap_high)
    
    cluster_info = []
    lipid_f = ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
    metab_f = ['BMI', '空腹血糖', '血尿酸']
    const_f = ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质']
    activ_f = [f for f in all_features if 'ADL' in f or 'IADL' in f or '活动量表' in f]
    
    for cid in range(n_clusters):
        mask = clusters == cid
        avg = shap_high[mask].mean(0)
        top5 = [(all_features[i], avg[i]) for i in np.argsort(np.abs(avg))[-5:][::-1]]
        
        contribs = {
            '血脂指标': sum(abs(avg[all_features.index(f)]) for f in lipid_f if f in all_features),
            '代谢指标': sum(abs(avg[all_features.index(f)]) for f in metab_f if f in all_features),
            '体质特征': sum(abs(avg[all_features.index(f)]) for f in const_f if f in all_features),
            '活动能力': sum(abs(avg[all_features.index(f)]) for f in activ_f if f in all_features)
        }
        dom = max(contribs, key=contribs.get)
        pname = {'血脂指标':'血脂主导型', '代谢指标':'代谢主导型', '体质特征':'体质主导型'}.get(dom, '活动能力相关型')
        
        cluster_info.append({'cluster_id': cid, 'pattern_name': pname, 'sample_count': int(mask.sum()),
                            'top5_features': top5, 'avg_shap': avg, 'contributions': contribs, 'dominant_type': dom})
        print(f"\n  模式{cid+1}({pname}): {mask.sum()}人")
        for fn, sv in top5: print(f"    {fn}: {'↑' if sv>0 else '↓'} (SHAP={sv:+.3f})")
    
    # 可视化1: 雷达图
    cats = ['血脂指标', '代谢指标', '体质特征', '活动能力']
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist() + [0]
    fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5))
    if n_clusters == 1: axes = [axes]
    for i, c in enumerate(cluster_info):
        vals = [c['contributions'][cat] for cat in cats] + [c['contributions'][cats[0]]]
        mx = max(vals[:-1]) or 1
        axes[i].plot(angles, [v/mx for v in vals], 'o-', linewidth=2)
        axes[i].fill(angles, [v/mx for v in vals], alpha=0.25)
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(cats, fontsize=10)
        axes[i].set_title(f'模式{i+1}: {c["pattern_name"]}\n({c["sample_count"]}人)', fontsize=12, fontweight='bold')
        axes[i].grid(True)
    plt.suptitle(f'高风险人群特征组合模式 ({mode_subdir})', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '09_高风险模式挖掘_特征组合雷达图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化2: 特征贡献对比
    fig, ax = plt.subplots(figsize=(12, 8))
    all_tops = []
    for c in cluster_info:
        for fn, _ in c['top5_features']:
            if fn not in all_tops: all_tops.append(fn)
    all_tops = all_tops[:10]
    x, w = np.arange(len(all_tops)), 0.25
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    for i, c in enumerate(cluster_info):
        vals = [c['avg_shap'][all_features.index(f)] if f in all_features else 0 for f in all_tops]
        ax.barh(x + i*w, vals, w, label=c['pattern_name'], color=colors[i%4], alpha=0.8)
    ax.set_yticks(x + w*(n_clusters-1)/2)
    ax.set_yticklabels(all_tops, fontsize=10)
    ax.set_xlabel('平均SHAP值', fontsize=12, fontweight='bold')
    ax.set_title(f'高风险模式特征贡献对比 ({mode_subdir})', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(axis='x', alpha=0.3, linestyle='--'); ax.axvline(0, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '10_高风险模式挖掘_特征贡献对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化3: 共现热力图
    n_feat = len(all_features)
    cooc = np.zeros((n_feat, n_feat))
    X_high = X[high_mask]
    for i in range(len(X_high)):
        t3 = np.argsort(np.abs(shap_high[i]))[-3:]
        for i1 in t3:
            for i2 in t3: cooc[i1, i2] += 1
    t15 = np.argsort(cooc.sum(1))[-15:]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cooc[t15][:, t15], annot=False, cmap='YlOrRd', 
               xticklabels=[all_features[i] for i in t15], yticklabels=[all_features[i] for i in t15], ax=ax)
    ax.set_title(f'高风险样本特征共现热力图 ({mode_subdir})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '11_高风险模式挖掘_特征共现热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_info


# ==================== 主分析函数 ====================
def run_analysis(activity_mode, df, feature_cols, target_col, output_dir, **attn_kwargs):
    print(f"\n{'='*80}\n活动量表模式: {activity_mode}\n{'='*80}")
    mode_map = {'items': '分项数据', 'subtotals': 'ADL_IADL总分', 'total': '活动量表总分'}
    mode_name = mode_map[activity_mode]
    img_dir = os.path.join(output_dir, mode_name)
    os.makedirs(img_dir, exist_ok=True)
    
    # 构建特征
    act_feats = {'items': ['ADL用厕','ADL吃饭','ADL步行','ADL穿衣','ADL洗澡','IADL购物','IADL做饭','IADL理财','IADL交通','IADL服药'],
                 'subtotals': ['ADL总分','IADL总分'], 'total': ['活动量表总分（ADL总分+IADL总分）']}[activity_mode]
    all_feats = feature_cols + act_feats
    X = df[all_feats].fillna(df[all_feats].median())
    y = (df[target_col] >= 60).astype(int)
    
    print(f"\n特征数: {len(all_feats)}, 高分组: {y.sum()}人 ({y.mean()*100:.1f}%)")
    
    # 决策树 - 调整参数以获取有效规则
    dt = DecisionTreeClassifier(
        max_depth=3,  # 降低深度，避免过拟合
        min_samples_leaf=50,  # 增加叶节点最小样本数
        min_samples_split=100,  # 增加分裂所需最小样本数
        class_weight={0: 1, 1: 5},  # 加大对少数类的权重
        criterion='entropy',  # 使用信息增益
        random_state=42
    ).fit(X, y)
    y_pred_dt = dt.predict(X)
    dt_imp = pd.DataFrame({'特征': all_feats, '重要性': dt.feature_importances_}).sort_values('重要性', ascending=False)
    rules = export_text(dt, feature_names=all_feats, max_depth=5)
    
    # 决策树可视化
    plt.figure(figsize=(24, 14))
    plot_tree(dt, feature_names=all_feats, class_names=['低分组','高分组'], filled=True, rounded=True, fontsize=8, max_depth=5)
    plt.title(f'决策树 ({mode_name})', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '01_决策树_树结构.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 随机森林
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1).fit(X, y)
    rf_imp = pd.DataFrame({'特征': all_feats, '重要性': rf.feature_importances_}).sort_values('重要性', ascending=False)
    
    # SHAP
    shap_vals = shap.TreeExplainer(rf).shap_values(X)
    shap_pos = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_pos, X, show=False, plot_type='dot', max_display=10)
    plt.title(f'SHAP摘要图 ({mode_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '02_SHAP摘要图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 注意力机制
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=all_feats)
    X_va_s = pd.DataFrame(scaler.transform(X_va), columns=all_feats)
    
    attn_model, attn_weights, val_acc = train_attention_model(X_tr_s, y_tr, X_va_s, y_va, **attn_kwargs)
    attn_imp = pd.DataFrame({'特征': all_feats, '注意力权重': attn_weights.mean(0)}).sort_values('注意力权重', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top10 = attn_imp.head(10)
    plt.barh(range(10), top10['注意力权重'], color='#9B59B6', alpha=0.8)
    for i, v in enumerate(top10['注意力权重']): plt.text(v+0.005, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    plt.yticks(range(10), top10['特征'], fontsize=11)
    plt.xlabel('注意力权重', fontsize=12, fontweight='bold')
    plt.title(f'注意力权重 Top 10 ({mode_name})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '03_注意力权重.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 高风险模式挖掘 - 增加聚类数量并优化
    cluster_patterns = analyze_high_risk_patterns(X, y, rf, all_feats, img_dir, mode_name, n_clusters=4)
    
    return {'mode': activity_mode, 'mode_name': mode_name, 'features': all_feats,
            'dt_accuracy': (y_pred_dt==y).mean(), 'rf_accuracy': (rf.predict(X)==y).mean(), 'attn_accuracy': val_acc,
            'dt_importance': dt_imp, 'rf_importance': rf_imp, 'attn_importance': attn_imp,
            'tree_rules': rules, 'cluster_patterns': cluster_patterns}


# ==================== Markdown报告 ====================
def generate_report(all_results, output_dir):
    md_path = os.path.join(output_dir, 'q2_2_2', '分析报告.md')
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 痰湿体质高风险人群核心特征组合识别分析报告\n\n")
        f.write("**分析日期**: 2026-04-18\n\n---\n\n")
        f.write("## 1. 分析概述\n\n")
        f.write("### 1.1 研究目标\n识别痰湿体质高风险人群（痰湿质评分≥60分）的核心特征组合模式，通过SHAP聚类挖掘不同的高风险人群亚型。\n\n")
        f.write("### 1.2 分析方法\n- **主分析**: 决策树（提供识别规则）\n- **佐证方法1**: 随机森林 + SHAP（验证稳定性）\n- **佐证方法2**: 注意力机制（独立验证）\n- **高风险模式挖掘**: SHAP值聚类分析（识别特征组合模式）\n\n")
        f.write("### 1.3 活动量表处理方式\n1. **分项数据**: 10个ADL/IADL分项指标\n2. **ADL+IADL总分**: 2个总分指标\n3. **活动量表总分**: 1个综合指标\n\n---\n\n")
        
        for res in all_results:
            mn = res['mode_name']
            f.write(f"## 2. 分析结果 - {mn}\n\n")
            f.write(f"### 2.1 模型性能\n| 模型 | 准确率 |\n|------|--------|\n| 决策树 | {res['dt_accuracy']:.4f} |\n| 随机森林 | {res['rf_accuracy']:.4f} |\n| 注意力机制 | {res['attn_accuracy']:.4f} |\n\n")
            
            f.write("### 2.2 决策树Top 5特征\n| 排名 | 特征 | 重要性 |\n|------|------|--------|\n")
            for i, r in res['dt_importance'].head(5).iterrows(): f.write(f"| {i+1} | {r['特征']} | {r['重要性']:.4f} |\n")
            f.write("\n")
            
            f.write("### 2.3 随机森林Top 5特征\n| 排名 | 特征 | 重要性 |\n|------|------|--------|\n")
            for i, r in res['rf_importance'].head(5).iterrows(): f.write(f"| {i+1} | {r['特征']} | {r['重要性']:.4f} |\n")
            f.write("\n")
            
            f.write("### 2.4 注意力机制Top 5特征\n| 排名 | 特征 | 权重 |\n|------|------|------|\n")
            for i, r in res['attn_importance'].head(5).iterrows(): f.write(f"| {i+1} | {r['特征']} | {r['注意力权重']:.4f} |\n")
            f.write("\n")
            
            f.write("### 2.5 决策树规则\n```\n")
            f.write(res['tree_rules'])
            f.write("```\n\n")
            
            f.write("### 2.6 高风险模式挖掘\n\n")
            if res.get('cluster_patterns'):
                for p in res['cluster_patterns']:
                    f.write(f"**模式{p['cluster_id']+1}：{p['pattern_name']}** ({p['sample_count']}人)\n\n")
                    f.write("| 排名 | 特征 | SHAP值 | 方向 |\n|------|------|--------|------|\n")
                    for rk, (fn, sv) in enumerate(p['top5_features'], 1):
                        f.write(f"| {rk} | {fn} | {sv:+.3f} | {'↑增加风险' if sv>0 else '↓降低风险'} |\n")
                    f.write(f"\n**主导类型**: {p['dominant_type']}\n\n")
            f.write("---\n\n")
        
        # 综合对比
        f.write("## 3. 综合对比\n\n### 3.1 性能对比\n| 模式 | 决策树 | 随机森林 | 注意力 |\n|------|--------|---------|--------|\n")
        for r in all_results: f.write(f"| {r['mode_name']} | {r['dt_accuracy']:.4f} | {r['rf_accuracy']:.4f} | {r['attn_accuracy']:.4f} |\n")
        f.write("\n### 3.2 高风险模式对比\n\n")
        for r in all_results:
            f.write(f"**{r['mode_name']}**:\n")
            if r.get('cluster_patterns'):
                pnames = [p['pattern_name'] for p in r['cluster_patterns']]
                f.write(f"- 发现模式: {', '.join(pnames)}\n")
                cnt = Counter(pnames)
                f.write(f"- 最常见: {cnt.most_common(1)[0]}\n")
            f.write("\n")
        f.write("---\n\n## 4. 结论\n\n基于SHAP聚类分析，识别出痰湿质高风险人群的多种特征组合模式，临床可针对不同模式采取差异化干预策略。\n\n*本报告由自动化分析流程生成*\n")
    
    print(f"\n✓ Markdown报告: {md_path}")


# ==================== 主函数 ====================
def main():
    print("="*80 + "\n痰湿体质高风险人群核心特征组合识别 - SHAP聚类模式挖掘版\n" + "="*80)
    
    # 配置
    ATTN_PARAMS = {'epochs': 300, 'lr': 0.0005, 'batch_size': 32, 
                   'hidden_dim': 128, 'dropout': 0.3}
    
    df = pd.read_pickle('data/data.pkl')
    print(f"数据: {df.shape}")
    
    base_feats = ['TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
                  '空腹血糖', '血尿酸', 'BMI', '平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', 
                  '血瘀质', '气郁质', '特禀质', '年龄组', '性别', '吸烟史', '饮酒史',
                  '高血脂症二分类标签', '血脂异常分型标签（确诊病例）']
    
    results = []
    for mode in ['items', 'subtotals', 'total']:
        results.append(run_analysis(mode, df, base_feats, '痰湿质', 'images/q2_2_2', **ATTN_PARAMS))
    
    generate_report(results, 'output')
    print("\n✓ 全部分析完成！")


if __name__ == '__main__':
    main()

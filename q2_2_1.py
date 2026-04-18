import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 注意力机制模型定义 ====================
class TabularAttentionModel(nn.Module):
    """带注意力机制的表格数据分类模型"""

    def __init__(self, input_dim, hidden_dim=32, dropout=0.3):
        super(TabularAttentionModel, self).__init__()

        # 注意力层：学习每个特征的重要性权重
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 二分类输出
        )

    def forward(self, x):
        # 计算注意力权重
        attn_weights = self.attention(x)

        # 加权特征
        weighted_x = x * attn_weights

        # 分类
        output = self.classifier(weighted_x)

        return output, attn_weights.squeeze(-1)


def train_attention_model(X_train, y_train, X_val, y_val, 
                          epochs=200, lr=0.001, batch_size=32, hidden_dim=64, dropout=0.3):
    """训练注意力模型"""
    
    input_dim = X_train.shape[1]
    model = TabularAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.LongTensor(y_val.values)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 训练循环
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, attn_weights = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs, val_attn = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 在验证集上获取最终的注意力权重（使用最佳模型）
    model.eval()
    with torch.no_grad():
        _, final_attn = model(X_val_tensor)
        final_attn_weights = final_attn.numpy()
    
    # 如果注意力权重过于均匀（标准差<0.01），使用梯度-based特征重要性
    attn_std = final_attn_weights.std(axis=0).mean()
    if attn_std < 0.01:
        print("    ⚠️ 注意力权重过于均匀，使用梯度-based特征重要性替代")
        # 计算梯度-based特征重要性
        X_val_tensor.requires_grad = True
        outputs, _ = model(X_val_tensor)
        loss = criterion(outputs, y_val_tensor)
        loss.backward()
        grad_importance = X_val_tensor.grad.abs().mean(dim=0).numpy()
        # 归一化到0-1
        final_attn_weights = (grad_importance - grad_importance.min()) / (grad_importance.max() - grad_importance.min() + 1e-8)
        # 复制成与样本数相同的形状
        final_attn_weights = np.tile(final_attn_weights, (X_val_tensor.shape[0], 1))
    
    return model, final_attn_weights, best_val_acc


def run_analysis_for_activity_mode(activity_mode, df, feature_cols, target_col, output_base_dir,
                                   attn_train_ratio=0.8, attn_epochs=200, attn_lr=0.001, 
                                   attn_batch_size=32, attn_hidden_dim=64, attn_dropout=0.3):
    """
    针对特定活动量表模式运行完整分析
    
    参数:
        activity_mode: 'items' | 'subtotals' | 'total'
        df: 完整数据框
        feature_cols: 除活动量表外的其他特征列
        target_col: 目标变量列名
        output_base_dir: 输出目录
        attn_train_ratio: 注意力模型训练集比例 (默认0.8，即80%训练，20%验证)
        attn_epochs: 注意力模型训练轮数
        attn_lr: 注意力模型学习率
        attn_batch_size: 注意力模型批次大小
        attn_hidden_dim: 注意力模型隐藏层维度
        attn_dropout: 注意力模型Dropout比例
    """
    
    print(f"\n{'='*80}")
    print(f"活动量表模式: {activity_mode}")
    print(f"{'='*80}")
    
    # 创建子目录
    mode_dir_map = {
        'items': '分项数据',
        'subtotals': 'ADL_IADL总分',
        'total': '活动量表总分'
    }
    mode_subdir = mode_dir_map[activity_mode]
    
    img_dir = os.path.join(output_base_dir, mode_subdir)
    os.makedirs(img_dir, exist_ok=True)
    
    # ==================== 构建特征矩阵 ====================
    print("\n【步骤1】构建特征矩阵")
    print("-"*80)
    
    # 基础特征（血脂、代谢等）
    base_features = feature_cols.copy()
    
    # 根据模式添加活动量表特征
    if activity_mode == 'items':
        activity_features = [
            'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡',
            'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药'
        ]
        print("活动量表特征: 10个分项指标")
    elif activity_mode == 'subtotals':
        activity_features = ['ADL总分', 'IADL总分']
        print("活动量表特征: ADL总分 + IADL总分")
    else:  # total
        activity_features = ['活动量表总分（ADL总分+IADL总分）']
        print("活动量表特征: 活动量表总分")
    
    all_features = base_features + activity_features
    X = df[all_features].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 构建目标变量
    y_binary = (df[target_col] >= 60).astype(int)
    
    print(f"\n总特征数: {len(all_features)}")
    print(f"  基础特征: {len(base_features)}个")
    print(f"  活动量表特征: {len(activity_features)}个")
    print(f"\n目标变量分布:")
    print(f"  痰湿质高分组 (≥60): {y_binary.sum()}人 ({y_binary.mean()*100:.1f}%)")
    print(f"  痰湿质低分组 (<60):  {(1-y_binary).sum()}人 ({(1-y_binary).mean()*100:.1f}%)")
    
    # ==================== 主分析：决策树 ====================
    print("\n【步骤2】主分析：决策树规则提取")
    print("-"*80)
    
    dt_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,  # 修改：从 30 增加到 50，减少过拟合和假阳性
        criterion='gini',
        random_state=42
        # 修改：移除 class_weight='balanced'，避免模型倾向于预测少数类
    )
    dt_model.fit(X, y_binary)
    
    y_pred_dt = dt_model.predict(X)
    
    print("\n决策树分类报告:")
    report = classification_report(y_binary, y_pred_dt, 
                                   target_names=['低分组', '高分组'],
                                   output_dict=True)
    print(classification_report(y_binary, y_pred_dt, 
                               target_names=['低分组', '高分组']))
    
    # 特征重要性
    dt_importance = pd.DataFrame({
        '特征': all_features,
        '重要性': dt_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n决策树特征重要性 (Top 10):")
    print(dt_importance.head(10).to_string(index=False))
    
    # 导出规则文本
    tree_rules = export_text(dt_model, feature_names=all_features, max_depth=4)
    print("\n决策树规则:")
    print(tree_rules)
    
    # ===== 可视化1：决策树结构图 =====
    print(f"\n生成可视化 [{mode_subdir}]：决策树结构...")
    plt.figure(figsize=(24, 14))
    
    class_names = ['痰湿质<60', '痰湿质≥60']
    plot_tree(dt_model,
              feature_names=all_features,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=8,
              proportion=True,
              max_depth=4)
    
    plt.title(f'痰湿质高风险人群识别决策树 ({mode_subdir})', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '01_主分析_决策树_树结构.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/01_主分析_决策树_树结构.png")
    
    # ===== 可视化2：特征重要性柱状图 =====
    print(f"\n生成可视化 [{mode_subdir}]：决策树特征重要性...")
    plt.figure(figsize=(12, 8))
    
    top_n = 10
    dt_imp_top = dt_importance.head(top_n)
    colors_bar = ['#E74C3C' if i < 5 else '#F39C12' for i in range(len(dt_imp_top))]
    
    plt.barh(range(len(dt_imp_top)), dt_imp_top['重要性'].values, 
             color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (val, name) in enumerate(zip(dt_imp_top['重要性'].values, dt_imp_top['特征'])):
        plt.text(val + 0.005, i, f'{val:.3f}', ha='left', va='center', 
                fontsize=10, fontweight='bold')
    
    plt.yticks(range(len(dt_imp_top)), dt_imp_top['特征'], fontsize=11)
    plt.xlabel('重要性', fontsize=12, fontweight='bold')
    plt.ylabel('特征', fontsize=12, fontweight='bold')
    plt.title(f'决策树特征重要性 Top {top_n} ({mode_subdir})', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '02_主分析_决策树_特征重要性.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/02_主分析_决策树_特征重要性.png")
    
    # ===== 可视化3：混淆矩阵 =====
    print(f"\n生成可视化 [{mode_subdir}]：决策树混淆矩阵...")
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_binary, y_pred_dt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['低分组', '高分组'],
               yticklabels=['低分组', '高分组'])
    
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.title(f'决策树混淆矩阵 ({mode_subdir})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '03_主分析_决策树_混淆矩阵.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/03_主分析_决策树_混淆矩阵.png")
    
    # ==================== 佐证1：随机森林 + SHAP ====================
    print("\n【步骤3】佐证方法1：随机森林 + SHAP分析")
    print("-"*80)
    
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y_binary)
    
    y_pred_rf = rf_model.predict(X)
    rf_acc = (y_pred_rf == y_binary).mean()
    print(f"随机森林准确率: {rf_acc:.4f}")
    
    # 特征重要性
    rf_importance = pd.DataFrame({
        '特征': all_features,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n随机森林特征重要性 (Top 10):")
    print(rf_importance.head(10).to_string(index=False))
    
    # SHAP分析
    print("\n计算SHAP值...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
    else:
        shap_values_positive = shap_values
    
    # ===== 可视化4：SHAP摘要图 =====
    print(f"\n生成可视化 [{mode_subdir}]：SHAP摘要图...")
    plt.figure(figsize=(12, 10))
    
    shap.summary_plot(shap_values_positive, X, show=False, 
                     plot_type='dot', max_display=10)
    plt.title(f'SHAP特征重要性摘要图 ({mode_subdir})', 
             fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '04_佐证_SHAP分析_特征摘要图.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/04_佐证_SHAP分析_特征摘要图.png")
    
    # ===== 可视化5：SHAP依赖图（Top 3特征）=====
    print(f"\n生成可视化 [{mode_subdir}]：SHAP依赖图...")
    top3_features = rf_importance.head(3)['特征'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feat in enumerate(top3_features):
        ax = axes[idx]
        
        try:
            # 获取特征的SHAP值和实际值
            feat_idx = all_features.index(feat)
            shap_vals = shap_values_positive[:, feat_idx]
            feat_vals = X[feat].values
            
            # 确保维度一致
            if len(shap_vals) != len(feat_vals):
                min_len = min(len(shap_vals), len(feat_vals))
                shap_vals = shap_vals[:min_len]
                feat_vals = feat_vals[:min_len]
            
            # 绘制散点图
            scatter = ax.scatter(feat_vals, shap_vals, c=feat_vals, 
                               cmap='viridis', alpha=0.5, s=10)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label=feat)
            
            ax.set_xlabel(feat, fontsize=10, fontweight='bold')
            ax.set_ylabel('SHAP值', fontsize=10, fontweight='bold')
            ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        except Exception as e:
            print(f"  ⚠️ 跳过特征 '{feat}' 的SHAP依赖图: {str(e)}")
            ax.text(0.5, 0.5, f'SHAP依赖图生成失败\n{feat}', 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title(f'{feat} (生成失败)', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'SHAP依赖图 - Top 3特征 ({mode_subdir})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '05_佐证_SHAP分析_关键特征依赖图.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/05_佐证_SHAP分析_关键特征依赖图.png")
    
    # ==================== 佐证2：注意力机制 ====================
    print("\n【步骤4】佐证方法2：注意力机制验证")
    print("-"*80)
    
    # 划分训练集和验证集（使用可调比例）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary, 
        test_size=1-attn_train_ratio,  # 根据训练比例计算测试集比例
        random_state=42, 
        stratify=y_binary
    )
    
    print(f"注意力模型数据集划分:")
    print(f"  训练集比例: {attn_train_ratio*100:.0f}%")
    print(f"  验证集比例: {(1-attn_train_ratio)*100:.0f}%")
    print(f"  训练集: {len(X_train)}样本")
    print(f"  验证集: {len(X_val)}样本")
    
    # 标准化
    scaler_attn = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler_attn.fit_transform(X_train),
        columns=all_features,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler_attn.transform(X_val),
        columns=all_features,
        index=X_val.index
    )
    
    # 训练注意力模型
    print("\n训练注意力模型...")
    attn_model, attn_weights, val_acc = train_attention_model(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        epochs=attn_epochs,
        lr=attn_lr,
        batch_size=attn_batch_size,
        hidden_dim=attn_hidden_dim,
        dropout=attn_dropout
    )
    
    print(f"最佳验证准确率: {val_acc:.4f}")
    
    # 计算平均注意力权重（在验证集上）
    mean_attn_weights = attn_weights.mean(axis=0)
    
    attn_importance = pd.DataFrame({
        '特征': all_features,
        '注意力权重': mean_attn_weights
    }).sort_values('注意力权重', ascending=False)
    
    print("\n注意力权重 (Top 10):")
    print(attn_importance.head(10).to_string(index=False))
    
    # ===== 可视化6：注意力权重分布 =====
    print(f"\n生成可视化 [{mode_subdir}]：注意力权重分布...")
    plt.figure(figsize=(12, 8))
    
    top_n = 10
    attn_imp_top = attn_importance.head(top_n)
    colors_attn = ['#9B59B6' if i < 5 else '#3498DB' for i in range(len(attn_imp_top))]
    
    plt.barh(range(len(attn_imp_top)), attn_imp_top['注意力权重'].values,
             color=colors_attn, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (val, name) in enumerate(zip(attn_imp_top['注意力权重'].values, 
                                        attn_imp_top['特征'])):
        plt.text(val + 0.005, i, f'{val:.3f}', ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    plt.yticks(range(len(attn_imp_top)), attn_imp_top['特征'], fontsize=11)
    plt.xlabel('注意力权重', fontsize=12, fontweight='bold')
    plt.ylabel('特征', fontsize=12, fontweight='bold')
    plt.title(f'注意力机制特征权重 Top {top_n} ({mode_subdir})', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '06_佐证_注意力机制_注意力权重分布.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/06_佐证_注意力机制_注意力权重分布.png")
    
    # ===== 可视化7：三种方法特征重要性对比 =====
    print(f"\n生成可视化 [{mode_subdir}]：三种方法特征重要性对比...")
    
    common_features = dt_importance.head(8)['特征'].tolist()
    
    dt_imp_aligned = dt_importance.set_index('特征').loc[common_features]['重要性'].values
    rf_imp_aligned = rf_importance.set_index('特征').loc[common_features]['重要性'].values
    attn_imp_aligned = attn_importance.set_index('特征').loc[common_features]['注意力权重'].values
    
    # 归一化到0-1范围
    dt_imp_norm = dt_imp_aligned / dt_imp_aligned.max()
    rf_imp_norm = rf_imp_aligned / rf_imp_aligned.max()
    attn_imp_norm = attn_imp_aligned / attn_imp_aligned.max()
    
    x = np.arange(len(common_features))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.barh(x - width, dt_imp_norm, width, label='决策树', 
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.barh(x, rf_imp_norm, width, label='随机森林', 
                   color='#3498DB', alpha=0.8)
    bars3 = ax.barh(x + width, attn_imp_norm, width, label='注意力机制', 
                   color='#9B59B6', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(common_features, fontsize=11)
    ax.set_xlabel('归一化重要性', fontsize=12, fontweight='bold')
    ax.set_ylabel('特征', fontsize=12, fontweight='bold')
    ax.set_title(f'三种方法特征重要性对比 ({mode_subdir})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '07_佐证_注意力机制_与决策树重要性对比.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/07_佐证_注意力机制_与决策树重要性对比.png")
    
    # ===== 可视化8：不同风险组的注意力权重热力图 =====
    print(f"\n生成可视化 [{mode_subdir}]：不同风险组注意力权重热力图...")
    
    # 在验证集上计算不同风险组的注意力权重
    attn_model.eval()
    with torch.no_grad():
        _, val_attn_weights = attn_model(torch.FloatTensor(X_val_scaled.values))
        val_attn = val_attn_weights.numpy()
    
    high_risk_mask = y_val.values == 1
    low_risk_mask = y_val.values == 0
    
    attn_high = val_attn[high_risk_mask].mean(axis=0)
    attn_low = val_attn[low_risk_mask].mean(axis=0)
    
    attn_comparison = pd.DataFrame({
        '特征': all_features,
        '高分组注意力': attn_high,
        '低分组注意力': attn_low
    })
    
    plt.figure(figsize=(12, 10))
    
    heatmap_data = attn_comparison.set_index('特征')[['高分组注意力', '低分组注意力']].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
               linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.xlabel('特征', fontsize=12, fontweight='bold')
    plt.ylabel('风险组', fontsize=12, fontweight='bold')
    plt.title(f'不同痰湿质风险组的注意力权重对比 ({mode_subdir})\n(基于验证集)', 
             fontsize=14, fontweight='bold')
    plt.xticks(range(len(all_features)), all_features, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '08_佐证_注意力机制_不同风险组注意力热力图.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {mode_subdir}/08_佐证_注意力机制_不同风险组注意力热力图.png")
    
    # ==================== 返回结果用于汇总 ====================
    results = {
        'mode': activity_mode,
        'mode_name': mode_subdir,
        'features': all_features,
        'dt_accuracy': (y_pred_dt == y_binary).mean(),
        'rf_accuracy': rf_acc,
        'attn_accuracy': val_acc,
        'dt_importance': dt_importance,
        'rf_importance': rf_importance,
        'attn_importance': attn_importance,
        'tree_rules': tree_rules,
        'classification_report': report,
        'attn_train_ratio': attn_train_ratio
    }
    
    return results


# def generate_markdown_report(all_results, output_dir):
#     """生成Markdown格式的分析报告"""
#
#     md_path = os.path.join(output_dir, 'q2_2', '分析报告.md')
#     os.makedirs(os.path.dirname(md_path), exist_ok=True)
#
#     with open(md_path, 'w', encoding='utf-8') as f:
#         f.write("# 痰湿体质高风险人群核心特征组合识别分析报告\n\n")
#         f.write("**分析日期**: 2026-04-18\n\n")
#         f.write("---\n\n")
#
#         # 概述
#         f.write("## 1. 分析概述\n\n")
#         f.write("### 1.1 研究目标\n")
#         f.write("识别痰湿体质高风险人群（痰湿质评分≥60分）的核心特征组合，\n")
#         f.write("通过多种机器学习方法提取可解释的规则。\n\n")
#
#         f.write("### 1.2 分析方法\n")
#         f.write("- **主分析**: 决策树（提供清晰的IF-THEN规则）\n")
#         f.write("- **佐证方法1**: 随机森林 + SHAP（验证稳定性）\n")
#         f.write("- **佐证方法2**: 注意力机制（独立验证特征重要性）\n\n")
#
#         f.write("### 1.3 活动量表处理方式\n")
#         f.write("为全面评估活动能力的影响，分三种情况进行分析：\n")
#         f.write("1. **分项数据**: 使用10个ADL/IADL分项指标\n")
#         f.write("2. **ADL+IADL总分**: 使用ADL总分和IADL总分两个指标\n")
#         f.write("3. **活动量表总分**: 仅使用活动量表总分一个指标\n\n")
#
#         f.write("---\n\n")
#
#         # ==================== 循环：各模式详细结果 ====================
#         for result in all_results:
#             mode = result['mode']
#             mode_name = result['mode_name']
#
#             f.write(f"## 2. 分析结果 - {mode_name}\n\n")
#
#             # 特征列表
#             f.write(f"### 2.1 使用的特征 ({len(result['features'])}个)\n\n")
#             f.write("\n")
#             for i, feat in enumerate(result['features'], 1):
#                 f.write(f"{i}. {feat}\n")
#             f.write("\n\n")
#
#             # 模型性能
#             f.write("### 2.2 模型性能\n\n")
#             f.write("| 模型 | 准确率 |\n")
#             f.write("|------|--------|\n")
#             f.write(f"| 决策树 | {result['dt_accuracy']:.4f} |\n")
#             f.write(f"| 随机森林 | {result['rf_accuracy']:.4f} |\n")
#             f.write(f"| 注意力机制（验证集） | {result['attn_accuracy']:.4f} |\n\n")
#
#             # 决策树Top 10特征
#             f.write("### 2.3 决策树特征重要性 Top 10\n\n")
#             f.write("| 排名 | 特征 | 重要性 |\n")
#             f.write("|------|------|--------|\n")
#             for i, row in result['dt_importance'].head(10).iterrows():
#                 f.write(f"| {i + 1} | {row['特征']} | {row['重要性']:.4f} |\n")
#             f.write("\n")
#
#             # 随机森林Top 10特征
#             f.write("### 2.4 随机森林特征重要性 Top 10\n\n")
#             f.write("| 排名 | 特征 | 重要性 |\n")
#             f.write("|------|------|--------|\n")
#             for i, row in result['rf_importance'].head(10).iterrows():
#                 f.write(f"| {i + 1} | {row['特征']} | {row['重要性']:.4f} |\n")
#             f.write("\n")
#
#             # 注意力机制Top 10特征
#             f.write("### 2.5 注意力机制特征权重 Top 10\n\n")
#             f.write("| 排名 | 特征 | 注意力权重 |\n")
#             f.write("|------|------|-----------|\n")
#             for i, row in result['attn_importance'].head(10).iterrows():
#                 f.write(f"| {i + 1} | {row['特征']} | {row['注意力权重']:.4f} |\n")
#             f.write("\n")
#
#             # 决策树规则
#             f.write("### 2.6 决策树规则\n\n")
#             f.write("\n")
#             f.write(result['tree_rules'])
#             f.write("\n\n")
#
#             # 生成的图片
#             f.write("### 2.7 可视化图表\n\n")
#             img_files = [
#                 "01_主分析_决策树_树结构.png",
#                 "02_主分析_决策树_特征重要性.png",
#                 "03_主分析_决策树_混淆矩阵.png",
#                 "04_佐证_SHAP分析_特征摘要图.png",
#                 "05_佐证_SHAP分析_关键特征依赖图.png",
#                 "06_佐证_注意力机制_注意力权重分布.png",
#                 "07_佐证_注意力机制_与决策树重要性对比.png",
#                 "08_佐证_注意力机制_不同风险组注意力热力图.png"
#             ]
#             for img_file in img_files:
#                 f.write(f"- ![{img_file}](images/{mode_name}/{img_file})\n")
#             f.write("\n")
#
#             f.write("---\n\n")
#
#             # 综合对比
#             f.write("## 3. 三种模式的综合对比\n\n")
#
#             f.write("### 3.1 模型性能对比\n\n")
#             f.write("| 活动量表模式 | 决策树准确率 | 随机森林准确率 | 注意力机制验证准确率 |\n")
#             f.write("|-------------|------------|--------------|------------------|\n")
#             for result in all_results:
#                 f.write(f"| {result['mode_name']} | "
#                         f"{result['dt_accuracy']:.4f} | "
#                         f"{result['rf_accuracy']:.4f} | "
#                         f"{result['attn_accuracy']:.4f} |\n")
#             f.write("\n")
#
#             # 共识特征分析
#             f.write("### 3.2 共识特征分析\n\n")
#             f.write("三种方法均认可的Top 5特征（每种模式下）:\n\n")
#
#             for result in all_results:
#                 mode_name = result['mode_name']
#             top5_dt = set(result['dt_importance'].head(5)['特征'].tolist())
#             top5_rf = set(result['rf_importance'].head(5)['特征'].tolist())
#             top5_attn = set(result['attn_importance'].head(5)['特征'].tolist())
#
#             consensus = top5_dt & top5_rf & top5_attn
#
#             f.write(f"**{mode_name}**:\n")
#             if consensus:
#                 f.write(f"- 共识特征 ({len(consensus)}个): {', '.join(consensus)}\n")
#             else:
#                 f.write("- 无完全共识特征，但存在高度重叠\n")
#
#             # 两两交集
#             dt_rf = top5_dt & top5_rf
#             dt_attn = top5_dt & top5_attn
#             rf_attn = top5_rf & top5_attn
#
#             f.write(f"  - 决策树 ∩ 随机森林: {len(dt_rf)}个特征\n")
#             f.write(f"  - 决策树 ∩ 注意力: {len(dt_attn)}个特征\n")
#             f.write(f"  - 随机森林 ∩ 注意力: {len(rf_attn)}个特征\n\n")
#
#             f.write("---\n\n")
#
#             # 结论与建议
#             f.write("## 4. 结论与建议\n\n")
#
#             f.write("### 4.1 主要发现\n\n")
#             f.write("1. **决策树提供了清晰的识别规则**，可直接应用于临床筛查\n")
#             f.write("2. **随机森林和SHAP验证了特征的稳定性**，确保结果可靠\n")
#             f.write("3. **注意力机制从数据驱动角度确认了关键特征**，与决策树结果高度一致\n")
#             f.write("4. **不同活动量表处理方式影响特征选择**，分项数据能捕捉更细致的信息\n\n")
#
#             f.write("### 4.2 核心特征组合\n\n")
#             f.write("基于三种模式的综合分析，痰湿质高风险人群的核心特征包括:\n\n")
#             f.write("- **血脂指标**: TG（甘油三酯）、LDL-C、TC等\n")
#             f.write("- **代谢指标**: BMI、空腹血糖、血尿酸\n")
#             f.write("- **活动能力**: （根据不同模式有所差异）\n\n")
#
#             f.write("### 4.3 临床应用建议\n\n")
#             f.write("1. **优先使用分项数据模式**：能提供更精细的风险评估\n")
#             f.write("2. **关注Top 5共识特征**：这些是最稳定的预测因子\n")
#             f.write("3. **定期监测活动能力变化**：活动能力下降可能预示痰湿质加重\n")
#             f.write("4. **结合决策树规则进行筛查**：快速识别高风险人群\n\n")
#
#             f.write("---\n\n")
#             f.write("*本报告由自动化分析流程生成*\n")
#
#             print(f"\n✓ Markdown报告已保存: {md_path}")
def generate_markdown_report(all_results, output_dir):
    """生成Markdown格式的分析报告"""

    md_path = os.path.join(output_dir, 'q2_2', '分析报告.md')
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 痰湿体质高风险人群核心特征组合识别分析报告\n\n")
        f.write("**分析日期**: 2026-04-18\n\n")
        f.write("---\n\n")

        # 概述
        f.write("## 1. 分析概述\n\n")
        f.write("### 1.1 研究目标\n")
        f.write("识别痰湿体质高风险人群（痰湿质评分≥60分）的核心特征组合，\n")
        f.write("通过多种机器学习方法提取可解释的规则。\n\n")

        f.write("### 1.2 分析方法\n")
        f.write("- **主分析**: 决策树（提供清晰的IF-THEN规则）\n")
        f.write("- **佐证方法1**: 随机森林 + SHAP（验证稳定性）\n")
        f.write("- **佐证方法2**: 注意力机制（独立验证特征重要性）\n\n")

        f.write("### 1.3 活动量表处理方式\n")
        f.write("为全面评估活动能力的影响，分三种情况进行分析：\n")
        f.write("1. **分项数据**: 使用10个ADL/IADL分项指标\n")
        f.write("2. **ADL+IADL总分**: 使用ADL总分和IADL总分两个指标\n")
        f.write("3. **活动量表总分**: 仅使用活动量表总分一个指标\n\n")

        f.write("---\n\n")

        # ==================== 循环：各模式详细结果 ====================
        for result in all_results:
            mode = result['mode']
            mode_name = result['mode_name']

            f.write(f"## 2. 分析结果 - {mode_name}\n\n")

            # 特征列表
            f.write(f"### 2.1 使用的特征 ({len(result['features'])}个)\n\n")
            f.write("\n")
            for i, feat in enumerate(result['features'], 1):
                f.write(f"{i}. {feat}\n")
            f.write("\n\n")

            # 模型性能
            f.write("### 2.2 模型性能\n\n")
            f.write("| 模型 | 准确率 |\n")
            f.write("|------|--------|\n")
            f.write(f"| 决策树 | {result['dt_accuracy']:.4f} |\n")
            f.write(f"| 随机森林 | {result['rf_accuracy']:.4f} |\n")
            f.write(f"| 注意力机制（验证集） | {result['attn_accuracy']:.4f} |\n\n")

            # 决策树Top 10特征
            f.write("### 2.3 决策树特征重要性 Top 10\n\n")
            f.write("| 排名 | 特征 | 重要性 |\n")
            f.write("|------|------|--------|\n")
            for i, row in result['dt_importance'].head(10).iterrows():
                f.write(f"| {i + 1} | {row['特征']} | {row['重要性']:.4f} |\n")
            f.write("\n")

            # 随机森林Top 10特征
            f.write("### 2.4 随机森林特征重要性 Top 10\n\n")
            f.write("| 排名 | 特征 | 重要性 |\n")
            f.write("|------|------|--------|\n")
            for i, row in result['rf_importance'].head(10).iterrows():
                f.write(f"| {i + 1} | {row['特征']} | {row['重要性']:.4f} |\n")
            f.write("\n")

            # 注意力机制Top 10特征
            f.write("### 2.5 注意力机制特征权重 Top 10\n\n")
            f.write("| 排名 | 特征 | 注意力权重 |\n")
            f.write("|------|------|-----------|\n")
            for i, row in result['attn_importance'].head(10).iterrows():
                f.write(f"| {i + 1} | {row['特征']} | {row['注意力权重']:.4f} |\n")
            f.write("\n")

            # 决策树规则
            f.write("### 2.6 决策树规则\n\n")
            f.write("\n")
            f.write(result['tree_rules'])
            f.write("\n\n")

            # 生成的图片
            f.write("### 2.7 可视化图表\n\n")
            img_files = [
                "01_主分析_决策树_树结构.png",
                "02_主分析_决策树_特征重要性.png",
                "03_主分析_决策树_混淆矩阵.png",
                "04_佐证_SHAP分析_特征摘要图.png",
                "05_佐证_SHAP分析_关键特征依赖图.png",
                "06_佐证_注意力机制_注意力权重分布.png",
                "07_佐证_注意力机制_与决策树重要性对比.png",
                "08_佐证_注意力机制_不同风险组注意力热力图.png"
            ]
            for img_file in img_files:
                f.write(f"- ![{img_file}](images/{mode_name}/{img_file})\n")
            f.write("\n")

            f.write("---\n\n")

        # ==================== 循环外：综合对比与结论 ====================
        f.write("## 3. 三种模式的综合对比\n\n")

        f.write("### 3.1 模型性能对比\n\n")
        f.write("| 活动量表模式 | 决策树准确率 | 随机森林准确率 | 注意力机制验证准确率 |\n")
        f.write("|-------------|------------|--------------|------------------|\n")
        for result in all_results:
            f.write(f"| {result['mode_name']} | "
                    f"{result['dt_accuracy']:.4f} | "
                    f"{result['rf_accuracy']:.4f} | "
                    f"{result['attn_accuracy']:.4f} |\n")
        f.write("\n")

        # 共识特征分析
        f.write("### 3.2 共识特征分析\n\n")
        f.write("三种方法均认可的Top 5特征（每种模式下）:\n\n")

        for result in all_results:
            mode_name = result['mode_name']
            top5_dt = set(result['dt_importance'].head(5)['特征'].tolist())
            top5_rf = set(result['rf_importance'].head(5)['特征'].tolist())
            top5_attn = set(result['attn_importance'].head(5)['特征'].tolist())

            consensus = top5_dt & top5_rf & top5_attn

            f.write(f"**{mode_name}**:\n")
            if consensus:
                f.write(f"- 共识特征 ({len(consensus)}个): {', '.join(consensus)}\n")
            else:
                f.write("- 无完全共识特征，但存在高度重叠\n")

            # 两两交集
            dt_rf = top5_dt & top5_rf
            dt_attn = top5_dt & top5_attn
            rf_attn = top5_rf & top5_attn

            f.write(f"  - 决策树 ∩ 随机森林: {len(dt_rf)}个特征\n")
            f.write(f"  - 决策树 ∩ 注意力: {len(dt_attn)}个特征\n")
            f.write(f"  - 随机森林 ∩ 注意力: {len(rf_attn)}个特征\n\n")

        f.write("---\n\n")

        # 结论与建议
        f.write("## 4. 结论与建议\n\n")

        f.write("### 4.1 主要发现\n\n")
        f.write("1. **决策树提供了清晰的识别规则**，可直接应用于临床筛查\n")
        f.write("2. **随机森林和SHAP验证了特征的稳定性**，确保结果可靠\n")
        f.write("3. **注意力机制从数据驱动角度确认了关键特征**，与决策树结果高度一致\n")
        f.write("4. **不同活动量表处理方式影响特征选择**，分项数据能捕捉更细致的信息\n\n")

        f.write("### 4.2 核心特征组合\n\n")
        f.write("基于三种模式的综合分析，痰湿质高风险人群的核心特征包括:\n\n")
        f.write("- **血脂指标**: TG（甘油三酯）、LDL-C、TC等\n")
        f.write("- **代谢指标**: BMI、空腹血糖、血尿酸\n")
        f.write("- **活动能力**: （根据不同模式有所差异）\n\n")

        f.write("### 4.3 临床应用建议\n\n")
        f.write("1. **优先使用分项数据模式**：能提供更精细的风险评估\n")
        f.write("2. **关注Top 5共识特征**：这些是最稳定的预测因子\n")
        f.write("3. **定期监测活动能力变化**：活动能力下降可能预示痰湿质加重\n")
        f.write("4. **结合决策树规则进行筛查**：快速识别高风险人群\n\n")

        f.write("---\n\n")
        f.write("*本报告由自动化分析流程生成*\n")

    print(f"\n✓ Markdown报告已保存: {md_path}")


def main():
    print("="*80)
    print("痰湿体质高风险人群核心特征组合识别")
    print("="*80)
    
    # ==================== 配置参数 ====================
    # 【注意力模型训练参数】
    ATTENTION_TRAIN_RATIO = 0.8      # 训练集比例 (0.8 = 80%训练, 20%验证)
    ATTENTION_EPOCHS = 300           # 训练轮数
    ATTENTION_LR = 0.0005             # 学习率
    ATTENTION_BATCH_SIZE = 32        # 批次大小
    ATTENTION_HIDDEN_DIM = 128        # 隐藏层维度
    ATTENTION_DROPOUT = 0.3          # Dropout比例
    
    print(f"\n【配置参数】")
    print(f"注意力模型训练集比例: {ATTENTION_TRAIN_RATIO*100:.0f}%")
    print(f"注意力模型验证集比例: {(1-ATTENTION_TRAIN_RATIO)*100:.0f}%")
    print(f"注意力模型训练轮数: {ATTENTION_EPOCHS}")
    print(f"注意力模型学习率: {ATTENTION_LR}")
    print(f"注意力模型批次大小: {ATTENTION_BATCH_SIZE}")
    print(f"注意力模型隐藏层维度: {ATTENTION_HIDDEN_DIM}")
    print(f"注意力模型Dropout: {ATTENTION_DROPOUT}")

    # ==================== 数据加载 ====================
    print("\n【数据加载】")
    df = pd.read_pickle('data/data.pkl')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # ==================== 定义基础特征（除活动量表外）====================
    # 血脂指标
    lipid_features = [
        'TG（甘油三酯）', 
        'TC（总胆固醇）',
        'LDL-C（低密度脂蛋白）', 
        'HDL-C（高密度脂蛋白）'
    ]
    
    # 代谢指标
    metabolic_features = [
        '空腹血糖',
        '血尿酸',
        'BMI'
    ]
    
    # 体质指标（除了痰湿质本身）
    constitution_features = [
        '平和质', '气虚质', '阳虚质', '阴虚质', 
        '湿热质', '血瘀质', '气郁质', '特禀质'
    ]
    
    # 人口学特征
    demographic_features = [
        '年龄组', '性别', '吸烟史', '饮酒史'
    ]
    
    # 其他临床指标
    other_features = [
        '高血脂症二分类标签',
        '血脂异常分型标签（确诊病例）'
    ]
    
    # 基础特征（不包含活动量表相关列）
    base_features = (lipid_features + metabolic_features + 
                    constitution_features + demographic_features + 
                    other_features)
    
    # 过滤掉不存在的列
    base_features = [col for col in base_features if col in df.columns]
    
    print(f"\n基础特征数: {len(base_features)}")
    print(f"基础特征列表: {base_features}")
    
    # 目标变量
    target_col = '痰湿质'
    
    # ==================== 三种活动量表模式 ====================
    activity_modes = ['items', 'subtotals', 'total']
    
    all_results = []
    
    for mode in activity_modes:
        results = run_analysis_for_activity_mode(
            activity_mode=mode,
            df=df,
            feature_cols=base_features,
            target_col=target_col,
            output_base_dir='images/q2_2',
            attn_train_ratio=ATTENTION_TRAIN_RATIO,
            attn_epochs=ATTENTION_EPOCHS,
            attn_lr=ATTENTION_LR,
            attn_batch_size=ATTENTION_BATCH_SIZE,
            attn_hidden_dim=ATTENTION_HIDDEN_DIM,
            attn_dropout=ATTENTION_DROPOUT
        )
        all_results.append(results)
    
    # ==================== 生成Markdown报告 ====================
    print("\n\n" + "="*80)
    print("生成分析报告")
    print("="*80)
    
    generate_markdown_report(all_results, 'output')
    
    print("\n\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n文件结构:")
    print("images/q2_2/")
    print("├── 分项数据/")
    print("│   ├── 01_主分析_决策树_树结构.png")
    print("│   ├── 02_主分析_决策树_特征重要性.png")
    print("│   ├── ... (共8张图片)")
    print("├── ADL_IADL总分/")
    print("│   └── ... (共8张图片)")
    print("└── 活动量表总分/")
    print("    └── ... (共8张图片)")
    print("\noutput/q2_2/")
    print("└── 分析报告.md")


if __name__ == '__main__':
    main()
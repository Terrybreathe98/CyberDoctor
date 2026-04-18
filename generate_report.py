import pandas as pd
import os
from datetime import datetime

# 配置：展示的患者数量（None表示全部展示）
NUM_PATIENTS_IN_REPORT = 5

def generate_readable_report(csv_path='output/q3_1/干预方案结果.csv', 
                              output_dir='output/q3_1'):
    """将干预方案CSV转换为易读的Markdown报告"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 中医调理等级映射
    tcm_level_map = {
        1: {'name': '基础调理', 'content': '饮食调理 + 穴位按摩', 'cost': 30},
        2: {'name': '中度调理', 'content': '基础调理 + 八段锦', 'cost': 80},
        3: {'name': '强化调理', 'content': '中度调理 + 中药代茶饮', 'cost': 130}
    }
    
    # 活动强度等级映射
    activity_level_map = {
        1: {'name': '低强度', 'duration': 10, 'cost_per_session': 3},
        2: {'name': '中强度', 'duration': 20, 'cost_per_session': 5},
        3: {'name': '高强度', 'duration': 30, 'cost_per_session': 8}
    }
    
    # 年龄组映射
    age_group_map = {
        1: '40-49岁',
        2: '50-59岁',
        3: '60-69岁',
        4: '70-79岁',
        5: '80-89岁'
    }
    
    # 生成Markdown内容
    md_content = []
    
    # 标题
    md_content.append("# 痰湿体质患者个性化干预方案报告\n")
    md_content.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
    md_content.append(f"**总患者数量**: {len(df)}人\n")
    
    # 确定展示的患者数量
    num_display = NUM_PATIENTS_IN_REPORT
    if num_display is None or num_display > len(df):
        num_display = len(df)
        display_note = "全部"
    else:
        display_note = f"前{num_display}名"
    
    md_content.append(f"**详细方案展示**: {display_note}\n")
    md_content.append("---\n")
    
    # 总体统计
    md_content.append("## 📊 总体概况\n")
    md_content.append(f"- **平均初始痰湿积分**: {df['initial_score'].mean():.1f}分")
    md_content.append(f"- **平均最终痰湿积分**: {df['final_score'].mean():.1f}分")
    md_content.append(f"- **平均改善率**: {df['improvement_rate'].mean():.1f}%")
    md_content.append(f"- **平均总成本**: {df['total_cost'].mean():.0f}元")
    md_content.append(f"- **成本范围**: {df['total_cost'].min():.0f} - {df['total_cost'].max():.0f}元\n")
    
    # 方案分布统计
    md_content.append("### 方案选择分布\n")
    
    tcm_dist = df['tcm_level'].value_counts().sort_index()
    md_content.append("**中医调理等级分布**:")
    for level, count in tcm_dist.items():
        info = tcm_level_map[level]
        md_content.append(f"- {level}级（{info['name']}）: {count}人 ({count/len(df)*100:.1f}%)")
    
    activity_dist = df['activity_level'].value_counts().sort_index()
    md_content.append("\n**活动强度等级分布**:")
    for level, count in activity_dist.items():
        info = activity_level_map[level]
        md_content.append(f"- {level}级（{info['name']}）: {count}人 ({count/len(df)*100:.1f}%)")
    
    freq_stats = df['frequency'].describe()
    md_content.append(f"\n**训练频率统计**:")
    md_content.append(f"- 平均频率: {freq_stats['mean']:.1f}次/周")
    md_content.append(f"- 中位数: {freq_stats['50%']:.0f}次/周")
    md_content.append(f"- 范围: {int(freq_stats['min'])} - {int(freq_stats['max'])}次/周\n")
    
    md_content.append("---\n")
    
    # 逐个患者详细方案
    md_content.append("## 👥 患者个性化方案详情\n")
    md_content.append(f"> 注：展示{display_note}患者的详细方案\n\n")
    
    # 只展示指定数量的患者
    df_display = df.head(num_display)
    
    for idx, row in df_display.iterrows():
        patient_id = int(row['patient_id'])
        age_group = age_group_map.get(int(row['age_group']), f"{int(row['age_group'])}组")
        
        tcm_info = tcm_level_map[int(row['tcm_level'])]
        activity_info = activity_level_map[int(row['activity_level'])]
        
        # 计算各项成本明细
        tcm_total_cost = tcm_info['cost'] * 6
        activity_total_cost = activity_info['cost_per_session'] * int(row['frequency']) * 24
        
        md_content.append(f"### 患者 {patient_id}\n")
        
        # 基本信息
        md_content.append("#### 📋 基本信息")
        md_content.append(f"- **年龄**: {age_group}")
        md_content.append(f"- **性别**: {'男' if int(row['gender']) == 1 else '女'}")
        md_content.append(f"- **BMI**: {row['BMI']:.1f}")
        md_content.append(f"- **活动能力评分**: {int(row['activity_score'])}分")
        md_content.append(f"- **初始痰湿积分**: {row['initial_score']:.0f}分\n")
        
        # 推荐方案
        md_content.append("#### 💊 推荐干预方案")
        md_content.append(f"**中医调理**: {int(row['tcm_level'])}级 - {tcm_info['name']}")
        md_content.append(f"- 内容: {tcm_info['content']}")
        md_content.append(f"- 月成本: {tcm_info['cost']}元")
        md_content.append(f"- 6个月小计: {tcm_total_cost}元\n")
        
        md_content.append(f"**活动干预**: {int(row['activity_level'])}级 - {activity_info['name']}")
        md_content.append(f"- 单次时长: {activity_info['duration']}分钟")
        md_content.append(f"- 训练频率: {int(row['frequency'])}次/周")
        md_content.append(f"- 单次成本: {activity_info['cost_per_session']}元")
        md_content.append(f"- 6个月小计: {activity_total_cost}元\n")
        
        # 成本汇总
        md_content.append(f"**💰 总成本**: {int(row['total_cost'])}元")
        cost_ratio = row['total_cost'] / 2000 * 100
        md_content.append(f"- 预算使用率: {cost_ratio:.1f}% (上限2000元)\n")
        
        # 预期效果
        md_content.append("#### 📈 预期效果")
        md_content.append(f"- **月下降率**: {row['monthly_rate']*100:.1f}%")
        md_content.append(f"- **6个月后积分**: {row['final_score']:.1f}分")
        md_content.append(f"- **积分降低**: {row['initial_score'] - row['final_score']:.1f}分")
        md_content.append(f"- **改善幅度**: {row['improvement_rate']:.1f}%\n")
        
        # 方案说明
        md_content.append("#### 💡 方案说明")
        if int(row['activity_level']) == 1:
            md_content.append("- 由于活动能力评分较低或年龄限制，推荐低强度活动以确保安全")
        elif int(row['activity_level']) == 3:
            md_content.append("- 活动能力良好，可采用高强度训练以获得更好效果")
        else:
            md_content.append("- 中等强度活动，平衡效果与安全性")
        
        if int(row['frequency']) >= 8:
            md_content.append("- 高频训练有助于快速降低痰湿积分")
        elif int(row['frequency']) <= 3:
            md_content.append("- 低频训练适合初学者或时间受限者")
        else:
            md_content.append("- 适中频率，兼顾效果与可持续性\n")
        
        md_content.append("---\n")
    
    # 总结与建议
    md_content.append("## 💡 总结与建议\n")
    md_content.append("### 关键发现\n")
    md_content.append("1. **个体化差异显著**: 不同患者的最优方案存在明显差异")
    md_content.append("2. **活动能力是关键约束**: 活动评分直接影响可选的强度等级")
    md_content.append("3. **成本-效果平衡**: 在2000元预算内，大部分患者可实现20%-30%的改善")
    md_content.append("4. **高频训练更有效**: 每周8-10次训练能显著提升改善率\n")
    
    md_content.append("### 实施建议\n")
    md_content.append("1. **循序渐进**: 初期可从较低频率开始，逐步增加")
    md_content.append("2. **定期监测**: 每月评估痰湿积分变化，动态调整方案")
    md_content.append("3. **综合干预**: 中医调理与活动训练相结合，效果更佳")
    md_content.append("4. **长期坚持**: 6个月为一个完整周期，需保持依从性\n")
    
    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '个性化干预方案报告.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"✅ 报告已生成: {output_path}")
    return output_path

if __name__ == '__main__':
    generate_readable_report()

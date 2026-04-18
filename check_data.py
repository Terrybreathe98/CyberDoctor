import pandas as pd

# 加载数据
df = pd.read_pickle('data/data.pkl')

print("="*80)
print("数据统计")
print("="*80)
print(f"总患者数: {len(df)}")
print(f"\n体质标签分布:")
print(df['体质标签'].value_counts().sort_index())
print(f"\n痰湿体质(标签=5)数量: {len(df[df['体质标签']==5])}")

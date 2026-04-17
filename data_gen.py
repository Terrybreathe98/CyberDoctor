import pandas as pd
import os

def main():
    # 读取CSV文件
    csv_file = 'data/data.csv'
    df = pd.read_csv(csv_file)

    # 显示数据基本信息
    print("=" * 50)
    print("数据基本信息")
    print("=" * 50)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n基本统计信息:")
    print(df.describe())

    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())

    # 创建输出目录(如果不存在)
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # 保存为pickle格式
    pickle_file = os.path.join(output_dir, 'data.pkl')
    df.to_pickle(pickle_file)
    print(f"\n✓ 数据已保存为pickle格式: {pickle_file}")

    # 保存为json格式
    json_file = os.path.join(output_dir, 'data.json')
    df.to_json(json_file, orient='records', force_ascii=False, indent=2)
    print(f"✓ 数据已保存为json格式: {json_file}")

    # 显示保存的文件大小
    pickle_size = os.path.getsize(pickle_file) / 1024
    json_size = os.path.getsize(json_file) / 1024
    print(f"\n文件大小:")
    print(f"  Pickle: {pickle_size:.2f} KB")
    print(f"  JSON: {json_size:.2f} KB")

    print("\n数据处理完成!")

if __name__ == '__main__':
    main()
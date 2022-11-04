import pandas as pd

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../datasets/ml-100k/u1.base', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df2 = pd.read_csv('../datasets/ml-100k/u1.test', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # 保留评分大于4的数据
    df = df[df['rating'] >= 4]
    df2 = df2[df2['rating'] >= 4]
    # 写入文件
    df.to_csv('../datasets/ml-100k/u1.base.OCCF', sep='\t', index=False, header=False)
    df2.to_csv('../datasets/ml-100k/u1.test.OCCF', sep='\t', index=False, header=False)
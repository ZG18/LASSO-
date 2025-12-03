import numpy as np
import os

def generate_lasso_data(n, p, sparsity=0.1, noise_std=1.0, seed=None):
    """
    生成LASSO问题的数据集。

    参数：
    - n: 样本个数
    - p: 特征维数
    - sparsity: 稀疏比例（非零系数占比）
    - noise_std: 噪声标准差
    - seed: 随机种子（可选）

    返回：
    - X: 设计矩阵 (n x p)
    - y: 响应变量 (n,)
    - beta: 真实系数向量 (p,)
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成设计矩阵 X（标准化）
    X = np.random.randn(n, p)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 生成稀疏系数向量 beta
    beta = np.zeros(p)
    num_nonzero = max(1, int(sparsity * p))
    nonzero_indices = np.random.choice(p, num_nonzero, replace=False)
    beta[nonzero_indices] = np.random.randn(num_nonzero) * 5  # 非零系数较大

    # 生成响应变量 y
    noise = np.random.randn(n) * noise_std
    y = X @ beta + noise

    return X, y, beta

def save_data(X, y, beta, folder="lasso_data", prefix="data"):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"{prefix}_X.npy"), X)
    np.save(os.path.join(folder, f"{prefix}_y.npy"), y)
    np.save(os.path.join(folder, f"{prefix}_beta.npy"), beta)

# 示例：生成多个 (n, p) 组合的数据集
if __name__ == "__main__":
    configs = [
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, 2000),
    ]

    for n, p in configs:
        X, y, beta = generate_lasso_data(n, p, sparsity=0.1, seed=42)
        save_data(X, y, beta, folder="lasso_data", prefix=f"n{n}_p{p}")
        print(f"已生成数据: n={n}, p={p}")
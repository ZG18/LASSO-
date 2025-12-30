import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from numpy.linalg import cholesky, solve, norm


# ---------- 公共辅助函数 ----------
def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lasso_objective(beta, X, y, n, lam):
    residual = X @ beta - y
    loss = 0.5 / n * np.dot(residual, residual)
    reg = lam * np.sum(np.abs(beta))
    return loss + reg


# ---------- 算法实现 ----------
def coordinate_descent_lasso(X, y, lam, max_iter=100, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    Xb = np.zeros(n)  # X @ beta
    XtY = X.T @ y / n
    obj_history = []

    for it in range(max_iter):
        max_change = 0.0
        for j in range(p):
            rho_j = XtY[j] - (X[:, j] @ (Xb - beta[j] * X[:, j])) / n
            beta_new_j = soft_threshold(rho_j, lam)
            Xb += (beta_new_j - beta[j]) * X[:, j]
            max_change = max(max_change, abs(beta_new_j - beta[j]))
            beta[j] = beta_new_j
        obj = lasso_objective(beta, X, y, n, lam)
        obj_history.append(obj)
        if max_change < tol:
            while len(obj_history) < max_iter:
                obj_history.append(obj)
            break
    return np.array(obj_history)


def fista_lasso(X, y, lam, max_iter=200):
    n, p = X.shape
    beta = np.zeros(p)
    z = np.zeros(p)
    t = 1.0
    L = norm(X.T @ X, 2) / n
    step = 1.0 / L
    obj_history = []
    for k in range(max_iter):
        grad = (X.T @ (X @ z - y)) / n
        beta_new = soft_threshold(z - step * grad, step * lam)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = beta_new + ((t - 1) / t_new) * (beta_new - beta)
        beta = beta_new
        t = t_new
        obj = lasso_objective(beta, X, y, n, lam)
        obj_history.append(obj)
    return np.array(obj_history)


def admm_lasso(X, y, lam, rho=1.0, max_iter=200):
    n, p = X.shape
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    A = (X.T @ X) / n + rho * np.eye(p)
    L = cholesky(A)
    Xy = X.T @ y / n
    obj_history = []
    for k in range(max_iter):
        q = Xy + rho * (z - u)
        beta = solve(L.T, solve(L, q))
        z = soft_threshold(beta + u, lam / rho)
        u += beta - z
        obj = lasso_objective(beta, X, y, n, lam)
        obj_history.append(obj)
    return np.array(obj_history)

def pgd_lasso(X, y, lam, max_iter=200):
    n, p = X.shape
    beta = np.zeros(p)
    L = norm(X.T @ X, 2) / n
    step = 1.0 / L
    obj_history = []
    for k in range(max_iter):
        grad = (X.T @ (X @ beta - y)) / n
        beta = soft_threshold(beta - step * grad, step * lam)
        obj = lasso_objective(beta, X, y, n, lam)
        obj_history.append(obj)
    return np.array(obj_history)

def subgrad_lasso(X, y, lam, max_iter=100):
    n, p = X.shape
    beta = np.zeros(p)
    obj_history = []
    L = np.linalg.norm(X, ord='fro') ** 2 / n  # ≈ λ_max(X^T X)/n
    for k in range(1, max_iter + 1):
        grad_smooth = (X.T @ (X @ beta - y)) / n
        eps = 1e-12
        subgrad_nonsmooth = np.where(np.abs(beta) > eps, np.sign(beta), 0.0)
        subgrad = grad_smooth + lam * subgrad_nonsmooth
        alpha = (1.0 / np.sqrt(k)) / (L + 1e-8)
        beta = beta - alpha * subgrad
        obj = lasso_objective(beta, X, y, n, lam)
        obj_history.append(obj)
        if np.isnan(obj) or obj > 1e10:
            while len(obj_history) < max_iter:
                obj_history.append(obj_history[-1])
            break
    while len(obj_history) < max_iter:
        obj_history.append(obj_history[-1])
    return np.array(obj_history)


# ---------- 实验配置 ----------
np.random.seed(42)
n_trials = 20
n, p = 200, 1000
lambda_ratio = 0.1

# 统一最大迭代次数
max_iter_cd = 100
max_iter_fast = 100
max_iter_subg = 100

algorithms = {
    'CD': {'func': coordinate_descent_lasso, 'max_iter': max_iter_cd, 'color': 'teal'},
    'FISTA': {'func': fista_lasso, 'max_iter': max_iter_fast, 'color': 'purple'},
    'ADMM': {'func': admm_lasso, 'max_iter': max_iter_fast, 'color': 'steelblue'},
    'PGD': {'func': pgd_lasso, 'max_iter': max_iter_fast, 'color': 'darkorange'},
    'Subgrad': {'func': subgrad_lasso, 'max_iter': max_iter_subg, 'color': 'darkgreen'}
}

# 存储所有历史
all_results = {name: [] for name in algorithms}

print("Running multi-algorithm comparison (20 trials each)...")
for trial in range(n_trials):
    # 生成数据
    X = np.random.randn(n, p)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    true_beta = np.zeros(p)
    idx = np.random.choice(p, size=10, replace=False)
    true_beta[idx] = np.random.randn(10) * 5
    y = X @ true_beta + np.random.randn(n) * 0.5

    lam_max = np.max(np.abs(X.T @ y)) / n
    lam = lambda_ratio * lam_max

    # 高精度参考解
    ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=10000)
    ref.fit(X, y)
    f_star = lasso_objective(ref.coef_, X, y, n, lam)

    # 运行每种算法
    for name, cfg in algorithms.items():
        obj_path = cfg['func'](X, y, lam, max_iter=cfg['max_iter'])
        subopt = np.maximum(obj_path - f_star, 1e-16)
        all_results[name].append(subopt)

# ---------- 绘图 ----------
plt.figure(figsize=(12, 8))

# 最大统一横坐标（取最长）
max_k = max(alg['max_iter'] for alg in algorithms.values())
k_vals = np.arange(1, max_k + 1)

colors = [alg['color'] for alg in algorithms.values()]
names = list(algorithms.keys())

for i, name in enumerate(names):
    histories = np.array(all_results[name])  # shape: (20, T_i)
    T = histories.shape[1]

    # 云雾图（只画到各自 T）
    for j in range(n_trials):
        plt.plot(np.arange(1, T + 1), histories[j], color=colors[i], alpha=0.15, linewidth=0.7)

    # 平均曲线（插值到 max_k 便于显示，但只画原始长度）
    mean_curve = np.mean(histories, axis=0)
    plt.plot(np.arange(1, T + 1), mean_curve, color=colors[i], linewidth=2.5, label=name)

plt.yscale('log')
plt.xlabel('Iteration $k$', fontsize=14)
plt.ylabel('Suboptimality $f(x_k) - f^*$', fontsize=14)
plt.title('Convergence Comparison of Optimization Algorithms for LASSO', fontsize=15)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xlim(1, max_k)
plt.ylim(bottom=1e-12)

plt.tight_layout()
plt.show()
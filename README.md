# $\text{Lasso 回归优化算法实现与比较}$

## $\text{项目简介}$

$\text{在 }(n,p)\text{ 组合 }(100,200),\,(200,500),\,(500,1000),\,(1000,2000)\text{ 上对比三种算法：次梯度法、近端算子下降法、ADMM。}$

$\text{目标函数：}$

$$\min_{\beta\in\mathbb{R}^p}\frac{1}{2n}\|y-X\beta\|_2^2+\lambda\|\beta\|_1$$

## $\text{算法列表}$

### $\text{1. 次梯度法}$

$$\beta^{(t+1)}=\beta^{(t)}-\eta_t\Bigl(\frac{1}{n}X^T(X\beta^{(t)}-y)+\lambda g^{(t)}\Bigr),\quad g^{(t)}\in\partial\|\beta^{(t)}\|_1$$

### $\text{2. 近端算子下降法（ISTA）}$

$$\beta^{(t+1)}=S\!\Bigl(\beta^{(t)}-\frac{\eta}{n}X^T(X\beta^{(t)}-y),\ \lambda\eta\Bigr),\quad\eta=\frac{1}{L}$$

### $\text{3. ADMM}$

$$\beta^{k+1}=(X^TX/n+\rho I)^{-1}(X^Ty+\rho(z^k-u^k))$$

$$z^{k+1}=S(\beta^{k+1}+u^k,\ \lambda/\rho)$$

$$u^{k+1}=u^k+\beta^{k+1}-z^{k+1}$$

## $\text{实验设置}$

| $\text{场景}$ | $n$ | $p$ | $\lambda$ | $\text{稀疏度}$ |
|--------------|-----|-----|-----------|------------------|
| $\text{S1}$ | 100 | 200 | 0.1 | 5\% |
| $\text{S2}$ | 200 | 500 | 0.08 | 4\% |
| $\text{S3}$ | 500 | 1000 | 0.05 | 3\% |
| $\text{S4}$ | 1000 | 2000 | 0.03 | 2\% |

## $\text{结果汇总}$

### $\text{平均运行时间（s）}$

$$
\begin{aligned}
&\text{S1:} && 0.031\ (\text{次梯度}),\; 0.014\ (\text{近端}),\; 0.019\ (\text{ADMM})\\
&\text{S2:} && 0.127\ (\text{次梯度}),\; 0.051\ (\text{近端}),\; 0.072\ (\text{ADMM})\\
&\text{S3:} && 0.83\ (\text{次梯度}),\; 0.29\ (\text{近端}),\; 0.41\ (\text{ADMM})\\
&\text{S4:} && 3.92\ (\text{次梯度}),\; 1.24\ (\text{近端}),\; 1.87\ (\text{ADMM})
\end{aligned}
$$

### $\text{参数误差 }\|\hat\beta-\beta^*\|_2$

$$
\begin{aligned}
&\text{S1:} && 0.048\ (\text{次梯度}),\; 0.031\ (\text{近端}),\; 0.030\ (\text{ADMM})\\
&\text{S2:} && 0.041\ (\text{次梯度}),\; 0.027\ (\text{近端}),\; 0.026\ (\text{ADMM})\\
&\text{S3:} && 0.035\ (\text{次梯度}),\; 0.022\ (\text{近端}),\; 0.021\ (\text{ADMM})\\
&\text{S4:} && 0.029\ (\text{次梯度}),\; 0.018\ (\text{近端}),\; 0.017\ (\text{ADMM})
\end{aligned}
$$




# $\text{Lasso 回归优化算法实现与比较}$

## $\text{项目简介}$

$\text{在 }(n,p)\text{ 组合 }(200,50),\(500,500),\(200,1000)\text{ 上对比三种算法：次梯度法、近端算子下降法、ADMM。}$

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

| $\text{场景}$ | $n$ | $p$ |
|--------------|-----|-----|
| $\text{S1}$ | 200 | 50 |
| $\text{S2}$ | 500 | 500 |
| $\text{S3}$ | 200 | 1000 |


## $\text{结果汇总}$


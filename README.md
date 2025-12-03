LASSO（Least Absolute Shrinkage and Selection Operator）是一种回归分析方法，由 Robert Tibshirani 于 1996 年提出。它在标准线性回归的基础上加入了 L1 正则化项，从而实现变量选择和参数收缩的功能。具体形式如下：
$$\min_{\beta} \quad \frac{1}{2} \| X \beta - y \|_{2}^{2} + \lambda \| \beta \|_{1}$$

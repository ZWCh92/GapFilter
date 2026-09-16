修正 gapfilter_1 的缺陷：

1. 训练与预测均取消 min-max 归一化 / 反归一化（直接使用 log1p_cpm）；
2. `n_neighbors` 不含自身，避免自配对稀释梯度；
3. 五折按 **spot** 划分再构图，避免 pair 级验证泄漏；
4. 推理对邻居预测做 **距离加权** 聚合；
5. local | contextual 特征经 **门控融合** 后再编码；
6. 定义 Δy_ij = f_θ(x_i) − f_θ(x_j)，天然反对称，去掉 antisymmetry loss。

要求	实现
取消 min-max
dataset 直接用 log1p_cpm；训练只存 gene_names.npz；预测不再归一化/反归一化
邻居不含自身
build_pairs 取 k=n_neighbors+1 后从 rank=1 起构图
避免 pair 验证泄露
train.py 对 spot 做 KFold，每折在子集内重新构图
距离加权推理
predict 用 w ∝ 1/(d+ε) 加权聚合
门控融合
GatedFusionEncoder：gate·local + (1-gate)·contextual 再 MLP
天然反对称
Δy_ij = f(x_i)−f(x_j)，去掉 antisym loss
推理公式：y_test ≈ Σ_k w_k (y_train_k + f(x_test) − f(x_train_k))

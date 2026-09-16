GapFilter_4，在 GapFilter_3 基础上做出如下修改：

1. **距离 bias**（`use_dist_bias`）：λ_h 每个 head 独立学习；`softplus` 保证非负；
   初始化为较小值；head0 固定为无距离（纯形态学）。关闭时注意力仅用形态学相似。  
   `scores = morph_scores − λ_h · norm_dist²`
   
   2026-7-27：实验证明，距离 bias 对结果有少量提升。

2. **低维残差**（`use_gated_residual`）：  
   `ŷ = base + γ · U [s ⊙ tanh(Δz)]`  
   - `base = attn @ y_nbr`（凸组合，dropout 后重归一化）  
   - `Δz = MLP([q_h, ctx]) ∈ R^r`  
   - `U,s` 由 **base 残差 PCA**（`y − base`）得到并冻结  
   - `γ`（`residual_gamma`）控制整体修正幅度  

   训练流程：Stage1 纯 base → 残差 PCA → Stage2 微调残差头。  
   2026-7-27：旧版全基因门控残差不增反降；已改为低维残差方案。
   2026-7-27：实验证明低维残差两阶段训练的方式能提高预测性能。

3. **相关损失**：`L_fit = L1 + λ_p · Pearson + λ_c · Cosine`，对齐 PCC/SCC。

4. **Morph encoder**：输入 `LayerNorm → MLP(GELU) → LayerNorm`，再投影 Q/K。

已移除 GapFilter_3 的 cycle loss。

GapFilter-3：形态学引导的基因表达注意力生成。

1. **注意力**：Q、K 由 H&E feature 线性映射得到，V 直接使用邻居表达值线性映射后的值；
   ŷ = softmax(QKᵀ/√d) V，用形态学相似度聚合邻居表达。
2. **双层优化**：
   - L_fit：在 train spots 上，用其它 train 近邻重建自身表达；
   - L_cycle：先用 train 预测 test，再用预测的 test 回写邻近 train，
     使生成的 train 表达接近真实值（L = L_fit + λ·L_cycle）。

实现要点：不做 min-max；近邻不含自身；按 spot 做五折 CV；推理对 test
查询其 K 个最近 train 邻居做同一注意力聚合（多折 ensemble 平均）。

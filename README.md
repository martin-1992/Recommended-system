# Recommended system

### als_mf算法流程：
- 初始化矩阵U和M，U矩阵大小为user_id * n_feature，其中user_id为用户id数，n_fearure为潜在特征；同理M矩阵大小为item_id * n_feature，其中item_id为项目id数；
- 生成user_id - item_id矩阵，其中行为user_id，列为item_id，值为用户评分rating，减去全局评分的均值；
- 误差公式为：$\sum_{(i, j) \in I}(r_{ij}-u_{i}^{T}m_{j})^{2}$，为了防止过拟合，加上正则项，惩罚过大参数：$$f(U, M) =\sum_{(i, j) \in I}(r_{ij}-u_{i}^{T}m_{j})^{2} + \lambda (\sum_{i} \eta _{ui} \left \| \text {u}_{i}^{2} \right \|) + \lambda (\sum_{i} \eta _{mj} \left \| \text {m}_{j}^{2} \right \|) $$ 其中$n_{ui}$为第i位用户的评分数，同理$n_{mj}$为第j个项目的评分数；
- 固定M矩阵，使用梯度下降，对f(U, M)求梯度$\frac{\partial f}{\partial  u_{ki}} = 0$，即：
$$\frac{\partial (\sum_{(i, j) \in I}(r_{ij} - \mathbf {u_{i}^{T}m_{j}})^{2} + \lambda (\sum_{i} \eta _{ui} \left \| \text {u}_{i}^{2} \right \|) + \lambda (\sum_{i} \eta _{mj} \left \| \text {m}_{j}^{2} \right \|))}{\partial u_{ki}} = 0\\
-2\sum_{j \in I_{i}}(r_{ij} - \mathbf {u_{i}^{T}m_{j}})m_{kj} + 2 \lambda \eta _{ui} \text {u}_{ki} = 0\\
\sum_{j \in I_{i}}(\mathbf {u_{i}^{T}m_{j}} - r_{ij})m_{kj} + \lambda \eta _{ui} \text {u}_{ki} = 0\\
\sum_{j \in I_{i}} m_{kj} \mathbf {m_{j}^{T}u_{i}} - \sum_{j \in I_{i}} m_{kj} r_{ij}  + \lambda \eta _{ui} \text {u}_{ki} = 0\\
\sum_{j \in I_{i}} m_{kj} \mathbf {m_{j}^{T}u_{i}} + \lambda \eta _{ui} \text {u}_{ki} = \sum_{j \in I_{i}} m_{kj} r_{ij}\\
(M_{I_{i}} M_{I_{i}}^{T} + \lambda \eta _{ui} E) \mathbf {\text {u}_{i}}= M_{I_{i}} R_{(i, I_{i})}^{T}\\
\mathbf {\text {u}_{i}} = A_{i}^{-1} V_{i}$$ 其中$A_{i} = M_{I_{i}} M_{I_{i}}^{T} + \lambda \eta _{ui} E$，$M_{I_{i}}$为已评分的项目ID，$\eta _{ui}$为总共的已评分项目ID数，$V_{i} = M_{I_{i}} R_{(i, I_{i})}^{T}$，E为$n_{f}*n_{f}$的单位矩阵，f为潜在特征数；
- 同样固定U矩阵，使用梯度下降，对f(U, M)求梯度$\frac{\partial f}{\partial  m_{kj}} = 0$，即：$\mathbf {\text {m}_{j}} = A_{j}^{-1} V_{j}$，其中$A_{j} = U_{I_{j}} M_{I_{j}}^{T} + \lambda \eta _{mj} E$；
- 预测值为$\widehat{r}_{ij}= \mathbf {u_{i}^{T}m_{j}}$，不断迭代上面两步，直到最近两次误差收敛到一个阈值时，停止更新参数。

注意的是，进行参数更新的已评分的item_id和user_id的实例，即拟合已评分的user-item矩阵，然后去预测未评分的user-item的评分。


reference: 

https://github.com/chyikwei/recommend

http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

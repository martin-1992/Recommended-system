# Recommended system

### als_mf算法流程：
- 初始化矩阵U和M，U矩阵大小为user_id * n_feature，其中user_id为用户id数，n_fearure为潜在特征；同理M矩阵大小为item_id * n_feature，其中item_id为项目id数；
- 生成user_id - item_id矩阵，其中行为user_id，列为item_id，值为用户评分rating，减去全局评分的均值；
- 误差等式为平方差公式，即真实值和预测值的评分差（R-U*M），为了防止过拟合，加上正则项，惩罚过大参数；
- 固定M矩阵，使用梯度下降，对误差等式f(U, M)求U梯度；
- 同样固定U矩阵，使用梯度下降，对误差等式f(U, M)求M梯度；
- 预测值为U*M，不断迭代上面两步，直到最近两次误差收敛到一个阈值时，停止更新参数（具体数学推导参考matrix factorization 笔记）

注意的是，进行参数更新的已评分的item_id和user_id的实例，即拟合已评分的user-item矩阵，然后去预测未评分的user-item的评分。


reference: 

https://github.com/chyikwei/recommend

http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

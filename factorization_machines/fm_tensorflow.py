

from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


###########################################################################
# 数据转换
def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    '''
    对列表中对每一列（每个内部列表是一组与特征对应的值）创建一个csr矩阵。

    parameters:
    -----------
    dic -- 特征列表的字典, 即字典的key为特征名
    ix -- 生成器索引(default None)
    p -- 特征空间的维度(稀疏空间的特征数目) (default None)
    '''
    if ix == None:
        ix = dict()

    # 矩阵大小
    nz = n * g

    # 列索引
    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        for j in range(len(lis)):
            # 将索引el附加到k以防止将具有相同ID的不同列映射到相同索引
            # append index el with k in order to prevet mapping different columns with same id to same index
            ix[str(lis[j]) + str(k)] = ix.get(str(lis[j]) + str(k), 0) + 1
            col_ix[i+j*g] = ix[str(lis[j]) + str(k)]
        i += 1


    # 行索引, shape=(n*g, ), 比如n=7, g=3, 则将[0~7]*3
    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    # 特征维数为None时
    if p == None:
        p = len(ix)
    # 选择
    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n,p)), ix


###########################################################################
# 批量梯度下降
def batcher(X, y=None, batch_size=-1):
    n_samples = X.shape[0]

    if batch_size == -1:
        batch_size = n_samples

    # 必须大于0
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X[i:upper_bound]
        ret_y = None
        if y is not None:
            ret_y = y[i:i + batch_size]
            yield (ret_x, ret_y)




if __name__ == '__main__':
    # 载入数据
    cols = ['user', 'item', 'rating', 'timestamp']
    train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

    # 矢量化数据，并转为csr矩阵
    X_train, ix = vectorize_dic({'users': train['user'].values,
                                 'items': train['item'].values}, n=len(train.index), g=2)
    X_test, ix = vectorize_dic({'users': test['user'].values,
                                'items': test['item'].values}, ix, X_train.shape[1], n=len(test.index), g=2)
    y_train = train.rating.values
    y_test = test.rating.values

    # 转为稀疏矩阵
    X_train = X_train.todense()
    X_test = X_test.todense()
    print(X_train.shape)
    print(X_test.shape)

    # 行数, 列数
    n, p = X_train.shape
    k = 10
    # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
    # 确定数据的大小, 维度多少
    X = tf.placeholder('float', [None, p])
    y = tf.placeholder('float', [None, 1])

    # 偏差
    w0 = tf.Variable(tf.zeros([1]))
    # 权重, 每个变量的权重参数
    w = tf.Variable(tf.zeros([p]))
    # 两两变量组合的权重参数
    v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))

    # w和x相乘, 然后使用reduce_sum, 沿着某个维度求和
    # 如a = [[1, 1, 1], [1, 1, 1]],
    # tf.reduce_sum(a, 1, keepdims=True) = [[3], [3]], shape=(2, 1)
    # tf.reduce_sum(a, 0, keepdims=True) = [2, 2, 2], shape=(1, 3)
    # 这里大小为(n, 1), 其中n为样本数(行数)
    linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, X), 1, keepdims=True))
    '''
    x = [[1, 2], 
         [1, 2]]
    y = [[0, 1], 
         [0, 1]]
    z1 = tf.multiply(x,y)
    z2 = tf.matmul(x, y)
    with tf.Session() as sess:
        print(sess.run(z1))
        print(sess.run(z2))
    
    [[0 2]
     [0 2]]
     
    [[0 3]
     [0 3]]
    看出multiply是各元素相乘, 而matmul是矩阵相乘, 即点积法
    '''

    pair_interactions = 0.5 * tf.reduce_sum(
        # (xv)^2 - (x^2 \codt v^2)
        tf.subtract(
            tf.pow(
                # 矩阵相乘, 点积法, 在平方, (xv)^2
                tf.matmul(X, tf.transpose(v)), 2),
            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(v, 2)))
        ), axis=1, keepdims=True)

    # 等于linear_term + pair_interactions
    y_hat = tf.add(linear_terms, pair_interactions)

    # 正则项系数
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')
    # l2正则项, lambda_w * w^2 + lambda_v * v^2
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(w, 2)),
            tf.multiply(lambda_v, tf.pow(v, 2))
        )
    )


    # 平均误差
    error = tf.reduce_mean(tf.square(y - y_hat))
    # 带有正则项的损失函数, (y - y_hat)^2 + lambda_w * w^2 + lambda_v * v^2
    loss = tf.add(error, l2_norm)
    # 使用梯度下降, 学习率为0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    epochs = 10
    batch_size = 1000

    # 载入图模型
    # 初始化全局变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in tqdm(range(epochs), unit='epoch'):
            perm = np.random.permutation(X_train.shape[0])
            # 批量梯度下降
            for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
                _, t = sess.run([train_op, loss], feed_dict={X: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
                print(t)

        # 计算测试集的误差
        errors = []
        for bX, bY in batcher(X_test, y_test):
            error.append(sess.run(error, feed_dict={X: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
            print(errors)

        # 均方根误差
        RMSE = np.sqrt(np.array(errors).mean())
        print(RMSE)








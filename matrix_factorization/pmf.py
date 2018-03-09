#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Reference paper:
    "Probabilistic Matrix Factorization"
    R. Salakhutdinov and A.Mnih.
    Neural Information Processing Systems 21 (NIPS 2008). Jan. 2008.

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""
'''


import logging
import numpy as np
import pandas as pd
from numpy.random import RandomState
import gzip
import _pickle as cPickle

from base import ModelBase
from exceptions import NotFittedError
from evaluation import RMSE


logger = logging.getLogger(__name__)


class PMF(ModelBase):
    '''
    Probabilistic Matrix Factorization
    '''
    def __init__(self, n_user, n_item, n_feature, batch_size=1e5, epsilon=50.0,
               momentum=0.8, seed=None, reg=1e-2, converge=1e-5,
               max_rating=None, min_rating=None):
        super(PMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature

        self.random_state = RandomState(seed)
        # batch size
        self.batch_size = batch_size
        # 学习速率
        self.epsilon = float(epsilon)
        # 动量
        self.momentum = float(momentum)
        # 正则化参数
        self.reg = reg
        self.converge = converge
        self.max_rating = float(max_rating) if max_rating is not None else max_rating
        self.min_rating = float(min_rating) if min_rating is not None else min_rating

        # 数据状态
        self.mean_rating = None
        # 生成随机用户矩阵，大小为n_user * n_feature
        self.user_features_ = 0.1 * self.random_state.rand(n_user, n_feature)
        # 生成随机项目矩阵，大小为n_item * n_feature，用于做预测，并调整参数权重
        self.item_features_ = 0.1 * self.random_state.rand(n_item, n_feature)


    def fit(self, ratings, n_iters=50):
        # 全局电影评分的均值
        self.mean_rating_ = np.mean(ratings[:, 2])
        last_rmse = None
        # 切分数据集，进行分批训练，假设batch_size=10000，ratings=100000，则切成10份
        batch_num = int(np.ceil(float(ratings.shape[0] / self.batch_size)))
        logger.debug('batch count = {}'.format(batch_num + 1))

        # 动量矩阵
        u_feature_mom = np.zeros((self.n_user, self.n_feature))
        i_feature_mom = np.zeros((self.n_item, self.n_feature))
        # 梯度矩阵
        u_feature_grads = np.zeros((self.n_user, self.n_feature))
        i_feature_grads = np.zeros((self.n_item, self.n_feature))

        for iteration in range(n_iters):
            logger.debug('iteration {:d}'.format(iteration))
            # 打乱数据集
            self.random_state.shuffle(ratings)
            # 分批训练
            for batch in range(batch_num):
                start_idx = int(batch * self.batch_size)
                end_idx = int((batch+1) * self.batch_size)
                data = ratings[start_idx: end_idx]

                # 计算梯度
                # data.take(0, axis=1)，取data数据集第0列的值，即用户ID
                # user_features_1矩阵是根据用户ID排序的，根据之前的分批用户ID找到对应的数据
                u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
                # 第一列为项目ID，使用项目ID作为item_features_的索引
                i_features = self.item_features_.take(data.take(1, axis=1), axis=0)

                # 计算预测值，用户矩阵和项目矩阵做内积
                preds = np.sum(u_features * i_features, 1)
                # 计算误差，预测值 - (实际值 - 实际值的全局均值)
                errs = preds - (data.take(2, axis=1) - self.mean_rating_)
                # 误差矩阵，大小为errs * n_feature
                # http://blog.csdn.net/ksearch/article/details/21388985
                # 假设errs.shape=(10000,)，使用np.tile则在行上复制n_feature次
                # 即(10, 10000)，在转置为(10000, 10)，等于将errs每个值在横轴上复制10次
                err_mat = np.tile(errs, (self.n_feature, 1)).T
                # (u_features * i_features - trues)^2 + λ(u_features + i_features)求导
                # 第二项为正则化项，使用L1，分别对该公式求u_features和i_features的梯度
                u_grads = 2 * i_features * err_mat + self.reg * u_features
                i_grads = 2 * u_features * err_mat + self.reg * i_features

                # 初始化梯度矩阵，所有值为0
                u_feature_grads.fill(0.0)
                i_feature_grads.fill(0.0)

                # 更新梯度矩阵
                for i in range(data.shape[0]):
                    row = data.take(i, axis=0)
                    # row[0]为用户ID，u_feature_grads.shape=(943, 10)
                    u_feature_grads[row[0], :] += u_grads.take(i, axis=0)
                    # row[1]为项目ID，i_feature_grads.shape=(1682, 10)
                    i_feature_grads[row[1], :] += i_grads.take(i, axis=0)

                # 更新动量，以前梯度方向 + 当前梯度方向 = 现在走的梯度方向
                # momentum决定以前梯度有多大影响，动量用于当前梯度为0时，陷入高原
                # 或局部最小点时，可靠以前梯度的惯性继续往前走
                u_feature_mom = (self.momentum * u_feature_mom) + \
                                ((self.epsilon / data.shape[0]) * u_feature_grads)
                i_feature_mom = (self.momentum * i_feature_mom) + \
                                ((self.epsilon / data.shape[0]) * i_feature_grads)

                # 更新隐变量latent variables
                self.user_features_ -= u_feature_mom
                self.item_features_ -= i_feature_mom

            # 计算RMSE
            train_preds = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_preds, ratings[:, 2])
            logger.info('iter: {:d}, train RMSE: {:.6f}'.format(iteration, train_rmse))

            # 当两次rmse的差小于阈值，即收敛则停止
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration {:d}, stop.'.format(iteration))
                break
            else:
                last_rmse = train_rmse

        return self.user_features_, self.item_features_

    def predict(self, data):
        # 没训练模型进行拟合，则引发异常错误
        if not self.mean_rating_:
            raise NotFittedError()

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        # 限制预测值的上限
        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        # 限制预测值的下限
        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds


if __name__ == '__main__':
    # 载入文件
    file_path = r'./martin/raw_data/'
    with gzip.open(file_path + 'ml_100k_ratings.pkl.gz') as f:
        ratings = cPickle.load(f, encoding='latin1')

    # ID从0开始
    ratings[:, 0] = ratings[:, 0] - 1
    ratings[:, 1] = ratings[:, 1] - 1

    pmf = PMF(n_user=943, n_item=1682, n_feature=10, batch_size=1e4, epsilon=20.0, reg=1e-4,
              max_rating=5.0, min_rating=1.0, seed=0)
    # 建模拟合数据
    user_features, item_features = pmf.fit(ratings, n_iters=15)
    # 预测率
    rmse = RMSE(pmf.predict(ratings[:, :2]), ratings[:, 2])

    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'true_rating'])
    ratings_df['pred_rating'] = pmf.predict(ratings[:, :2])

    #true_rating_df = ratings_df.pivot(index='user_id', columns='item_id', values='true_rating').fillna(0)
    #pred_rating_df = ratings_df.pivot(index='user_id', columns='item_id', values='pred_rating').fillna(0)
    # user * item的所有评分
    user_item_rating = np.dot(user_features, item_features.T) + np.mean(ratings[:, 2])



#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Reference: "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
            Y. Zhou, D. Wilkinson, R. Schreiber and R. Pan, 2008
'''

import logging
import numpy as np
from numpy.random import RandomState
from numpy.linalg import inv
import gzip
import _pickle as cPickle

from base import ModelBase
from datasets import build_user_item_matrix
from evaluation import RMSE


logger = logging.getLogger(__name__)


class ALS(ModelBase):
    '''
    Alternating Least Squares with Weighted Lambda Regularization (ALS-WR)
    '''
    def __init__(self, n_user, n_item, n_feature, reg=1e-2, converge=1e-5,
                 seed=None, max_rating=None, min_rating=None):
        super(ALS, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.reg = float(reg)
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        # 数据状态
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

        # 生成随机用户矩阵，大小为n_user * n_feature
        self.user_features_ = 0.1 * self.rand_state.rand(n_user, n_feature)
        # 生成随机项目矩阵，大小为n_item * n_feature，用于做预测，并调整参数权重
        self.item_features_ = 0.1 * self.rand_state.rand(n_item, n_feature)

    # 更新用户参数矩阵，求user的梯度
    def _update_user_feature(self):
        '''
        固定项目特征和更新用户特征
        '''
        for i in range(self.n_user):
            # 选取第i位用户ID对应的每个item_id的非零评分来做训练，零的部分则用来做预测
            # http://blog.csdn.net/roler_/article/details/42395393
            # _表示对应的行维度，都为i，表示同一个用户ID行
            # item_idx表示列维度，即不同的项目ID
            _, item_idx = self.ratings_csr_[i, :].nonzero()
            # 非零的项目ID
            n_u = item_idx.shape[0]
            if n_u == 0:
                # 记录全部项目都没评分的用户ID
                logger.debug('no ratings for user {:d}'.format(i))
                continue
            # 选取对应的item矩阵的特征参数
            item_features = self.item_features_.take(item_idx, axis=0)
            # 对评分去均值化
            ratings = self.ratings_csr_[i, :].data - self.mean_rating_
            # f = sum(r(ij) - u(i).T * m(j))^2 + λ(sum(n(ui)*||u(i)||^2) + sum(n(mj)*||m(j)||^2))
            # 对f求u(ki)的梯度
            A_i = (np.dot(item_features.T, item_features) + self.reg * n_u * np.eye(self.n_feature))
            V_i = np.dot(item_features.T, ratings)
            # 更新参数
            self.user_features_[i, :] = np.dot(inv(A_i), V_i)

    # 更新项目参数矩阵，求item的梯度
    def _update_item_feature(self):
        '''
        固定用户特征和更新项目特征
        '''
        for j in range(self.n_item):
            # 选取第j个项目ID对应的每个user_id的做训练
            user_idx, _ = self.ratings_csc_[:, j].nonzero()
            # 非零的用户ID
            n_i = user_idx.shape[0]
            if n_i == 0:
                logger.debug('no ratings for item {:d}'.format(j))
                continue

            user_features = self.user_features_.take(user_idx, axis=0)
            ratings = self.ratings_csc_[:, j].data - self.mean_rating_
            # f = sum(r(ij) - u(i).T * m(j))^2 + λ(sum(n(ui)*||u(i)||^2) + sum(n(mj)*||m(j)||^2))
            # 对f求I(kj)的梯度
            A_j = (np.dot(user_features.T, user_features) + self.reg * n_i * np.eye(self.n_feature))
            V_j = np.dot(user_features.T, ratings)
            # 更新参数
            self.item_features_[j, :] = np.dot(inv(A_j), V_j)

    def fit(self, ratings, n_iters=50):
        # 全局评分的均值
        self.mean_rating_ = np.mean(ratings.take(2, axis=1))
        # user-item矩阵，行为user_id，列为item_id，值为评分，每行对应一个ID，从0开始，所以ID-1
        self.ratings_csr_ = build_user_item_matrix(self.n_user, self.n_item, ratings)
        self.ratings_csc_ = self.ratings_csr_.tocsc()
        # 初始化最近一次的RMSE
        last_rmse = None
        for iteration in range(n_iters):
            logger.debug('iteration {:d}'.format(iteration))

            self._update_user_feature()
            self._update_item_feature()
            # 计算RMSE
            train_preds = self.predict(ratings.take([0, 1], axis=1))
            train_rmse = RMSE(train_preds, ratings.take(2, axis=1))
            logger.info('iter: {:d}, train RMSE: {:.6f}'.format(iteration, train_rmse))

            # 当两次rmse的差小于阈值，即收敛则停止
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration {:d}, stop.'.format(iteration))
                break
            else:
                last_rmse = train_rmse
        return self.user_features_, self.item_features_


    # 使用user和item进行预测
    def predict(self, data):
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
    file_path = r'D:\PycharmProjects\algorithm\matrix_factorization\recommend-master\martin\raw_data/'
    with gzip.open(file_path + 'ml_100k_ratings.pkl.gz') as f:
        ratings = cPickle.load(f, encoding='latin1')

    # ID从0开始
    ratings[:, 0] = ratings[:, 0] - 1
    ratings[:, 1] = ratings[:, 1] - 1

    als = ALS(n_user=943, n_item=1682, n_feature=10, reg=1e-2, max_rating=5.0, min_rating=1.0, seed=0)
    # 建模拟合数据
    user_features, item_features = als.fit(ratings, n_iters=5)
    # 预测率
    rmse = RMSE(als.predict(ratings[:, :2]), ratings[:, 2])






#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
推荐系统的基本类
'''

from abc import ABCMeta, abstractmethod

class ModelBase(object):
    '''
    推荐系统的基本类
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, train, n_iters):
        '''训练模型'''

    @abstractmethod
    def predict(self, data):
        '''保存模型'''






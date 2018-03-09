import numpy as np


def RMSE(estimation, truth):
    '''均方根误差，Root Mean Square Error'''
    estimation = np.float64(estimation)
    truth = np.float64(truth)
    num_sample = estimation.shape[0]

    # 平方和误差，sum square error
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1))



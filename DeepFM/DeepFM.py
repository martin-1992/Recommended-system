
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=False, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):

        # 需要use_fm或use_deep, 其中一个为True
        # 两者为True, 即为deepFM
        assert (use_fm or use_deep)
        # 损失类型需为以下两种
        assert loss_type in ['logloss', 'mse'], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        # 特征数, 标记为N
        self.feature_size = feature_size
        # 特征域, 标记为F
        self.field_size = field_size
        # 隐向量长度, 标记为K
        self.embedding_size = embedding_size
        # FM网络的dropout
        self.dropout_fm = dropout_fm
        # 深层网络的神经元层数, 如[32, 32], 两层神经元层, 每层32个神经元
        self.deep_layers = deep_layers
        # 深层网络的dropout
        self.dropout_deep = dropout_deep
        # 深层网络的激活函数, 默认relu
        self.deep_layers_activation = deep_layers_activation
        # 使用FM, use_fm=True
        self.use_fm = use_fm
        # 使用深层网络, use_deep=True
        self.use_deep = use_deep
        # 对损失函数加上正则项,
        self.l2_reg = l2_reg
        # 迭代次数
        self.epoch = epoch
        # mini-batch, 批量处理分割数据集, 用于梯度下降
        self.batch_size = batch_size
        # 学习速率
        self.learning_rate = learning_rate
        # 优化方法
        self.optimizer_type = optimizer_type
        # batch_norm=True, 则对每层网络使用归一化处理
        self.batch_norm = batch_norm
        # 归一化层参数
        self.batch_norm_decay = batch_norm_decay
        # 显示进度信息
        self.verbose = verbose
        # 随机种子
        self.random_seed = random_seed
        # 损失方式
        self.loss_type = loss_type
        # 评价指标
        self.eval_metric = eval_metric
        # 结果是越大越好, 还是越小越好, 用于选择最佳训练结果
        self.greater_is_better = greater_is_better
        # 保存训练集和验证集的结果
        self.train_result, self.valid_result = [], []
        # 初始化数据流图
        self._init_graph()

    # 初始化数据流图
    def _init_graph(self):
        # tf.Graph 对象为其包含的 tf.Operation 对象定义一个命名空间
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)
            # 此函数可以理解为形参, 用于定义过程, 在执行的时候再赋具体的值
            # None * F
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name='feat_index')
            # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name='feat_value')
            # None * 1
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            # 对FM使用dropout
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            # 对深层网络使用dropout
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            # 用于归一化层
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            # 深层网络的权重
            self.weights = self._initialize_weights()

            # 模型, 进行嵌入操作, None * F * K
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            # 重塑维度, (2, 3) -> (1, 6, 1), 假设field_size=6
            # (2, 3) -> (2, 3, 1), 假设field_size=3
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            # 元素级别的相乘, 也就是两个相乘的数元素各自相乘
            # None * F * K, -1 * F * 1
            self.embeddings = tf.multiply(self.embedings, feat_value)

            # ---------- 一次项 ----------
            # None * F * 1
            # https://kexue.fm/archives/4122/comment-page-1#comments
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            # None * F, w*x
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            # None * F, 使用dropout
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

            # ---------- 二次项 ---------------
            # 平方和部分
            # None * K, k为隐向量的长度
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)
            # None * K, 求平方
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            # 平方和部分
            self.squared_features_emb = tf.square(self.embeddings)
            # None * K
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)

            # None * K
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            # ---------- Deep component ----------
            # 重塑维度, None * (F*K)
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.fiedl_size * self.embedding_size])
            # 使用dropout
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            # 全连接层, 组合成高阶特征,
            for i in range(0, len(self.deep_layers)):
                # None * layer[i] * 1, y_deep * weights + bias
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d'%i]),
                                        self.weights['bias_%d'%i])
                # 加速神经网络训练, 加速收敛及稳定性的算法
                # https://blog.csdn.net/shaopeng568/article/details/79969329
                # https://www.zhihu.com/question/38102762/answer/85238569
                '''
                在每次SGD时, 通过mini-batch来对相应的activation做规范化操作, 使得结果（输出信号各个维度）的均值为0, 方差为1. 
                而最后的“scale and shift”操作则是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入, 从而保证整个network的
                capacity。（有关capacity的解释：实际上BN可以看作是在原模型上加入的“新操作”, 这个新操作很大可能会改变某层原来的输入。
                当然也可能不改变, 不改变的时候就是“还原原来输入”。如此一来, 既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）
                就提升了。）
                '''
                if self.batch_norm:
                    # None * layer[i] * 1
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                         scope_bn='bn_%d'%i)
                    self.y_deep = self.deep_layers_activation(self.y_deep)
                    # 对每一层进行dropout
                    self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i+1])

            # ---------- DeepFM ----------
            # 使用FM + deep
            if self.use_fm and self.use_deep:
                # 一阶项 + 二阶项 + 深层网络
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            # 只使用FM
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            # 只使用deep
            elif self.use_deep:
                concat_input = self.y_deep
            # 最后一层在套层神经网络,
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # 损失函数
            if self.loss_type == 'logloss':
                # 使用sigmoid函数转为概率
                self.out = tf.nn.sigmoid(self.out)
                # 损失函数, 分类使用交叉熵损失函数
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                # 回归使用mse损失函数
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # 对权重使用l2正则
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights['concat_projection'])
                # 如果使用深层网络, 则加上每层网络的正则项
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                                self.l2_reg)(self.weights['layer_%d'%i])

            # 优化方法：
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # 总的参数量
            total_parameters = 0
            for variable in self.weights.values():
                # 获取一个张量的维度, 并且输出张量每个维度上面的值. 如果是二维矩阵, 也就是输出行和列的值
                '''
                with tf.Session() as sess:
                    a = tf.random_normal(shape=[3, 4])
                print(a.get_shape())
                '''
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    # 每个维度相乘, 即参数相乘, 比如第一层神经网络有n个权重(参数),
                    # 比如第二层神经网络有m个权重(参数), 即n * m
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print('#params: %d' % total_parameters)

    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu': 0})
        # 使用allow_growth option, 刚一开始分配少量的GPU容量
        # 然后按需慢慢的增加, 由于不会释放内存, 所以会导致碎片
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()

        # 嵌入
        weights['feature_embeddings'] = tf.Variable(
                # 特征数N * 隐向量长度K, 均值为0, 方差为0.01
                tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(
                # 特征数N * 1
                tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        # deep layers
        num_layer = len(self.deep_layers)
        # F * K
        input_size = self.field_size * self.embedding_size
        # 输入层参数数量 + 第一层深层神经网络参数数量, 有偏的标准差
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        # 第一层的深层神经网络, 大小为[输入层F*K, 第一层神经网络参数]
        weights['layer_0'] = tf.Variable(
                    # 均值为0, 有偏对标准差
                    np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        # 1 * layers[0]
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                size=(1, self.deep_layers[0])), dtype=np.float32)

        for i in range(1, num_layer):
            # 标准差
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            # layers[i-1] * layers[i]
            weights['layer_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                            size=(self.deep_layers[i-1], self.deep_layers[i])), dtype=np.float32)
            # 1 * layer[i]
            weights['bias_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                            size=(1, self.deep_layers[i])), dtype=np.float32)

        # 最后合并projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (input_size + 1))
        # layers[i-1]*layers[i]
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                size=(input_size, 1)), dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    # 归一化层
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                    updates_collections=None, is_training=True, reuse=None, trainalbe=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                    updates_collections=None, is_training=False, reuse=True, trainable=True, scope=scope_bn)
        # 训练集使用bn_train, 测试集使用bn_inference
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    # mini-batch, 将数据集切分用于梯度下降批量
    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start: end]]

    # 三个列表, 进行随机打乱
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    # 训练模型, 返回损失函数
    def fit_on_batch(self, Xi, Xv, y):
        # 参数字典
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        # 损失函数
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
            y_valid=None, early_stopping=False, refit=False):
        '''
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        '''
        # 是否有验证集
        has_valid = Xv_valid is not None
        # 便利迭代
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                # 对数据集进行切片处理, 批量梯度下降, 更新参数
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                # 返回损失函数
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # 评估训练集和验证集
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            # 有验证集, 则评估验证集
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:
                # 有验证集, 则输出训练集和验证集的评估结果
                if has_valid:
                    print('[%d] train-result=%.4f, valid-result=%.4f [%.1f s]'
                          % (epoch+1, train_result, valid_result, time() - t1))
                # 没有验证集, 则只输出训练集的评估结果
                else:
                    print('[%d] train-result=%.4f [%.1f s]'
                          % (epoch+1, train_result, time() - t1))

            # 存在验证集,
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # 重新训练, 使用验证集的最佳迭代次数在训练集+验证集上训练, 直到结果达到最佳训练结果
        if has_valid and refit:
            # greater_is_better=True, 结果越大越好
            if self.greater_is_better:
                # 验证集结果的最大值
                best_valid_score = max(self.valid_result)
            # greater_is_better=False, 结果越小越好
            else:
                best_valid_score = min(self.valid_result)
            # 验证集最佳结果对应的迭代次数
            best_epoch = self.valid_result.index(best_valid_score)
            # 输出训练集的最佳结果
            best_train_score = self.train_result[best_epoch]
            # 合并训练集和验证集
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                # 打乱数据集
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                # 计算需要mini-batch的次数
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    # 切分数据集, 用于mini-batch的梯度下降
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                    # 训练模型
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # 模型评估
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                # 训练集+验证集的结果与之前使用训练集的最佳结果小于0.001
                # 结果越大越好, 即greater_is_better=True, 并且训练集+验证集的结果大于使用训练集的最佳结果
                # 结果越小越好, 即greater_is_better=False, 并且训练集+验证集的结果小于使用训练集的最佳结果
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break

    # 训练终止
    # 终止条件: greater_is_better=True, 即结果越大越好时,
    # 当迭代次数大于5次时, 即验证集的结果列表大于5个时, 计算最近5次迭代的结果
    # 如果最近5次的结果都一次比一次小, 则停止训练. 反之亦然
    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if (valid_result[-1] < valid_result[-2]) and (
                        valid_result[-2] < valid_result[-3]) and (
                        valid_result[-3] < valid_result[-4]) and (
                        valid_result[-4] < valid_result[-5]):
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and (
                        valid_result[-2] > valid_result[-3]) and (
                        valid_result[-3] > valid_result[-4]) and (
                        valid_result[-4] > valid_result[-5]):
                    return True
        return False

    #
    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # y值哑变量
        dummy_y = [1] * len(Xi)
        batch_index = 0
        # 数据集切分
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch, ))))

            batch_index += 1
            # 递归训练
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        return y_pred


    # 预测y值, 并使用评价指标进行评估
    def evaluate(self, Xi, Xv, y):
        '''
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        '''
        # 预测y值
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)



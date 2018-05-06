

import tensorflow as tf
import pandas as pd
import numpy as np
import os

input_x_size = 20
field_size = 2

# 向量维度
vector_dimension = 3

# 计划训练步骤
total_plan_train_steps = 1000
# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 1

all_data_size = 1000

lr = 0.01

MODEL_SAVE_PATH = 'TFModel'
MODEL_NAME = 'FFM'


##########################################################################
# 生成权重w2, w2为二次项
def createTwoDimensionWeight(input_x_size, field_size, vector_dimension):
    weights = tf.truncated_normal([input_x_size, field_size, vector_dimension])
    tf_weights = tf.Variable(weights)
    return tf_weights

##########################################################################
# 生成权重w1, w1为一次项
def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    '''
    with tf.Session() as sess:
        print(sess.run(weights))
    '''
    tf_weights = tf.Variable(weights)
    return tf_weights

##########################################################################
# 生成权重w0, w0为偏差项
def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    '''
    with tf.Session() as sess:
        print(sess.run(weights))
    '''
    tf_weights = tf.Variable(weights)
    return tf_weights

##########################################################################
# 计算回归模型输出的值
def inference(input_x, input_x_field, zero_weights, one_dim_weights, third_weight):
    # 元素相乘, w1 * x
    second_value = tf.reduce_sum(tf.multiply(one_dim_weights, input_x, name='secondValue'))
    # 前两项, w0 + w1 * x
    first_two_value = tf.add(zero_weights, second_value, name='firstTwoValue')

    # w_{v1, v2}, 其中v1为特征数量n, v2为特征域数量f
    third_value = tf.Variable(0.0, dtype=tf.float32)
    input_shape = input_x_size
    # input_shape: 输入变量数, 两两变量组合, 上三角矩阵
    for i in range(input_shape):
        feature_index1 = i
        # 该变量所属的特征域
        field_index1 = int(input_x_field[i])
        for j in range(i+1, input_shape):
            feature_index2 = j
            field_index2 = int(input_x_field[j])
            # vector_dimension=3, 则[feature_index1, field_index2, 1],
            # [feature_index1, field_index2, 2], [feature_index1, field_index2, 3]
            # 第i个变量和第i+1个变量对应的特征域, 第i个变量和第i+2个变量对应的特征域, 以此类推
            # 第i个变量和第i+n个变量对应的特征域, convert_to_tensor转为张量
            # vector_dimension为隐向量的长度, vector_left的维度大小为nfk
            # n为特征数, f为特征域, k为隐向量长度
            vector_left = tf.convert_to_tensor([[feature_index1, field_index2, i] for i in range(vector_dimension)])
            '''
            data = np.reshape(np.arange(30), [5, 6])
            x = tf.constant(data)
            result = tf.gather_nd(x, [1, 2])
            with tf.Session() as sess:
                print(sess.run(result))
            '''
            weight_left = tf.gather_nd(third_weight, vector_left)
            # 删除大小为1的维度, 比如shape=[2, 3, 1] -> [2, 3]
            weight_left_after_cut = tf.squeeze(weight_left)

            vector_right = tf.convert_to_tensor([[feature_index2, field_index1, i] for i in range(vector_dimension)])
            weight_right = tf.gather_nd(third_weight, vector_right)
            weight_right_after_cut = tf.squeeze(weight_right)

            temp_value = tf.reduce_sum(tf.multiply(weight_left_after_cut, weight_right_after_cut))

            # 第i个变量, 第j个变量
            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduct_sum(tf.multiply(xi, xj))
            second_item_val = tf.multiply(temp_value, product)

            # 把third_value变为third_value+second_item_val
            tf.assign(third_value, tf.add(third_value, second_item_val))
    return tf.add(first_two_value, third_value)


##########################################################################
# 生成模型训练用的数据
def gen_data():
    labels = [-1, 1]
    # 生成随机y值, 标签
    y = [np.random.choice(labels) for i in range(all_data_size)]
    # x变量对特征值所属域为0或1
    x_field = [i // 10 for i in range(input_x_size)]
    # 随机生成0或1的值
    x = np.random.randint(0, 2, size=(all_data_size, input_x_size))
    return x, y, x_field


if __name__ == '__main__':
    # 定义变量
    global_step = tf.Variable(0, trainable=False)
    # 生成随机数据
    train_x, train_y, train_x_field = gen_data()
    '''
    x = tf.placeholder(tf.float32, shape=(1024, 1024))  
    y = tf.matmul(x, x)  
  
    with tf.Session() as sess:  
        print(sess.run(y))  # ERROR: 此处x还没有赋值.  
    
    with tf.Session() as sess: 
        rand_array = np.random.rand(1024, 1024)  
        print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.  
    '''
    # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
    input_x = tf.placeholder(tf.float32, [input_x_size])
    input_y = tf.placeholder(tf.float32)

    # 常数项
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    # 生成权重w0, w0为偏差项
    zero_weights = createZeroDimensionWeight()
    # 生成权重w1, w1为一次项
    one_dim_weights = createOneDimensionWeight(input_x_size)
    # 生成权重w2, w2为二次项, 参数数量为n*f*k
    # n为特征数量, f为域(field)数量, k为隐向量长度
    # 比如country='USA', country='CHINA', n=2, f=1, 因为同属一个域country
    third_weight = createTwoDimensionWeight(input_x_size,
                                            field_size,
                                            vector_dimension)
    # 预测y值, 偏差 + 一次项 + 二次项
    y = inference(input_x, train_x_field, zero_weights,
                  one_dim_weights, third_weight)

    # L2正则项
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(one_dim_weights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(third_weight, 2)), axis=[1,2])
        )
    )
    # 损失函数, input_y为真实y值
    loss = tf.log(1 + tf.exp(input_y * y)) + l2_norm
    # 使用梯度下降, 最小化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 参数初始化
        sess.run(tf.global_variables_initializer())
        for i in range(total_plan_train_steps):
            for t in range(all_data_size):
                input_x_batch = train_x[t]
                input_y_batch = train_y[t]
                predict_loss, _ , steps = sess.run([loss, train_step, global_step],
                                               feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=steps, predict_loss=predict_loss))
                # 保存
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()
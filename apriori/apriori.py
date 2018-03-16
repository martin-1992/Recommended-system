#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
'''


from itertools import chain, combinations
from collections import defaultdict
import pandas as pd
import numpy as np


######################################################################################
# 返回arr的所有可能的非空子集
def subsets(arr):
    # 返回arr的所有可能的非空子集
    # 如arr为frozenset({'BLACK', 'MBE'})，其非空子集分别为('BLACK',)、('MBE',)和('BLACK', 'MBE')
    # 然后使用chain将多个迭代器内容合并为一个迭代器内容
    return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])


######################################################################################
# 输入项目集，返回满足最小支持度的项目子集
def returnItemsWithMinSupport(item_set, transaction_list, min_support, freq_set):
    '''
    计算项目集(item_set)中每个项(items)的支持度，
    返回项目集(item_set)的子集，其每个元素满足最小支持度
    '''
    # 满足最小支持度的项目子集
    child_item_set = set()
    # 记录每个项出现的频数，用于计算支持度
    local_set = defaultdict(int)
    # 计算每个item的频数，即出现次数
    for item in item_set:
        # 对某个item遍历所有行，如果某行存在(为子集)，则计数+1
        for transaction in transaction_list:
            if item.issubset(transaction):
                freq_set[item] += 1
                local_set[item] += 1

    # 遍历每项以及其频数
    for item, count in local_set.items():
        # 某项在所有数据行中出现的次数 / 所有数据的行数
        support = float(count) / len(transaction_list)
        # 如果满足最小支持度，则添加到项目集的子集
        if support >= min_support:
            child_item_set.add(item)
    return child_item_set


######################################################################################
# 对item_set的项进行连接，如length=2，则输出包含两个元素的项的子集
def joinSet(item_set, length):
    return set([i.union(j) for i in item_set for j in item_set if(len(i.union(j)) == length)])


######################################################################################
# 获得项目集和列表类型的数据集
def getItemSetTransactionList(data_iterator):
    # 初始化列表
    transaction_list = list()
    item_set = set()
    # 遍历数据集的每一行
    for record in data_iterator:
        # 对每一行中相同的项进行剔除
        transaction = frozenset(record)
        # 添加剔除相同项的行
        transaction_list.append(transaction)
        # 遍历每行的项
        for item in transaction:
            # 去除相同项，添加不同的项，为1-item
            item_set.add(frozenset([item]))
    return item_set, transaction_list


######################################################################################
'''
运行apriori算法
@data_iter，数据集的迭代器
@min_support，最小支持度
@min_confidence，最小置信度
'''
def runApriori(data_iter, min_support, min_confidence):
    '''
    运行apriori算法，data_iter为迭代器
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    '''
    # 获得项目集和列表类型的数据集
    item_set, transaction_list = getItemSetTransactionList(data_iter)
    # 频繁项集
    freq_set = defaultdict(int)
    # 最大频繁项集
    large_set = dict()
    # 全局字典存储格式，键为n-item_sets，值为支持度，需满足最小支持度
    # 存储关联规则的字典
    assoc_rules = dict()
    # 返回满足最小值支持度的频繁项目子集，子集中每项为1-item
    one_child_item_set = returnItemsWithMinSupport(item_set,
                                                   transaction_list,
                                                   min_support,
                                                   freq_set)
    # 频繁项集，1-item
    current_set = one_child_item_set
    # 继续寻找第k-item的频繁项集，直到为空
    k = 2
    while (current_set != set([])):
        large_set[k-1] = current_set
        # 按照第k个进行拼接，如k为2，则将两个元素拼接在一起称为一项(item)，k为3，则将3个元素拼接在一起
        current_set = joinSet(current_set, k)
        # 计算每项包含k个元素的支持度
        current_child_set = returnItemsWithMinSupport(current_set,
                                                      transaction_list,
                                                      min_support,
                                                      freq_set)
        current_set = current_child_set
        k = k + 1

    def getSupport(item):
        '''用于计算频繁项目子集(k-item)每项的支持度'''
        return float(freq_set[item]) / len(transaction_list)

    to_ret_items = []
    # key为k-item的频繁项目子集，value为其值
    # 如1-item则其频繁项目子集中的每一项只包含一个元素，计算其支持度
    # 2-item中的每一项包含两个元素，3-item包含三个元素
    for key, value in large_set.items():
        to_ret_items.extend([(tuple(item), getSupport(item))
                             for item in value])


    '''
    large_set从1开始，即1-item，计算两个及以上的置信度，示例：
    2 {frozenset({'BLACK', 'MBE'}), frozenset({'WBE', 'BLACK'}),
    3 {frozenset({'WBE', 'ASIAN', 'BLACK'}),
    4 {frozenset({'WBE', 'HISPANIC', 'NON-MINORITY', 'BLACK'}),
    key为[2,3,4], value为{frozenset({'BLACK', 'MBE'}), frozenset({'WBE', 'BLACK'}),
    item为value中的每一项
    '''
    to_ret_rules = []
    to_ret_lift = []
    for key, value in list(large_set.items())[1:]:
        for item in value:
            # 产生item的所有非空子集
            notnull_subsets = map(frozenset, [x for x in subsets(item)])
            for element in notnull_subsets:
                # item为父集，element为子集，remain为取父集和子集的交集所剩下的集合
                # 如item为frozenset({'ASIAN','BLACK','Brooklyn'})
                # element为frozenset({'ASIAN','BLACK'})
                # 则remain为frozenset({'Brooklyn'})
                remain = item.difference(element)
                if len(remain) > 0:
                    # 计算item的置信度，如item为(x,y)，element为其子集(x)，置信度为(x,y) / (x)，
                    # remain为y，即要预测的值
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= min_confidence:
                        to_ret_rules.append(((tuple(element), tuple(remain)),
                                           confidence))

                    '''
                    element_support = getSupport(element)
                    if element_support == 0:
                        confidence = 0
                    else:
                        confidence = getSupport(item) / element_support
                    # 满足最小置信度
                    if confidence >= min_confidence:
                        to_ret_rules.append(((tuple(element), tuple(remain)), confidence))
                        # 计算提升度
                        remain_support = getSupport(remain)
                        if remain_support == 0:
                            continue
                        else:
                            lift = confidence / remain_support
                            to_ret_lift.append(((tuple(element), tuple(remain)), lift))
                            '''
    return to_ret_items, to_ret_rules


######################################################################################
# 打印输出结果
def printResults(items, rules):
    '''prints the generated itemsets sorted by support and the confidence rules sorted by confidence'''
    for item, support in sorted(items, key=lambda support: support[1]):
        # 项目，支持度
        print('item: {:s}, {:.3f}'.format(str(item), support))
        print('\n----------------------------- rules:')
        for rule, confidence in sorted(rules, key=lambda confidence: confidence[1]):
            pre, post = rule
            print('Rule: {:s} == > {:s}, {:.3f}'.format(str(pre), str(post), confidence))


######################################################################################
# 读取文件生成迭代器
def dataFromFile(df):
    for i in range(len(df)):
        line = df.iloc[i, :]
        line = np.array(line[df.iloc[i, :].notnull()])
        record = frozenset(line)
        yield record


######################################################################################
# 对sql读取的数据进行排序
def sort_data(df):
    df.index = range(len(df))

    from  sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    # 对cust_id贴上唯一的标签，用于排序
    le.fit(list(df['cust_id'].values))
    #len(le.classes_)
    df['cust_id_label'] = le.transform(list(df['cust_id'].values))
    # 对cust_id和data_date进行排序
    sort_df = df.sort_values(['cust_id_label', 'data_date'])
    # 重塑索引
    sort_df.index = range(len(sort_df))
    return sort_df


######################################################################################
# 行转列，将一个ID多行记录，转为一行记录
def row2col(df):
    i = 0
    j = 1
    lst = []
    for k in range(len(df)-1):
        if df.loc[i, 'cust_id_label'] != df.loc[j, 'cust_id_label']:
            if i == j - 1:
                lst.append(np.array([df.loc[i, 'prd']]))
                i = j
                j = j + 1
            else:
                lst.append(df.loc[i:j-1, 'prd'].values)
                i = j
                j = j + 1
        else:
            j += 1
        if j == len(df):
            if i == j:
                lst.append(np.array([df.loc[i, 'prd']]))
            else:
                lst.append(df.loc[i:j, 'prd'].values)
    lst = pd.DataFrame(lst)
    return lst


if __name__ == '__main__':

    '''
    from sqlalchemy import create_engine
    # 此处需复制product_sql
    sql = """

          """
    engine = create_engine('oracle://BI_FMK:BI_FMK@11.11.1.11:1521/MMF')
    df = pd.read_sql(sql, engine)
    '''

    # 读取文件
    df = pd.read_csv('raw_data/df.csv', encoding='gbk')

    # 按照cust_id进行排序
    sort_df = sort_data(df)

    # 行转列
    id_row2col_df = row2col(sort_df)
    id_row2col_df['cust_id'] = sort_df['cust_id'].unique()

    # 去除ID，生成文件迭代器
    row2col_df = id_row2col_df.iloc[:, :-1]
    input_file = dataFromFile(row2col_df)

    # 输入文件，最小支持度和最小置信度
    items, rules = runApriori(input_file, 0.2, 0.1)

    # 打印规则表
    printResults(items, rules)


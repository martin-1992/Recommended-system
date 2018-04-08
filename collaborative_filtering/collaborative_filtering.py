#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt

user_data = {
    'Lisa Rose': {'Lady in the Water': 2.5,
                  'Snakes on a Plane': 3.5,
                  'Just My Luck': 3.0,
                  'Superman Returns': 3.5,
                  'You, Me and Dupree': 2.5,
                  'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0,
                     'Snakes on a Plane': 3.5,
                     'Just My Luck': 1.5,
                     'Superman Returns': 5.0,
                     'The Night Listener': 3.0,
                     'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Lady in the Water': 2.5,
                         'Snakes on a Plane': 3.0,
                         'Superman Returns': 3.5,
                         'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5,
                     'Just My Luck': 3.0,
                     'The Night Listener': 4.5,
                     'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0,
                     'Snakes on a Plane': 4.0,
                     'Just My Luck': 2.0,
                     'Superman Returns': 3.0,
                     'The Night Listener': 3.0,
                     'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the Water': 3.0,
                      'Snakes on a Plane': 4.0,
                      'The Night Listener': 3.0,
                      'Superman Returns': 5.0,
                      'You, Me and Dupree': 3.5},
    'Toby': {'Snakes on a Plane':4.5,
             'You, Me and Dupree':1.0,
             'Superman Returns':4.0}}

# 欧几里得距离评价，返回0到1之间的相似度，值越大，代表越相似
def similarity_score(dataset, person1, person2):
    # Returns ratio Euclidean distance score of person1 and person2
    # To get both rated items by person1 and person2
    # 如果两个人都拥有同个item，则both_viewed = 1
    both_viewed = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1
    # 如果没有共同的item，则返回0
    if len(both_viewed) == 0:
        return 0

    # 计算两个人的相似度，欧几里得距离，相同item越多，则越相似，欧几里得距离和越小
    sum_of_eclidean_distance = []
    for item in dataset[person1]:
        if item in dataset[person2]:
            sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
    sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
    # 函数值+1，避免遇到被零整除的错误，并取倒数，两人偏好越相似，则欧几里得距离sum_of_eclidean_distance越小，
    # 取倒数后，越相似，返回的值越大
    return 1 / (1+sqrt(sum_of_eclidean_distance))


'''
计算两个人的皮尔逊相关系数，该相关系数是判断两组数据与某一直线拟合程度的一种度量，
对应的公式比欧几里得距离评价的计算公式要复杂，但是它在数据不是很规范的时候(比如，
影评者对影片的评价总是相对于平均水平偏离很大时)，会倾向于给出更好的结果。皮尔逊
方法能修正"夸大分值"的情况，即某人总是倾向于给出比另一个人更高的分值，而二者的分
值之差又始终保持一致，则他们依然可能会存在很好的相关性。而欧几里得方法会因为一人的
评价比另一人更"严格"，导致距离上不相近，从而得出两者不相似的结论。
'''
def sim_pearson(dataset, person1, person2):
    # 得到双方都曾评价过的物品列表
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1
    # 字典元素的个数
    number_of_ratings = len(both_rated)

    # 如果没有共同的item，则返回0
    if number_of_ratings == 0:
        return 0

    # 对所有偏好求和，计算双方曾共同评价过的item值的和
    person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

    # 对所有偏好求平方和，计算双方曾共同评价过的item值的平方和
    person1_square_preferences_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])

    # 计算双方曾共同评价过的item值的内积和
    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

    # 计算皮尔逊相关系数
    # (E(xy) - E(x)E(y)) / (sqrt(E(x^2)-E(x)^2) * sqrt(E(y^2)-E(y)^2)
    numerator_value = product_sum_of_both_users - (person1_preferences_sum * person2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
                                person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))
    # 如果分母为0，返回0
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r


# 指定一个用户，返回一群相似的用户，number_of_users指定返回多少个相似用户
def topMatches(dataset, person, number_of_users=None, similarity=sim_pearson):
    scores = [(similarity(dataset, person, other_person), other_person) for other_person in dataset if other_person != person]
    # 对相似度按照从高到低进行排序
    scores.sort()
    scores.reverse()
    # 如果没有指定返回多少相似用户，则返回全部
    if number_of_users == None:
        return scores
    return scores[0: number_of_users]

# 给某位用户推荐电影
def getRecommendations(dataset, person):
    # 通过使用每个其他用户的排名的加权平均值，为某人获得建议
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset:
        # 不用和自己比较
        if other == person:
            continue
        sim = sim_pearson(dataset, person, other)
        # 忽视比0小的值
        if sim <= 0:
            continue
        # 只标记没看过的电影
        for item in dataset[other]:
            if item not in dataset[person] or dataset[person][item] == 0:
                # 初始化没看过的电影为0
                totals.setdefault(item, 0)
                # 添加每部电影分数(评价) * 相似度，如{'Just My Luck': 8.07,'Lady in the Water': 8.38, 'The Night Listener': 12.89}
                totals[item] += dataset[other][item] * sim
                # 添加每部电影的相似度，如{'Just My Luck': 3.19,'Lady in the Water': 2.95, 'The Night Listener': 3.85}
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # 归一化列表，(每部电影分数(评价)*相似度) / 每部电影的相似度
    rankings = [(total/simSums[item], item) for item, total in totals.items()]
    # 按从大到小进行排序
    rankings.sort()
    rankings.reverse()
    # 去掉分数，返回推荐电影
    #recommend_list = [recommend_item for score, recommend_item in rankings]
    return rankings

print(getRecommendations(user_data, 'Toby'))


######################################################################
# 查看哪些物品相近
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # 将物品和人员对调
            result[item][person] = prefs[person][item]
    return result

item_data = transformPrefs(user_data)
print(getRecommendations(item_data, 'Just My Luck'))

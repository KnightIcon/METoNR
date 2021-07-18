# todo
# 1 给新闻找邻居
# - category_neighbor
# - publish_date_neighbor
# - word_count_neighbor
# 2 给用户找邻居
# - city_neighbor
# - environment_neighbor
# - device_neighbor
# - os_neighbor
import os
import time

import pandas as pd
import random
import csv


def _list2str(l):
    return ' '.join('%s' % x for x in l)


def _str2list(s):
    return [int(i) for i in s.split()]


# write nested list of dict to csv
def _to_csv(data, out_file):
    with open(out_file, 'w', newline='') as f:
        w = csv.writer(f)
        fieldnames = data[0].keys()  # solve the problem to automatically write the header
        w.writerow(fieldnames)
        for row in data:
            w.writerow(row.values())


def _calc_word_count_type(word_count: int) -> int:
    """
    通过字数计算属于的分类
    :param word_count:  字数
    :return:  type of word_count 字数的分类
    """
    word_count = int(word_count)
    if word_count < 158:
        return 0
    elif word_count < 187:
        return 1
    elif word_count < 219:
        return 2
    else:
        return 3


def _calc_timestamp_to_date(timestamp: int) -> str:
    """
    通过created_at_ts这个时间戳计算文章发布于哪一天
    :param timestamp:
    :return:
    """
    timestamp = int(timestamp)
    return time.strftime('%Y-%m-%d', time.localtime(timestamp/1000))


def _random_pick(inputs: list, num: int, current: int) -> list:
    """
    从inputs中随机挑选num个不重复样本，且不能挑选值为current的样本
    tips: nums一定要大于等于len(inputs) + 1
    :param inputs:
    :param num:
    :param current:
    :return: 挑选后的样本
    """
    result = []
    while num > 0:
        idx = random.choice(inputs)
        if idx != current and idx not in result:
            num -= 1
            result.append(idx)
    return result


data_path = r'D:/data/dataset/globo'
neighbor_num = 4

article_file = os.path.join(data_path, 'articles_1_10.csv')

article_data = pd.read_csv(article_file)
article_data = article_data.to_dict('records')

category_dict = dict()
publish_date_dict = dict()
word_count_dict = dict()
for i in range(4):
    word_count_dict[i] = []

for article in article_data:
    article_id = int(article['article_id'])
    category_id = int(article['category_id'])
    word_count = int(article['words_count'])
    timestamp = int(article['created_at_ts'])

    if category_id not in category_dict.keys():
        category_dict[category_id] = [article_id]
    else:
        category_dict[category_id].append(article_id)

    word_count_type = _calc_word_count_type(word_count)
    word_count_dict[word_count_type].append(article_id)

    date = _calc_timestamp_to_date(timestamp)
    if date not in publish_date_dict.keys():
        publish_date_dict[date] = [article_id]
    else:
        publish_date_dict[date].append(article_id)

all_other_category_article_ids = []
for aids in category_dict.values():
    if len(aids) < neighbor_num + 1:
        all_other_category_article_ids += aids

all_other_publish_date_article_ids = []
for aids in publish_date_dict.values():
    if len(aids) < neighbor_num + 1:
        all_other_publish_date_article_ids += aids

for article in article_data:
    article_id = int(article['article_id'])
    category_id = int(article['category_id'])
    word_count = int(article['words_count'])
    timestamp = int(article['created_at_ts'])
    remain = neighbor_num

    if len(category_dict[category_id]) < neighbor_num + 1:
        category_neighbor_aids = _random_pick(all_other_category_article_ids, neighbor_num, article_id)
    else:
        category_neighbor_aids = _random_pick(category_dict[category_id], neighbor_num, article_id)
    article['category_neighbor'] = _list2str(category_neighbor_aids)

    word_count_type = _calc_word_count_type(word_count)
    remain = neighbor_num
    word_count_neighbor_aids = list()
    while remain > 0:
        neighbor_aid = random.choice(word_count_dict[word_count_type])
        if neighbor_aid != article_id and neighbor_aid not in word_count_neighbor_aids:
            remain -= 1
            word_count_neighbor_aids.append(neighbor_aid)
    article['word_count_neighbor'] = _list2str(word_count_neighbor_aids)

    remain = neighbor_num
    publish_date = _calc_timestamp_to_date(timestamp)
    if len(publish_date_dict[publish_date]) < neighbor_num + 1:
        publish_date_neighbor_aids = _random_pick(all_other_publish_date_article_ids, neighbor_num, article_id)
    else:
        publish_date_neighbor_aids = _random_pick(publish_date_dict[publish_date], neighbor_num, article_id)
    article['publish_date_neighbor'] = _list2str(publish_date_neighbor_aids)


_to_csv(article_data, os.path.join(data_path, 'articles_1_10_with_neighbors.csv'))

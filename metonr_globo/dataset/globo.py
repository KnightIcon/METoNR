import csv

import pickle
import random

import pandas as pd
import glob
import os
from itertools import groupby
from operator import itemgetter
from log.global_logger import get_logger

config = {
    'np_ratio': 14, # 正负采样比 设大一点比较灵活
    'data_path': r'D:\data\dataset\globo',
    'train_test_threshold': 0.8,
    'min_article_appear': 5,
    'min_user_appear': [6, 7, 8, 9, 10],
    'test_compact': 0.1
}


logger = get_logger()


def _list2str(l):
    return ' '.join('%s' % x for x in l)


def _generate_negative(dataset, positive_newss, index, np_ratio):
    origin_user_id = int(dataset[index]['user_id'])
    result = []
    # 先向后搜索，够了就停
    # 如果不够再向前搜索
    remain = np_ratio
    dataset_size = len(dataset)

    # 先向后搜索
    right_search_index = index + 1
    while remain != 0 and right_search_index < dataset_size:
        data = dataset[right_search_index]
        right_search_index += 1
        user_id = int(data['user_id'])
        news_id = int(data['click_article_id'])
        # 如果当前数据的user_id和待生成news_list的这个user，id不同
        # 并且当前的news_id不在user的正样本集合里面
        # 并且不在当前的结果集里面
        # 就放到结果集里面
        if origin_user_id != user_id \
                and news_id not in positive_newss \
                and news_id not in result:
            remain -= 1
            result.append(news_id)

    # 再向前搜索
    left_search_index = index - 1
    while remain != 0 and left_search_index >= 0:
        data = dataset[left_search_index]
        left_search_index -= 1
        user_id = int(data['user_id'])
        news_id = int(data['click_article_id'])
        # 如果当前数据的user_id和待生成news_list的这个user，id不同
        # 并且当前的news_id不在user的正样本集合里面
        # 并且不在当前的结果集里面
        # 就放到结果集里面
        if origin_user_id != user_id \
                and news_id not in positive_newss \
                and news_id not in result:
            remain -= 1
            result.append(news_id)

    return result


# write nested list of dict to csv
def _to_csv(data, out_file):
    with open(out_file, 'w', newline='') as f:
        w = csv.writer(f)
        fieldnames = data[0].keys()  # solve the problem to automatically write the header
        w.writerow(fieldnames)
        for row in data:
            w.writerow(row.values())


# 读取数据

articles_metas = pd.read_csv(os.path.join(config['data_path'], 'articles_metadata.csv'))
articles_metas = articles_metas.to_dict('records')
articles_emb = pickle.load(open(os.path.join(config['data_path'], 'articles_embeddings.pickle'), 'rb'))
logger.info('articles loaded: total amount %d', len(articles_metas))

# 把新闻向量和新闻meta聚集于一处
for i, a in enumerate(articles_metas):
    a['embedding'] = _list2str(articles_emb[i])
    a['appear_count'] = 0

# 转成dict方便统计出现次数
k = [a['article_id'] for a in articles_metas]
article_dict = dict(zip(k, articles_metas))


all_files = glob.glob(os.path.join(config['data_path'], 'clicks/clicks/*'))
clicks = pd.concat([pd.read_csv(_) for _ in all_files], ignore_index=True)
# clicks = pd.read_csv(os.path.join(config['data_path'], 'clicks_sample.csv'))
clicks['index'] = clicks.index
clicks = clicks.to_dict('records')
logger.info('clicks loaded: total amount %d', len(clicks))


# 统计article出现的次数并且去掉出现次数小于10次的article
for click in clicks:
    article_id = click['click_article_id']
    article_dict[article_id]['appear_count'] += 1
filtered_article_dict = {k: v for k, v in article_dict.items() if v['appear_count'] >= config['min_article_appear']}
less_than_10_article_ids = [v['article_id'] for _, v in article_dict.items() if v['appear_count'] < config['min_article_appear']]
filtered_article_list = list(filtered_article_dict.values())
logger.info('article filtered less than %d: remain %d', config['min_article_appear'], len(filtered_article_list))

filter_count = 0


def _filter_func(x):
    global filter_count
    filter_count += 1
    if filter_count % 1000 == 0:
        logger.info("%d/%d", filter_count, len(clicks))
    return x['click_article_id'] in less_than_10_article_ids


clicks = list(filter(_filter_func, clicks))
clicks_sort_by_user_id = sorted(clicks, key=lambda x: x['user_id'])
clicks.sort(key=lambda x: x['click_timestamp'])
logger.info('click sorted')


for i, click in enumerate(clicks):
    click['index'] = i


# 将group中的多条数据聚焦为一条数据
# 排除小于10条的group
# 测试集的candidate为后2条, 测试集的history为前n-2条
# 训练集的candidate为[n-4:n-2], 训练集的history为前n-4条
for min_user_appear in config['min_user_appear']:
    trains = list()
    tests = list()
    total_clicks = 0
    sessions = groupby(clicks_sort_by_user_id, itemgetter('user_id'))
    logger.info('click grouped by user_id')
    for _, group in sessions:
        group = list(group)
        if len(group) < min_user_appear:
            continue
        total_clicks += len(group)
        train = dict()
        test = dict()
        trains.append(train)
        tests.append(test)
        need_copy = ['session_id', 'user_id', 'click_environment', 'click_deviceGroup',
                     'click_country', 'click_region', 'click_referrer_type']
        group_clicks = list()
        for attr in need_copy:
            train[attr] = group[0][attr]
            test[attr] = group[0][attr]
        for item in group:
            group_clicks.append(item['click_article_id'])
        test['negative'] = _generate_negative(clicks, group_clicks, group[-1]['index'], config['np_ratio']) \
                                    + _generate_negative(clicks, group_clicks, group[-2]['index'], config['np_ratio'])
        test['negative'] = _list2str(test['negative'])
        test['positive'] = group_clicks[-2:]
        test['positive'] = _list2str(test['positive'])
        test['history'] = group_clicks[:-2]
        test['history'] = _list2str(test['history'])
        train['negative'] = _generate_negative(clicks, group_clicks, group[-3]['index'], config['np_ratio']) \
                                     + _generate_negative(clicks, group_clicks, group[-4]['index'], config['np_ratio'])
        train['negative'] = _list2str(train['negative'])
        train['positive'] = group_clicks[-4:-2]
        train['positive'] = _list2str(train['positive'])
        train['history'] = group_clicks[:-4]
        train['history'] = _list2str(train['history'])

    logger.info('trains and tests generated@%d %d', min_user_appear, len(trains))
    # tests不用全部的，随机采一些
    tests = random.sample(tests, int(config['test_compact'] * len(tests)))

    # 输出为合适的格式
    _to_csv(trains, os.path.join(config['data_path'], 'train_behavior_%d_%d.csv' % (config['min_article_appear'], min_user_appear)))
    _to_csv(tests, os.path.join(config['data_path'], 'test_behavior_%d_%d.csv' % (config['min_article_appear'], min_user_appear)))
    _to_csv(filtered_article_list, os.path.join(config['data_path'], 'articles_%d_%d.csv' % (config['min_article_appear'], min_user_appear)))
    logger.info('saved to file@%d', min_user_appear)

    logger.info("total clicks@%d, %d", min_user_appear, total_clicks)
    logger.info("total users@%d: %d", min_user_appear, len(trains))
    logger.info("total articles@%d: %d", min_user_appear, len(filtered_article_list))

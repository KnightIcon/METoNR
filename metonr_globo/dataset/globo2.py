"""
临时脚本, 用过即删
"""
import os
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


test_compact = 0.1
data_path = r'D:/data/dataset/globo'
samples_add_per_behavior = 5

news_file = os.path.join(data_path, 'articles_1_10.csv')
test_behaviors_file = os.path.join(data_path, 'test_behavior_1_10.csv')

news_data = pd.read_csv(news_file)
news_data = news_data.to_dict('records')

tests = pd.read_csv(test_behaviors_file)
tests = tests.to_dict('records')

tests = random.sample(tests, int(test_compact * len(tests)))

for test in tests:
    negative_samples = _str2list(test['negative'])
    positive_samples = _str2list(test['positive'])
    all_samples = negative_samples + positive_samples
    count = samples_add_per_behavior
    add_samples = []
    while count > 0:
        sample = random.choice(news_data)
        nid = int(sample['article_id'])
        if nid not in all_samples:
            count -= 1
            add_samples.append(nid)
    test['negative'] = _list2str(add_samples + negative_samples)

_to_csv(tests, os.path.join(data_path, 'test_behavior_1_10_2.csv'))

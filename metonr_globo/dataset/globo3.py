import os
import pandas as pd
import glob
import csv


def _to_csv(data, out_file):
    with open(out_file, 'w', newline='') as f:
        w = csv.writer(f)
        fieldnames = data[0].keys()  # solve the problem to automatically write the header
        w.writerow(fieldnames)
        for row in data:
            w.writerow(row.values())


data_path = r'D:/data/dataset/globo'

all_files = glob.glob(os.path.join(data_path, 'clicks/clicks/*'))
clicks = pd.concat([pd.read_csv(_) for _ in all_files], ignore_index=True)
# clicks = pd.read_csv(os.path.join(config['data_path'], 'clicks_sample.csv'))
clicks['index'] = clicks.index
clicks = clicks.to_dict('records')

uid2os = dict()
for click in clicks:
    uid = int(click['user_id'])
    click_os = int(click['click_os'])
    if uid not in uid2os.keys():
        uid2os[uid] = click_os

train_behaviors_file = os.path.join(data_path, 'train_behavior_1_10.csv')
test_behaviors_file = os.path.join(data_path, 'test_behavior_1_10_2.csv')
train_behaviors_out_file = os.path.join(data_path, 'train_behavior_1_10_os.csv')
test_behaviors_out_file = os.path.join(data_path, 'test_behavior_1_10_2_os.csv')


def append_os_info(behavior_file, uid2os, out_file):
    behaviors = pd.read_csv(os.path.join(behavior_file))
    behaviors = behaviors.to_dict('records')
    for behavior in behaviors:
        uid = int(behavior['user_id'])
        behavior['click_os'] = uid2os[uid]
    _to_csv(behaviors, out_file)


append_os_info(train_behaviors_file, uid2os, train_behaviors_out_file)
append_os_info(test_behaviors_file, uid2os, test_behaviors_out_file)

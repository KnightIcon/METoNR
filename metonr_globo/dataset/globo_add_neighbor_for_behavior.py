import os
import pandas as pd
import random
import csv

data_path = r'D:/data/dataset/globo'
test_behaviors_file = os.path.join(data_path, 'test_behavior_1_10_2_os.csv')
train_behaviors_file = os.path.join(data_path, 'train_behavior_1_10_os.csv')
test_behaviors_out_file = os.path.join(data_path, 'test_behavior_1_10_2_metonr.csv')
train_behaviors_out_file = os.path.join(data_path, 'train_behavior_1_10_metonr.csv')
neighbor_num = 4


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


def add_neighbor_for_behavior(file_path, neighbor_num, out_file):
    # step 0: load file
    behavior_data = pd.read_csv(file_path)
    behavior_data = behavior_data.to_dict('records')

    # step 1: 构建uid2history
    uid2history = dict()
    for behavior in behavior_data:
        uid = int(behavior['user_id'])
        history = _str2list(behavior['history'])
        uid2history[uid] = history

    # step 2: 构建每种neighbor_kind的dict
    # step 2.1 构建region_neighbor字典
    region_dict = dict()
    for behavior in behavior_data:
        uid = int(behavior['user_id'])
        region_id = int(behavior['click_region'])
        if region_id not in region_dict.keys():
            region_dict[region_id] = [uid]
        else:
            region_dict[region_id].append(uid)

    # step 2.2 构建os_neighbor字典
    os_dict = dict()
    for behavior in behavior_data:
        uid = int(behavior['user_id'])
        os_id = int(behavior['click_os'])
        if os_id not in os_dict.keys():
            os_dict[os_id] = [uid]
        else:
            os_dict[os_id].append(uid)

    # step 2.3 构建device_neighbor字典
    device_dict = dict()
    for behavior in behavior_data:
        uid = int(behavior['user_id'])
        device_id = int(behavior['click_deviceGroup'])
        if device_id not in device_dict.keys():
            device_dict[device_id] = [uid]
        else:
            device_dict[device_id].append(uid)
    # step 3: 选邻居
    # step 3.1: 选region的邻居
    for behavior in behavior_data:
        uid = int(behavior['user_id'])
        region_id = int(behavior['click_region'])
        region_neighbor_ids = _random_pick(region_dict[region_id], neighbor_num, uid)
        for i in range(neighbor_num):
            behavior['region_neighbor_' + str(i)] = _list2str(uid2history[region_neighbor_ids[i]])
        os_id = int(behavior['click_os'])
        os_neighbor_ids = _random_pick(os_dict[os_id], neighbor_num, uid)
        for i in range(neighbor_num):
            behavior['os_neighbor_' + str(i)] = _list2str(uid2history[os_neighbor_ids[i]])
        device_id = int(behavior['click_deviceGroup'])
        device_neighbor_ids = _random_pick(device_dict[device_id], neighbor_num, uid)
        for i in range(neighbor_num):
            behavior['device_neighbor_' + str(i)] = _list2str(uid2history[device_neighbor_ids[i]])
    # step 4: to_csv
    _to_csv(behavior_data, out_file)


add_neighbor_for_behavior(train_behaviors_file, neighbor_num, train_behaviors_out_file)
# add_neighbor_for_behavior(test_behaviors_file, neighbor_num, test_behaviors_out_file)

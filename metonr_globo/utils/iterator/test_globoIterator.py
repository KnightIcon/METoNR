import os
from unittest import TestCase

from config.config_loader import prepare_hparams
from utils.iterator.globo_iterator import GloboIterator
from tqdm import tqdm


class TestGloboIterator(TestCase):
    def test_load_data_from_file(self):
        yaml_file = r'../../config/metonr.yml'
        yaml_file = os.path.abspath(yaml_file)
        hparams = prepare_hparams(yaml_file)
        data_path = r'D:\data\dataset\globo'
        train_news_file = os.path.join(data_path, 'articles_1_10_n.csv')
        test_news_file = os.path.join(data_path, 'articles_1_10_n.csv')
        train_behaviors_file = os.path.join(data_path, 'train_behavior_1_10_n.csv')
        test_behaviors_file = os.path.join(data_path, 'test_behavior_1_10_n.csv')
        it = GloboIterator(hparams, npratio=3)
        # data = it.load_data_from_file(train_news_file, train_behaviors_file)
        tqdm_util = tqdm(
            # todo 看内部的这个函数
            it.load_data_from_file(
                train_news_file, train_behaviors_file
            ),
        )

        for batch_data_input in tqdm_util:
            pass
        print()

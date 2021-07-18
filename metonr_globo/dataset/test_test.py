from unittest import TestCase
import pickle
import pandas as pd
from log.global_logger import get_logger


class TestTest(TestCase):
    def test_load_word_dict(self):
        with open(r'C:\Users\MSI-NB\Desktop\MINDdemo_utils\word_dict.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data)

    def test_load_vert_dict(self):
        with open(r'C:\Users\MSI-NB\Desktop\MINDdemo_utils\vert_dict.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data)
        with open(r'C:\Users\MSI-NB\Desktop\MINDdemo_utils\subvert_dict.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data)

    def test_group_by(self):
        l = [["a", 12, 12], [None, 12.3, 33.], ["b", 12.3, 123], ["a", 1, 1]]
        df = pd.DataFrame(l, columns=["a", "b", "c"])
        group = df.groupby(by="a")
        for key, value in group:
            print(key)
            print(value)
            print("")

    def test_load_article_emb(self):
        with open(r'D:\data\dataset\globo\articles_embeddings.pickle', 'rb') as f:
            data = pickle.load(f)
            line = list(data[0])
            line = ' '.join('%s' % x for x in line)

    def test_arr(self):
        logger = get_logger()
        l = [i for i in range(10)]
        print(l[-2:])
        print(l[-4:-2])
        print(l[:-4])
        logger.info('hello %d', 1)

    def test_load_word_dict(self):
        with open(r'C:\Users\MSI-NB\Desktop\MINDdemo_utils\uid2index.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data)

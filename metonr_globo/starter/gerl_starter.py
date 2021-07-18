import sys
import os

root_path = os.path.abspath(r'../')
sys.path.append(root_path)


from config.config_loader import prepare_hparams
from utils.iterator.globo_iterator import GloboIterator
from models.gerl import GERLModel


if __name__ == '__main__':
    yaml_file = r'../config/gerl.yml'
    yaml_file = os.path.abspath(yaml_file)
    seed = 42
    data_path = r'/home/tg/data/globo'
    # data_path = r'D:/data/dataset/globo'

    train_news_file = os.path.join(data_path, 'articles_1_10_n.csv')
    test_news_file = os.path.join(data_path, 'articles_1_10_n.csv')
    train_behaviors_file = os.path.join(data_path, 'train_behavior_1_10_n.csv')
    test_behaviors_file = os.path.join(data_path, 'test_behavior_1_10_n.csv')

    hparams = prepare_hparams(yaml_file)

    iterator = GloboIterator

    model = GERLModel(hparams, iterator, seed=seed)

    eval_res = model.run_eval(test_news_file, test_behaviors_file)
    eval_info = ", ".join(
        [
            str(item[0]) + ":" + str(item[1])
            for item in sorted(eval_res.items(), key=lambda x: x[0])
        ]
    )
    print("\neval without train")
    print("\neval info: " + eval_info)
    model.fit(train_news_file, train_behaviors_file, test_news_file, test_behaviors_file)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import math

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras
from utils.evaluate.eval_utils import cal_metric

__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters. 超参数
        iterator_creator_train (obj): An iterator to load the data in training steps. 用于加载trainData的iterator
        iterator_creator_test (obj): An iterator to load the data in testing steps. 用于加载testData的iterator
        seed (int): Random seed. 随机种子
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all models, such as loss function,
        parameter set.
        初始化模型。创建所有模型都需要的通用逻辑，例如损失函数，参数设置等

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator_train (obj): An iterator to load the data in training steps.
            iterator_creator_test (obj): An iterator to load the data in testing steps.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            npratio=hparams.npratio
        )
        self.test_iterator = iterator_creator(
            hparams,
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        # set GPU use with on demand growth 设置GPU
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        # set this TensorFlow session as the default session for Keras
        tf.compat.v1.keras.backend.set_session(sess)

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.model, self.scorer = self._build_model()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    @abc.abstractmethod
    def _build_model(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
            获取损失函数的名称
        Returns:
            obj: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
            获取优化器
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(lr=lr)

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        根据不同的任务来获取最后的预测得分
        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.
            完成一次优化步骤。
        Args:

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.

        Args:

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    # todo 修改这个函数
    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.
        使用 train_file 拟合模型。 每个时期在 valid_file 上评估模型以观察训练状态。如果 test_news_file 不是 None，也评估它。

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.

        Returns:
            obj: An instance of self.
        """

        # epoch
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            tqdm_util = tqdm(
                # todo 看内部的这个函数
                self.train_iterator.load_data_from_file(
                    train_news_file, train_behaviors_file
                ),
                total=math.ceil(self.hparams.total_train / self.hparams.batch_size)
            )

            for batch_data_input in tqdm_util:
                # 训练一个batch
                step_result = self.train(batch_data_input)
                step_data_loss = step_result
                # 累加loss
                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description("train: ")

            train_end = time.time()
            train_time = train_end - train_start

            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )
            # 验证过程
            eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            # 如果测试文件不是None, 也跑一下测试集
            if test_news_file is not None:
                test_res = self.run_eval(test_news_file, test_behaviors_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_news_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            all_labels: labels after group.
            all_preds: preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        _, group_labels, group_preds = self.run_slow_eval(
                news_filename, behaviors_file
        )
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res

    def run_slow_eval(self, news_filename, behaviors_file):
        preds = []
        labels = []
        imp_indexes = []

        for batch_data_input in tqdm(
            self.test_iterator.load_data_from_file(news_filename, behaviors_file),
            total=math.ceil(self.hparams.total_test / self.hparams.batch_size)
        ):
            step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexes.extend(np.reshape(step_imp_index, -1))

        group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes
        )
        return group_impr_indexes, group_labels, group_preds

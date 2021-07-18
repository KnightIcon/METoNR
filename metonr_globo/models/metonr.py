# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os

import keras
from keras import layers

from config.config_loader import prepare_hparams
from models.base_model import BaseModel
from models.layers import SelfAttention
from keras.regularizers import l2
from keras import backend as K

from utils.iterator.globo_iterator import GloboIterator

__all__ = ["METONRModel"]


class METONRModel(BaseModel):
    """GRU Model

    Attributes:
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for METONR.

        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train(obj): NAML data loader class for train data.
            iterator_creator_test(obj): NAML data loader class for test and validation data
        """

        # 加载超参数
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["click_embs_batch"],
            batch_data["user_neighbor_embs_batch"],
            batch_data["candidate_embs_batch"],
            batch_data["candidate_neighbor_embs_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_neighbor_encoder(self):
        # todo 加dim_size之类的参数
        """The main function to create neighbor encoder of METONR.
        Return:
            obj: the neighbor encoder of METONR.
        """
        hparams = self.hparams
        user_neighbor_encoder_input = keras.Input(shape=(hparams.neighbor_size + 1, hparams.article_emb_size), dtype="float32")

        neighbor_input_emb = layers.Lambda(lambda x: x[:, :hparams.neighbor_size, :])(user_neighbor_encoder_input)
        news_input_emb = layers.Lambda(lambda x: x[:, -1:, :])(user_neighbor_encoder_input)
        neighbor_emb = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [news_input_emb] + [neighbor_input_emb] * 2
        )
        neighbor_emb = layers.Reshape((-1,))(neighbor_emb)
        model = keras.Model(user_neighbor_encoder_input, neighbor_emb)
        return model

    def _build_news_encoder(self):
        """The main function to create news encoder of MeToNR.
            建模新闻
        Return:
            obj: the news encoder of MeToNR.
        """

        hparams = self.hparams
        # todo news的shape需要调
        encoder_input = keras.Input(shape=(hparams.neighbor_kind + 1, hparams.head_num * hparams.head_dim), dtype="float32")
        all_kind_news_neighbors_emb = layers.Lambda(lambda x: x[:, :hparams.neighbor_kind, :])(encoder_input)
        news_input_emb = layers.Lambda(lambda x: x[:, -1:, :])(encoder_input)
        all_kind_neighbors_attention_emb = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [news_input_emb] + [all_kind_news_neighbors_emb] * 2
        )
        all_kind_neighbors_attention_emb = layers.Reshape((hparams.head_num * hparams.head_dim,))(all_kind_neighbors_attention_emb)
        news_input_emb_flatten = layers.Flatten()(news_input_emb)
        news_present = layers.concatenate([all_kind_neighbors_attention_emb, news_input_emb_flatten], axis=1)

        for dense_dim in hparams.dense_dim_list:
            dense_layer = layers.Dense(dense_dim, kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal',
                                        activation=hparams.dense_activation)
            news_present = dense_layer(news_present)

        model = keras.Model(encoder_input, news_present, name="news_encoder")
        return model

    def _build_user_encoder(self):
        """The main function to create user encoder of MeToNR.
            建模用户
        Args:
        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        user_gru_emb = keras.Input(shape=(hparams.article_emb_size,), dtype="float32")
        all_kind_user_neighbors_emb = keras.Input(shape=(hparams.neighbor_kind, hparams.head_num * hparams.head_dim), dtype="float32")
        user_gru_reshape_emb = layers.Reshape((1, hparams.article_emb_size))(user_gru_emb)
        all_kind_neighbors_attention_emb = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [user_gru_reshape_emb] + [all_kind_user_neighbors_emb] * 2
        )
        all_kind_neighbors_attention_emb = layers.Reshape((hparams.head_num * hparams.head_dim,))(all_kind_neighbors_attention_emb)
        user_present = layers.concatenate([all_kind_neighbors_attention_emb, user_gru_emb], axis=1)

        for dense_dim in hparams.dense_dim_list:
            dense_layer = layers.Dense(dense_dim, kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal',
                                        activation=hparams.dense_activation)
            user_present = dense_layer(user_present)

        model = keras.Model([user_gru_emb, all_kind_user_neighbors_emb], user_present, name="user_encoder")
        return model

    def _build_user_gru_encoder(self):
        hparams = self.hparams
        user_input_emb = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size), dtype="float32"
        )
        # todo 这个size也不是article_emb_size
        user_gru_emb = layers.GRU(hparams.article_emb_size)(user_input_emb)
        model = keras.Model(user_input_emb, user_gru_emb, name="user_gru_encoder")
        return model

    def _build_metonr(self):
        hparams = self.hparams
        user_input_emb = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size), dtype="float32"
        )
        user_neighbor_input_emb = keras.Input(
            shape=(hparams.neighbor_kind, hparams.neighbor_size, hparams.his_size, hparams.article_emb_size),
            dtype="float32"
        )
        pred_news_neighbor_input_emb = keras.Input(
            shape=(hparams.npratio + 1, hparams.neighbor_kind, hparams.neighbor_size, hparams.article_emb_size),
            dtype="float32"
        )
        pred_input_emb = keras.Input(
            shape=(hparams.npratio + 1, hparams.article_emb_size), dtype="float32"
        )
        pred_input_emb_one_not_reshape = keras.Input(
            shape=(1, hparams.article_emb_size,), dtype="float32"
        )
        pred_input_emb_one = layers.Reshape((hparams.article_emb_size,))(
            pred_input_emb_one_not_reshape
        )
        pred_news_neighbor_input_emb_one_not_reshape = keras.Input(
            shape=(1, hparams.neighbor_kind, hparams.neighbor_size, hparams.article_emb_size),
            dtype="float32"
        )
        pred_news_neighbor_input_emb_one = layers.Reshape((hparams.neighbor_kind, hparams.neighbor_size, hparams.article_emb_size))(
            pred_news_neighbor_input_emb_one_not_reshape
        )

        # 编码user
        user_gru_encoder = self._build_user_gru_encoder()
        user_gru_emb = user_gru_encoder(user_input_emb)
        user_neighbor_gru_emb = layers.TimeDistributed(layers.TimeDistributed(user_gru_encoder))(user_neighbor_input_emb)

        user_gru_emb_repeat = layers.RepeatVector(hparams.neighbor_kind)(user_gru_emb)
        user_gru_emb_repeat = layers.Reshape((hparams.neighbor_kind, 1, -1))(user_gru_emb_repeat)
        user_neighbor_encoder_input = layers.concatenate([user_neighbor_gru_emb, user_gru_emb_repeat], axis=2)
        user_all_kinds_neighbor_present = layers.TimeDistributed(self._build_neighbor_encoder())(user_neighbor_encoder_input)

        user_encoder = self._build_user_encoder()
        user_present = user_encoder([user_gru_emb, user_all_kinds_neighbor_present])

        # 编码news
        news_input_emb_repeat = layers.Flatten()(pred_input_emb)
        news_input_emb_repeat = layers.RepeatVector(hparams.neighbor_kind)(news_input_emb_repeat)
        news_input_emb_repeat = layers.Reshape((hparams.neighbor_kind, hparams.npratio + 1, 1, hparams.article_emb_size))(news_input_emb_repeat)
        news_input_emb_repeat = layers.Lambda(lambda x: K.permute_dimensions(x, pattern=(0, 2, 1, 3, 4)))(news_input_emb_repeat)
        news_neighbor_encoder_input = layers.concatenate([pred_news_neighbor_input_emb, news_input_emb_repeat], axis=3)
        news_neighbor_encoder_list = [self._build_neighbor_encoder() for _ in range(hparams.neighbor_kind)]
        news_all_kinds_neighbor_present_list = []
        for i in range(hparams.npratio + 1):
            news_one_kinds_neighbor_present_list = []
            for j in range(hparams.neighbor_kind):
                sl = layers.Lambda(lambda x: x[:, i, j, :, :])(news_neighbor_encoder_input)
                temp = news_neighbor_encoder_list[j](sl)
                temp = layers.Reshape([1, hparams.head_num * hparams.head_dim])(temp)
                news_one_kinds_neighbor_present_list.append(temp)
            news_one_kinds_neighbor_present = layers.concatenate(news_one_kinds_neighbor_present_list, axis=1)
            news_one_kinds_neighbor_present = layers.Reshape([1, hparams.neighbor_kind, hparams.head_num * hparams.head_dim])(news_one_kinds_neighbor_present)
            news_all_kinds_neighbor_present_list.append(news_one_kinds_neighbor_present)
        news_all_kinds_neighbor_present = layers.concatenate(news_all_kinds_neighbor_present_list, axis=1)
        dense_dim = hparams.head_num * hparams.head_dim
        dense_layer = layers.Dense(
                dense_dim,
                kernel_regularizer=l2(0.001),
                kernel_initializer='glorot_normal',
                activation=hparams.dense_activation)
        pred_dense_emb = layers.TimeDistributed(dense_layer)(pred_input_emb)
        pred_dense_emb = layers.Reshape((hparams.npratio + 1, 1, dense_dim))(pred_dense_emb)
        news_encoder_input = layers.concatenate([news_all_kinds_neighbor_present, pred_dense_emb], axis=2)
        news_encoder = self._build_news_encoder()
        news_present = layers.TimeDistributed(news_encoder)(news_encoder_input)

        news_input_emb_repeat_one = layers.RepeatVector(hparams.neighbor_kind)(pred_input_emb_one)
        news_input_emb_repeat_one = layers.Reshape((hparams.neighbor_kind, 1, hparams.article_emb_size))(news_input_emb_repeat_one)
        news_neighbor_encoder_input_one = layers.concatenate([pred_news_neighbor_input_emb_one, news_input_emb_repeat_one], axis=2)
        news_all_kinds_neighbor_present_one_list = []
        for j in range(hparams.neighbor_kind):
            sl = layers.Lambda(lambda x: x[:, j, :, :])(news_neighbor_encoder_input_one)
            temp = news_neighbor_encoder_list[j](sl)
            temp = layers.Reshape([1, hparams.head_num * hparams.head_dim])(temp)
            news_all_kinds_neighbor_present_one_list.append(temp)
        news_all_kinds_neighbor_present_one = layers.concatenate(news_all_kinds_neighbor_present_one_list, axis=1)
        pred_dense_emb_one = dense_layer(pred_input_emb_one)
        pred_dense_emb_one = layers.Reshape((1, dense_dim))(pred_dense_emb_one)
        news_encoder_input_one = layers.concatenate([news_all_kinds_neighbor_present_one, pred_dense_emb_one], axis=1)
        news_present_one = news_encoder(news_encoder_input_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                user_input_emb,
                user_neighbor_input_emb,
                pred_input_emb,
                pred_news_neighbor_input_emb,
            ],
            preds,
        )
        scorer = keras.Model(
            [
                user_input_emb,
                user_neighbor_input_emb,
                pred_input_emb_one_not_reshape,
                pred_news_neighbor_input_emb_one_not_reshape,
            ],
            pred_one,
        )

        return model, scorer

    def _build_model(self):
        """Build METONR model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_metonr()
        return model, scorer


if __name__ == '__main__':
    yaml_file = r'../config/metonr.yml'
    seed = 24
    yaml_file = os.path.abspath(yaml_file)
    hparams = prepare_hparams(yaml_file)
    iterator = GloboIterator
    model = METONRModel(hparams, iterator, seed=seed)
    # model._bulid_neighbor_encoder()

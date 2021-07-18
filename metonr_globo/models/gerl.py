# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os

import keras
from keras import layers
from keras.regularizers import l2

from config.config_loader import prepare_hparams
from models.base_model import BaseModel
from models.layers import AttLayer2, SelfAttention
from utils.iterator.globo_iterator import GloboIterator

__all__ = ["GERLModel"]


class GERLModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for NAML.
        Compared with the BaseModel, NAML need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

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
            batch_data["candidate_embs_batch"],
            batch_data["candidate_neighbor_embs_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_model(self):
        """Build NAML model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_gerl()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NAML.

        Args:
            newsencoder(obj): the news encoder of NAML.

        Return:
            obj: the user encoder of NAML.
        """
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size),
            dtype="float32",
        )

        click_news_presents = layers.TimeDistributed(newsencoder)(
            his_input_title_body_verts
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            click_news_presents
        )

        model = keras.Model(
            his_input_title_body_verts, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self):
        """The main function to create news encoder of GRU.
            建模词之间的交互
            因为globo直接提供article_embbeding, 这一部分啥也不干

        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_emb = keras.Input(shape=(hparams.article_emb_size,), dtype="float32")
        news_present = layers.Dense(hparams.dense_dim,
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation=hparams.dense_activation)(sequences_input_emb)
        model = keras.Model(sequences_input_emb, news_present, name="news_encoder")
        return model

    def _build_news_neighbor_encoder(self):
        """The main function to create news encoder of MeToNR.
            建模新闻
        Return:
            obj: the news encoder of MeToNR.
        """

        hparams = self.hparams
        # todo news的shape需要调
        encoder_input = keras.Input(shape=(hparams.neighbor_kind * hparams.neighbor_size + 1, hparams.article_emb_size), dtype="float32")
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

        model = keras.Model(encoder_input, news_present, name="news_neighbor_encoder")
        return model

    # def _build_news_id_encoder(self):
    #     """build user id encoder of GERL news encoder.
    #
    #     Return:
    #         obj: the user id encoder of GERL.
    #     """
    #     hparams = self.hparams
    #     input_news_id = keras.Input(shape=(1,), dtype="float32")
    #
    #     news_id_embedding = layers.Embedding(
    #         hparams.news_num, hparams.news_id_dim, trainable=True
    #     )
    #
    #     news_id_emb = news_id_embedding(input_news_id)
    #     pred = layers.Dense(
    #         hparams.filter_num,
    #         activation=hparams.dense_activation,
    #         bias_initializer=keras.initializers.Zeros(),
    #         kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
    #     )(news_id_emb)
    #     pred = layers.Reshape((1, hparams.filter_num))(pred)
    #
    #     model = keras.Model(input_news_id, pred, name="news_id_encoder")
    #     return model

    def _build_gerl(self):
        """The main function to create NAML's logic. The core of NAML
        is a user encoder and a news encoder.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_article_emb = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size), dtype="float32"
        )

        pred_input_article_emb = keras.Input(
            shape=(hparams.npratio + 1, hparams.article_emb_size), dtype="float32"
        )

        pred_input_article_emb_one = keras.Input(
            shape=(1, hparams.article_emb_size,), dtype="float32"
        )

        pred_news_neighbor_input_emb_not_reshape = keras.Input(
            shape=(hparams.npratio + 1, hparams.neighbor_kind, hparams.neighbor_size, hparams.article_emb_size),
            dtype="float32"
        )

        pred_news_neighbor_input_emb = layers.Reshape((hparams.npratio + 1, hparams.neighbor_kind * hparams.neighbor_size, hparams.article_emb_size))(
            pred_news_neighbor_input_emb_not_reshape
        )

        pred_news_neighbor_input_emb_one_not_reshape = keras.Input(
            shape=(1, hparams.neighbor_kind, hparams.neighbor_size, hparams.article_emb_size),
            dtype="float32"
        )
        pred_news_neighbor_input_emb_one = layers.Reshape((hparams.neighbor_kind * hparams.neighbor_size, hparams.article_emb_size))(
            pred_news_neighbor_input_emb_one_not_reshape
        )

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(self.newsencoder)
        news_neighbor_encoder = self._build_news_neighbor_encoder()

        user_present = self.userencoder(his_input_article_emb)
        user_present = layers.Dense(hparams.dense_dim,
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation=hparams.dense_activation)(user_present)
        pred_input_article_emb_one_reshape = layers.Reshape((hparams.article_emb_size,))(pred_input_article_emb_one)
        pred_input_article_emb_reshape = layers.Reshape((hparams.npratio + 1, 1, hparams.article_emb_size))(pred_input_article_emb)
        news_neighbor_encoder_input = layers.concatenate([pred_news_neighbor_input_emb, pred_input_article_emb_reshape], axis=2)
        news_neighbor_encoder_input_one = layers.concatenate([pred_news_neighbor_input_emb_one, pred_input_article_emb_one], axis=1)
        news_present_emb = layers.TimeDistributed(self.newsencoder)(pred_input_article_emb)
        news_present_neighbor = layers.TimeDistributed(news_neighbor_encoder)(news_neighbor_encoder_input)
        news_present = layers.concatenate([news_present_emb, news_present_neighbor], axis=2)
        news_present_emb_one = self.newsencoder(pred_input_article_emb_one_reshape)
        news_present_neighbor_one = news_neighbor_encoder(news_neighbor_encoder_input_one)
        news_present_one = layers.concatenate([news_present_emb_one, news_present_neighbor_one], axis=1)
        dense_layer = layers.Dense(
                hparams.dense_dim,
                kernel_regularizer=l2(0.001),
                kernel_initializer='glorot_normal',
                activation=hparams.dense_activation)
        news_present = layers.TimeDistributed(dense_layer)(news_present)
        news_present_one = dense_layer(news_present_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_article_emb,
                pred_input_article_emb,
                pred_news_neighbor_input_emb_not_reshape,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_article_emb,
                pred_input_article_emb_one,
                pred_news_neighbor_input_emb_one_not_reshape,
            ],
            pred_one,
        )
        model.summary()
        return model, scorer


if __name__ == '__main__':
    yaml_file = r'../config/gerl.yml'
    seed = 24
    yaml_file = os.path.abspath(yaml_file)
    hparams = prepare_hparams(yaml_file)
    iterator = GloboIterator
    model = GERLModel(hparams, iterator, seed=seed)
    # model._bulid_neighbor_encoder()
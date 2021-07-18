# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import keras
from keras import layers

from models.base_model import BaseModel
from models.layers import AttLayer2
from keras.regularizers import l2

__all__ = ["DANModel"]


class DANModel(BaseModel):
    """GRU Model

    Attributes:
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
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

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

    def _build_userencoder(self, news_encoder):
        """The main function to create user encoder of NRMS.
            建模候选新闻之间的交互

        Args:
            newsencoder(obj): the news encoder of NRMS.

        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_emb = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size), dtype="float32"
        )
        his_emb = news_encoder(his_input_emb)
        attention_emb = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            his_emb
        )
        lstm_emb = layers.RNN(layers.LSTMCell(hparams.rnn_layer_dim), return_sequences=True)(his_emb)
        attention_lstm_emb = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            lstm_emb
        )
        user_present = layers.Concatenate(axis=-1)(
            [attention_emb,  attention_lstm_emb]
        )
        user_present = layers.Dense(hparams.dense_dim,
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='glorot_normal',
                     activation=hparams.dense_activation)(user_present)
        model = keras.Model(his_input_emb, user_present, name="user_encoder")
        return model

    def _build_model(self):
        """Build NAML model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_dan()
        return model, scorer

    def _build_dan(self):
        hparams = self.hparams
        his_input_emb = keras.Input(
            shape=(hparams.his_size, hparams.article_emb_size), dtype="float32"
        )
        pred_input_emb = keras.Input(
            shape=(hparams.npratio + 1, hparams.article_emb_size), dtype="float32"
        )
        pred_input_emb_one = keras.Input(
            shape=(1, hparams.article_emb_size,), dtype="float32"
        )
        pred_emb_one_reshape = layers.Reshape((hparams.article_emb_size,))(
            pred_input_emb_one
        )
        news_encoder = self._build_newsencoder()
        user_encoder = self._build_userencoder(news_encoder)

        news_present = news_encoder(pred_input_emb)
        news_present_one = news_encoder(pred_emb_one_reshape)
        user_present = user_encoder(his_input_emb)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_emb,
                pred_input_emb,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_emb,
                pred_input_emb_one,
            ],
            pred_one,
        )

        return model, scorer

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import keras
from keras import layers

from models.base_model import BaseModel
from models.layers import AttLayer2

__all__ = ["NAMLModel"]


class NAMLModel(BaseModel):
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
            batch_data["clicked_vert_batch"],
            batch_data["candidate_embs_batch"],
            batch_data["candidate_vert_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_model(self):
        """Build NAML model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_naml()
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
            shape=(hparams.his_size, hparams.article_emb_size + 1),
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

    def _build_news_encoder(self):
        """The main function to create news encoder of NAML.
        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder

        Args:
            embedding_layer(obj): a word embedding layer.

        Return:
            obj: the news encoder of NAML.
        """
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.article_emb_size + 1,), dtype="float32"
        )

        sequences_input_article = layers.Lambda(lambda x: x[:, : hparams.article_emb_size])(
            input_title_body_verts
        )
        sequences_input_article = layers.Reshape((1, hparams.article_emb_size))(sequences_input_article)

        input_vert = layers.Lambda(
            lambda x: x[:, hparams.article_emb_size:]
        )(input_title_body_verts)

        vert_repr = self._build_vert_encoder()(input_vert)

        concate_repr = layers.Concatenate(axis=-2)(
            [sequences_input_article, vert_repr]
        )
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            concate_repr
        )

        model = keras.Model(input_title_body_verts, news_repr, name="news_encoder")
        model.summary()
        return model

    def _build_vert_encoder(self):
        """build vert encoder of NAML news encoder.

        Return:
            obj: the vert encoder of NAML.
        """
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype="float32")

        vert_embedding = layers.Embedding(
            hparams.vert_num, hparams.vert_emb_dim, trainable=True
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_naml(self):
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

        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype="float32")

        pred_input_article_emb = keras.Input(
            shape=(hparams.npratio + 1, hparams.article_emb_size), dtype="float32"
        )

        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="float32")

        pred_input_article_emb_one = keras.Input(
            shape=(1, hparams.article_emb_size,), dtype="float32"
        )

        pred_input_vert_one = keras.Input(shape=(1, 1), dtype="float32")

        his_article_vert = layers.Concatenate(axis=-1)(
            [his_input_article_emb,  his_input_vert]
        )

        pred_article_vert = layers.Concatenate(axis=-1)(
            [pred_input_article_emb, pred_input_vert]
        )

        pred_article_vert_one = layers.Concatenate(axis=-1)(
            [
                pred_input_article_emb_one, pred_input_vert_one,
            ]
        )
        pred_article_vert_one = layers.Reshape((-1,))(pred_article_vert_one)

        self.newsencoder = self._build_news_encoder()
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_article_vert)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_article_vert)
        news_present_one = self.newsencoder(pred_article_vert_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_article_emb,
                his_input_vert,
                pred_input_article_emb,
                pred_input_vert,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_article_emb,
                his_input_vert,
                pred_input_article_emb_one,
                pred_input_vert_one,
            ],
            pred_one,
        )
        model.summary()
        return model, scorer

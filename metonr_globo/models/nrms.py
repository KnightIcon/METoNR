# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import keras
from keras import layers

from models.base_model import BaseModel
from models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(
            self, hparams, iterator_creator, seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train(obj): NRMS data loader class for train data.
            iterator_creator_test(obj): NRMS data loader class for test and validation data
        """
        super().__init__(
            hparams, iterator_creator, seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            array: labels
        """
        input_feat = [
            batch_data["click_embs_batch"],
            batch_data["candidate_embs_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_model(self):
        """Build NRMS model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, newsencoder):
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

        click_emb_presents = layers.TimeDistributed(newsencoder)(his_input_emb)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_emb_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_emb, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self):
        """The main function to create news encoder of NRMS.
            建模词之间的交互
            因为globo直接提供article_embbeding, 这一部分啥也不干

        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_emb = keras.Input(shape=(hparams.article_emb_size,), dtype="float32")
        model = keras.Model(sequences_input_emb, sequences_input_emb, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
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

        # todo 这里是否需要reshape
        pred_emb_one_reshape = layers.Reshape((hparams.article_emb_size,))(
            pred_input_emb_one
        )

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_input_emb)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_emb)
        news_present_one = self.newsencoder(pred_emb_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_emb, pred_input_emb], preds)
        scorer = keras.Model([his_input_emb, pred_input_emb_one], pred_one)

        return model, scorer

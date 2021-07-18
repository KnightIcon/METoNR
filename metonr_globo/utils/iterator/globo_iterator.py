# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import random

import tensorflow as tf
import numpy as np

from utils.iterator.base_iterator import BaseIterator
from utils.utils import newsample

__all__ = ["GloboIterator"]


class GloboIterator(BaseIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts. 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        batch_size (int): the samples num in one batch.
        his_size (int): max clicked news num in user click history.
    """

    def __init__(
            self, hparams, npratio=-1, col_spliter=",",
    ):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.batch_size = hparams.batch_size
        self.his_size = hparams.his_size
        self.article_emb_size = hparams.article_emb_size
        self.npratio = npratio
        self.neighbor_size = hparams.neighbor_size
        self.neighbor_kind = hparams.neighbor_kind

    def init_news(self, news_file):
        """ init news information given news file, such as news_title_index, news_abstract_index.
        Args:
            news_file: path of news file
        """
        self.nid2index = {}
        self.nid2index[0] = len(self.nid2index)
        self.news_vert = list()
        self.news_vert.append(0)
        self.news_emb = list()
        self.news_emb.append([0 for _ in range(self.article_emb_size)])
        temp_category_neighbors = [[0 for _ in range(self.neighbor_size)]]
        temp_word_count_neighbors = [[0 for _ in range(self.neighbor_size)]]
        temp_publish_date_neighbors = [[0 for _ in range(self.neighbor_size)]]

        with tf.io.gfile.GFile(news_file, "r") as rd:
            rd.readline()
            for line in rd:
                # nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                #     self.col_spliter
                # )
                nid, vert, _, _, _, embedding, _, category_neighbor, word_count_neighbor, publish_date_neighbor \
                    = line.strip("\n").split(self.col_spliter)

                if nid in self.nid2index:
                    continue

                # 将userId转换为按顺序的index
                self.nid2index[nid] = len(self.nid2index)
                self.news_vert.append(int(vert))
                embedding = [float(x) for x in embedding.split()]
                self.news_emb.append(embedding)
                temp_category_neighbors.append([x for x in category_neighbor.split()])
                temp_word_count_neighbors.append([x for x in word_count_neighbor.split()])
                temp_publish_date_neighbors.append([x for x in publish_date_neighbor.split()])

        self.category_neighbors = list()
        self.word_count_neighbors = list()
        self.publish_date_neghbors = list()
        for cn in temp_category_neighbors:
            self.category_neighbors.append([self.nid2index[i] for i in cn])
        for wn in temp_word_count_neighbors:
            self.word_count_neighbors.append([self.nid2index[i] for i in wn])
        for dn in temp_publish_date_neighbors:
            self.publish_date_neghbors.append([self.nid2index[i] for i in dn])

    def init_behaviors(self, behaviors_file):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []
        self.user_neighbors = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            rd.readline()
            for line in rd:
                _, uid, _, _, _, _, _, negative, positive, history, _, *temp_all_kind_neighbors = line.strip("\n").split(self.col_spliter)

                all_kind_neighbors = []
                for single_kind_neighbors in temp_all_kind_neighbors:
                    neighbor_history = [self.nid2index[i] for i in single_kind_neighbors.split()]
                    neighbor_history = [0] * (self.his_size - len(neighbor_history)) + neighbor_history[: self.his_size]
                    all_kind_neighbors.append(neighbor_history)

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[: self.his_size]

                impr_news = [self.nid2index[i] for i in positive.split()] + [self.nid2index[i] for i in negative.split()]
                label = [1 for _ in positive.split()] + [0 for _ in negative.split()]

                self.user_neighbors.append(all_kind_neighbors)
                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uid)
                impr_index += 1

    def parser_one_line(self, line):

        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                impr_index = []
                user_index = []
                user_neighbor_emb = []
                click_emb = []
                candidate_emb = []
                candidate_neighbor_emb = []
                click_vert_index = []
                candidate_vert_index = []
                ori_label = [1] + [0] * self.npratio
                label = []

                n = newsample(negs, self.npratio)
                candidates = [p] + n
                d = dict(zip(candidates, ori_label))
                random.shuffle(candidates)
                for candidate in candidates:
                    label.append(d[candidate])
                    candidate_emb.append(self.news_emb[candidate])
                    candidate_vert_index.append([self.news_vert[candidate]])
                    candidate_neighbor_emb.append([
                        [self.news_emb[nidx] for nidx in self.category_neighbors[candidate]],
                        [self.news_emb[nidx] for nidx in self.word_count_neighbors[candidate]],
                        [self.news_emb[nidx] for nidx in self.publish_date_neghbors[candidate]],
                    ])
                for single_kind_neighbors in self.user_neighbors[line]:
                    user_neighbor_emb.append([self.news_emb[nidx] for nidx in single_kind_neighbors])
                for history in self.histories[line]:
                    click_emb.append(self.news_emb[history])
                    click_vert_index.append([self.news_vert[history]])
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    user_neighbor_emb,
                    candidate_vert_index,
                    candidate_neighbor_emb,
                    candidate_emb,
                    click_vert_index,
                    click_emb
                )

        else:

            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                impr_index = []
                user_index = []
                click_emb = []
                click_vert_index = []
                user_neighbor_emb = []
                label = [label]
                candidate_emb = [self.news_emb[news]]
                candidate_vert_index = [[self.news_vert[news]]]
                candidate_neighbor_emb = [[
                    [self.news_emb[nidx] for nidx in self.category_neighbors[news]],
                    [self.news_emb[nidx] for nidx in self.word_count_neighbors[news]],
                    [self.news_emb[nidx] for nidx in self.publish_date_neghbors[news]],
                ]]
                for single_kind_neighbors in self.user_neighbors[line]:
                    user_neighbor_emb.append([self.news_emb[nidx] for nidx in single_kind_neighbors])
                for history in self.histories[line]:
                    click_emb.append(self.news_emb[history])
                    click_vert_index.append([self.news_vert[history]])
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    user_neighbor_emb,
                    candidate_vert_index,
                    candidate_neighbor_emb,
                    candidate_emb,
                    click_vert_index,
                    click_emb
                )

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from a file.
        
        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """

        # 初始化新闻
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        # 初始化behaviors
        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        user_neighbor_embs = []
        candidate_embs = []
        candidate_vert_indexes = []
        candidate_neighbor_embs = []
        click_vert_indexes = []
        click_embs = []
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                    label,
                    impr_index,
                    user_index,
                    user_neighbor_emb,
                    candidate_vert_index,
                    candidate_neighbor_emb,
                    candidate_emb,
                    click_vert_index,
                    click_emb
            ) in self.parser_one_line(index):
                user_neighbor_embs.append(user_neighbor_emb)
                candidate_neighbor_embs.append(candidate_neighbor_emb)
                candidate_vert_indexes.append(candidate_vert_index)
                click_vert_indexes.append(click_vert_index)
                imp_indexes.append(impr_index)
                user_indexes.append(user_index)
                label_list.append(label)
                candidate_embs.append(candidate_emb)
                click_embs.append(click_emb)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        user_neighbor_embs,
                        candidate_embs,
                        candidate_vert_indexes,
                        candidate_neighbor_embs,
                        click_vert_indexes,
                        click_embs
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    user_neighbor_embs = []
                    candidate_embs = []
                    candidate_vert_indexes = []
                    candidate_neighbor_embs = []
                    click_vert_indexes = []
                    click_embs = []
                    cnt = 0

    def _convert_data(
            self,
            label_list,
            imp_indexes,
            user_indexes,
            user_neighbor_embs,
            candidate_embs,
            candidate_vert_indexes,
            candidate_neighbor_embs,
            click_vert_indexes,
            click_embs
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels. 标签列表
            imp_indexes (list): a list of impression indexes. impression索引列表
            user_indexes (list): a list of user indexes. 用户索引列表
            candidate_vert_indexes (list): the candidate news verts' words indices. 候选新闻分类词索引
            click_vert_indexes (list): indices for user's clicked news verts.
            candidate_embs (list): a list of candidate emb: 候选新闻嵌入
            click_embs (list): a list of click emb: 点击新闻嵌入
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.int32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_vert_index_batch = np.asarray(candidate_vert_indexes, dtype=np.int64)
        click_vert_index_batch = np.asarray(click_vert_indexes, dtype=np.int64)
        candidate_embs_batch = np.asarray(candidate_embs, dtype=np.float64)
        click_embs_batch = np.asarray(click_embs, dtype=np.float64)
        user_neighbor_embs_batch = np.asarray(user_neighbor_embs, dtype=np.float64)\
            .reshape((self.batch_size, self.neighbor_kind, self.neighbor_size, self.his_size, self.article_emb_size))
        candidate_neighbor_embs_batch = np.asarray(candidate_neighbor_embs, dtype=np.float64)
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_vert_batch": click_vert_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "labels": labels,
            "candidate_embs_batch": candidate_embs_batch,
            "click_embs_batch": click_embs_batch,
            "user_neighbor_embs_batch": user_neighbor_embs_batch,
            "candidate_neighbor_embs_batch": candidate_neighbor_embs_batch
        }

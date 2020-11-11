"""Data iterators
"""

from .tools import word_tokenize, newsample, NUMPY
import numpy as np
import pickle
from ..model import training

class BasicIterator:
    def __init__(self,
                 params:training.params):
        self.params = params

    def open(self, *file, **options):
        """Access to data file.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def batch(self, *args, **options):
        """Generator of batch data
        """
        raise NotImplementedError

    def check_data_bag(self):
        """Which parts of data are needed?
        """
        raise NotImplementedError

    def offer_constrains(self):
        return {}

# Adapted from MS/Recommenders / MindAllIterator
# The original is really sucks.
class MindIterator(BasicIterator):
    def __init__(self,
                 params : training.params,
                 col_spliter = '\t',
                 ID_spliter = "%"):
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = params.batch_size
        self.title_size = params.title_size
        self.body_size = params.body_size
        if self.body_size is None:
            self.body_size = 0
        self.his_size = params.his_size
        self.npratio = params.npratio if params.npratio > 0 else 0

        self.word_dict = self.load_dict(params.wordDict_file)
        self.vert_dict = self.load_dict(params.vertDict_file)
        self.subvert_dict = self.load_dict(params.subvertDict_file)
        self.uid2index = self.load_dict(params.userDict_file)
        self.nid2index = {}
        self._data_bag = {}
        self.size = 0
        self.Mind_data_bag = [
            'impression clicked',
            'impression index',
            'user index',
            'impression title',
            'impression abstract',
            'impression category',
            'impression subcatgory',
            'history title',
            'history abstract',
            'history catgory',
            'history subcatgory',
        ]

    def load_dict(self, file_path):
        """ load pickle file
        Args:
            file path (str): file path
        
        Returns:
            (obj): pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def check_data_bag(self, data_bag):
        for name in data_bag:
            if name not in self.Mind_data_bag:
                raise KeyError(f"Don't have {name} data sources, " \
                               f"available data: {self.Mind_data_bag}")

    def open(self, news_file, behavior_file):
        self._open_news(news_file)
        self._open_behavior(behavior_file)

    def _open_news(self, news_file):
        news_data_bag = {
            "title" : [""],
            "abstract": [""],
            "category": [""],
            "subcategory": [""]
        }

        news_title = [""]
        news_ab = [""]
        news_vert = [""]
        news_subvert = [""]

        with open(news_file, "r", encoding='utf-8') as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip(
                    "\n").split(self.col_spliter)

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                ab = word_tokenize(ab)
                news_title.append(title)
                news_ab.append(ab)
                news_vert.append(vert)
                news_subvert.append(subvert)

        news_title_index = np.zeros((len(news_title), self.title_size),dtype="int32")
        news_ab_index = np.zeros((len(news_ab), self.body_size),dtype="int32")
        news_vert_index = np.zeros((len(news_vert), 1), dtype="int32")
        news_subvert_index = np.zeros((len(news_subvert), 1),
                                           dtype="int32")

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            ab = news_ab[news_index]
            vert = news_vert[news_index]
            subvert = news_subvert[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    news_title_index[news_index,word_index] = self.word_dict[title[word_index].lower()]
            for word_index_ab in range(min(self.body_size, len(ab))):
                if ab[word_index_ab] in self.word_dict:
                    news_ab_index[news_index,word_index_ab] = self.word_dict[ab[word_index_ab].lower()]
            if vert in self.vert_dict:
                news_vert_index[news_index, 0] = self.vert_dict[vert]
            if subvert in self.subvert_dict:
                news_subvert_index[news_index,0] = self.subvert_dict[subvert]

        news_data_bag = {
            "title" : news_title_index,       # (total_news, title_size)
            "abstract": news_ab_index,        # (total_news, body_size)
            "category": news_vert_index,      # (total_news, 1)
            "subcategory": news_subvert_index # (total_news, 1)
        }
        self._data_bag.update(news_data_bag)
    def _open_behavior(self, behaviors_file):
        histories = []
        imprs = []
        labels = []
        uindexes = []

        with open(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[:self.his_size]

                impr_news = [
                    self.nid2index[i.split("-")[0]] for i in impr.split()
                ]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0
                histories.append(history)
                imprs.append(impr_news)
                labels.append(label)
                uindexes.append(uindex)
                impr_index += 1
                
                self.size += len(label)
        users_data_bag = {
            "history" : histories,
            "impression": imprs,
            "clicked" : labels,
            "user" : uindexes,
            "impression index" : np.arange(len(labels))
        }
        self._data_bag.update(users_data_bag)

    def Bag(self, bag, data_bag):
        results = {
            name:np.asanyarray(bag[name]) for name in data_bag
        }
        return results

    def impression_batch(self, data_bag):
        """Diff from batch, yield all data from one user at once
        """
        Iter_bag = {name: [] for name in self.Mind_data_bag}
        total_size = len(self._data_bag['user'])
        print(total_size)
        indexes = np.arange(total_size)
        np.random.shuffle(indexes)
        for index in indexes:
            # user_impression, 1
            bag = self.parser_one_line(index, whole=True)
            for i, name in enumerate(self.Mind_data_bag):
                Iter_bag[name].extend(bag[i])
            yield self.Bag(Iter_bag, data_bag=data_bag)
            for i, name in enumerate(self.Mind_data_bag):
                Iter_bag[name] = []

    def batch(self, data_bag, test=False):
        self.check_data_bag(data_bag)
        if test == True:
            for bag in self.impression_batch(data_bag):
                yield bag
        else:
            Iter_bag = {
                name:[] for name in self.Mind_data_bag
            }

            total_size = len(self._data_bag['user'])
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            count = 0
            for index in indexes:
                bag = self.parser_one_line(index)
                length = len(bag[0])
                if count + length >= self.batch_size:
                    take = self.batch_size - count
                    left = length - take
                else:
                    take = length
                count += take
                for i, name in enumerate(self.Mind_data_bag):
                    Iter_bag[name].extend(bag[i][:take])

                if count >= self.batch_size:
                    yield self.Bag(
                        Iter_bag,
                        data_bag = data_bag
                    )
                    count = left
                    for i, name in enumerate(self.Mind_data_bag):
                        Iter_bag[name] = bag[i][-left:]

    def parser_one_line(self, index, data_bag = {}, whole=False):
        """Parse index into feature values.
        
        Args:
            index (str): a string indicating one instance.

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index, 
            candidate_ab_index, clicked_ab_index,
            candidate_vert_index, clicked_vert_index,
            candidate_subvert_index, clicked_subvert_index,
        """
        npratio = self.npratio

        impr_label = self._data_bag['clicked'][index]
        impr = self._data_bag['impression'][index]

        poss = []
        negs = []

        for news, click in zip(impr, impr_label):
            if click == 1:
                poss.append(news)
            else:
                negs.append(news)
        user_index = []
        impr_index = []
        labels = []
        candidate_title_index = []
        candidate_ab_index = []
        candidate_vert_index = []
        candidate_subvert_index = []
        click_title_index = []
        click_ab_index = []
        click_vert_index = []
        click_subvert_index = []
        if whole == True:
            poss_first = len(poss)
            whole = poss + negs
            for i, new in enumerate(whole):
                labels.append([1] if i < poss_first else [0])
                candidate_title_index.append(self._data_bag['title'][[new]])
                candidate_ab_index.append(self._data_bag['abstract'][[new]])
                candidate_vert_index.append(self._data_bag['category'][[new]])
                candidate_subvert_index.append(self._data_bag['subcategory'][[new]])
                click_title_index.append(self._data_bag['title'][self._data_bag["history"][index]])
                click_ab_index.append(self._data_bag['abstract'][self._data_bag["history"][index]])
                click_vert_index.append(self._data_bag['category'][self._data_bag["history"][index]])
                click_subvert_index.append(self._data_bag['subcategory'][self._data_bag["history"][index]])
                impr_index.append(self._data_bag['impression index'][index])
                user_index.append(self._data_bag['user'][index])
        else:
            for p in poss:
                labels.append([1]+[0] * npratio)

                n = newsample(negs, npratio)
                candidate_title_index.append(self._data_bag['title'][[p] + n])
                candidate_ab_index.append(self._data_bag['abstract'][[p] + n])
                candidate_vert_index.append(self._data_bag['category'][[p] + n])
                candidate_subvert_index.append(self._data_bag['subcategory'][[p] + n])
                click_title_index.append(self._data_bag['title'][self._data_bag["history"][index]])
                click_ab_index.append(self._data_bag['abstract'][self._data_bag["history"][index]])
                click_vert_index.append(self._data_bag['category'][self._data_bag["history"][index]])
                click_subvert_index.append(self._data_bag['subcategory'][self._data_bag["history"][index]])
                impr_index.append(self._data_bag['impression index'][index])
                user_index.append(self._data_bag['user'][index])

        return (
            labels,
            impr_index,
            user_index,
            candidate_title_index,
            candidate_ab_index,
            candidate_vert_index,
            candidate_subvert_index,
            click_title_index,
            click_ab_index,
            click_vert_index,
            click_subvert_index,
        )
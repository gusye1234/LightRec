"""Store models here
Some methods should to be bound to model
- the method to offer which type of hyperparameters
- the method to construct loss (loss function)
- the method to eval itself.
"""
import abc
import torch
from torch import nn
import numpy as np
from .base import BasicModel
from .training import *
from ..data import iterator
from .layer import NRMS_News_Encoder, NRMS_User_Encoder

class NRMS(BasicModel):
    """
    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference 
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference 
    on Natural Language Processing (EMNLP-IJCNLP)
    """
    def __init__(self,
                 param : params):
        super(NRMS, self).__init__(name="NRMS")
        param.check_constrains(self.offer_constarins())
        self.param = param
        self.init_weight()

    def init_weight(self):
        self.word2vec = np.load(self.param.wordEmb_file)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(self.word2vec), freeze=False, padding_idx=0)
        self.sigmoid = nn.Sigmoid()
        self.word_emb_dim = self.word2vec.shape[-1]
        self.news = NRMS_News_Encoder(self.param)
        self.user = NRMS_User_Encoder(self.param)
        self.loss_function = nn.CrossEntropyLoss()

    def offer_data_bag(self):
        return [
            'user index', 'impression clicked', 'impression title', 'history title'
        ]

    def offer_label_bag(self):
        return 'impression clicked'

    def offer_constarins(self):
        return {'wordEmb_file': str, 'dropout': float}

    def loss(self, pred, truth):
        # (pred,
        #  truth) = TO(pred,truth, device=self.device)
        
        label = torch.zeros(truth.shape[0]).to(self.device).long()
        cate_loss = self.loss_function(pred, label)
        
        return cate_loss

    def forward(self, batch_data_bag):
        """
        'impression clicked', (batch, 1+npratio)
        'impression title', (batch 1+npratio, title)
        'history title', (batch, his, title)
        'user index', (batch)
        """
        (im_news, im_label,his_news) = TO(batch_data_bag['impression title'],
                                          batch_data_bag['impression clicked'],
                                          batch_data_bag['history title'],
                                          device=self.device)
        _, K_1, title_size = im_news.shape
        _, his_size, _ = his_news.shape
        word_emb = self.word_emb_dim

        '(batch, 1+npratio, title, word_emb)'
        im_word = self.word_embedding(im_news.long())
        '(batch*(1+npratio), title, word_emb)'
        im_word = im_word.view(-1, title_size, word_emb)
        '(batch, history, title, word_emb)'
        his_word = self.word_embedding(his_news.long())
        '(batch*history, title, word_emb)'
        his_word = his_word.view(-1, title_size, word_emb)

        '(batch*(1+npratio), k*dim)'
        im_R = self.news(im_word)
        '(batch, 1+npratio, k*dim)'
        im_R = im_R.view(-1, K_1, im_R.shape[-1])
        '(batch*history, k*dim)'
        his_R = self.news(his_word)
        '(batch, history, k*dim)'
        his_R = his_R.view(-1, his_size, his_R.shape[-1])

        '(batch, k*dim)'
        user_R = self.user(his_R)

        # ----------------
        user_R = user_R.unsqueeze_(-1)
        '(batch, 1+npratio)'
        CTR = torch.matmul(im_R, user_R).squeeze()
        return CTR

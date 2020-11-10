import torch
from torch import nn
import torch.nn.functional as F
from .base import BasicLayer
from .training import params

class AttLayer2(BasicLayer):
    """
        Take:
            (batch, title, in_dim)
        Out:
            (batch, in_dim)
    """
    def __init__(self,
                 in_dim,
                 param :  params):
        super(AttLayer2, self).__init__()
        self.in_dim = in_dim
        self.dim = param.attention_hidden_dim
        self.init_weight()

    def init_weight(self):
        self.soft = nn.Softmax(dim=-1)
        self.trans = nn.Linear(
            self.in_dim, self.dim
        ).double()
        self.q = nn.Linear(
            self.dim, 1, bias=False
        ).double()
        for l in [self.trans, self.q]:
            nn.init.xavier_uniform_(l.weight, gain=1.0)
        nn.init.zeros_(self.trans.bias)

    def forward(self, x):
        """
        Args:
            x (tensor): shape (batch, title, in_dim)

        Returns:
            shape (batch, in_dim)
        """
        inputs = x
        x = self.trans(x.double())
        x = torch.tanh(x)
        x = self.q(x)
        x = x.squeeze() # (batch, title)
        x = self.soft(x)
        x = x.unsqueeze(-1)
        # inputs*x : (batch, title, in_dim)
        # sum(inputs*x) : (batch, in_dim)
        return torch.sum(inputs*x, dim=1)

class SelfAttention(BasicLayer):
    """Simplied for NRMS
        Take:
            (batch, title, word_emb)
        Out:
            (batch, head*dim)
    """
    def __init__(self,
                 param : params):
        super(SelfAttention, self).__init__()
        param.check_constrains(self.offer_constarins())
        self.word_emb_dim = param.word_emb_dim
        self.heads = param.head_num
        self.dims  = param.head_dim
        self.output_dim = param.head_num * param.head_dim
        self.init_weight()

    def offer_constarins(self):
        return {'head_num': int,
                'head_dim':int,
                'word_emb_dim': int}

    def init_weight(self):
        """
        """
        self.soft = nn.Softmax(dim=-1)
        self.WQ = nn.Linear(
            self.word_emb_dim, self.output_dim, bias=False,
        ).double()
        self.WK = nn.Linear(
            self.word_emb_dim, self.output_dim, bias=False,
        ).double()
        self.WV = nn.Linear(
            self.word_emb_dim, self.output_dim, bias=False,
        ).double()
        '''WQ * WK'''
        for layer in [self.WQ, self.WK, self.WV]:
            nn.init.xavier_uniform_(layer.weight, gain=1.0)

    def forward(self, Q, K=None, V=None):
        # Q : (batch, title, word_dim)
        Q : torch.tensor
        K = K or Q
        V = V or Q
        batch_size = Q.shape[0]
        title_size = Q.shape[1]
        tmp = {
            "Q_seq" : (Q, self.WQ),
            "K_seq" : (K, self.WK),
            "V_seq" : (V, self.WV)
        }
        for name, form in tmp.items():
            seq : torch.tensor
            seq = form[1](form[0].double())
            seq = seq.view(
                batch_size,
                seq.shape[1],
                self.heads,
                self.dims
            )
            tmp[name] = seq.permute(0,2,1,3)
            # batch, heads, title_size, dims

        Attention_matrix = torch.matmul(
            tmp['Q_seq'],
            tmp['V_seq'].transpose(3,2)
        )/self.heads**0.5 # batch, heads, title_size, title_size

        Attention_matrix = self.soft(Attention_matrix)
        Repsent = torch.matmul(Attention_matrix,
                               tmp['V_seq'])
        Repsent = Repsent.permute(0, 2, 1, 3).contiguous()
        Repsent = Repsent.view(batch_size,
                               Repsent.shape[1],
                               self.output_dim)
        # Repsent: (batch, title, k*dim)
        return Repsent



class NRMS_News_Encoder(BasicLayer):
    """
        Take: 
            (batch, title_size, word_emb)
        Out : 
            (batch, k*dim)
    
    """
    def __init__(self,
                 param : params):
        super(NRMS_News_Encoder, self).__init__()
        self.param = param
        self.init_weight()

    def init_weight(self):
        inter_dim = self.param.head_num * self.param.head_dim
        self.selfattention = SelfAttention(self.param)
        self.additive = AttLayer2(inter_dim, self.param)
        self.drop = nn.Dropout(p=self.param.dropout)

    def forward(self, x):
        # x : (batch, title, word_dim)
        x = self.drop(x)
        x = self.selfattention(x)
        x = self.drop(x)
        x = self.additive(x)
        # x: (batch, head*num)
        return x


class NRMS_User_Encoder(BasicLayer):
    """
        Take: (batch, history, k*dim)
        Out : (batch, k*dim)
    """
    def __init__(self,
                 param: params):
        super(NRMS_User_Encoder, self).__init__()
        self.param = param
        self.init_weight()

    def init_weight(self):
        user_params = params(
            word_emb_dim = self.param.head_num * self.param.head_dim,
            head_num  = self.param.head_num,
            head_dim  = self.param.head_dim
        )
        inter_dim = self.param.head_num * self.param.head_dim
        self.selfattention = SelfAttention(user_params)
        self.additive = AttLayer2(inter_dim, self.param)
        self.drop = nn.Dropout(p=self.param.dropout)

    def forward(self, x):
        # x : (batch, history, k*head)
        x = self.drop(x)
        x = self.selfattention(x)
        x = self.drop(x)
        x = self.additive(x)

        return x

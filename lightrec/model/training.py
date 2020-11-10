"""Including
- manage hyperparameters
- sample
- testing for certain methods
- pretty report
...
"""
import yaml
import torch
import numpy as np
from tabulate import tabulate
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
########################################################
# evaluation
########################################################
# Adapted from recommenders/reco_utils/recommedner/deeprec/deeprec_utils.py
def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean([
                mrr_score(each_labels, each_preds)
                for each_labels, each_preds in zip(labels, preds)
            ])
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean([
                    ndcg_score(each_labels, each_preds, k)
                    for each_labels, each_preds in zip(labels, preds)
                ])
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean([
                    hit_score(each_labels, each_preds, k)
                    for each_labels, each_preds in zip(labels, preds)
                ])
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean([
                roc_auc_score(each_labels, each_preds)
                for each_labels, each_preds in zip(labels, preds)
            ])
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res
########################################################
# hyperparameters manager
########################################################
class params:
    """Manager for hyper parameters
    """
    def __init__(self, for_model=None, file=None, **kwargs):
        self._dict = {}
        self._dict.update(kwargs)
        self._model = for_model or "General"
        if file is not None:
            self.open(file)

    def open(self, yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = flat_config(config)
            self._dict.update(config)

    def check_constrains(self, types):
        """Check this params instance satisfy the need of certain model

        Args:
            types (dict): key: name of hyperparameter, value: type
        Raises:
            KeyError
            TypeError
        """
        for name, Type in types.items():
            if name not in self._dict:
                raise KeyError(
                    f"Expect '{name}'' from {self._model} params, but not founded."
                )
            elif not isinstance(self._dict[name], Type):
                raise TypeError(f"Expect '{name}' from {self._model} params to be {Type}, " \
                                f"but got '{type(self._dict[name])}'")

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, index):
        return self._dict.get(index, None)

    def __repr__(self):
        return dict2table(self._dict, headers="row")


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    
    Args:
        config (dict): Configuration loaded from a yaml file.
    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config

def dict2table(table: dict, headers='row'):
    if headers == 'row':
        tab = []
        for key in sorted(list(table)):
            tab.append([key, table[key]])
        return tabulate(tab)
    elif headers == 'firstrow':
        head = []
        data = []
        for key in sorted(list(table)):
            head.append(key)
            data.append(table[key])
        return tabulate([head, data], headers='firstrow', floatfmt=".4f")

########################################################
# utils
########################################################
class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def TO(*tensors, **kwargs):
    if kwargs.get("device"):
        device = torch.device(kwargs['device'])
    else:
        device = torch.device('cpu')
    results = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        results.append(tensor.to(device))
    return results
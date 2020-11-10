import torch
from torch.nn import Module


class BasicModel(Module):
    """The base of model
    Three inference should be satisfied
        offer_constrains: optional, offer api to check types
        loss: design loss function
        evaluate: design eval methods
    """
    def __init__(self, name=None):
        super(BasicModel, self).__init__()
        self.device = torch.device('cpu') # default device
        self._name = name or self.__class__.__name__

    def to(self, device: torch.cuda.device):
        self.device = device
        return self.to(device)

    def offer_constarins(self):
        """offer hyperparameters type constrains
        
        Return:
            dict: key: name of param, value: type
        """
        return {}

    def loss(self, *args, **kwargs):
        raise NotImplementedError(
            f"model {self._name} not implemented loss function")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            f"model {self._name} not implemented evaluation function")


class BasicLayer(Module):
    """The base of layer, wrapped functions operation
    """
    def __init__(self, name=None):
        super(BasicLayer, self).__init__()
        self._name = name or self.__class__.__name__

    def to(self, device: torch.cuda.device):
        self.device = device
        return self.to(device)

    def offer_constarins(self):
        return {}
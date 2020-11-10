"""Including
- manage hyperparameters
- sample
- testing for certain methods
- pretty report
...
"""
import yaml
from tabulate import tabulate

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
        results.append(tensor.to(device))
    return results
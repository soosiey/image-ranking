from torch import nn


class GeneralModel(nn.Module):
    def __init__(self):
        super(GeneralModel, self).__init__()
        self._name = None

    @property
    def name(self):
        if self._name is None:
            raise Exception("Name not set")
        return self._name

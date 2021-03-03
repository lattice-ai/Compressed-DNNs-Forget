""" Config Class """


import json

class Config:

    """
    Config class containing data, train and model hyperparameters
    """

    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """
        Creates config from json
        """

        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)


class HelperObject(object):
    """
    Simple Helper Object to convert json into Python Object
    """

    def __init__(self, dict_):
        self.__dict__.update(dict_)
import yaml


# NOTE: shamelessly stolen from https://github.com/jaywalnut310/glow-tts/blob/master/utils.py
# modified to accept yaml input instead of json
# converts dictionary to a convenient class instance

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def create_hparams(model_dir: str) -> HParams:
    with open(model_dir, "r") as f:
        config = yaml.safe_load(f)

    hparams = HParams(**config)

    return hparams



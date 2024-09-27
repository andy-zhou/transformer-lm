from .encoder import Encoder
from .splits import make_splits


def load(filename: str):
    with open(filename, "r") as f:
        data = f.read()
        encoder = Encoder(data)
        train, test = make_splits(data, encoder)

    return encoder, train, test

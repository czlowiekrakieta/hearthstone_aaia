import os
from os.path import join as pj

from random import sample
import numpy as np
import pandas as pd

from hearth.backend.config import TRAINING_GAMES, TRAINING_DECKS, BASE_PATH


def build_association_table(df):
    """
    will be used to determining what split is optimal, i.e. does not lose too much data

    does absolutely nothing, placeholder
    :param df:
    :return:
    """
    pass


def load_and_split(training_decks=None, test_decks=None, split_ratio=.7):
    """

    :param training_decks:
    :param test_decks:
    :param float split_ratio:
    :return df of matches, training idx, test idx:
    """
    decks = open(pj(BASE_PATH, "trainingDecks.json"), 'r').read().split('\n')
    decks = list(map(eval, decks[:-1]))
    decks = {x['deckName'][0]: x for x in decks}

    N = int(len(decks) * split_ratio)
    if training_decks is None:
        training_decks = sample(list(decks.keys()), N)

    if test_decks is None:
        test_decks = set(decks) - set(training_decks)

    if set(training_decks) & set(test_decks):
        raise ValueError("Training and test set have to be mutually exclusive. "
                         "If you are not sure why it did not happen, just "
                         "let me build them on my own, without passing "
                         "training_decks and test_decks arguments")

    df = pd.read_csv(TRAINING_GAMES, sep=";", header=None)
    train_idx_series = df[2].isin(training_decks) | df[4].isin(training_decks)
    test_idx_series = df[2].isin(test_decks) | df[4].isin(test_decks)

    train_idx = np.where(train_idx_series & ~test_idx_series)
    test_idx = np.where(test_idx_series & ~train_idx_series)

    return df, train_idx, test_idx
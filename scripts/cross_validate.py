import os
from os.path import join as pj

from random import sample
import numpy as np
import pandas as pd

BASE_PATH = os.environ['HEARTHSTONE_BASE']


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
    decks = open(pj(BASE_PATH, 'trainingDecks.json'), 'r').read().split('\n')
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

    df = pd.read_csv(pj(BASE_PATH, 'training_games.csv'), sep=';', header=None)
    train_idx = df[1].isin(training_decks) | df[3].isin(training_decks)
    test_idx = df[1].isin(test_decks) | df[3].isin(test_decks)

    train_idx = np.where(train_idx & ~test_idx)
    test_idx = np.where(test_idx & ~train_idx)

    return df, train_idx, test_idx
from hearth.backend.cross_validate import load_and_split
from hearth.backend.utils import build_full_cards_and_heroes_list, vectorize_deck, get_card_idx_map
from hearth.backend.config import TRAINING_GAMES, DECKS_JSON

import numpy as np
import pandas as pd
import json
from os.path import exists

from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.ensemble import GradientBoostingClassifier

N_SPLITS = 5


def main(model_type='gbc', layers=None, dropout=None,
         player_one_file=None, player_two_file=None, fraction=0.25):
    decks_dicts = []

    if not 0 < fraction <= 1.:
        raise ValueError("fraction has to be between 0 and 1")

    f = open(DECKS_JSON, "r")

    for r in f.readlines():
        decks_dicts.append(json.loads(r))

    decks_named_dicts = {x['deckName'][0]: x for x in decks_dicts}

    def include_bot(bot_deck_str):
        bot, deck = bot_deck_str.split('_')
        z = {'bot': bot}
        z.update(decks_named_dicts[deck])
        return z

    cards, heroes = build_full_cards_and_heroes_list(DECKS_JSON)
    card_map = get_card_idx_map(cards)
    hero_map = get_card_idx_map(heroes)  # misleading name
    #     decks_named_dicts =

    # loads battle file
    battles_df = pd.read_csv(TRAINING_GAMES, sep=";", header=None)
    if not exists(player_one_file):
        battles_df['bot_with_deck_p1'] = battles_df[1] + '_' + battles_df[2]
        p1_arr = battles_df['bot_with_deck_p1'].apply(include_bot).apply(vectorize_deck,
                                                                         args=(card_map, hero_map),
                                                                         one_hot=False)
        p1_arr = np.array(p1_arr)
        p1_arr = csr_matrix(p1_arr)
        save_npz(player_one_file, p1_arr)
    else:
        print("READING: ", player_one_file)
        p1_arr = load_npz(player_one_file)

    if not exists(player_two_file):
        battles_df['bot_with_deck_p2'] = battles_df[3] + '_' + battles_df[4]
        p2_arr = battles_df['bot_with_deck_p2'].apply(include_bot).apply(vectorize_deck,
                                                                         args=(card_map, hero_map),
                                                                         one_hot=False)
        p2_arr = np.array(p2_arr)
        p2_arr = csr_matrix(p2_arr)
        save_npz(player_two_file, p2_arr)
    else:
        print("READING: ", player_two_file)
        p2_arr = load_npz(player_one_file)

    p1_arr, p2_arr, y = np.array(p1_arr.todense()), np.array(p2_arr.todense()), battles_df[5]
    if fraction < 1:
        idx = np.random.permutation(p1_arr.shape[0])
        p1_arr = p1_arr[idx, :]
        p2_arr = p2_arr[idx, :]

    y = (y == 'PLAYER_1 WON').astype(int)
    y = y.as_matrix()

    probas = []
    ys = []

    for n in range(N_SPLITS):
        print(n)
        train, test = load_and_split()[1:]
        p1_train = p1_arr[train]
        p2_train = p2_arr[train]

        p1_test = p1_arr[test]
        p2_test = p2_arr[test]

        X_train_ord = np.hstack((p1_train, p2_train))
        X_train_rev = np.hstack((p2_train, p1_train))

        X_test_ord = np.hstack((p1_test, p2_test))
        X_test_rev = np.hstack((p2_test, p1_test))

        y_train = y[train]
        y_train = np.hstack((y_train, 1-y_train))

        y_test = y[test]
        y_test = np.hstack((y_test, 1-y_test))

        X_train = np.vstack((X_train_ord, X_train_rev))
        X_test = np.vstack((X_test_ord, X_test_rev))

        if model_type == 'gbc':
            model = GradientBoostingClassifier().fit(X_train, y_train)
        elif model_type == 'siamese':
            from hearth.backend.siamese import build_siam
            p = X_train.shape[1] // 2
            model = build_siam(layers=layers, dropout=dropout, input_shape=p)
            Xtr = (X_train[:, :p], X_train[:, p:])
            X_test = (X_test[:, :p], X_test[:, p:])

            model.fit(Xtr, y_train)
        else:
            raise NameError("model_type should be either gbc or siamese")

        probas.append(model.predict_proba(X_test))
        ys.append(y_test)

    return probas, ys

#     for n in range(N_SPLITS):
#         tr = all_decks[offset:offset+fold_size]
#         offset += fold_size

#         df, train, test = load_and_split()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='siamese')
parser.add_argument('--layers', type=int, nargs='+', default=[100, 200])
parser.add_argument('--dropout', type=float, nargs='+', default=[.1, .1])
parser.add_argument('--player_one_file', type=str, required=True)
parser.add_argument('--player_two_file', type=str, required=True)
parser.add_argument('--fraction', type=float, default=0.25)

if __name__ == '__main__':
    probas, ys = main(**vars(parser.parse_known_args()[0]))
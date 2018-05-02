from hearth.backend.cross_validate import load_and_split
from hearth.backend.utils import build_full_cards_and_heroes_list, vectorize_deck, get_card_idx_map
from hearth.backend.config import TRAINING_GAMES, DECKS_JSON

import numpy as np
import pandas as pd
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

N_SPLITS = 5


def main():
    decks_dicts = []

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
    battles_df['bot_with_deck_p1'] = battles_df[1] + '_' + battles_df[2]
    p1_arr = battles_df['bot_with_deck_p1'].apply(include_bot).apply(vectorize_deck,
                                                                     args=(card_map, hero_map),
                                                                     one_hot=False)
    battles_df['bot_with_deck_p2'] = battles_df[3] + '_' + battles_df[4]
    print("player one vectorized")
    p2_arr = battles_df['bot_with_deck_p2'].apply(include_bot).apply(vectorize_deck,
                                                                     args=(card_map, hero_map),
                                                                     one_hot=False)
    print("player two vectorized")
    p1_arr, p2_arr, y = np.array(p1_arr.tolist()), np.array(p2_arr.tolist()), battles_df[5]

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

        model = GradientBoostingClassifier().fit(X_train, y_train)
        probas.append(model.predict_proba(X_test))
        ys.append(y_test)

    return probas, ys

#     for n in range(N_SPLITS):
#         tr = all_decks[offset:offset+fold_size]
#         offset += fold_size

#         df, train, test = load_and_split()


if __name__ == '__main__':
    probas, ys = main()
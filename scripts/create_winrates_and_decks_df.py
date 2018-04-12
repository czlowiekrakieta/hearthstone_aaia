import os
from os.path import join as pj
from functools import reduce

import pandas as pd
import numpy as np


BASE_PATH = os.environ['HEARTHSTONE_BASE']

df = pd.read_csv(pj(BASE_PATH, 'training_games.csv'), sep=';', header=None)

decks = open(pj(BASE_PATH, 'trainingDecks.json'), 'r').read().split('\n')
decks = list(map(eval, decks[:-1]))
decks = {x['deckName'][0]: x for x in decks}

decks_with_bots = [b + '_' + d for d in decks for b in ['A1', 'A2', 'B1', 'B2']]
df['player_one'] = df[1] + '_' + df[2]
df['player_two'] = df[3] + '_' + df[4]

df_winrates = pd.DataFrame(columns=['deck', 'bot', 'win_count', 'battle_count'], index=decks_with_bots)

for i, db in enumerate(decks_with_bots):
    rows = df.iloc[np.where(df.player_one == db)[0], :]
    battle_count = rows.shape[0]
    win_count = (rows[5] == 'PLAYER_0 WON').sum()

    rows = df.iloc[np.where(df.player_two == db)[0], :]
    battle_count += rows.shape[0]
    win_count += (rows[5] == 'PLAYER_1 WON').sum()

    bot, deck = db.split('_')
    df_winrates.loc[db, :] = [deck, bot, win_count, battle_count]

    if i % 20 == 0:
        print("DONE: ", i, "FROM: ", len(decks_with_bots))

df_winrates['groups'] = df_winrates['deck'].map(dict(zip(decks.keys(),
                                                         range(len(decks)))))
df_winrates.reset_index().to_csv(pj(BASE_PATH, "decks_winrates.csv"), index=False)

uniq_cards = list(set(reduce(lambda x, y: x + list(y['cards'].keys()), decks.values(), [])))
card_count = len(uniq_cards)

map_card_idx = dict(zip(uniq_cards, range(len(uniq_cards))))
deck_card_df = pd.DataFrame(columns=['hero'] + uniq_cards, index=decks.keys())

for dname, properties in decks.items():
    row = np.zeros(card_count)
    for c, cnt in properties['cards'].items():
        row[map_card_idx[c]] += cnt[0]
    deck_card_df.loc[dname, uniq_cards] = row
    deck_card_df.loc[dname, 'hero'] = properties['hero'][0]

deck_card_df['deck'] = deck_card_df.index
deck_card_df.reset_index()
deck_card_df.to_csv(pj(BASE_PATH, 'decks_with_cards.csv'), index=False)
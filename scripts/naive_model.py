import os
import pandas as pd

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_validate

BASE = os.environ['HEARTHSTONE_BASE']

df = pd.read_csv(os.path.join(BASE, 'decks_with_cards.csv')).\
    merge(pd.read_csv(os.path.join(BASE, 'decks_winrates.csv')), on='deck')

df['win_ratio'] = df.win_count / df.battle_count
df.drop(['win_count', 'battle_count'], axis=1, inplace=True)
df = pd.concat((df, pd.get_dummies(df['hero']), pd.get_dummies(df['bot'])), axis=1).drop(['hero', 'bot'], axis=1)

groups = df.pop('groups')
groups = df['deck']

X, y = df.drop(['deck', 'index', 'win_ratio'], axis=1), df['win_ratio']
model = ElasticNetCV(l1_ratio=[.25, .5, .75],
                     cv=10)

print(cross_validate(model, X, y, groups=groups, n_jobs=4, verbose=10, scoring=['r2', 'neg_mean_squared_error']))
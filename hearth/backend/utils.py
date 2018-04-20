import json
import numpy as np


def build_full_cards_and_heroes_list(path: str):
    """

    :param str path:
    :return:
    """

    cards = []
    heroes = []
    file = open(path, "r")
    for r in file.readlines():
        this_cards = json.loads(r)
        cards.extend(this_cards['cards'].keys())
        heroes.append(this_cards['hero'][0])

    return set(cards), set(heroes)


def get_card_idx_map(*args):
    all_cards = []
    for l in args:
        all_cards.extend(l)
    return dict(zip(all_cards, range(len(all_cards))))


def vectorize_deck(deck_dict: dict, card_idx_map: dict,
                   hero_mapping: dict, N = None, H = None,
                   one_hot=True, bot=False):
    cards = deck_dict['cards']
    bmap = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3}
    bot = deck_dict['bot'] if 'bot' in deck_dict else False

    if bot and one_hot:
        B = 4
        bots = np.zeros(4)
        bots[bmap[bot]] = 1
    elif bot:
        B = 1
        bots = np.array(bmap[bot])
    else:
        B = 0

    if N is None:
        N = len(card_idx_map)

    if H is None and one_hot:
        H = len(hero_mapping)
        hero = np.zeros(H)
        hero[hero_mapping[deck_dict['hero'][0]]] = 1
    elif not one_hot:
        hero = np.array(hero_mapping[deck_dict['hero'][0]])
        H = 1
    else:
        raise ValueError

    vec = np.zeros(N+H+B)
    vec[-H:-B] = hero

    if bot:
        vec[-B:] = bots

    for card_name, card_count in cards.items():
        vec[card_idx_map[card_name]] = card_count[0]

    return vec
import os

BASE_PATH = os.environ['HEARTHSTONE_BASE']
TRAINING_DECKS = os.path.join(BASE_PATH, "decks_with_cards.csv")
TRAINING_GAMES = os.path.join(BASE_PATH, "training_games.csv")
DECKS_JSON = os.path.join(BASE_PATH, "trainingDecks.json")
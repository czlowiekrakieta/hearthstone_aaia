from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K


def build_leg(layers=None, dropout=None, input_shape=300):
    if layers is None and dropout is None:
        layers = [300, 300]
        dropout = [.5, .5]

    if len(layers) != len(dropout):
        raise ValueError

    if isinstance(input_shape, int):
        input_shape = (input_shape, )

    leg = Sequential()
    leg.add(Dense(layers[0], activation='relu', input_shape=input_shape))
    leg.add(Dropout(dropout[0]))
    for l, d in zip(layers[1:], dropout[1:]):
        leg.add(Dense(l, activation='relu'))
        leg.add(Dropout(d))
    return leg


def build_siam(layers=None, dropout=None, input_shape=300):
    leg = build_leg(layers=layers, dropout=dropout, input_shape=input_shape)

    i1, i2 = Input(shape=(input_shape,)), Input(shape=(input_shape,))
    l1, l2 = leg(i1), leg(i2)

    l1_distance_layer = Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([l1, l2])

    pred = Dense(1, activation='sigmoid')(l1_distance)
    model = Model(inputs=[i1, i2], outputs=pred)
    return model
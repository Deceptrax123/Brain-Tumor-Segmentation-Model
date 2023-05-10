import keras
from keras.losses import categorical_crossentropy
import numpy as np
from keras import backend as K


def dice_coef(y_true, y_pred):
    # ytrue_reshaped = keras.backend.flatten(y_true)
    # ypred_reshaped = keras.backend.flatten(y_pred)

    intersection = keras.backend.sum(
        y_true*y_pred, axis=[1, 2, 3])

    E = keras.backend.epsilon()
    dice = keras.backend.mean((2.*intersection+E)/(keras.backend.sum(y_true, axis=[1, 2, 3]) +
                                                   keras.backend.sum(y_pred, axis=[1, 2, 3])+E), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    dice_loss = 1-dice_coef(y_true, y_pred)
    return dice_loss


def overall_loss(y_true, y_pred):
    loss = (0.5*categorical_crossentropy(y_true, y_pred)) + \
        dice_loss(y_true, y_pred)
    return loss

import keras
from keras.losses import categorical_crossentropy


def dice_coef(y_true, y_pred, smooth=1e-4):
    intersection = keras.backend.sum(y_true*y_pred, axis=[0, 1, 2, 3])
    union = keras.backend.sum(
        y_true, axis=[0, 1, 2, 3])+keras.backend.sum(y_pred, axis=[0, 1, 2, 3])

    mean = keras.backend.mean((2.*intersection+smooth)/(union+smooth), axis=0)
    return mean


def dice_loss(y_true, y_pred):
    loss = 1-dice_coef(y_true, y_pred)
    return loss


def categorical_loss(y_true, y_pred):
    loss = (0.5*categorical_crossentropy(y_true, y_pred)) + \
        dice_loss(y_true, y_pred)
    return loss

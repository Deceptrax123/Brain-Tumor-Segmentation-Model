import keras
from keras.losses import categorical_crossentropy
import numpy as np
from keras import backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred):
    # ytrue_reshaped = keras.backend.flatten(y_true)
    # ypred_reshaped = keras.backend.flatten(y_pred)

    intersection = keras.backend.sum(
        y_true*y_pred, axis=[1, 2, 3, 4])

    E = keras.backend.epsilon()
    dice = keras.backend.mean((2.*intersection+E)/(keras.backend.sum(y_true, axis=[1, 2, 3, 4]) +
                                                   keras.backend.sum(y_pred, axis=[1, 2, 3, 4])+E), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    dice_loss = 1-dice_coef(y_true, y_pred)
    return dice_loss


def dice_coef_necrotic(y_true, y_pred):
    ytrue_f = keras.backend.batch_flatten(y_true[:, :, :, :, 1])
    ypred_f = keras.backend.batch_flatten(y_pred[:, :, :, :, 1])

    intersection = keras.backend.sum(ytrue_f*ypred_f, axis=1)
    union = keras.backend.sum(
        ytrue_f, axis=1)+keras.backend.sum(keras.backend.square(ypred_f), axis=1)
    E = keras.backend.epsilon()

    dice = keras.backend.mean((2.*intersection+E)/(union+E))

    return dice


def dice_coef_edema(y_true, y_pred):
    ytrue_f = keras.backend.batch_flatten(y_true[:, :, :, :, 2])
    ypred_f = keras.backend.batch_flatten(y_pred[:, :, :, :, 2])

    intersection = keras.backend.sum(ytrue_f*ypred_f, axis=1)
    union = keras.backend.sum(
        ytrue_f, axis=1)+keras.backend.sum(keras.backend.square(ypred_f), axis=1)
    E = keras.backend.epsilon()

    dice = keras.backend.mean((2.*intersection+E)/(union+E))

    return dice


def dice_coef_enhancing(y_true, y_pred):
    ytrue_f = keras.backend.batch_flatten(y_true[:, :, :, :, 3])
    ypred_f = keras.backend.batch_flatten(y_pred[:, :, :, :, 3])

    intersection = keras.backend.sum(ytrue_f*ypred_f, axis=1)
    union = keras.backend.sum(
        ytrue_f, axis=1)+keras.backend.sum(keras.backend.square(ypred_f), axis=1)
    E = keras.backend.epsilon()

    dice = keras.backend.mean((2.*intersection+E)/(union+E))

    return dice


def dice_coef_complete(y_true, y_pred):
    dice = 0
    for i in range(1, 4):
        dice = dice+dice_coef(y_true[:, :, :, :, i], y_pred[:, :, :, :, i])
    return (dice/3)


def dice_cosh_loss(ytrue, ypred):
    loss = tf.math.log1p(tf.math.cosh(1-(3*dice_coef_complete(ytrue, ypred))))

    return loss


def overall_loss(y_true, y_pred):
    loss = (0.5*categorical_crossentropy(y_true, y_pred)) + \
        dice_loss(y_true, y_pred)
    return loss

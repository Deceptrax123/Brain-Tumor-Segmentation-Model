import tensorflow as tf
import numpy as np
from keras import backend as K


class Complete_Dice_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        E = K.epsilon()
        dice = 0
        for i in range(1, 4):
            # ypred is a probability map->Convert to hard labels
            ytrue_channel, ypred_channel = y_true[:,
                                                  :, :, :, i], y_pred[:, :, :, :, i]

            ytrue_f = K.batch_flatten(ytrue_channel)
            ypred_f = K.batch_flatten(ypred_channel)

            intersection = K.sum(ypred_f*ytrue_f, axis=1)
            union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

            dice_channel = 1-K.mean((2.*intersection+E)/(union+E))
            dice = dice+dice_channel
        dice_logcosh = tf.math.log(tf.math.cosh(dice))
        return dice_logcosh

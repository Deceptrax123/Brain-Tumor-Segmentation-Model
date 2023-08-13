import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.losses import categorical_crossentropy


class Complete_Dice_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        E = 1
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


class Dice_cross_entropy(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        E = 1
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
        entropy = categorical_crossentropy(y_true, y_pred)

        loss = entropy+dice
        return loss


class CrossEntropyDiceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        regional_dice_loss = 0
        regional_cross_loss = 0
        E = 1
        for i in range(1, 4):
            ytrue_channel, ypred_channel = y_true[:, :,
                                                  :, :, i], y_pred[:, :, :, :, i]

            slices = ytrue_channel.shape[3]

            slice_losses = 0
            slice_dice = 0
            for j in range(slices):
                # cross entropy slice loss
                loss = K.mean(K.sum((ytrue_channel[:, :, :, j]*tf.math.log(ypred_channel[:, :, :, j]))+(
                    (1-ytrue_channel[:, :, :, j])*tf.math.log((1-ypred_channel[:, :, :, j]))), axis=0))
                slice_losses += loss

                # dice slice loss
                truth = K.batch_flatten(ytrue_channel[:, :, :, j])
                pred = K.batch_flatten(ypred_channel[:, :, :, j])

                intersection = K.sum(truth*pred, axis=1)
                union = K.sum(truth, axis=1)+K.sum(pred, axis=1)

                dice = 1-(2.*intersection)/(union+E)
                slice_dice += dice

            regional_cross_loss += (slice_losses/slices)
            regional_dice_loss += (slice_dice/slices)

        cross_entropy_loss = regional_cross_loss/3
        dice_loss = regional_dice_loss/3

        combined = cross_entropy_loss+dice_loss
        return combined

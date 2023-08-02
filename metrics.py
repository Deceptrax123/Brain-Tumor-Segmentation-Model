import tensorflow as tf
from keras import backend as K


class Complete_Dice_Coef(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef_complete', **kwargs):
        super(Complete_Dice_Coef, self).__init__(name=name, **kwargs)
        self.coef = self.add_weight(name='complete', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # convert probability map to one hot labels
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=4), depth=4)

        E = K.epsilon()
        dice = 0
        for i in range(1, 4):
            ytrue_channel, ypred_channel = y_true[:,
                                                  :, :, :, i], y_pred[:, :, :, :, i]

            ytrue_f = K.batch_flatten(ytrue_channel)
            ypred_f = K.batch_flatten(ypred_channel)

            intersection = K.sum(ypred_f*ytrue_f, axis=1)
            union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

            dice_channel = K.mean((2.*intersection+E)/(union+E))
            dice = dice+dice_channel
        self.coef.assign_add(dice)

    def result(self):
        return self.coef

    def reset_states(self):
        self.coef.assign(0.)


class Necrotic_Dice_Coef(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef_necrotic', **kwargs):
        super(Necrotic_Dice_Coef, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='necrotic', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert probability map to one hot labels
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=4), depth=4)

        ytrue_f = K.batch_flatten(y_true[:, :, :, :, 1])
        ypred_f = K.batch_flatten(y_pred[:, :, :, :, 1])

        intersection = K.sum(ytrue_f*ypred_f, axis=1)
        union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

        E = K.epsilon()

        dice = K.mean(2.*(intersection+E)/(union+E))

        self.loss.assign_add(dice)

    def result(self):
        return self.loss

    def reset_state(self):
        self.loss.assign(0.)


class Enhancing_Dice_Coef(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef_enhancing', **kwargs):
        super(Enhancing_Dice_Coef, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='enhancing', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert probability map to one hot labels
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=4), depth=4)

        ytrue_f = K.batch_flatten(y_true[:, :, :, :, 3])
        ypred_f = K.batch_flatten(y_pred[:, :, :, :, 3])

        intersection = K.sum(ytrue_f*ypred_f, axis=1)
        union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

        E = K.epsilon()

        dice = K.mean(2.*(intersection+E)/(union+E))

        self.loss.assign_add(dice)

    def result(self):
        return self.loss

    def reset_state(self):
        self.loss.assign(0.)


class Edema_Dice_Coef(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef_edema', **kwargs):
        super(Edema_Dice_Coef, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='edema', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert probability map to one hot labels
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=4), depth=4)

        ytrue_f = K.batch_flatten(y_true[:, :, :, :, 2])
        ypred_f = K.batch_flatten(y_pred[:, :, :, :, 2])

        intersection = K.sum(ytrue_f*ypred_f, axis=1)
        union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

        E = K.epsilon()

        dice = K.mean(2.*(intersection+E)/(union+E))

        self.loss.assign_add(dice)

    def result(self):
        return self.loss

    def reset_state(self):
        self.loss.assign(0.)


class Dice_coef(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef', **kwargs):
        super(Dice_coef, self).__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='dc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert probability map to one hot labels
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=4), depth=4)

        ytrue_f = K.batch_flatten(y_true)
        ypred_f = K.batch_flatten(y_pred)

        intersection = K.sum(ytrue_f*ypred_f, axis=1)
        union = K.sum(ytrue_f, axis=1)+K.sum(ypred_f, axis=1)

        E = K.epsilon()

        dice = K.mean(2.*(intersection+E)/(union+E))

        self.loss.assign_add(dice)

    def result(self):
        return self.loss

    def reset_states(self):
        self.loss.assign_add(0.)

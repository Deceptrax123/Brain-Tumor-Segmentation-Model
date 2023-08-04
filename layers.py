from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Dropout, MaxPooling3D, Input, BatchNormalization, Add, Concatenate, ConvLSTM2D, Bidirectional, Attention
import tensorflow as tf


class ConvBlockEnc(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, padding='same'):
        super(ConvBlockEnc, self).__init__()

        self.conv = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding,
                           kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')
        self.bn = BatchNormalization()

    def call(self, input, training=False):
        x = self.conv(input)
        x1 = self.bn(x)
        x2 = tf.nn.relu(x1)

        x3 = self.conv(x2)
        x4 = self.bn(x3)
        x5 = tf.nn.relu(x4)

        return x5


class ConvBlockDec(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, padding='same'):
        super(ConvBlockDec, self).__init__()
        self.deconv = Conv3DTranspose(filters=filters, kernel_size=kernel_size, padding=padding,
                                      kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')
        self.bn = BatchNormalization()

    def call(self, input, training=False):
        x = self.deconv(input)
        x = self.bn(x)
        x = tf.nn.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

        return x


class Lstm(tf.keras.layers.layer):
    def __init__(self, kernel_size, filters, padding='same'):
        super(Lstm, self).__init__()
        self.lstm = ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='tanh',
                               recurrent_activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')

    def call(self, input, training=False):
        x = self.lstm(input)

        return x

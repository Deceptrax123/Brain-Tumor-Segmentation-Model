from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
import keras


def model():
    input_layer = Input((128, 128, 128, 3))

    initializer = tf.keras.initializers.HeNormal(seed=0)
    # Encoding network
    conv_1 = Conv3D(filters=8, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=1, activation='relu')(input_layer)
    conv_2 = Conv3D(filters=16, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=2, activation='relu')(conv_1)
    conv_3 = Conv3D(filters=32, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=2, activation='relu')(conv_2)  # 30X30X30X32

    # Decoding network
    de_avg_1 = UpSampling3D(size=2)(conv_3)  # 60X60X60X32
    de_conv_1 = Conv3DTranspose(
        filters=16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer=initializer, strides=2, padding='same')(de_avg_1)

    de_conv_2 = Conv3DTranspose(filters=8, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, activation='relu', strides=1)(de_conv_1)

    de_conv_3 = Conv3DTranspose(filters=8, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, activation='relu', strides=1)(de_conv_2)
    de_conv_4 = Conv3DTranspose(filters=4, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, activation='relu', strides=1)(de_conv_3)
    de_conv_5 = Conv3DTranspose(filters=4, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, strides=1, activation='softmax')(de_conv_4)

    model = Model(inputs=input_layer, outputs=de_conv_5)

    return model

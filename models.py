from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D
from keras.layers import Dropout, Input, BatchNormalization
import tensorflow as tf
from keras.models import Model
import keras


def create():
    input_layer = Input((128, 128, 128, 3))

    initializer = tf.keras.initializers.HeNormal(seed=0)
    # Encoding network
    conv_1 = Conv3D(filters=8, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=2, activation='relu')(input_layer)
    conv_2 = Conv3D(filters=16, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=2, activation='relu')(conv_1)
    conv_3 = Conv3D(filters=32, kernel_size=(3, 3, 3),
                    kernel_initializer=initializer, strides=2, activation='relu')(conv_2)  # 30X30X30X32

    # Decoding network
    de_conv_1 = Conv3DTranspose(
        filters=32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer=initializer, strides=2)(conv_3)

    de_conv_2 = Conv3DTranspose(filters=16, kernel_size=(
        3, 3, 3), strides=2, kernel_initializer=initializer, activation='relu')(de_conv_1)

    de_conv_3 = Conv3DTranspose(filters=8, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, activation='relu', strides=2, padding='same')(de_conv_2)

    output_layer = Conv3DTranspose(filters=4, kernel_size=(
        3, 3, 3), kernel_initializer=initializer, activation='softmax', strides=1)(de_conv_3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

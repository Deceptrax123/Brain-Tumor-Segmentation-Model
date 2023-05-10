import tensorflow as tf
from keras.layers import Conv3D, Reshape, Conv3DTranspose, UpSampling3D, MaxPooling3D, Concatenate
from keras.layers import Dropout, Input, BatchNormalization, Dense, Flatten
from keras.models import Model
import keras


def create_2pathed():
    input_layer = Input((128, 128, 128, 3))

    # first encoding block
    # convoluted layers
    conv1_e_1 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv1_e_2 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_1)
    conv1_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_2)
    conv1_e_4 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_3)
    conv1_e_5 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_4)

    # down sampling by a double strided convolution
    pool1_e_7 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv1_e_4)

    # convoluted layers
    conv1_e_8 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(pool1_e_7)
    conv1_e_9 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_8)
    conv1_e_10 = Conv3D(filters=32, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_9)
    conv1_e_11 = Conv3D(filters=32, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_10)

    # second encoding block
    # convoluted layers
    conv2_e_1 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv2_e_2 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_1)
    conv2_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_2)
    conv2_e_4 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_3)
    conv2_e_5 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_4)

    # Pooling layer
    pool2_e_5 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv2_e_4)

    # convoluated layers
    conv2_e_5 = Conv3D(filters=16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(pool2_e_5)
    conv2_e_6 = Conv3D(filters=16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv2_e_5)
    conv2_e_7 = Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv2_e_6)
    conv2_e_8 = Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv2_e_7)

    # Concatenate both the encoding blocks
    concat_layer = Concatenate(axis=4)([conv1_e_11, conv2_e_8])

    # Combined convoluted layers
    conv_e_9 = Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                      strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(concat_layer)
    conv_e_10 = Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_9)
    conv_e_11 = Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_10)
    conv_e_12 = Conv3D(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_11)

    # Max pooling layer
    pool_e_13 = MaxPooling3D(pool_size=(2, 2, 2))(conv_e_12)

    # Reduce to 16X16X16
    conv_e_13 = Conv3D(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(pool_e_13)
    conv_e_14 = Conv3D(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_13)
    conv_e_15 = Conv3D(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal', activation='relu',
                       strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_14)

    # Decoder
    # Deconvolutions
    conv_d_1 = Conv3DTranspose(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_e_15)
    conv_d_2 = Conv3DTranspose(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_d_1)
    conv_d_3 = Conv3DTranspose(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_d_2)

    # Upsampling
    up_d_1 = UpSampling3D(size=(2, 2, 2))(conv_d_3)

    # Deconvolutions
    conv_d_4 = Conv3DTranspose(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(up_d_1)
    conv_d_5 = Conv3DTranspose(filters=256, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_d_4)
    conv_d_6 = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_d_5)
    conv_d_7 = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', strides=1, use_bias=True, padding='valid', bias_initializer='he_normal')(conv_d_6)

    # deconvolve the 2pathed block
    # first decoding block
    conv_d_8 = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_7)
    conv_d_9 = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                               activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_8)
    conv_d_10 = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_9)
    conv_d_11 = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_10)

    up_d_2 = UpSampling3D(size=(2, 2, 2))(conv_d_11)

    conv_d_12 = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(up_d_2)
    conv_d_13 = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_12)
    conv_d_14 = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_13)
    conv_d_15 = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), kernel_initializer='he_normal',
                                activation='relu', use_bias=True, bias_initializer='he_normal')(conv_d_14)

    # shape obtained,flatten and dense
    flat = Flatten()(conv_d_15)

    models = Model(inputs=input_layer, outputs=flat)
    return models

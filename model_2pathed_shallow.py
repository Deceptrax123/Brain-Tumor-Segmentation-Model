import tensorflow as tf
from keras.layers import Conv3D, Reshape, Conv3DTranspose, UpSampling3D, MaxPooling3D, Concatenate, Input
from keras.layers import BatchNormalization, Dropout, Dense, Flatten, Attention, Add
from keras.models import Model
import keras


def create_2pathed_shallow():
    input_layer = Input((128, 128, 128, 3))

    # first encoding block
    # Convolution layers
    conv1_e_1 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv1_e_2 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_1)
    conv1_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_2)

    # Downsample
    pool1_e_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv1_e_3)

    # Convolution Layers
    conv1_e_4 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(pool1_e_1)
    conv1_e_5 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_4)
    conv1_e_6 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), strides=1, activation='relu', use_bias=True, bias_initializer='he_normal')(conv1_e_5)

    # Second encoding block
    # Convolutions
    conv2_e_1 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv2_e_2 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_1)
    conv2_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_2)

    # Pooling
    pool2_e_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv2_e_3)

    # Convolutions
    conv2_e_4 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(pool2_e_1)
    conv2_e_5 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_4)
    conv2_e_6 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_5)

    # Concatenate
    path_combined = Concatenate(axis=4)([conv1_e_6, conv2_e_6])

    # Combined convolutions
    conv_7 = Conv3D(filters=32, kernel_initializer='he_normal', kernel_size=(3, 3, 3), activation='relu',
                    strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(path_combined)
    conv_8 = Conv3D(filters=64, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv_7)
    conv_9 = Conv3D(filters=128, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv_8)

    # Deconvolution
    d_conv = Conv3DTranspose(filters=128, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, use_bias=True, bias_initializer='he_normal')(conv_9)
    d_conv_1 = Conv3DTranspose(filters=64, kernel_initializer='he_normal', activation='relu',
                               strides=1, kernel_size=(3, 3, 3), padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv)
    d_conv_2 = Conv3DTranspose(filters=32, kernel_initializer='he_normal', activation='relu',
                               strides=1, kernel_size=(3, 3, 3), padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv_1)

    # first decoding block
    dconv1_e_1 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv_2)
    dconv1_e_2 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_1)
    dconv1_e_3 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_2)

    upsample_e_1 = UpSampling3D(size=(2, 2, 2))(dconv1_e_3)

    dconv1_e_4 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(upsample_e_1)
    dconv1_e_5 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_4)
    dconv1_e_6 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_5)

    # reconstruct 128X128X128X4
    recon = Conv3DTranspose(filters=4, kernel_initializer='glorot_normal', kernel_size=(
        3, 3, 3), activation='softmax', strides=1, padding='valid', use_bias=True, bias_initializer='glorot_normal')(dconv1_e_6)

    model = Model(inputs=input_layer, outputs=recon)
    return model

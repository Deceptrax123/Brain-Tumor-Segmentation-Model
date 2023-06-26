import tensorflow as tf
from keras.layers import Conv3D, Reshape, Conv3DTranspose, UpSampling3D, MaxPooling3D, Concatenate, ConvLSTM2D, Bidirectional
from keras.layers import Dropout, Input, BatchNormalization, Dense, Flatten, Reshape, Add
from keras.models import Model
import keras


def two_path_bi_lstm():
    input_layer = Input((128, 128, 128, 3))
    # first encoding block
    # Convolution layers
    conv1_e_1 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv1_e_2 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_1)
    conv1_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_2)
    conv1_e_4 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_3)
    conv1_e_5 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_4)

    # Downsample
    pool1_e_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv1_e_5)

    # Convolution Layers
    conv1_e_6 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(pool1_e_1)
    conv1_e_7 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_6)
    conv1_e_8 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), strides=1, activation='elu', use_bias=True, bias_initializer='he_normal')(conv1_e_7)
    conv1_e_9 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_8)
    conv1_e_10 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv1_e_9)

    # Batch normalization to get mean 0 activations
    batch1 = BatchNormalization()(conv1_e_10)

    # Second encoding block
    # Convolutions
    conv2_e_1 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(input_layer)
    conv2_e_2 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_1)
    conv2_e_3 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_2)
    conv2_e_4 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_3)
    conv2_e_5 = Conv3D(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_4)

    # Pooling
    pool2_e_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(conv2_e_5)

    # Convolutions
    conv2_e_6 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(pool2_e_1)
    conv2_e_7 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_6)
    conv2_e_8 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_7)
    conv2_e_9 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_8)
    conv2_e_10 = Conv3D(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv2_e_9)

    batch2 = BatchNormalization()(conv2_e_10)

    # Concatenate
    path_combined = Concatenate(axis=4)([batch1, batch2])

    # Combined convolutions
    conv_7 = Conv3D(filters=32, kernel_initializer='he_normal', kernel_size=(3, 3, 3), activation='elu',
                    strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(path_combined)
    conv_8 = Conv3D(filters=64, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv_7)
    conv_9 = Conv3D(filters=128, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(conv_8)

    batch3 = BatchNormalization()(conv_9)

    # Bidirectional-lstm block
    bi_dir_1 = Bidirectional(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', kernel_initializer='glorot_normal',
                                        recurrent_activation='hard_sigmoid', recurrent_initializer='he_normal', padding='same', use_bias=True, bias_initializer='glorot_normal', return_sequences=True))(batch3)
    bi_dir_2 = Bidirectional(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', kernel_initializer='glorot_normal',
                                        recurrent_activation='hard_sigmoid', recurrent_initializer='he_normal', padding='same', use_bias=True, bias_initializer='glorot_normal', return_sequences=True))(batch3)

    # Combined layer(attention)
    bidir_comb = Add()([bi_dir_1, bi_dir_2])

    # Deconvolution
    d_conv = Conv3DTranspose(filters=128, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, use_bias=True, bias_initializer='he_normal')(bidir_comb)
    d_conv_1 = Conv3DTranspose(filters=64, kernel_initializer='he_normal', activation='elu',
                               strides=1, kernel_size=(3, 3, 3), padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv)
    d_conv_2 = Conv3DTranspose(filters=32, kernel_initializer='he_normal', activation='elu',
                               strides=1, kernel_size=(3, 3, 3), padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv_1)

    # first decoding block
    dconv1_e_1 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(d_conv_2)
    dconv1_e_2 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_1)
    dconv1_e_3 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_2)
    dconv1_e_4 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_3)
    dconv1_e_5 = Conv3DTranspose(filters=16, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_4)

    upsample_e_1 = UpSampling3D(size=(2, 2, 2))(dconv1_e_5)

    dconv1_e_6 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(upsample_e_1)
    dconv1_e_7 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_6)
    dconv1_e_8 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_7)
    dconv1_e_9 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_8)
    dconv1_e_10 = Conv3DTranspose(filters=8, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='elu', strides=1, padding='valid', use_bias=True, bias_initializer='he_normal')(dconv1_e_9)

    batchnorm = BatchNormalization()(dconv1_e_10)

    # Feature learning feature mapping
    fl1 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', padding='same', strides=1, use_bias=True, bias_initializer='he_normal')(batchnorm)
    fl1_batch = BatchNormalization()(fl1)

    fl2 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', padding='same', strides=1, use_bias=True, bias_initializer='he_normal')(fl1_batch)
    fl2_batch = BatchNormalization()(fl2)

    fl3 = Conv3D(filters=4, kernel_initializer='he_normal', kernel_size=(
        3, 3, 3), activation='relu', padding='same', strides=1, use_bias=True, bias_initializer='he_normal')(fl2_batch)
    fl3_batch = BatchNormalization()(fl3)

    mask = Conv3D(filters=4, kernel_initializer='glorot_normal', kernel_size=(
        3, 3, 3), activation='softmax', strides=1, padding='same', use_bias=True, bias_initializer='glorot_normal')(fl3_batch)

    model = Model(inputs=input_layer, outputs=mask)
    return model

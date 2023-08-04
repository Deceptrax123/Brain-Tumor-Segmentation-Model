from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Dropout, MaxPooling3D, Input, BatchNormalization, Add, Concatenate, ConvLSTM2D, Bidirectional, Attention
import tensorflow as tf
from keras.models import Model


def ResidualBlockEnc(input_layer, no_of_filters):

    conv_1 = Conv3D(filters=no_of_filters, kernel_size=(3, 3, 3), padding="same", activation="relu",
                    use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(input_layer)
    BN_1 = BatchNormalization()(conv_1)
    conv_2 = Conv3D(filters=no_of_filters, kernel_size=(3, 3, 3), padding="same", activation="relu",
                    use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(BN_1)
    BN_2 = BatchNormalization()(conv_2)
    conv_skip = Conv3D(filters=no_of_filters, kernel_size=(
        1, 1, 1), bias_initializer="he_normal", kernel_initializer="he_normal")(input_layer)

    add_layer = Add()([BN_2, conv_skip])
    act = Activation(tf.keras.activations.relu)(add_layer)
    act = BatchNormalization()(act)

    return act


def ResidualBlockDec(input_layer, no_of_filters):

    deconv_1 = Conv3DTranspose(filters=no_of_filters, kernel_size=(3, 3, 3), padding="same", activation="relu",
                               use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(input_layer)
    BN_1 = BatchNormalization()(deconv_1)
    deconv_2 = Conv3DTranspose(filters=no_of_filters, kernel_size=(3, 3, 3), padding="same", activation="relu",
                               use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(BN_1)
    BN_2 = BatchNormalization()(deconv_2)
    deconv_skip = Conv3DTranspose(filters=no_of_filters, kernel_size=(
        1, 1, 1), bias_initializer="he_normal", kernel_initializer="he_normal")(input_layer)

    add_layer = Add()([BN_2, deconv_skip])
    act = Activation(tf.keras.activations.relu)(add_layer)
    act = BatchNormalization()(act)

    return act


def encoder(input_layer):

    res_block_1 = ResidualBlockEnc(input_layer, no_of_filters=8)
    res_block_2 = ResidualBlockEnc(res_block_1, no_of_filters=8)

    down_1 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_2)

    res_block_3 = ResidualBlockEnc(down_1, no_of_filters=16)
    res_block_4 = ResidualBlockEnc(res_block_3, no_of_filters=16)

    down_2 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_4)

    res_block_5 = ResidualBlockEnc(down_2, no_of_filters=32)
    res_block_6 = ResidualBlockEnc(res_block_5, no_of_filters=32)

    down_3 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_6)

    res_block_7 = ResidualBlockEnc(down_3, no_of_filters=64)
    res_block_8 = ResidualBlockEnc(res_block_7, no_of_filters=64)

    down_4 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_8)

    res_block_9 = ResidualBlockEnc(down_4, no_of_filters=128)
    res_block_10 = ResidualBlockEnc(res_block_9, no_of_filters=128)

    down_5 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_10)

    res_block_11 = ResidualBlockEnc(down_5, no_of_filters=256)
    res_block_12 = ResidualBlockEnc(res_block_11, no_of_filters=256)

    down_6 = MaxPooling3D(pool_size=(2, 2, 2))(res_block_12)

    res_block_13 = ResidualBlockEnc(down_6, no_of_filters=512)
    res_block_14 = ResidualBlockEnc(res_block_13, no_of_filters=512)

    return res_block_14


def decoder(input_layer):

    up_0 = UpSampling3D(size=(2, 2, 2))(input_layer)

    res_block_0_0 = ResidualBlockDec(up_0, no_of_filters=256)
    res_block_0_0_1 = ResidualBlockDec(res_block_0_0, no_of_filters=256)

    up_1 = UpSampling3D(size=(2, 2, 2))(res_block_0_0_1)

    res_block_0 = ResidualBlockDec(up_1, no_of_filters=128)
    res_block_0_1 = ResidualBlockDec(res_block_0, no_of_filters=128)

    up_2 = UpSampling3D(size=(2, 2, 2))(res_block_0_1)

    res_block = ResidualBlockDec(up_2, no_of_filters=64)
    resblock = ResidualBlockDec(res_block, no_of_filters=64)

    up_3 = UpSampling3D(size=(2, 2, 2))(resblock)

    res_block_1 = ResidualBlockDec(up_3, no_of_filters=32)
    res_block_2 = ResidualBlockDec(res_block_1, no_of_filters=32)

    up_4 = UpSampling3D(size=(2, 2, 2))(res_block_2)

    res_block_3 = ResidualBlockDec(up_4, no_of_filters=16)
    res_block_4 = ResidualBlockDec(res_block_3, no_of_filters=16)

    up_5 = UpSampling3D(size=(2, 2, 2))(res_block_4)

    res_block_5 = ResidualBlockDec(up_5, no_of_filters=8)
    res_block_6 = ResidualBlockDec(res_block_5, no_of_filters=8)

    return res_block_6


def lstms(input_layer):
    bi_dir_1 = Bidirectional(ConvLSTM2D(filters=256, kernel_size=(3, 3), activation='tanh', kernel_initializer='glorot_normal',
                                        recurrent_activation='hard_sigmoid', recurrent_initializer='glorot_normal', padding='same', use_bias=True, bias_initializer='glorot_normal', return_sequences=True))(input_layer)

    return bi_dir_1


def convolution_blocks(input_layer):
    conv = Conv3D(filters=4, kernel_size=(3, 3, 3), padding='same', activation='relu',
                  kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')(input_layer)

    batchnorm = BatchNormalization()(conv)

    return batchnorm


def Convolution_enc(input_layer):
    conv = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same', activation='relu',
                  kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')(input_layer)

    batchnorm = BatchNormalization()(conv)

    return batchnorm


def make_model():
    input_layer = Input((128, 128, 128, 3))
    input_layer = BatchNormalization()(input_layer)
    # Encoding
    enc1 = encoder(input_layer)
    enc2 = encoder(input_layer)

    # Concatenate
    enc = Concatenate()([enc1, enc2])

    # Learn smaller volumetric regions
    ce1 = Convolution_enc(enc)
    ce2 = Convolution_enc(ce1)
    ce3 = Convolution_enc(ce2)
    ce4 = Convolution_enc(ce3)
    ce5 = Convolution_enc(ce4)

    # Attention to smaller regions
    att2 = Attention()([ce5, ce4])

    # Bilstm layer
    bilstm = lstms(att2)

    # Decoder
    dec1 = decoder(bilstm)

    combined_layer = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), padding="same", activation="relu",
                                     use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(dec1)
    combined_layer = BatchNormalization()(combined_layer)

    # Few convolution layers
    conv1 = convolution_blocks(combined_layer)
    conv2 = convolution_blocks(conv1)
    conv3 = convolution_blocks(conv2)

    output_layer = Conv3D(filters=4, kernel_size=(1, 1, 1), kernel_initializer='glorot_normal',
                          padding='same', activation='softmax', use_bias=True, bias_initializer='glorot_normal')(conv3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

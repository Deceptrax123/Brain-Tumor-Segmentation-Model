from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Dropout, Input, BatchNormalization, Add, Concatenate, ConvLSTM2D, Bidirectional
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

    return act


def encoder(input_layer):

    res_block_1 = ResidualBlockEnc(input_layer, no_of_filters=8)
    res_block_2 = ResidualBlockEnc(res_block_1, no_of_filters=8)

    down_1 = MaxPool3D(pool_size=(2, 2, 2))(res_block_2)

    res_block_3 = ResidualBlockEnc(down_1, no_of_filters=16)
    res_block_4 = ResidualBlockEnc(res_block_3, no_of_filters=16)

    down_2 = MaxPool3D(pool_size=(2, 2, 2))(res_block_4)

    res_block_5 = ResidualBlockEnc(down_2, no_of_filters=32)
    res_block_6 = ResidualBlockEnc(res_block_5, no_of_filters=32)

    down_3 = MaxPool3D(pool_size=(2, 2, 2))(res_block_6)

    res_block_7 = ResidualBlockEnc(down_3, no_of_filters=64)
    res_block_8 = ResidualBlockEnc(res_block_7, no_of_filters=64)

    down_4 = MaxPool3D(pool_size=(2, 2, 2))(res_block_8)

    return down_4


def decoder(input_layer):
    up_0 = UpSampling3D(size=(2, 2, 2))(input_layer)

    res_block_0 = ResidualBlockDec(up_0, no_of_filters=64)
    res_block_0_1 = ResidualBlockDec(res_block_0, no_of_filters=64)

    up_1 = UpSampling3D(size=(2, 2, 2))(res_block_0_1)

    res_block_1 = ResidualBlockDec(up_1, no_of_filters=32)
    res_block_2 = ResidualBlockDec(res_block_1, no_of_filters=32)

    up_2 = UpSampling3D(size=(2, 2, 2))(res_block_2)

    res_block_3 = ResidualBlockDec(up_2, no_of_filters=16)
    res_block_4 = ResidualBlockDec(res_block_3, no_of_filters=16)

    up_3 = UpSampling3D(size=(2, 2, 2))(res_block_4)

    res_block_5 = ResidualBlockDec(up_3, no_of_filters=8)
    res_block_6 = ResidualBlockDec(res_block_5, no_of_filters=8)

    return res_block_6


def lstms(input_layer):
    bi_dir_1 = Bidirectional(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', kernel_initializer='glorot_normal',
                                        recurrent_activation='hard_sigmoid', recurrent_initializer='glorot_normal', padding='same', use_bias=True, bias_initializer='glorot_normal', return_sequences=True))(input_layer)

    return bi_dir_1


def convolution_blocks(input_layer):
    conv = Conv3D(filters=4, kernel_size=(3, 3, 3), padding='same', activation='relu',
                  kernel_initializer='he_normal', use_bias=True, bias_initializer='he_normal')(input_layer)

    batchnorm = BatchNormalization()(conv)

    return batchnorm


def make_model():
    input_layer = Input((128, 128, 128, 3))

    # Encoding
    enc1 = encoder(input_layer)
    enc2 = encoder(input_layer)
    enc = Concatenate(axis=4)([enc1, enc2])

    # Bilstm layer
    bilstm = lstms(enc)

    # Decoder
    dec1 = decoder(bilstm)
    dec2 = decoder(bilstm)

    dec = Concatenate(axis=4)([dec1, dec2])

    combined_layer = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), padding="same", activation="relu",
                                     use_bias=True, bias_initializer="he_normal", kernel_initializer="he_normal")(dec)

    # Few convolution la
    conv1 = convolution_blocks(combined_layer)
    conv2 = convolution_blocks(conv1)
    conv3 = convolution_blocks(conv2)

    output_layer = Conv3D(filters=4, kernel_size=(3, 3, 3), kernel_initializer='glorot_normal',
                          padding='same', activation='softmax', use_bias=True, bias_initializer='glorot_normal')(conv3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

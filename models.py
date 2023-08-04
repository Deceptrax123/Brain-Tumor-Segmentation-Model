from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Input, Dropout, MaxPooling3D, BatchNormalization, Add, Concatenate, ConvLSTM2D, Bidirectional, Attention
import tensorflow as tf
from layers import ConvBlockEnc, ConvBlockDec, Lstm, Final
from keras.models import Model


class DualPathCNNLstm(tf.keras.Model):
    def __init__(self):
        super(DualPathCNNLstm, self).__init__()

        self.input_layer = Conv3D(kernel_size=(3, 3, 3), use_bias=True, kernel_initializer='he_normal',
                                  bias_initializer='he_normal', padding='same', filters=3)
        self.convenc_start = ConvBlockEnc(kernel_size=(3, 3, 3), filters=4)
        self.convenc_1 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=8)
        self.convenc_2 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=16)
        self.convenc_3 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=32)
        self.convenc_4 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=64)
        self.convenc_5 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=128)
        self.convenc_6 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=256)

        self.maxpool = MaxPooling3D(pool_size=(2, 2, 2))

        self.dconv_1 = ConvBlockDec(kernel_size=(3, 3, 3), filters=128)
        self.dconv_2 = ConvBlockDec(kernel_size=(3, 3, 3), filters=64)
        self.dconv_3 = ConvBlockDec(kernel_size=(3, 3, 3), filters=32)
        self.dconv_4 = ConvBlockDec(kernel_size=(3, 3, 3), filters=16)
        self.dconv_5 = ConvBlockDec(kernel_size=(3, 3, 3), filters=8)
        self.dconv_6 = ConvBlockDec(kernel_size=(3, 3, 3), filters=4)

        self.upsample = UpSampling3D(size=(2, 2, 2))

        self.lstm_encode = Lstm(kernel_size=(3, 3), filters=4)
        self.lstm_decode = Lstm(kernel_size=(3, 3), filters=4)
        self.lstm_down = Lstm(kernel_size=(3, 3), filters=256)

        self.add = Add()
        self.concatenate = Concatenate()

        self.output_layer = Final(kernel_size=(3, 3, 3), filters=4)

    def call(self, input, training=False, **kwargs):
        # Forward pass

        # encoding step
        # initial embedding
        x11 = self.convenc_start(input)

        # pass it to lstm
        x11 = self.lstm_encode(x11)
        x11_down = self.maxpool(x11)

        x21 = self.convenc_1(x11_down)
        x21_down = self.maxpool(x21)

        x31 = self.convenc_2(x21_down)
        x31_down = self.maxpool(x31)

        x41 = self.convenc_3(x31_down)
        x41_down = self.maxpool(x41)

        x51 = self.convenc_4(x41_down)
        x51_down = self.maxpool(x51)

        x61 = self.convenc_5(x51_down)
        x61_down = self.maxpool(x61)

        x71 = self.convenc_6(x61_down)

        # embeded lstm
        xlstm = self.lstm_down(x71)

        # decoding step and bring in the skip connections
        xlstm_up = self.upsample(xlstm)

        x62 = self.concatenate([x61, xlstm_up])
        x62 = self.dconv_1(xlstm_up)
        x62_up = self.upsample(x62)

        x52 = self.concatenate([x62_up, x51])
        x52 = self.dconv_2(x62_up)
        x52_up = self.upsample(x52)

        x42 = self.concatenate([x52_up, x41])
        x42 = self.dconv_3(x52_up)
        x42_up = self.upsample(x42)

        x32 = self.concatenate([x42_up, x31])
        x32 = self.dconv_4(x42_up)
        x32_up = self.upsample(x32)

        x22 = self.concatenate([x32_up, x21])
        x22 = self.dconv_5(x32_up)
        x22_up = self.upsample(x22)

        x12 = self.concatenate([x22_up, x11])
        x12 = self.dconv_6(x22_up)

        # decoding lstm
        lstm_decode = self.lstm_decode(x12)

        # classifier
        classifier = self.output_layer(lstm_decode)

        return classifier

    def summary(self):
        x = Input(shape=(128, 128, 128, 3))
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()


model = DualPathCNNLstm()
k = model.summary()

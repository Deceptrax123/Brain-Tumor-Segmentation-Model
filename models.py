from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Input, Dropout, MaxPooling3D, BatchNormalization, Add, ConvLSTM2D, Bidirectional, Attention
import tensorflow as tf
from layers import ConvBlockEnc, ConvBlockDec, Lstm, Final
from keras.models import Model
from keras.utils import plot_model


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

        self.maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))
        self.maxpool2 = MaxPooling3D(pool_size=(2, 2, 2))
        self.maxpool3 = MaxPooling3D(pool_size=(2, 2, 2))
        self.maxpool4 = MaxPooling3D(pool_size=(2, 2, 2))
        self.maxpool5 = MaxPooling3D(pool_size=(2, 2, 2))
        self.maxpool6 = MaxPooling3D(pool_size=(2, 2, 2))

        self.dconv_0 = ConvBlockDec(kernel_size=(3, 3, 3), filters=256)
        self.dconv_1 = ConvBlockDec(kernel_size=(3, 3, 3), filters=128)
        self.dconv_2 = ConvBlockDec(kernel_size=(3, 3, 3), filters=64)
        self.dconv_3 = ConvBlockDec(kernel_size=(3, 3, 3), filters=32)
        self.dconv_4 = ConvBlockDec(kernel_size=(3, 3, 3), filters=16)
        self.dconv_5 = ConvBlockDec(kernel_size=(3, 3, 3), filters=8)
        self.dconv_6 = ConvBlockDec(kernel_size=(3, 3, 3), filters=4)

        self.upsample1 = UpSampling3D(size=(2, 2, 2))
        self.upsample2 = UpSampling3D(size=(2, 2, 2))
        self.upsample3 = UpSampling3D(size=(2, 2, 2))
        self.upsample4 = UpSampling3D(size=(2, 2, 2))
        self.upsample5 = UpSampling3D(size=(2, 2, 2))
        self.upsample6 = UpSampling3D(size=(2, 2, 2))

        self.lstm_encode = Lstm(kernel_size=(3, 3), filters=4)
        self.lstm_decode = Lstm(kernel_size=(3, 3), filters=4)
        self.lstm_down = Lstm(kernel_size=(3, 3), filters=256)

        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()
        self.add4 = Add()
        self.add5 = Add()
        self.add6 = Add()

        self.output_layer = Final(kernel_size=(3, 3, 3), filters=4)

    def call(self, input, training=False, **kwargs):
        # Forward pass

        # encoding step
        # initial embedding
        x11 = self.convenc_start(input)

        # pass it to lstm
        x11 = self.lstm_encode(x11)
        x11_down = self.maxpool1(x11)

        x21 = self.convenc_1(x11_down)
        x21_down = self.maxpool2(x21)

        x31 = self.convenc_2(x21_down)
        x31_down = self.maxpool3(x31)

        x41 = self.convenc_3(x31_down)
        x41_down = self.maxpool4(x41)

        x51 = self.convenc_4(x41_down)
        x51_down = self.maxpool5(x51)

        x61 = self.convenc_5(x51_down)
        x61_down = self.maxpool6(x61)

        x71 = self.convenc_6(x61_down)

        # embeded lstm
        xlstm = self.lstm_down(x71)

        # decoding step and bring in the skip connections
        xlstm = self.dconv_0(xlstm)
        xlstm_up = self.upsample1(xlstm)

        x62 = self.dconv_1(xlstm_up)
        x62 = self.add1([x62, x61])
        x62_up = self.upsample2(x62)

        x52 = self.dconv_2(x62_up)
        x52 = self.add2([x51, x52])
        x52_up = self.upsample3(x52)

        x42 = self.dconv_3(x52_up)
        x42 = self.add3([x41, x42])
        x42_up = self.upsample4(x42)

        x32 = self.dconv_4(x42_up)
        x32 = self.add4([x31, x32])
        x32_up = self.upsample5(x32)

        x22 = self.dconv_5(x32_up)
        x22 = self.add5([x22, x21])
        x22_up = self.upsample6(x22)

        x12 = self.dconv_6(x22_up)
        x12 = self.add6([x12, x11])

        # decoding lstm
        lstm_decode = self.lstm_decode(x12)

        # classifier
        classifier = self.output_layer(lstm_decode)

        return classifier

    def build_graph(self):
        x = Input(shape=(128, 128, 128, 3))
        model = Model(inputs=x, outputs=self.call(x))
        return model


model = DualPathCNNLstm()
model.build(input_shape=(None, 128, 128, 128, 3))
model.build_graph().summary()

plot_model(model.build_graph(), to_file='model.png', dpi=96,
           show_shapes=True, show_layer_names=True, expand_nested=False)

from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, AveragePooling3D, Conv3DTranspose, UpSampling3D, Activation
from keras.layers import Dropout, MaxPooling3D, Input, BatchNormalization, Add, Concatenate, ConvLSTM2D, Bidirectional, Attention
import tensorflow as tf
from keras.models import Model
from layers import ConvBlockEnc, ConvBlockDec, Lstm, Down


class DualCNNLstm(tf.keras.models.Model):
    def __init__(self):
        super(DualCNNLstm, self).__init__()

        self.convenc_1 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=8)
        self.convenc_2 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=16)
        self.convenc_3 = ConvBlockEnc(kernel_size=(3, 3, 3), filter=32)
        self.convenc_4 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=64)
        self.convenc_5 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=128)
        self.convenc_6 = ConvBlockEnc(kernel_size=(3, 3, 3), filters=256)

        self.maxpool = MaxPool3D(pool_size=(2, 2, 2))

        self.dconv_1 = ConvBlockDec(kernel_size=(3, 3, 3), filters=128)
        self.dconv_2 = ConvBlockDec(kernel_size=(3, 3, 3), filters=64)
        self.dconv_3 = ConvBlockDec(kernel_size=(3, 3, 3), filters=32)
        self.dconv_4 = ConvBlockDec(kernel_size=(3, 3, 3), filters=16)
        self.dconv_5 = ConvBlockDec(kernel_size=(3, 3, 3), filters=8)
        self.dconv_6 = ConvBlockDec(kernel_size=(3, 3, 3), filters=4)

        self.upsample = UpSampling3D(size=(2, 2, 2))

        self.lstm = Lstm(kernel_size=(2, 2), filters=256)

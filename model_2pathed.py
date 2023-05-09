import tensorflow as tf
from keras.layers import Conv3D,Reshape,Conv3DTranspose,UpSampling3D,MaxPooling3D,Concatenate
from keras.layers import Dropout,Input,BatchNormalization,Dense
from keras.models import Model
import keras

def create_2pathed():
    input_layer=Input((128,128,128,3))

    #first encoding block
    #convoluted layers
    conv1_e_1=Conv3D(filters=4,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(input_layer)
    conv1_e_2=Conv3D(filters=4,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_1)
    conv1_e_3=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_2)
    conv1_e_4=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_3)
    conv1_e_5=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_4)

    #down sampling by a double strided convolution
    pool1_e_7=MaxPooling3D(pool_size=(2,2,2),strides=2)(conv1_e_4)

    #convoluted layers
    conv1_e_8=Conv3D(filters=16,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(pool1_e_7)
    conv1_e_9=Conv3D(filters=16,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_8)
    conv1_e_10=Conv3D(filters=32,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_9)
    conv1_e_11=Conv3D(filters=32,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv1_e_10)

    #second encoding block
    #convoluted layers
    conv2_e_1=Conv3D(filters=4,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(input_layer)
    conv2_e_2=Conv3D(filters=4,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv2_e_1)
    conv2_e_3=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv2_e_2)
    conv2_e_4=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv2_e_3)
    conv2_e_5=Conv3D(filters=8,kernel_initializer='he_normal',kernel_size=(3,3,3),activation='relu',strides=1,padding='valid',use_bias=True,bias_initializer='he_normal')(conv2_e_4)


    #Pooling layer
    pool2_e_5=MaxPooling3D(pool_size=(2,2,2),strides=2)(conv2_e_4)

    #convoluated layers
    conv2_e_5=Conv3D(filters=16,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(pool2_e_5)
    conv2_e_6=Conv3D(filters=16,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv2_e_5)
    conv2_e_7=Conv3D(filters=32,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv2_e_6)
    conv2_e_8=Conv3D(filters=32,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv2_e_7)

    #Concatenate both the encoding blocks
    concat_layer=Concatenate(axis=4)([conv1_e_11,conv2_e_8])

    #Decoder
    conv_e_9=Conv3D(filters=64,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(concat_layer)
    conv_e_10=Conv3D(filters=128,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv_e_9)
    conv_e_11=Conv3D(filters=256,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv_e_10)
    conv_e_12=Conv3D(filters=512,kernel_size=(3,3,3),kernel_initializer='he_normal',activation='relu',strides=1,use_bias=True,padding='valid',bias_initializer='he_normal')(conv_e_11)

    model=Model(inputs=input_layer,outputs=conv_e_12)
    return model

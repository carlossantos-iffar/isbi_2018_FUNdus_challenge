import os

from keras import backend as K
from keras import objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import Input
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
import numpy as np
from keras.optimizers import SGD

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')


def unet(img_size):
    """
    generate network based on unet
    """    
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    n_filters = 8
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv4_1 = Conv2D(16 * n_filters, (k, k), padding=padding)(pool4)
    conv4_1 = BatchNormalization(scale=False, axis=3)(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)    
    conv4_1 = Conv2D(16 * n_filters, (k, k), padding=padding)(conv4_1)
    conv4_1 = BatchNormalization(scale=False, axis=3)(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)    
    pool4_1 = MaxPooling2D(pool_size=(s, s))(conv4_1)
    
    conv4_2 = Conv2D(32 * n_filters, (k, k), padding=padding)(pool4_1)
    conv4_2 = BatchNormalization(scale=False, axis=3)(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)    
    conv4_2 = Conv2D(32 * n_filters, (k, k), padding=padding)(conv4_2)
    conv4_2 = BatchNormalization(scale=False, axis=3)(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)    
    pool4_2 = MaxPooling2D(pool_size=(s, s))(conv4_2)

    conv4_3 = Conv2D(64 * n_filters, (k, k), padding=padding)(pool4_2)
    conv4_3 = BatchNormalization(scale=False, axis=3)(conv4_3)
    conv4_3 = Activation('relu')(conv4_3)    
    conv4_3 = Conv2D(64 * n_filters, (k, k), padding=padding)(conv4_3)
    conv4_3 = BatchNormalization(scale=False, axis=3)(conv4_3)
    conv4_3 = Activation('relu')(conv4_3)    
    pool4_3 = MaxPooling2D(pool_size=(s, s))(conv4_3)

    conv5 = Conv2D(64 * n_filters, (k, k), padding=padding)(pool4_3)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(64 * n_filters, (k, k), padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    

    up0_1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4_3])
    conv4_3_up = Conv2D(64 * n_filters, (k, k), padding=padding)(up0_1)
    conv4_3_up = BatchNormalization(scale=False, axis=3)(conv4_3_up)
    conv4_3_up = Activation('relu')(conv4_3_up)    
    conv4_3_up = Conv2D(64 * n_filters, (k, k), padding=padding)(conv4_3_up)
    conv4_3_up = BatchNormalization(scale=False, axis=3)(conv4_3_up)
    conv4_3_up = Activation('relu')(conv4_3_up)   
    
    up0_2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv4_3_up), conv4_2])
    conv4_2_up = Conv2D(32 * n_filters, (k, k), padding=padding)(up0_2)
    conv4_2_up = BatchNormalization(scale=False, axis=3)(conv4_2_up)
    conv4_2_up = Activation('relu')(conv4_2_up)    
    conv4_2_up = Conv2D(32 * n_filters, (k, k), padding=padding)(conv4_2_up)
    conv4_2_up = BatchNormalization(scale=False, axis=3)(conv4_2_up)
    conv4_2_up = Activation('relu')(conv4_2_up)   
    
    up0_3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv4_2_up), conv4_1])
    conv4_1_up = Conv2D(16 * n_filters, (k, k), padding=padding)(up0_3)
    conv4_1_up = BatchNormalization(scale=False, axis=3)(conv4_1_up)
    conv4_1_up = Activation('relu')(conv4_1_up)    
    conv4_1_up = Conv2D(16 * n_filters, (k, k), padding=padding)(conv4_1_up)
    conv4_1_up = BatchNormalization(scale=False, axis=3)(conv4_1_up)
    conv4_1_up = Activation('relu')(conv4_1_up)   
    
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv4_1_up), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
     
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    unet = Model(inputs, outputs)
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=None)

    return unet


def conv_blocks(n_block, x, n_filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1)):
    conv = x
    for _ in range(n_block):
        conv = Conv2D(n_filters, filter_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(conv)
        conv = BatchNormalization(scale=False, axis=3)(conv)
        conv = Activation('relu')(conv) 
    return conv


def od_from_fundus_vessel(img_size):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    out_ch_fundus_vessel = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    conv1 = conv_blocks(2, input_vessel, n_filters_vessel, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_vessel, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_vessel, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_vessel, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_vessel, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)

    conv6 = conv_blocks(2, pool5, 16 * n_filters_vessel, (k, k), padding=padding)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(conv6))
    vessel_block = Concatenate(axis=3)(list_dilated_conv)  
    
    output_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    conv1 = conv_blocks(2, input_fundus, n_filters_fundus, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_fundus, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_fundus, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_fundus, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_fundus, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)
    
    conv6_1 = conv_blocks(2, pool5, 16 * n_filters_fundus, (k, k), padding=padding)
    concated_block = Concatenate(axis=3)([conv6_1, vessel_block])
    
    up0 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(concated_block), conv5])
    conv6_2 = conv_blocks(2, up0, 16 * n_filters_fundus, (k, k), padding=padding)
    
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6_2), conv4])
    conv6_2 = conv_blocks(2, up1, 8 * n_filters_fundus, (k, k), padding=padding)
     
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6_2), conv3])
    conv7 = conv_blocks(2, up2, 4 * n_filters_fundus, (k, k), padding=padding)
        
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = conv_blocks(2, up3, 2 * n_filters_fundus, (k, k), padding=padding)
    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = conv_blocks(2, up4, n_filters_fundus, (k, k), padding=padding)
    
    output_fundus_vessel = Conv2D(out_ch_fundus_vessel, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    model = Model([input_fundus, input_vessel], [output_fundus_vessel, output_vessel])
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    model.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=[seg_loss, seg_loss], loss_weights=[1, 1], metrics=None)

    return model


def od_from_fundus_vessel_v2(img_size):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    out_ch_fundus_vessel = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    conv1 = conv_blocks(2, input_vessel, n_filters_vessel, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_vessel, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_vessel, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_vessel, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 8 * n_filters_vessel, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)

    conv6 = conv_blocks(2, pool5, 8 * n_filters_vessel, (k, k), padding=padding)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(4 * n_filters_vessel, (k, k), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(4 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(4 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(conv6))
    vessel_block = Concatenate(axis=3)(list_dilated_conv)  
    
    output_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    conv1 = conv_blocks(2, input_fundus, n_filters_fundus, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_fundus, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_fundus, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_fundus, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_fundus, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)
    
    conv6_1 = conv_blocks(2, pool5, 16 * n_filters_fundus, (k, k), padding=padding)
    concated_block = Concatenate(axis=3)([conv6_1, vessel_block])
    
    up0 = UpSampling2D(size=(s, s))(concated_block)
    up0_conv = conv_blocks(1, up0, 16 * n_filters_fundus, (k, k), padding=padding)
    concat0 = Concatenate(axis=3)([up0_conv, conv5])
    conv6_2 = conv_blocks(2, concat0, 16 * n_filters_fundus, (k, k), padding=padding)
    
    up1 = UpSampling2D(size=(s, s))(conv6_2)
    up1_conv = conv_blocks(1, up1, 8 * n_filters_fundus, (k, k), padding=padding)
    concat1 = Concatenate(axis=3)([up1_conv, conv4])
    conv6_2 = conv_blocks(2, concat1, 8 * n_filters_fundus, (k, k), padding=padding)
     
    up2 = UpSampling2D(size=(s, s))(conv6_2)
    up2_conv = conv_blocks(1, up2, 4 * n_filters_fundus, (k, k), padding=padding)
    concat2 = Concatenate(axis=3)([up2_conv, conv3])
    conv7 = conv_blocks(2, concat2, 4 * n_filters_fundus, (k, k), padding=padding)
    
    up3 = UpSampling2D(size=(s, s))(conv7)
    up3_conv = conv_blocks(1, up3, 2 * n_filters_fundus, (k, k), padding=padding)
    concat3 = Concatenate(axis=3)([up3_conv, conv2])
    conv8 = conv_blocks(2, concat3, 2 * n_filters_fundus, (k, k), padding=padding)
    
    up4 = UpSampling2D(size=(s, s))(conv8)
    up4_conv = conv_blocks(1, up4, n_filters_fundus, (k, k), padding=padding)
    concat4 = Concatenate(axis=3)([up4_conv, conv1])
    conv9 = conv_blocks(2, concat4, n_filters_fundus, (k, k), padding=padding)
    
    output_fundus_vessel = Conv2D(out_ch_fundus_vessel, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    model = Model([input_fundus, input_vessel], [output_fundus_vessel, output_vessel])
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    model.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=[seg_loss, seg_loss], loss_weights=[1, 1], metrics=None)

    return model


def od_from_vessel(img_size, n_filters):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 1  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = conv_blocks(2, inputs, n_filters, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)

    conv6 = conv_blocks(2, pool5, 16 * n_filters, (k, k), padding=padding)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(2, 2), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(4, 4), padding=padding)(conv6))
    final_block = Concatenate(axis=3)(list_dilated_conv)  
    
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(final_block)
    
    model = Model(inputs, outputs)
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    model.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=['accuracy'])

    return model


def od_from_fundus_vessel_v4(img_size):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    out_ch_fundus_vessel = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    conv1 = conv_blocks(2, input_vessel, n_filters_vessel, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_vessel, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_vessel, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_vessel, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_vessel, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)

    conv6 = conv_blocks(2, pool5, 16 * n_filters_vessel, (k, k), padding=padding)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(conv6))
    vessel_block = Concatenate(axis=3)(list_dilated_conv)  
    
    output_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    conv1 = conv_blocks(2, input_fundus, n_filters_fundus, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_fundus, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_fundus, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_fundus, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_fundus, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)
    
    conv6_1 = conv_blocks(2, pool5, 16 * n_filters_fundus, (k, k), padding=padding)
    concated_block = Concatenate(axis=3)([conv6_1, vessel_block])
    
    up0 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(concated_block), conv5])
    conv6_2 = conv_blocks(2, up0, 16 * n_filters_fundus, (k, k), padding=padding)
    
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6_2), conv4])
    conv6_2 = conv_blocks(2, up1, 8 * n_filters_fundus, (k, k), padding=padding)
     
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6_2), conv3])
    conv7 = conv_blocks(2, up2, 4 * n_filters_fundus, (k, k), padding=padding)
        
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = conv_blocks(2, up3, 2 * n_filters_fundus, (k, k), padding=padding)
    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = conv_blocks(2, up4, n_filters_fundus, (k, k), padding=padding)
    
    output_fundus_vessel = Conv2D(out_ch_fundus_vessel, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    model = Model([input_fundus, input_vessel], [output_fundus_vessel, output_vessel])
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    model.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=[seg_loss, seg_loss], loss_weights=[1, 0.1], metrics=None)

    return model


def od_from_fundus_vessel_v3(img_size):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    out_ch_fundus_vessel = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    conv1 = conv_blocks(2, input_vessel, n_filters_vessel, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_vessel, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_vessel, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_vessel, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_vessel, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)

    conv6 = conv_blocks(2, pool5, 16 * n_filters_vessel, (k, k), padding=padding)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(conv6))
    list_dilated_conv.append(Conv2D(16 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(conv6))
    vessel_block = Concatenate(axis=3)(list_dilated_conv)  
    
    output_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    conv1 = conv_blocks(2, input_fundus, n_filters_fundus, (k, k), padding=padding)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = conv_blocks(2, pool1, 2 * n_filters_fundus, (k, k), padding=padding)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = conv_blocks(2, pool2, 4 * n_filters_fundus, (k, k), padding=padding)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = conv_blocks(2, pool3, 8 * n_filters_fundus, (k, k), padding=padding)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = conv_blocks(2, pool4, 16 * n_filters_fundus, (k, k), padding=padding)
    pool5 = MaxPooling2D(pool_size=(s, s))(conv5)
    
    conv6_1 = conv_blocks(2, pool5, 16 * n_filters_fundus, (k, k), padding=padding)
    vessel_block = conv_blocks(1, vessel_block, 16 * n_filters_fundus, (k, k), padding=padding)
    concated_block = Concatenate(axis=3)([conv6_1, vessel_block])
    
    up0 = UpSampling2D(size=(s, s))(concated_block)
    up0_conv = conv_blocks(1, up0, 16 * n_filters_fundus, (k, k), padding=padding)
    concat0 = Concatenate(axis=3)([up0_conv, conv5])
    conv6_2 = conv_blocks(2, concat0, 16 * n_filters_fundus, (k, k), padding=padding)
    
    up1 = UpSampling2D(size=(s, s))(conv6_2)
    up1_conv = conv_blocks(1, up1, 8 * n_filters_fundus, (k, k), padding=padding)
    concat1 = Concatenate(axis=3)([up1_conv, conv4])
    conv6_2 = conv_blocks(2, concat1, 8 * n_filters_fundus, (k, k), padding=padding)
     
    up2 = UpSampling2D(size=(s, s))(conv6_2)
    up2_conv = conv_blocks(1, up2, 4 * n_filters_fundus, (k, k), padding=padding)
    concat2 = Concatenate(axis=3)([up2_conv, conv3])
    conv7 = conv_blocks(2, concat2, 4 * n_filters_fundus, (k, k), padding=padding)
    
    up3 = UpSampling2D(size=(s, s))(conv7)
    up3_conv = conv_blocks(1, up3, 2 * n_filters_fundus, (k, k), padding=padding)
    concat3 = Concatenate(axis=3)([up3_conv, conv2])
    conv8 = conv_blocks(2, concat3, 2 * n_filters_fundus, (k, k), padding=padding)
    
    up4 = UpSampling2D(size=(s, s))(conv8)
    up4_conv = conv_blocks(1, up4, n_filters_fundus, (k, k), padding=padding)
    concat4 = Concatenate(axis=3)([up4_conv, conv1])
    conv9 = conv_blocks(2, concat4, n_filters_fundus, (k, k), padding=padding)
    
    output_fundus_vessel = Conv2D(out_ch_fundus_vessel, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    model = Model([input_fundus, input_vessel], [output_fundus_vessel, output_vessel])
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    model.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=[seg_loss, seg_loss], loss_weights=[1, 0.1], metrics=None)

    return model


def weights_for_activation2coord(act_h, act_w):
    l_h = 1. / act_h
    l_w = 1. / act_w
    weights = np.zeros((act_h * act_w, 2))
    for i in range(act_h * act_w):
        weights[i, 0] = (0.5 + i / act_w) * l_h
        weights[i, 1] = (0.5 + i % act_w) * l_w
    return np.expand_dims(weights, axis=0)


def od_fovea_from_vessel(img_size):
    # set image specifics
    k = 3  # kernel size
    img_ch = 1  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch))
    conv1 = conv_blocks(2, input_vessel, n_filters_vessel, (k, k), padding=padding)
    conv2 = conv_blocks(2, conv1, n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)
    conv3 = conv_blocks(2, conv2, n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)
    conv4 = conv_blocks(2, conv3, n_filters_vessel, (k, k), dilation_rate=(8, 8), padding=padding)
    conv5_1 = conv_blocks(2, conv4, n_filters_vessel, (k, k), dilation_rate=(16, 16), padding=padding)
    conv6_1 = conv_blocks(2, conv5_1, n_filters_vessel, (k, k), dilation_rate=(32, 32), padding=padding)
    conv7_1 = conv_blocks(2, conv6_1, n_filters_vessel, (k, k), dilation_rate=(64, 64), padding=padding)
    
    od_output_vessel = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv7_1)
    
    conv5_2 = conv_blocks(2, conv4, n_filters_vessel, (k, k), dilation_rate=(16, 16), padding=padding)
    conv6_2 = conv_blocks(2, conv5_2, n_filters_vessel, (k, k), dilation_rate=(32, 32), padding=padding)
    conv7_2 = conv_blocks(2, conv6_2, n_filters_vessel, (k, k), dilation_rate=(64, 64), padding=padding)
    
    fovea_output_vessel = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv7_2)
    
    flattened_od = Flatten()(od_output_vessel)
    flattened_fovea = Flatten()(fovea_output_vessel)
    
    normalized_flattened_od = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_od)
    normalized_flattened_fovea = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_fovea)

    img2coord = Dense(2, activation=None, use_bias=False, name="img2coord")
    img2coord.trainable = False
    coord_od = img2coord(normalized_flattened_od)
    coord_fovea = img2coord(normalized_flattened_fovea)
    outputs = Concatenate(axis=1)([coord_od, coord_fovea])

    model = Model(input_vessel, outputs)
    model.get_layer("img2coord").set_weights(weights_for_activation2coord(od_output_vessel._keras_shape[1], od_output_vessel._keras_shape[2]))

    model.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss="mean_absolute_error", metrics=None)

    return model


def localize_od_fv(img_size, branching_point):
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    list_out = [input_vessel]
    for index in range(branching_point):
        conv = conv_blocks(2, list_out[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool = MaxPooling2D(pool_size=(s, s))(conv)
        list_out.append(pool)
    
    list_out_od, list_out_fovea = [list_out[-1]], [list_out[-1]]
    for index in range(5 - branching_point):
        conv_od = conv_blocks(2, list_out_od[-1], 2 ** (index + branching_point) * n_filters_fundus, (k, k), padding=padding)
        pool_od = MaxPooling2D(pool_size=(s, s))(conv_od)
        list_out_od.append(pool_od)
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index + branching_point) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), padding=padding)(list_out_od[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(list_out_od[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(list_out_od[-1]))
    vessel_block_od = Concatenate(axis=3)(list_dilated_conv)  
    
    od_activation_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block_od)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(list_out_fovea[-1]))
    vessel_block_fovea = Concatenate(axis=3)(list_dilated_conv)  

    fovea_activation_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block_fovea)
    
    flattened_od_vessel = Flatten()(od_activation_vessel)
    flattened_fovea_vessel = Flatten()(fovea_activation_vessel)
    
    normalized_flattened_od_vessel = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_od_vessel)
    normalized_flattened_fovea_vessel = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_fovea_vessel)
    
    img2coord_vessel = Dense(2, activation=None, use_bias=False, name="img2coord_vessel")
    img2coord_vessel.trainable = False
    coord_od_vessel = img2coord_vessel(normalized_flattened_od_vessel)
    coord_fovea_vessel = img2coord_vessel(normalized_flattened_fovea_vessel)
    
    outputs_vessel = Concatenate(axis=1)([coord_od_vessel, coord_fovea_vessel])
        
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    list_out = [input_fundus]
    for index in range(branching_point):
        conv = conv_blocks(2, list_out[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool = MaxPooling2D(pool_size=(s, s))(conv)
        list_out.append(pool)
    
    list_out_od, list_out_fovea = [list_out[-1]], [list_out[-1]]
    for index in range(4 - branching_point):
        conv_od = conv_blocks(2, list_out_od[-1], 2 ** (index + branching_point) * n_filters_fundus, (k, k), padding=padding)
        pool_od = MaxPooling2D(pool_size=(s, s))(conv_od)
        list_out_od.append(pool_od)
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index + branching_point) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
        
    od_concat_block = Concatenate(axis=3)([list_out_od[-1], UpSampling2D(size=(s, s))(vessel_block_od)])
    conv_od_final = conv_blocks(2, od_concat_block, 16 * n_filters_fundus, (k, k), padding=padding)
    activation_od = Conv2D(1, (1, 1), padding=padding, activation='sigmoid')(conv_od_final)

    fovea_concat_block = Concatenate(axis=3)([list_out_fovea[-1], UpSampling2D(size=(s, s))(vessel_block_fovea)])
    conv_fovea_final = conv_blocks(2, fovea_concat_block, 16 * n_filters_fundus, (k, k), padding=padding)
    activation_fovea = Conv2D(1, (1, 1), padding=padding, activation='sigmoid')(conv_fovea_final)
    
    flattened_od = Flatten()(activation_od)
    flattened_fovea = Flatten()(activation_fovea)
    
    normalized_flattened_od = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_od)
    normalized_flattened_fovea = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_fovea)

    img2coord = Dense(2, activation=None, use_bias=False, name="img2coord")
    img2coord.trainable = False
    coord_od = img2coord(normalized_flattened_od)
    coord_fovea = img2coord(normalized_flattened_fovea)
    
    outputs = Concatenate(axis=1)([coord_od, coord_fovea])

    model = Model([input_fundus, input_vessel], [outputs, outputs_vessel])
    model.get_layer("img2coord").set_weights(weights_for_activation2coord(activation_od._keras_shape[1], activation_od._keras_shape[2]))
    model.get_layer("img2coord_vessel").set_weights(weights_for_activation2coord(od_activation_vessel._keras_shape[1], od_activation_vessel._keras_shape[2]))

    model.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=["mean_absolute_error", "mean_absolute_error"], loss_weights=[1, 1. / 3.], metrics=None)

    return model


def localize_fv(img_size):
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    list_out = [input_vessel]
    
    list_out_fovea = [list_out[-1]]
    for index in range(5):
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(list_out_fovea[-1]))
    vessel_block_fovea = Concatenate(axis=3)(list_dilated_conv)  

    fovea_activation_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block_fovea)
    
    flattened_fovea_vessel = Flatten()(fovea_activation_vessel)
    
    normalized_flattened_fovea_vessel = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_fovea_vessel)
    
    img2coord_vessel = Dense(2, activation=None, use_bias=False, name="img2coord_vessel")
    img2coord_vessel.trainable = False
    coord_fovea_vessel = img2coord_vessel(normalized_flattened_fovea_vessel)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    list_out = [input_fundus]
    list_out_fovea = [list_out[-1]]
    for index in range(4):
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
        
    fovea_concat_block = Concatenate(axis=3)([list_out_fovea[-1], UpSampling2D(size=(s, s))(vessel_block_fovea)])
    conv_fovea_final = conv_blocks(2, fovea_concat_block, 16 * n_filters_fundus, (k, k), padding=padding)
    activation_fovea = Conv2D(1, (1, 1), padding=padding, activation='sigmoid')(conv_fovea_final)
    
    flattened_fovea = Flatten()(activation_fovea)
    
    normalized_flattened_fovea = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True))(flattened_fovea)

    img2coord = Dense(2, activation=None, use_bias=False, name="img2coord")
    img2coord.trainable = False
    coord_fovea = img2coord(normalized_flattened_fovea)

    model = Model([input_fundus, input_vessel], [coord_fovea, coord_fovea_vessel])
    model.get_layer("img2coord").set_weights(weights_for_activation2coord(activation_fovea._keras_shape[1], activation_fovea._keras_shape[2]))
    model.get_layer("img2coord_vessel").set_weights(weights_for_activation2coord(fovea_activation_vessel._keras_shape[1], fovea_activation_vessel._keras_shape[2]))

    model.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=["mean_absolute_error", "mean_absolute_error"], loss_weights=[1, 1. / 3.], metrics=None)

    return model


def segment_fv(img_size):
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch_vessel = 1  # image channels
    out_ch_vessel = 1  # output channel
    img_ch_fundus = 3  # image channels
    img_height, img_width = img_size[0], img_size[1]
    n_filters_vessel = 16
    n_filters_fundus = 32
    padding = 'same'
    
    input_vessel = Input((img_height, img_width, img_ch_vessel))
    list_out = [input_vessel]
    
    list_out_fovea = [list_out[-1]]
    for index in range(5):
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
    
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(2, 2), padding=padding)(list_out_fovea[-1]))
    list_dilated_conv.append(Conv2D(8 * n_filters_vessel, (k, k), dilation_rate=(4, 4), padding=padding)(list_out_fovea[-1]))
    vessel_block_fovea = Concatenate(axis=3)(list_dilated_conv)  

    activation_fovea_vessel = Conv2D(out_ch_vessel, (1, 1), padding=padding, activation='sigmoid')(vessel_block_fovea)
    
    input_fundus = Input((img_height, img_width, img_ch_fundus))
    list_out = [input_fundus]
    list_out_fovea = [list_out[-1]]
    for index in range(4):
        conv_fovea = conv_blocks(2, list_out_fovea[-1], 2 ** (index) * n_filters_fundus, (k, k), padding=padding)
        pool_fovea = MaxPooling2D(pool_size=(s, s))(conv_fovea)
        list_out_fovea.append(pool_fovea)
        
    fovea_concat_block = Concatenate(axis=3)([list_out_fovea[-1], UpSampling2D(size=(s, s))(vessel_block_fovea)])
    conv_fovea_final = conv_blocks(2, fovea_concat_block, 16 * n_filters_fundus, (k, k), padding=padding)
    activation_fovea = Conv2D(1, (1, 1), padding=padding, activation='sigmoid')(conv_fovea_final)

    model = Model([input_fundus, input_vessel], [activation_fovea, activation_fovea_vessel])

    model.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1, 1. / 3.], metrics=None)

    return model


import os

from keras import backend as K, objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape, Permute, Input
from keras.layers.core import Activation, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')


def set_optimizer(network):

    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)

    network.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=None)
    return network


def split(X, r, axis):
    if axis == 1:
        return [Lambda(lambda x:x[:, index * (x._keras_shape[1] // r):(index + 1) * (x._keras_shape[1] // r), ...])(X) for index in range(r)]
    elif axis == 2:
        return [Lambda(lambda x:x[:, :, index * (x._keras_shape[2] // r):(index + 1) * (x._keras_shape[2] // r), ...])(X) for index in range(r)]
    elif axis == 3:
        return [Lambda(lambda x:x[:, :, :, index * (x._keras_shape[3] // r):(index + 1) * (x._keras_shape[3] // r), ...])(X) for index in range(r)]


def split_tensor(X, r, axis):
    if axis == 1:
        return [Lambda(lambda x:x[:, index * (K.shape(x)[1] // r):(index + 1) * (K.shape(x)[1] // r), ...], output_shape=(None, K.shape(X)[1] // r,))(X) for index in range(r)]
    elif axis == 2:
        return [Lambda(lambda x:x[:, :, index * (K.shape(x)[2] // r):(index + 1) * (K.shape(x)[2] // r), ...])(X) for index in range(r)]
    elif axis == 3:
        return [Lambda(lambda x:x[:, :, :, index * (K.shape(x)[3] // r):(index + 1) * (K.shape(x)[3] // r), ...])(X) for index in range(r)]

    
def inverse_pixel_shuffling(input_tensor, scale_factor):
    n, h, w, d = input_tensor._keras_shape
    r = int(scale_factor)
    ori_imgs = split(input_tensor, 3, 3)
    imgs = []
    for ori_img in ori_imgs:
        X = ori_img  # n,h,w,d
        X = Reshape((h // r, r, w // r, r))(X)
        X = Permute((1, 3, 2, 4))(X)
        X = Reshape((h // r, w // r, r * r))(X)
        imgs.append(X)

    return Concatenate(axis=3)(imgs)


def pixel_shuffling(input_tensor, scale_factor):
    n, h, w, d = input_tensor._keras_shape
    assert d % int(scale_factor ** 2) == 0
    r = int(scale_factor)

    X = Reshape((h, w, r, r))(input_tensor)
    X = Permute((1, 3, 2, 4))(X)
    X = Reshape((h * r, w * r, 1))(X)
        
    return X


def unet_atrous_sr(depth, atrous_depth, scale_factor, img_size, loss_weight_divisor=None):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = scale_factor ** 2  # output channel
    n_filters = 32
    padding = 'same'
    max_depth = 512
    h, w = img_size

    inputs = Input((h, w, img_ch))
    ips = inverse_pixel_shuffling(inputs, scale_factor)

    list_out = [ips]
    list_conv = []
    # encode with maxpooling
    for index in range(atrous_depth - 1):
        conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** index), max_depth), (k, k), padding=padding)
        pool = MaxPooling2D(pool_size=(s, s))(conv)
        list_out.append(pool)
        list_conv.append(conv)
    # conv before transition to atrous conv
    conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k), padding=padding)
    list_out.append(conv)
    list_conv.append(conv)
    # encode with dilated conv
    for index in range(atrous_depth, depth):
        conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k),
                           dilation_rate=(2 ** (index - atrous_depth + 1), 2 ** (index - atrous_depth + 1)), padding=padding)
        list_conv.append(conv)
        list_out.append(conv)
    # decode with concat 
    for index in range(depth, atrous_depth, -1):
        concat = Concatenate(axis=3)([list_out[-1], list_conv[index - 2]])
        conv = conv_blocks(2, concat, min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k),
                           dilation_rate=(2 ** (index - atrous_depth + 1), 2 ** (index - atrous_depth + 1)), padding=padding)
        list_conv.append(conv)
        list_out.append(conv)
    # decode with upsample
    for index in range(atrous_depth, 1, -1):
        up = UpSampling2D(size=(s, s))(list_out[-1])
        up_conv = conv_blocks(1, up, min(n_filters * (2 ** (index - 2)), max_depth), (k, k), padding=padding)
        concat = Concatenate(axis=3)([up_conv, list_conv[index - 2]])
        conv = conv_blocks(2, concat, min(n_filters * (2 ** (index - 2)), max_depth), (k, k), padding=padding)
        list_out.append(conv)
    # sigmoid 
    penultimate = Conv2D(out_ch, (3, 3), padding=padding)(list_out[-1])
    outputs = pixel_shuffling(penultimate, scale_factor)
    outputs = Conv2D(1, (3, 3), padding=padding)(outputs)
    outputs = Conv2D(1, (3, 3), padding=padding, activation='sigmoid')(outputs)

    unet = Model(inputs, outputs)
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        y_pred_flat = K.clip(y_pred_flat, K.epsilon(), 1 - K.epsilon())

        weight1 = K.sum(K.cast(K.equal(y_true_flat, 1), tf.float32))
        weight1 = K.maximum(weight1, 1)  # prevent division by zero
        weight0 = K.sum(K.cast(K.equal(y_true_flat, 0), tf.float32))
         
        weighted_binary_crossentropy = -K.mean(y_true_flat * K.log(y_pred_flat) * weight0 / (weight1 * loss_weight_divisor)
                                               + (1 - y_true_flat) * K.log(1 - y_pred_flat), axis=-1)
        return weighted_binary_crossentropy

    def seg_loss_no_weight(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    if loss_weight_divisor:
        unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=None)
    else:
        unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss_no_weight, metrics=None)
    return unet


def unet_atrous(depth, atrous_depth, loss_weight_divisor=None):
        
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    n_filters = 32
    padding = 'same'
    max_depth = 512
    
    inputs = Input((None, None, img_ch))
    list_out = [inputs]
    list_conv = []
    # encode with maxpooling
    for index in range(atrous_depth - 1):
        conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** index), max_depth), (k, k), padding=padding)
        pool = MaxPooling2D(pool_size=(s, s))(conv)
        list_out.append(pool)
        list_conv.append(conv)
    # conv before transition to atrous conv
    conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k), padding=padding)
    list_out.append(conv)
    list_conv.append(conv)
    # encode with dilated conv
    for index in range(atrous_depth, depth):
        conv = conv_blocks(2, list_out[-1], min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k),
                           dilation_rate=(2 ** (index - atrous_depth + 1), 2 ** (index - atrous_depth + 1)), padding=padding)
        list_conv.append(conv)
        list_out.append(conv)
    # decode with concat 
    for index in range(depth, atrous_depth, -1):
        concat = Concatenate(axis=3)([list_out[-1], list_conv[index - 2]])
        conv = conv_blocks(2, concat, min(n_filters * (2 ** (atrous_depth - 1)), max_depth), (k, k),
                           dilation_rate=(2 ** (index - atrous_depth + 1), 2 ** (index - atrous_depth + 1)), padding=padding)
        list_conv.append(conv)
        list_out.append(conv)
    # decode with upsample
    for index in range(atrous_depth, 1, -1):
        up = UpSampling2D(size=(s, s))(list_out[-1])
        up_conv = conv_blocks(1, up, min(n_filters * (2 ** (index - 2)), max_depth), (k, k), padding=padding)
        concat = Concatenate(axis=3)([up_conv, list_conv[index - 2]])
        conv = conv_blocks(2, concat, min(n_filters * (2 ** (index - 2)), max_depth), (k, k), padding=padding)
        list_out.append(conv)
    # sigmoid 
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(list_out[-1])
    
    unet = Model(inputs, outputs)
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        y_pred_flat = K.clip(y_pred_flat, K.epsilon(), 1 - K.epsilon())

        weight1 = K.sum(K.cast(K.equal(y_true_flat, 1), tf.float32))
        weight1 = K.maximum(weight1, 1)  # prevent division by zero
        weight0 = K.sum(K.cast(K.equal(y_true_flat, 0), tf.float32))
         
        weighted_binary_crossentropy = -K.mean(y_true_flat * K.log(y_pred_flat) * weight0 / (weight1 * loss_weight_divisor)
                                               + (1 - y_true_flat) * K.log(1 - y_pred_flat), axis=-1)
        return weighted_binary_crossentropy

    def seg_loss_no_weight(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        return objectives.binary_crossentropy(y_true_flat, y_pred_flat)
    
    if loss_weight_divisor:
        unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=None)
    else:
        unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss_no_weight, metrics=None)
    return unet


def atrous_conv_net(depth, loss_weight):

    # set image specifics
    k = 3  # kernel size
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    n_filters = 16
    padding = 'same'
    
    inputs = Input((None, None, img_ch))
    list_out = [inputs]
    # encode
    for index in range(depth):
        conv = conv_blocks(2, list_out[-1], n_filters, (k, k), dilation_rate=(2 ** index, 2 ** index), padding=padding)
        list_out.append(conv)
    # decode
    for index in range(depth, 2 * depth - 1):
        concat = Concatenate(axis=3)([list_out[index], list_out[2 * depth - 1 - index]])
        conv = conv_blocks(2, concat, n_filters, (k, k), dilation_rate=(2 ** (2 * depth - 2 - index), 2 ** (2 * depth - 2 - index)), padding=padding)
        list_out.append(conv)
    # sigmoid 
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(list_out[-1])
    
    unet = Model(inputs, outputs)
    
    def seg_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        y_pred_flat = K.clip(y_pred_flat, K.epsilon(), 1 - K.epsilon())
        
        weighted_binary_crossentropy = -K.mean(y_true_flat * K.log(y_pred_flat) * loss_weight
                                               + (1 - y_true_flat) * K.log(1 - y_pred_flat), axis=-1)
        return weighted_binary_crossentropy
    
    unet.compile(optimizer=Adam(lr=0, beta_1=0.5), loss=seg_loss, metrics=None)

    return unet


def conv_blocks(n_block, x, n_filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1)):
    conv = x
    for _ in range(n_block):
        conv = Conv2D(n_filters, filter_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(conv)
        conv = BatchNormalization(scale=False, axis=3)(conv)
        conv = Activation('relu')(conv) 
    return conv


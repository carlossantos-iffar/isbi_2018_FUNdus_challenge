import os

from keras import backend as K, objectives
from keras.layers import Conv2D, UpSampling2D, Dense, Multiply, AveragePooling2D, Input
from keras.layers.core import Activation, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras import regularizers

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')


def set_optimizer(network):
    def L2(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, 0, 4)
        return objectives.mean_squared_error(y_true, y_pred_clipped)
    network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=L2, metrics=['accuracy'])
    return network

def dr_dense_from_scratch(pretrained_network):
    # change names and fix layers
    for layer in pretrained_network.layers:
        layer.name = "pretrained_" + layer.name

    gap=pretrained_network.get_layer("pretrained_global_average_pooling2d_1").output
    output = Dense(1)(gap)

    network = Model(pretrained_network.get_layer("pretrained_input_1").input, output)
    
    network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.mean_squared_error, metrics=['accuracy'])

    return network 

def dme_classifier(EX_segmentor, fovea_segmentor):
    s = 2
    k = 3
    n_out = 3
    n_filters = 32
    padding = "same"
    
    # change names
    for layer in EX_segmentor.layers:
        layer.name = "EX_segmentor_" + layer.name
    for layer in fovea_segmentor.layers:
        layer.name = "fovea_segmentor_" + layer.name
    
    # set inputs
    ex_fundus_input = EX_segmentor.get_layer("EX_segmentor_input_1").input
    fovea_fundus_input = fovea_segmentor.get_layer("fovea_segmentor_input_2").input
    fovea_vessel_input = fovea_segmentor.get_layer("fovea_segmentor_input_1").input

    # get intermediate layers
    ex_penultimate_layer = EX_segmentor.get_layer("EX_segmentor_activation_32").output
    fovea_activation_layer = fovea_segmentor.get_layer("fovea_segmentor_conv2d_25").output
    ex_average_pooled = AveragePooling2D((16, 16))(ex_penultimate_layer)
    
    attended_layer = Multiply()([ex_average_pooled, fovea_activation_layer])
    attended_GAP = GlobalAveragePooling2D()(attended_layer)
    non_attended_GAP = GlobalAveragePooling2D()(ex_average_pooled)
    
    last_layer = ex_fundus_input
    for index in range(4):
        conv = conv_blocks(1, last_layer, n_filters * (2 ** index), (k, k), "relu", padding=padding)
        conv = conv_blocks(1, last_layer, n_filters * (2 ** index), (k, k), "relu", padding=padding, strides=(s, s))
        last_layer = conv
    last_layer = Activation("sigmoid")(last_layer)
    conv_gap = GlobalAveragePooling2D()(last_layer)
    concat = Concatenate(axis=1)([attended_GAP, non_attended_GAP, conv_gap])
 
    output = Dense(n_out, activation="softmax")(concat)
  
    network = Model([ex_fundus_input, fovea_fundus_input, fovea_vessel_input], output)
    
    network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.categorical_crossentropy, metrics=['accuracy'])

    return network


def dme_classifier_v0(EX_segmentor, fovea_localizer):
    s = 2
    k = 3
    n_out = 3
    
    # change names
    for layer in EX_segmentor.layers:
        layer.name = "EX_segmentor_" + layer.name
    for layer in fovea_localizer.layers:
        layer.name = "fovea_localizer_" + layer.name
    
    # set inputs
    ex_fundus_layer = EX_segmentor.get_layer("EX_segmentor_input_1")
    fovea_fundus_layer = fovea_localizer.get_layer("fovea_localizer_input_2")
    fovea_vessel_layer = fovea_localizer.get_layer("fovea_localizer_input_1")

    # get intermediate layers
    ex_segmentor_layer = EX_segmentor.get_layer("EX_segmentor_activation_32").output
#     ex_segmentor_activation_layer = EX_segmentor.get_layer("EX_segmentor_conv2d_33").output
#     fovea_penultimate_layer = fovea_localizer.get_layer("fovea_localizer_activation_20").output
    fovea_activation_layer = fovea_localizer.get_layer("fovea_localizer_conv2d_25").output
    
    # upsample 40x40x1 feature map 
    last_layer = fovea_activation_layer
    for _ in range(3):
        up = UpSampling2D(size=(s, s))(last_layer)
        last_layer = conv_blocks(1, up, 1, (k, k), "relu")
    up = UpSampling2D(size=(s, s))(last_layer)
    attention = conv_blocks(1, up, 1, (k, k), "sigmoid")
    
    attended_layer = Multiply()([ex_segmentor_layer, attention])
    attended_GAP = GlobalAveragePooling2D()(attended_layer)
    non_attended_GAP = GlobalAveragePooling2D()(ex_segmentor_layer)

    concat = Concatenate(axis=1)([attended_GAP, non_attended_GAP])
 
    output = Dense(n_out, activation="softmax")(concat)
  
    network = Model([ex_fundus_layer.input, fovea_fundus_layer.input, fovea_vessel_layer.input], output)
    
    network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.categorical_crossentropy, metrics=['accuracy'])

    return network


def basic_block(x, n_blocks, n_initial_filters, max_n_filters, k, s, padding):
    last_layer = x
    for index in range(n_blocks):
        conv = conv_blocks(2, last_layer, min(n_initial_filters * (2 ** index), max_n_filters), (k, k), "relu", padding=padding)
        pool = MaxPooling2D(pool_size=(s, s))(conv)
        last_layer = pool
    return conv_blocks(2, last_layer, min(n_initial_filters * (2 ** n_blocks) , max_n_filters), (k, k), "relu", padding=padding)


def dr_classifier(EX_segmentor, HE_segmentor, MA_segmentor, SE_segmentor, loss_type):
    s = 2
    k = 3
    padding = "same"
    max_n_filters = 512
    n_filters = 32
    depth = 3
    
    # build a convolution branch     
    conv_input = Input((None, None, depth))
    conv_branch = basic_block(conv_input, 6, n_filters, max_n_filters, k, s, padding)
    
    # change names and fix layers
    for layer in EX_segmentor.layers:
        layer.name = "EX_segmentor_" + layer.name
        layer.trainable = False
    for layer in HE_segmentor.layers:
        layer.name = "HE_segmentor_" + layer.name
        layer.trainable = False
    for layer in MA_segmentor.layers:
        layer.name = "MA_segmentor_" + layer.name
        layer.trainable = False
    for layer in SE_segmentor.layers:
        layer.name = "SE_segmentor_" + layer.name
        layer.trainable = False
    
    # set inputs
    ex_input = EX_segmentor.get_layer("EX_segmentor_input_1").input
    he_input = HE_segmentor.get_layer("HE_segmentor_input_1").input
    ma_input = MA_segmentor.get_layer("MA_segmentor_input_1").input
    se_input = SE_segmentor.get_layer("SE_segmentor_input_1").input

    # get first or second bottleneck layers 
    ex_bottleneck_layer = EX_segmentor.get_layer("EX_segmentor_activation_14").output
    he_bottleneck_layer = HE_segmentor.get_layer("HE_segmentor_activation_14").output
    ma_bottleneck_layer = MA_segmentor.get_layer("MA_segmentor_activation_6").output
    ma_bottleneck_layer_down = basic_block(ma_bottleneck_layer, 4, n_filters * (2 ** 2), max_n_filters, k, s, padding)
    se_bottleneck_layer = SE_segmentor.get_layer("SE_segmentor_activation_10").output
    se_bottleneck_layer_down = basic_block(se_bottleneck_layer, 2, n_filters * (2 ** 4), max_n_filters, k, s, padding)

    concat = Concatenate(axis=3)([ex_bottleneck_layer, he_bottleneck_layer, se_bottleneck_layer_down, ma_bottleneck_layer_down, conv_branch])
    final_conv = conv_blocks(2, concat, max_n_filters, (k, k), "relu", padding=padding)
    
    gap = GlobalAveragePooling2D()(final_conv)
    output = Dense(1)(gap)

    network = Model([ex_input, he_input, ma_input, se_input, conv_input], output)
    
    def smooth_L1(y_true, y_pred):
        x = K.abs(y_true - y_pred)
        HUBER_DELTA = 1
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  K.sum(x)

    if loss_type == "smooth_L1":
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=smooth_L1, metrics=['accuracy'])
    else:
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.mean_squared_error, metrics=['accuracy'])

    return network


def dr_classifier_from_features(loss_type):
    s = 2
    k = 3
    padding = "same"
    max_n_filters = 256
    n_filters = 32
    feature_shape_ex_he = (10, 10, 512)
    feature_shape_se = (10, 10, 512)
    feature_shape_ma = (10, 10, 512)
    img_shape = (640, 640, 3)
    
    # build a convolution branch     
    conv_input = Input(img_shape)
    conv_branch = basic_block(conv_input, 6, n_filters, max_n_filters, k, s, padding)
    conv_gap = GlobalAveragePooling2D()(conv_branch)
    
    # set inputs
    ex_input = Input(feature_shape_ex_he)
    he_input = Input(feature_shape_ex_he)
    ma_input = Input(feature_shape_ma)
    se_input = Input(feature_shape_se)

    ex_shrinked = Conv2D(n_filters, (1, 1))(ex_input)
    he_shrinked = Conv2D(n_filters, (1, 1))(he_input)
    ma_shrinked = Conv2D(n_filters, (1, 1))(ma_input)
    se_shrinked = Conv2D(n_filters, (1, 1))(se_input)
    
    ex_gap = GlobalAveragePooling2D()(ex_shrinked)
    he_gap = GlobalAveragePooling2D()(he_shrinked)
    ma_gap = GlobalAveragePooling2D()(ma_shrinked)
    se_gap = GlobalAveragePooling2D()(se_shrinked)
    
    concat = Concatenate(axis=1)([conv_gap, ex_gap, he_gap, ma_gap, se_gap])
    output = Dense(1)(concat)

    network = Model([ex_input, he_input, ma_input, se_input, conv_input], output)
    
    def smooth_L1(y_true, y_pred):
        x = K.abs(y_true - y_pred)
        HUBER_DELTA = 1
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  K.sum(x)

    if loss_type == "smooth_L1":
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=smooth_L1, metrics=['accuracy'])
    else:
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.mean_squared_error, metrics=['accuracy'])

    return network


def dr_classifier_from_features_no_conv_branch(loss_type):
    feature_shape_ex_he = (10, 10, 512)
    feature_shape_se = (10, 10, 512)
    feature_shape_ma = (10, 10, 512)
    
    # set inputs
    ex_input = Input(feature_shape_ex_he)
    he_input = Input(feature_shape_ex_he)
    ma_input = Input(feature_shape_ma)
    se_input = Input(feature_shape_se)

    feature_concat = Concatenate(axis=3)([ex_input, he_input, ma_input, se_input])
#     feature_comb = Conv2D(128, (1, 1))(feature_concat)
    feature_comb = conv_blocks(2, feature_concat, 256, (1, 1), "relu")
    feature_comb = GlobalAveragePooling2D()(feature_comb)
    output = Dense(1)(feature_comb)

    network = Model([ex_input, he_input, ma_input, se_input], output)
    
    def smooth_L1(y_true, y_pred):
        x = K.abs(y_true - y_pred)
        HUBER_DELTA = 1
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  x

    if loss_type == "smooth_L1":
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=smooth_L1, metrics=['accuracy'])
    else:
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=objectives.mean_squared_error, metrics=['accuracy'])

    return network


def conv_blocks(n_block, x, n_filters, filter_size, activation, padding='same', strides=(1, 1), dilation_rate=(1, 1)):
    conv = x
    for _ in range(n_block):
        conv = Conv2D(n_filters, filter_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(conv)
        conv = BatchNormalization(scale=False, axis=3)(conv)
        conv = Activation(activation)(conv) 
    return conv


def resnet_buliding_block(x, n_filters, filter_size, padding, l2_coeff):
    conv1 = Conv2D(n_filters, filter_size, padding=padding,
                   kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(x)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)   

    conv2 = Conv2D(n_filters, filter_size, padding=padding,
                   kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(conv1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Add()([conv2, x])
    
    out = Activation('relu')(conv2)   
    
    return out


def resnet_blocks(x, n, n_filters, filter_size, padding, l2_coeff):
    for _ in range(n):
        x = resnet_buliding_block(x, n_filters, filter_size, padding, l2_coeff)
    return x


def conv_block(x, n_filters, filter_size, strides, padding, l2_coeff):
    conv = Conv2D(n_filters, filter_size, strides=strides, padding=padding,
                  kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(x)
    conv = BatchNormalization(scale=False, axis=3)(conv)
    conv = Activation('relu')(conv) 
    return conv


def dr_network_segmentation_as_input():

    # set image specifics
    img_shape = (512, 512, 7)
    n_filters = 32
    n_out = 1
    k = 3  # kernel size
    s = 2  # stride
    img_h, img_w, img_ch = img_shape[0], img_shape[1], img_shape[2]
    padding = 'same'
    l2_coeff = 0.0005
    list_n_building_blocks = [2, 3]
    blocks = []
    
    inputs = Input((img_h, img_w, img_ch))
    reduction_conv = conv_block(inputs, n_filters, (k, k), (s, s), padding, l2_coeff)
    blocks.append(conv_block(reduction_conv, n_filters, (k, k), (s, s), padding, l2_coeff))
    for index, n_building_blocks in enumerate(list_n_building_blocks):
        blocks.append(resnet_blocks(blocks[index], n_building_blocks, (2 ** index) * n_filters, (k, k), padding, l2_coeff))
        blocks[index + 1] = conv_block(blocks[index + 1], 2 ** (index + 1) * n_filters, (k, k), (s, s), padding, l2_coeff)
    
    list_avg_pools = []
    for i in range(2):
        list_avg_pools.append(AveragePooling2D((4 // (2 ** i), 4 // (2 ** i)))(blocks[i]))
    blocks_concat = Concatenate(axis=3)(list_avg_pools + [blocks[-1]])
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), padding=padding,
                                    kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(2, 2), padding=padding,
                                     kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(4, 4), padding=padding,
                                    kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    final_block = Concatenate(axis=3)(list_dilated_conv)
    final_block = conv_block(final_block, 16 * n_filters, (k, k), (s, s), padding, l2_coeff)
    gap = GlobalAveragePooling2D()(final_block)
    outputs = Dense(n_out)(gap)
    
    network = Model(inputs, outputs)    

    def L2(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, 0, 4)
        return objectives.mean_squared_error(y_true, y_pred_clipped)

    network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=L2, metrics=['accuracy'])

    return network 


def dr_network(loss_type):

    # set image specifics
    img_shape = (512, 512, 3)
    n_filters = 32
    n_out = 1
    k = 3  # kernel size
    s = 2  # stride
    img_h, img_w, img_ch = img_shape[0], img_shape[1], img_shape[2]
    padding = 'same'
    l2_coeff = 0.0005
    list_n_building_blocks = [2, 3, 4]
    blocks = []
    
    inputs = Input((img_h, img_w, img_ch))
    blocks.append(conv_block(inputs, n_filters, (k, k), (s, s), padding, l2_coeff))
    for index, n_building_blocks in enumerate(list_n_building_blocks):
        blocks.append(resnet_blocks(blocks[index], n_building_blocks, (2 ** index) * n_filters, (k, k), padding, l2_coeff))
        blocks[index + 1] = conv_block(blocks[index + 1], 2 ** (index + 1) * n_filters, (k, k), (s, s), padding, l2_coeff)
    
    list_avg_pools = []
    for i in range(3):
        list_avg_pools.append(AveragePooling2D((8 // (2 ** i), 8 // (2 ** i)))(blocks[i]))
    blocks_concat = Concatenate(axis=3)(list_avg_pools + [blocks[3]])
    list_dilated_conv = []
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), padding=padding,
                                    kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(2, 2), padding=padding,
                                     kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    list_dilated_conv.append(Conv2D(16 * n_filters, (k, k), dilation_rate=(4, 4), padding=padding,
                                    kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))(blocks_concat))
    final_block = Concatenate(axis=3)(list_dilated_conv)
    final_block = conv_block(final_block, 16 * n_filters, (k, k), (s, s), padding, l2_coeff)
    gap = GlobalAveragePooling2D()(final_block)
    outputs = Dense(n_out)(gap)
    
    network = Model(inputs, outputs)    
    
    def smooth_L1(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, 0, 4)
        x = K.abs(y_true - y_pred_clipped)
        HUBER_DELTA = 1
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  x

    def L2(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, 0, 4)
        return objectives.mean_squared_error(y_true, y_pred_clipped)

    if loss_type == "smooth_L1":
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=smooth_L1, metrics=['accuracy'])
    elif loss_type == "L2":
        network.compile(optimizer=SGD(lr=0, momentum=0.9, nesterov=True), loss=L2, metrics=['accuracy'])

    return network 


def bottleneck_extractor(network, layer_name, additional_max_pool):
    # change names and fix layers
    for layer in network.layers:
        layer.name = "pretrained_" + layer.name
        layer.trainable = False
    
    # maxpool if necessary
    s = 2
    output = network.get_layer("pretrained_" + layer_name).output
    if additional_max_pool == 4:  # MA
        output_maxpool = output
        output_avgpool = output
        for _ in range(2):
            output_maxpool = MaxPooling2D(pool_size=(s, s))(output_maxpool)
            output_avgpool = AveragePooling2D(pool_size=(s, s))(output_avgpool)
        
        output_maxpool_maxpool = output_maxpool
        output_maxpool_avgpool = output_maxpool
        output_avgpool_maxpool = output_avgpool
        output_avgpool_avgpool = output_avgpool
        for _ in range(2):
            output_maxpool_maxpool = MaxPooling2D(pool_size=(s, s))(output_maxpool_maxpool)
            output_avgpool_maxpool = MaxPooling2D(pool_size=(s, s))(output_avgpool_maxpool)
            output_maxpool_avgpool = AveragePooling2D(pool_size=(s, s))(output_maxpool_avgpool)
            output_avgpool_avgpool = AveragePooling2D(pool_size=(s, s))(output_avgpool_avgpool)
            
        output = Concatenate(axis=3)([output_maxpool_maxpool, output_maxpool_avgpool, output_avgpool_maxpool, output_avgpool_avgpool])
    else:    
        for _ in range(additional_max_pool):
            output = MaxPooling2D(pool_size=(s, s))(output)

    return Model(network.get_layer("pretrained_input_1").input, output)


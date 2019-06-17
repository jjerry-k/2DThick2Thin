from keras import backend as K
from keras import layers, models, activations, optimizers
import sys
sys.path.append(['.','..'])
from utils import *

def generator(model = 'vgg', n_slice=6, case=1):
    '''
    model : 'vgg', 'resnet', 'xception', 'mobile', 'dense'
    '''
    block_dict = {
        "vgg" : ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'],
        "resnet" : ['activation_40', 'activation_22', 'activation_10', 'activation_1'],
        "xception" : ['block13_sepconv2_bn', 'block4_sepconv2_bn', 'block3_sepconv2_bn', 'block1_conv1_act'],
        "mobile" : ['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'], 
        "dense" : ['pool4_conv', 'pool3_conv', 'pool2_conv', 'conv1/relu']
    }
    # ========= Encoder ==========
    print("=========== Information about Backbone ===========")
    base_model = load_base_model(model, input_shape=(None, None, 3))
    x = layers.Conv2D(1024, 3, padding='same', activation='relu')(base_model.output) # H/32

    # ========= Decoder ==========
    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/16
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][0]).output], axis = -1)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/8
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][1]).output], axis = -1)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/4
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][2]).output], axis = -1)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/2
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][3]).output], axis = -1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H
    if model == 'vgg':
        x = layers.concatenate([x, base_model.get_layer(block_dict[model][4]).output], axis = -1)
    output = layers.Conv2D(n_slice, 3, padding='same', activation='relu')(x)

    #output = layers.DepthwiseConv2D(3, padding='same')(x)
    if case == 2:
        Net = models.Model(base_model.input, output)
    else : 
        Net = models.Model(base_model.input, [output, output])
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def discriminator(model = 'vgg', n_slice=6, case=1):
    '''
    model : 'vgg', 'resnet', 'xception', 'mobile', 'dense'
    '''
    block_dict = {
        "vgg" : ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'],
        "resnet" : ['activation_40', 'activation_22', 'activation_10', 'activation_1'],
        "xception" : ['block13_sepconv2_bn', 'block4_sepconv2_bn', 'block3_sepconv2_bn', 'block1_conv1_act'],
        "mobile" : ['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'], 
        "dense" : ['pool4_conv', 'pool3_conv', 'pool2_conv', 'conv1/relu']
    }
    # ========= Extractor ==========
    print("=========== Information about Backbone ===========")
    base_model = load_base_model(model, input_shape=(None, None, n_slice))
    texture = layers.Conv2D(1024, 3, padding='same', activation='relu')(base_model.output) # H/32
    
    # ========= Classifier ==========
    x = layers.GlobalAvgPool2D()(texture)
    output = layers.Dense(1, activation='sigmoid')(x)
    if case==2:
        Net = models.Model(base_model.input, [output, texture])
    else:
        Net = models.Model(base_model.input, output)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net
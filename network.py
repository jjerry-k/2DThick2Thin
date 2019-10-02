import tensorflow as tf
from tensorflow.keras import layers, models, activations, optimizers
import sys
sys.path.append(['.','..'])
from utils import *
from func import *

tf_version = float(tf.__version__[:-2])

UpSampling2D = layers.UpSampling2D(interpolation='bilinear') if tf_version > 1.12 else layers.UpSampling2D()

def generator(model = 'vgg', n_slice=6, case=None):
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
    x = UpSampling2D(x) # H/16
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][0]).output], axis = -1)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    x = UpSampling2D(x) # H/8
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][1]).output], axis = -1)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = UpSampling2D(x) # H/4
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][2]).output], axis = -1)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = UpSampling2D(x) # H/2
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][3]).output], axis = -1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    aux = layers.Conv2D(n_slice, 3, padding='same', activation='relu')(x)
    
    x = UpSampling2D(x) # H
    if model == 'vgg':
        x = layers.concatenate([x, base_model.get_layer(block_dict[model][4]).output], axis = -1)
    output = layers.Conv2D(n_slice, 3, padding='same', activation='relu')(x)

    #output = layers.DepthwiseConv2D(3, padding='same')(x)
    if case == 'aux':
        Net = models.Model(base_model.input, [aux, output])
    else:
        Net = models.Model(base_model.input, output)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def discriminator(model = 'vgg', n_slice=6):
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
    x = layers.Conv2D(1024, 3, padding='same')(base_model.output) # H/32
    x = layers.LeakyReLU()(x)
    
    # ========= Classifier ==========
    x = layers.GlobalAvgPool2D()(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    Net = models.Model(base_model.input, output)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net


def unet(in_slice = 3, out_slice=6, case=None, name = 'unet'):
    x = layers.Input(shape=(None, None, in_slice))
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(x)
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(block1)
    
    pool1 = layers.Conv2D(64, 2, strides=(2, 2), name='block1_conv3')(block1)
    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(pool1)
    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(block2)
    
    pool2 = layers.Conv2D(128, 2, strides=(2, 2), name='block2_conv3')(block2)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(pool2)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(block3)
    
    pool3 = layers.Conv2D(256, 2, strides=(2, 2), name='block3_conv4')(block3)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(pool3)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(block4)
    
    pool4 = layers.Conv2D(512, 2, strides=(2, 2), name='block4_conv4')(block4)
    block5 = layers.Conv2D(1024, 3, padding='same', activation='relu', name='block5_conv1')(pool4)
    block5 = layers.Conv2D(1024, 3, padding='same', activation='relu', name='block5_conv2')(block5)
    
    unpool1 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(block5)
    concat1 = layers.Concatenate(axis = 3)([unpool1, block4])
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv1')(concat1)
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv2')(block6)
    
    unpool2 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(block6)
    concat2 = layers.Concatenate(axis = 3)([unpool2, block3])
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv1')(concat2)
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv2')(block7)
    
    unpool3 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(block7)
    concat3 = layers.Concatenate(axis = 3)([unpool3, block2])
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv1')(concat3)
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv2')(block8)
    aux = layers.Conv2D(out_slice, 3, padding='same', activation='relu', name='auxilary')(block8)
    
    unpool4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(block8)
    concat4 = layers.Concatenate(axis = 3)([unpool4, block1])
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv1')(concat4)
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv2')(block9)
    
    output = layers.Conv2D(out_slice, 3, padding='same', activation='relu', name='output')(block9)
    
    if case == 'aux':
        Net = models.Model(x, [aux, output], name = name)
    else:
        Net = models.Model(x, output, name = name)
        
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def Gen2Dis(gen, dis, case=None):
    
    if case=='aux':
        dis_out = dis(gen.output[1])
        net = models.Model(inputs=gen.input, outputs=[gen.output[0], gen.output[1], dis_out])
    
    else:
        dis_out = dis(gen.output)
        net = models.Model(inputs=gen.input, outputs=[gen.output, dis_out])
    
    return net


def conv_block(input_layer, n_filters, ksize, strides=(1, 1), padding='same', mode='conv', name='conv_block'):
    if mode=='conv':
        output = layers.Conv2D(n_filters, ksize, strides=strides, padding=padding, name=name+"_conv")(input_layer)
    elif mode =='upconv':
        output = layers.Conv2DTranspose(n_filters, ksize, strides=strides, padding=padding, name=name+"_upconv")(input_layer)
    output = layers.BatchNormalization(name=name+"_bn")(output)
    output = layers.Activation('relu', name=name+"_act")(output)
    return output



def unet_cs(image_shape=(None, None, 1), case=None, name = 'unet'):
    x = layers.Input(shape=image_shape)
    #noise = layers.GaussianNoise(0.1, name="Gaussian_Noise")(x)
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(x)
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(block1)
    
    pool1 = layers.Conv2D(64, 2, strides=(2, 2), name='block1_conv3')(block1)
    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(pool1)
    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(block2)
    
    pool2 = layers.Conv2D(128, 2, strides=(2, 2), name='block2_conv3')(block2)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(pool2)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(block3)
    
    pool3 = layers.Conv2D(256, 2, strides=(2, 2), name='block3_conv4')(block3)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(pool3)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(block4)
    
    pool4 = layers.Conv2D(512, 2, strides=(2, 2), name='block4_conv4')(block4)
    block5 = layers.Conv2D(1024, 3, padding='same', activation='relu', name='block5_conv1')(pool4)
    block5 = layers.Conv2D(1024, 3, padding='same', activation='relu', name='block5_conv2')(block5)
    
    unpool1 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same', name='block6_upconv')(block5)
    concat1 = layers.Concatenate(axis = 3)([block4, unpool1])
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv1')(concat1)
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv2')(block6)
    
    unpool2 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same', name='block7_upconv')(block6)                          
    concat2 = layers.Concatenate(axis = 3)([block3, unpool2])
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv1')(concat2)
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv2')(block7)
    
    unpool3 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', name='block8_upconv')(block7)
    concat3 = layers.Concatenate(axis = 3)([block2, unpool3])
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv1')(concat3)
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv2')(block8)
    aux = layers.Conv2D(image_shape[-1], 3, padding='same', activation='relu', name='auxilary')(block8)
    
    unpool4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', name='block9_upconv')(block8)
    concat4 = layers.Concatenate(axis = 3)([block1, unpool4])
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv1')(concat4)
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv2')(block9)
    
    output = layers.Conv2D(image_shape[-1], 3, padding='same', name='output')(block9)
    
    if case == 'aux':
        Net = models.Model(x, [aux, output], name = name)
    else:
        Net = models.Model(x, output, name = name)
        
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def unet_cs_bn(input_shape=(None, None, 1, ), last_activation='linear', name="unet"):
    input_layer = layers.Input(shape=input_shape, name=name+"_input")
    
    en1 = conv_block(input_layer, 64, 3, name=name+"_en1_block1")
    en1 = conv_block(en1, 64, 3, name=name+"_en1_block2")
    
    en2 = layers.MaxPool2D(pool_size=(2, 2), name=name+"_pool1")(en1)
    en2 = conv_block(en2, 128, 3, name=name+"_en2_block1")
    en2 = conv_block(en2, 128, 3, name=name+"_en2_block2")
    
    en3 = layers.MaxPool2D(pool_size=(2, 2), name=name+"_pool2")(en2)
    en3 = conv_block(en3, 256, 3, name=name+"_en3_block1")
    en3 = conv_block(en3, 256, 3, name=name+"_en3_block2")
    
    en4 = layers.MaxPool2D(pool_size=(2, 2), name=name+"_pool3")(en3)
    en4 = conv_block(en4, 512, 3, name=name+"_en4_block1")
    en4 = conv_block(en4, 512, 3, name=name+"_en4_block2")
    
    en5 = layers.MaxPool2D(pool_size=(2, 2), name=name+"_pool4")(en4)
    en5 = conv_block(en5, 1024, 3, name=name+"_en5_block1")
    en5 = conv_block(en5, 1024, 3, name=name+"_en5_block2")
    
    de4 = conv_block(en5, 512, (2, 2), strides=(2, 2), padding='same', mode='upconv', name=name+"_de4_block1")
    de4 = layers.concatenate([en4, de4], name=name+"_de4_concat")
    de4 = conv_block(de4, 512, 3, name=name+"_de4_block2")
    de4 = conv_block(de4, 512, 3, name=name+"_de4_block3")
    
    de3 = conv_block(de4, 256, (2, 2), strides=(2, 2), padding='same', mode='upconv', name=name+"_de3_block1")
    de3 = layers.concatenate([en3, de3], name=name+"_de3_concat")
    de3 = conv_block(de3, 256, 3, name=name+"_de3_block2")
    de3 = conv_block(de3, 256, 3, name=name+"_de3_block3")
    
    de2 = conv_block(de3, 128, (2, 2), strides=(2, 2), padding='same', mode='upconv', name=name+"_de2_block1")
    de2 = layers.concatenate([en2, de2], name=name+"_de2_concat")
    de2 = conv_block(de2, 128, 3, name=name+"_de2_block2")
    de2 = conv_block(de2, 128, 3, name=name+"_de2_block3")
    
    de1 = conv_block(de2, 64, (2, 2), strides=(2, 2), padding='same', mode='upconv', name=name+"_de1_block1")
    de1 = layers.concatenate([en1, de1], name=name+"_de1_concat")
    de1 = conv_block(de1, 64, 3, name=name+"_de1_block2")
    de1 = conv_block(de1, 64, 3, name=name+"_de1_block3")
    
    output = layers.Conv2D(1, 1, activation=last_activation, name=name+"_output")(de1)
    return models.Model(inputs=input_layer, outputs=output, name=name)


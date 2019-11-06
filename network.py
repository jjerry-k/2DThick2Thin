from tensorflow.keras import models, layers

def conv_2d_block(input_layer, n_filter, ksize, padding='same', activation='relu', name='block'):
    output = layers.Conv2D(n_filter, ksize, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def upconv_2d_block(input_layer, n_filter, ksize, strides, padding='same', activation='relu', name='block'):
    output = layers.Conv2DTranspose(n_filter, ksize, strides, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def residual_2d_block(input_layer, n_filter, ksize, padding='same', activation='relu', mode=None, name='block'):
    '''
    Reference : https://arxiv.org/pdf/1612.02177.pdf
    '''
    if mode == "up":
        output = upconv_2d_block(input_layer, n_filter, (1, 6), strides=(1, 6), padding='same', 
                         activation=activation, name=name+'_up')
    else:
        output = conv_2d_block(input_layer, n_filter, ksize, activation=activation, name=name+'_conv1')
    output = conv_2d_block(output, n_filter, ksize, activation='linear', name=name+'_conv2')
    output = layers.Add(name=name+'_add')([output, input_layer])
    
    return output

def Axi_SR(in_slice = 3, out_slice=6, name = 'unet'):
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
    
    unpool4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(block8)
    concat4 = layers.Concatenate(axis = 3)([unpool4, block1])
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv1')(concat4)
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv2')(block9)
    
    output = layers.Conv2D(out_slice, 3, padding='same', activation='relu', name='output')(block9)
    
    Net = models.Model(x, output, name = name)
        
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def COR_SR2D(input_shape=(None, None, 1), layer_activation='relu', last_activation='linear', name='2D_SR'):
    
    '''
    input_shape : (Cor, Sag, Axi)
    '''
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, 64, 3, activation=layer_activation, name=name+'_en1')
    
    for_concat = layers.UpSampling2D(size=(6, 1), name=name+'_up_en1')(en1)
    
    en2 = conv_2d_block(en1, 128, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = conv_2d_block(en2, 256, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = conv_2d_block(en3, 512, 3, activation=layer_activation, name=name+'_en4')
    
    up = upconv_2d_block(en4, 64, (6, 1), strides=(6, 1), padding='same', 
                         activation=layer_activation, name=name+'_up')
    concat = layers.Concatenate(axis=-1, name=name+'_concat')([for_concat, up])
    
    refine = conv_2d_block(concat, 64, 3, activation=last_activation, name=name+'_refine')
    
    output = conv_2d_block(refine, 1, 1, activation=last_activation, name=name+'_output')
    
    Net = models.Model(input_layer, output, name = name)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    
    return Net



def COR_SR2D_res(input_shape=(None, None, 1), residual_channel=64, layer_activation='relu', last_activation='linear', name='2D_SR'):
    
    '''
    input_shape : (Sag, Cor, Axi)
    '''
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, residual_channel, 3, activation=layer_activation, name=name+'_en1')
    
    for_concat = layers.UpSampling2D(size=(6, 1), name=name+'_up_en1')(en1)
    
    en2 = residual_2d_block(en1, residual_channel, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = residual_2d_block(en2, residual_channel, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = residual_2d_block(en3, residual_channel, 3, activation=layer_activation, name=name+'_en4')
    
    up = upconv_2d_block(en4, residual_channel, (6, 1), strides=(6, 1), padding='same', 
                         activation=layer_activation, name=name+'_up')
    
    concat = layers.Concatenate(axis=-1, name=name+'_concat')([for_concat, up])
    
    refine = conv_2d_block(concat, residual_channel, 3, activation=last_activation, name=name+'_refine')
    
    output = conv_2d_block(refine, 1, 1, activation=last_activation, name=name+'_output')
    
    Net = models.Model(inputs=input_layer, outputs=output, name=name)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    
    return Net


def dis2D(input_shape=(None, None, 1), layer_activation='relu', last_activation='sigmoid', name='2D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, 64, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool2D(name=name+'_pool1')(en1)
    en2 = conv_2d_block(en1, 128, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool2D(name=name+'_pool2')(en2)
    en3 = conv_2d_block(en2, 256, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool2D(name=name+'_pool3')(en3)
    en4 = conv_2d_block(en3, 512, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool2D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    Net = models.Model(inputs=input_layer, outputs=output, name=name)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    
    return Net



def dis2D_res(input_shape=(None, None, 1), residual_channel=64, layer_activation='relu', last_activation='sigmoid', name='2D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, residual_channel, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool2D(name=name+'_pool1')(en1)
    en2 = residual_2d_block(en1, residual_channel, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool2D(name=name+'_pool2')(en2)
    en3 = residual_2d_block(en2, residual_channel, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool2D(name=name+'_pool3')(en3)
    en4 = residual_2d_block(en3, residual_channel, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool2D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    Net = models.Model(inputs=input_layer, outputs=output, name=name)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    
    return Net
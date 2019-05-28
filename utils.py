import numpy as np
import nibabel as nib

def load_nii(PATH):
    """
    Input
    PATH : Path of Nifti file
    
    Output
    output : Image of Nifti [Height, Width, Slices]
    """
    data = nib.load(PATH)
    img = data.get_data()
    img = np.rot90(img)
    
    return img

def load_nii_multi(PATH):
    """
    Input
    PATH : Path of patient
    
    Output
    t1_img : Image of patient's T1
    dante_img : Image of patient's DANTE
    t2_img : Image of patient's T2
    """
    SEQ_LISTS = ['t1setrafs', 'T1SPACE09mmISOPOSTwDANTE', 't2tsetra']
    t1_PATH = os.path.join(PAT_PATH, SEQ_LISTS[0])
    t1_rsl = [img for img in os.listdir(t1_PATH) if '_rsl' in img]
    t1_PATH = os.path.join(t1_PATH, t1_rsl[0])

    dante_PATH = os.path.join(PAT_PATH, SEQ_LISTS[1])
    dante_rsl = [img for img in os.listdir(dante_PATH) if '_rsl' in img]
    dante_PATH = os.path.join(dante_PATH, dante_rsl[0])

    t2_PATH = os.path.join(PAT_PATH, SEQ_LISTS[2])
    t2_rsl = [img for img in os.listdir(t2_PATH) if '_rsl' in img]
    t2_PATH = os.path.join(t2_PATH, t2_rsl[0])
    
    t1_img = load_nii(t1_PATH)
    dante_img = load_nii(dante_PATH)
    t2_img = load_nii(t2_PATH)
    
    return t1_img, dante_img, t2_img


from keras.applications import VGG16, ResNet50, Xception, MobileNet
from keras.applications import DenseNet121, NASNetMobile, InceptionResNetV2, InceptionV3
def load_base_model(backbone = 'vgg'):
    '''
    Loading backbone network
    ====== Input ======
    backbone : Network name of backbone 
        ['vgg', 'resnet', 'xception', 'mobile', 'IRv2', 'Iv3', 'dense', 'nas']
    weights : Path of Pretrained weights 
    include_top : whether to include the classifier(the 3 fully-connected layers)
    
    ====== Output ======
    base_model : Keras Model instance
    '''
    model_dict = {'vgg':'VGG16',
                 'resnet':'ResNet50',
                 'xception':'Xception',
                 'mobile':'MobileNet',
                 'IRv2':'InceptionResNetV2',
                 'Iv3':'InceptionV3',
                 'dense':'DenseNet121',
                 'nas':'NASNetMobile'}
    
    command = "base_model = %s(weights=None, include_top=False)"%model_dict[backbone]
    print("Loading %s model"%model_dict[backbone])
    exec(command, globals())
    non_train_params = [layer.shape.num_elements() for layer in base_model.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("Total Parameter of Model : ", format(base_model.count_params(), ','))
    print("Trainable Parameter of Model : ", format(base_model.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return base_model

from keras import backend as K

def layer_mean(x):
    x = K.mean(x, axis=-1, keepdims=True)
    return x
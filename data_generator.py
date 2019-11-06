import os, argparse
import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", dest="input_dir", type=str, help="Path of Input")
parser.add_argument("--val_num", dest="val_num", type=int, help="Validation Data Number", default=0)

args = parser.parse_args()

dict_args = vars(args)

for i in dict_args.keys():
    assert dict_args[i]!=None, '"%s" key is None Value!'%i
print("\n================ Training Options ================")
print("Input dir : ", args.input_dir)
print("Validation Number : ", args.val_num)
print("====================================================\n")



data_dir = args.input_dir
data_lists = sorted(os.listdir(data_dir))
si15_lists = [i for i in data_lists if '_siemens_15' in i]

print(len(si15_lists))


def check_data(img):
    output = None
    num_use = img.shape[-1]//6*6
    num_trash = img.shape[-1]%6
    num_bot = np.ceil(num_trash/2).astype(np.int8)
    num_top = np.floor(num_trash/2).astype(np.int8)
    check = 2*num_bot+num_top
    if check == 0:
        output = img
    elif check == 2:
        output = img[..., num_bot:]
    else :    
        output = img[...,num_bot:-(num_top)]
    return output

def DataLoader2D(root, data_list):

    '''
    Data Loader for Training, Validation data
    version.cor : complete
    '''
    labels = []
    for scan in data_list:
        img = nib.load(os.path.join(os.path.join(root, scan)))
        img = img.get_data().astype(np.uint16)
#             imgs.append(np.rot90(img, axes=(1,2)))
        labels.append(np.rot90(np.transpose(img, [1, 2, 0]), 2, axes=(1,2)))
        
    labels = np.array(labels)[:, :, 4:, :]
    n, c, a, s  = labels.shape
    labels = np.reshape(labels, [n*c, a, s])
    img = np.array(np.split(labels, 252//6, axis=1))
    img = np.mean(img, axis=2)
    img = np.transpose(img, [1, 0, 2])
    
    return img, labels


val_idx =[0, 18, 36]
test_idx = 54

if args.val_num == 3:
    img, lab = DataLoader2D(data_dir, si15_lists[test_idx:])
    np.save('./data/test_img', img)
    np.save('./data/test_lab', lab)
    print(img.shape, lab.shape)

else :
    start_idx = val_idx[args.val_num]
    img, lab = DataLoader2D(data_dir, si15_lists[start_idx:start_idx+18])
    np.save('./data/part_%02d_img'%(args.val_num), img)
    np.save('./data/part_%02d_lab'%(args.val_num), lab)
    print(img.shape, lab.shape)



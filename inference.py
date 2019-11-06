import os, tqdm, datetime, random, argparse
import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", dest="input_dir", type=str, help="Path of Input")
parser.add_argument("--model", dest="model", type=str, help="Path of Model")
parser.add_argument("--plain", dest="plain", type=str, help="Plain of Image (c or a)")
parser.add_argument("--val_num", dest="val_num", type=int, help="Validation Data Number", default=0)

args = parser.parse_args()

dict_args = vars(args)

for i in dict_args.keys():
    assert dict_args[i]!=None, '"%s" key is None Value!'%i
print("\n================ Training Options ================")
print("Input dir : ", args.input_dir)
print("Model : ", args.model)
print("Plain : ", args.plain)
print("Validation Number : ", args.val_num)
print("====================================================\n")

npy_list = [npy for npy in sorted(os.listdir(args.input_dir)) if '.npy' in npy and args.plain in npy and 'test' in npy]

test_x = np.load(os.path.join(args.input_dir, npy_list[0]))[..., np.newaxis]
test_y = np.load(os.path.join(args.input_dir, npy_list[1]))[..., np.newaxis]

print(test_x.shape, test_y.shape)

model_path = os.path.join(args.model, sorted(os.listdir(args.model))[args.val_num])

ckpt_lists = [ckpt for ckpt in sorted(os.listdir(model_path)) if '.h5' in ckpt]
print(ckpt_lists[-1])
ckpt_path = os.path.join(model_path, ckpt_lists[-1])

with open(os.path.join(model_path, "model.json"), "r") as json_file:
    model_json = json_file.read()

import tensorflow as tf
from pprint import pprint
from tensorflow.keras import models, layers, losses, optimizers

from loss import *
from utils import *
from metric import *
from network import *

random.seed(777)
tf.set_random_seed(777)
np.random.seed(777)
os.environ['PYTHONHASHSEED'] = '777'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

G = models.model_from_json(model_json)    
G.load_weights(ckpt_path)
G = models.Model(inputs = G.input, outputs = G.outputs[0])
G.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=mse_grad_loss, metrics=[mse, gradient_2d_loss, psnr])
print("\nModel Loaded!\n")

result = G.evaluate(test_x, test_y)

print(result)

pred = G.predict(test_x)

aff = np.eye(4)
aff[2, 2]=6


for i in range(6):
    start = i*256 
    tmp_x = np.transpose(np.rot90(test_x[i:i+256, ..., 0], 2, axes=(1,2)),  [2, 0, 1])
    tmp_y = np.transpose(np.rot90(test_y[i:i+256, ..., 0], 2, axes=(1,2)),  [2, 0, 1])
    tmp_pred = np.transpose(np.rot90(pred[i:i+256, ..., 0], 2, axes=(1,2)),  [2, 0, 1])
    
    nib.save(nib.Nifti1Image(tmp_x, aff), os.path.join(model_path.replace("checkpoint", "result"), '%02d_input.nii'%(i+1)))
    nib.save(nib.Nifti1Image(tmp_y, np.eye(4)), os.path.join(model_path.replace("checkpoint", "result"), '%02d_label.nii'%(i+1)))
    nib.save(nib.Nifti1Image(tmp_pred, np.eye(4)), os.path.join(model_path.replace("checkpoint", "result"), '%02d_pred.nii'%(i+1)))

    

#from IPython.display import clear_output
from tensorflow.keras import models, optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar
from loss import *
from utils import *
from network import *
from matplotlib import pyplot as plt

import os, cv2, random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(777)
np.random.seed(777)
tf.set_random_seed(777)
os.environ['PYTHONHASHSEED'] = '777'

sess = tf.Session()

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", dest="save_dir", type=str, help="Path of Save")
parser.add_argument("--input_dir", dest="input_dir", type=str, help="Path of Input")
parser.add_argument("--mode", dest="mode", type=str, help="Mode of I/O")
parser.add_argument("--val_num", dest="val_num", type=int, help="Validation Data Number", default=0)
parser.add_argument("--encoder", dest="encoder", type=str, help="Type of Encoder")
parser.add_argument("--loss", dest="loss", type=str, help="Type of Loss")
parser.add_argument("--batch", dest="batch", type=int, help="Size of Batch")
parser.add_argument("--epochs", dest="epochs", type=int, help="Number of Epochs")
parser.add_argument("--aux", dest="aux", type=bool, help="Number of Epochs", default=False)

args = parser.parse_args()

dict_args = vars(args)

for i in dict_args.keys():
    assert dict_args[i]!=None, '"%s" key is None Value!'%i
print("\n=======Training Options=======")
print("Save dir : ", args.save_dir)
print("Input dir : ", args.input_dir)
print("Mode : ", args.mode)
print("Encoder Type: ", args.encoder)
print("Loss Function: ", args.loss.upper())
print("Batch Size : ", args.batch)
print("Epochs : ", args.epochs)
print("Using Auxilary : ", args.aux)
print("==============================\n")
      
loss_dict = {"L1" : Custom_MAE, 
             "L2" : Custom_MSE,
             "RMSE" : Custom_RMSE,
             "SSIM" : Custom_SSIM,
             "L1SSIM" : multi_loss(1., 1., args.loss.upper()).loss,
             "L2SSIM" : multi_loss(1., 1., args.loss.upper()).loss, 
             "MI" : mutual_information, 
             "L1MI": multi_loss(1., 1., args.loss.upper()).loss,
             "L2MI" : multi_loss(1., 1000., args.loss.upper()).loss, 
             "L2GRAD": l2_and_gradient_loss}



if args.aux :
    case='aux'
else:
    case=None

if args.mode == '3to6':
    N_SLICES = 6
    Loader = data_loader_6
elif args.mode == '3to12':
    N_SLICES = 12
    Loader = data_loader_12

ROOT_DIR = args.save_dir
    
train_low, train_high, val_low, val_high, test_low, _ = Loader(args.input_dir, img=None)
if args.aux:
    b, h, w, s = train_high.shape
    train_aux = []
    for img in np.reshape(np.transpose(train_high, [0, 3, 1, 2]), [b*s, h, w]):
        train_aux.append(cv2.resize(img, None, fx=0.5, fy=0.5))
    train_aux = np.array(train_aux)
    train_aux = np.transpose(np.reshape(train_aux, [b, s, h//2, w//2]), [0, 2, 3, 1])
    
    b, h, w, s = val_high.shape
    val_aux = []
    for img in np.reshape(np.transpose(val_high, [0, 3, 1, 2]), [b*s, h, w]):
        val_aux.append(cv2.resize(img, None, fx=0.5, fy=0.5))
    val_aux = np.array(val_aux)
    val_aux = np.transpose(np.reshape(val_aux, [b, s, h//2, w//2]), [0, 2, 3, 1])
    
print("Train X's shape : ", train_low.shape)
print("Train Y's shape : ", train_high.shape)
print("Validation X's shape : ", val_low.shape)
print("Validation Y's shape : ", val_high.shape)
print("Test X's shape : ", test_low.shape)    
print("\n")



print("Build a Generator !")      
if args.encoder == "unet":
    G = unet(out_slice=N_SLICES, case=case)
else:
    G = generator(model=args.encoder, n_slice=N_SLICES, case=case)

    
print("\n===================================\n")


print("Build a Discriminator !")

D = discriminator(model='vgg', n_slice=N_SLICES)
D.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=losses.binary_crossentropy)

D.trainable=False

A = Gen2Dis(G, D, case=case)
if args.aux:
    A.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), 
          loss=[loss_dict[args.loss.upper()], loss_dict[args.loss.upper()], losses.binary_crossentropy], loss_weights=[5, 50, 1], metrics=[Custom_MSE, mutual_information])
else:
    A.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), 
              loss=[loss_dict[args.loss.upper()], losses.binary_crossentropy], loss_weights=[1, 1], metrics={'output':mutual_information})
#print(A.summary())
    
shuffle_idx = np.random.choice(len(train_low), len(train_low), replace=False)
train_low = train_low[shuffle_idx]
train_high = train_high[shuffle_idx]
if args.aux:
    train_aux = train_aux[shuffle_idx]
batch = args.batch
steps = len(train_low)//batch +1

epochs = args.epochs
train_loss = {"Generator_Total" : [], "Generator_Style" : [], "Generator_AD" : [], "mi": [], "mse": [], "Discriminator_AD" : []}
val_loss = {"Generator_Total" : [], "Generator_Style" : [], "Generator_AD" : [], "mi": [], "mse": []}


save_root = args.save_dir

try:
    os.makedirs(save_root)
    print("\nMake Save Directory!\n")
except:
    print("\nDirectory Already Exist!\n")
         

model_json = A.to_json()
with open(os.path.join(save_root, "model.json"), "w") as json_file:
    json_file.write(model_json)
    
print("\nModel Saved!\n")    
#A.save(os.path.join(save_root + "model.h5"))
total_progbar = Progbar(epochs)
for epoch in range(epochs):
    #print("Epochs : %03d/%03d"%(epoch+1, epochs))
#     epoch_progbar = Progbar(steps)
    epoch_t_g_total = 0
    epoch_t_g_style = 0
    epoch_t_g_dis = 0
    epoch_t_d_dis = 0
    epoch_t_mse = 0
    epoch_t_mi = 0
    
    epoch_v_g_total = 0
    epoch_v_g_style = 0
    epoch_v_g_dis = 0
    epoch_v_mse = 0
    epoch_v_mi = 0
    
    for step in range(steps):
        
        idx = step*batch
        
        if step+1 == steps:
            step_train_low = train_low[-batch:]
            step_train_high = train_high[-batch:]
            if args.aux:
                step_train_aux = train_aux[-batch:]
        else:
            step_train_low = train_low[idx:idx+batch]
            step_train_high = train_high[idx:idx+batch]
            if args.aux:
                step_train_aux = train_aux[idx:idx+batch]
        train_gen_label = np.ones([len(step_train_low), 1], dtype='float')
        train_dis_label = np.zeros([len(step_train_low)*2, 1])
        train_dis_label[len(step_train_low):] = 1.
#         print(step_train_low.shape)
#         print(step_train_high.shape)
#         print(train_gen_label.shape)
        # Traininig Phase
        D.trainable=False
        if args.aux:
            G_Loss = A.train_on_batch(step_train_low, [step_train_aux, step_train_high, train_gen_label])
            #G_eval = A.evaluate(step_train_low, [step_train_aux, step_train_high, train_gen_label])
            _, G_output, _= A.predict(step_train_low)
        else:
            G_Loss = A.train_on_batch(step_train_low, [step_train_high, train_gen_label])
            #G_eval = A.evaluate(step_train_low, [step_train_high, train_gen_label])
            G_output, _= A.predict(step_train_low)
        #print(G_Loss)
        D.trainable=True
        train_dis_input = np.concatenate([step_train_high, G_output], 0)
        D_Loss = D.train_on_batch(train_dis_input, train_dis_label)
        
        epoch_t_g_total += G_Loss[0]
        epoch_t_g_style += G_Loss[-3]
        epoch_t_g_dis += G_Loss[-2]
        epoch_t_mse += np.mean(np.square(G_output-step_train_high))
        epoch_t_mi += G_Loss[-1]
        epoch_t_d_dis += D_Loss
        
        #epoch_progbar.update(step+1, [("G_Style", G_Loss[1]), ("G_Dis", G_Loss[2]), ("D_Dis", D_Loss)])
    
    
    train_loss["Generator_Total"].append(epoch_t_g_total/steps)
    train_loss["Generator_Style"].append(epoch_t_g_style/steps)
    train_loss["Generator_AD"].append(epoch_t_g_dis/steps)
    train_loss["Discriminator_AD"].append(epoch_t_d_dis/steps)
    train_loss["mse"].append(epoch_t_mse/steps)
    train_loss["mi"].append(epoch_t_mi/steps)
    for idx_v in range(0, len(val_low), batch):
        if idx_v == len(val_low)//batch * batch:
            if args.aux:
                V_loss = A.test_on_batch(val_low[idx_v:], [val_aux[idx_v:], val_high[idx_v:], np.ones([len(val_low[idx_v:]), 1], dtype='float')])
                _, V_output, _= A.predict(val_low[idx_v:])
            else:
                V_loss = A.test_on_batch(val_low[idx_v:], [val_high[idx_v:], np.ones([len(val_low[idx_v:]), 1], dtype='float')])
                V_output, _= A.predict(val_low[idx_v:])
            epoch_v_mse += np.mean(np.square(V_output-val_high[idx_v:]))
        else:
            if args.aux:
                V_loss = A.test_on_batch(val_low[idx_v:idx_v+batch], [val_aux[idx_v:idx_v+batch], val_high[idx_v:idx_v+batch], np.ones([len(val_low[idx_v:idx_v+batch]), 1], dtype='float')])
                _, V_output, _= A.predict(val_low[idx_v:idx_v+batch])
            else:
                V_loss = A.test_on_batch(val_low[idx_v:idx_v+batch], [val_high[idx_v:idx_v+batch], np.ones([len(val_low[idx_v:idx_v+batch]), 1], dtype='float')])
                V_output, _= A.predict(val_low[idx_v:idx_v+batch])
            epoch_v_mse += np.mean(np.square(V_output-val_high[idx_v:idx_v+batch]))

        epoch_v_g_total += V_loss[0]
        epoch_v_g_style += V_loss[-3]
        epoch_v_g_dis += V_loss[-2]
        epoch_v_mi += V_loss[-1]
        
    val_loss["Generator_Total"].append(epoch_v_g_total/len(val_low))
    val_loss["Generator_Style"].append(epoch_v_g_style/len(val_low))
    val_loss["Generator_AD"].append(epoch_v_g_dis/len(val_low))
    val_loss["mse"].append(epoch_v_mse/len(val_low))
    val_loss["mi"].append(epoch_v_mi/len(val_low))
    
#     ran_idx = np.random.choice(len(train_low)-1, 1)
#     test_in = train_low[ran_idx[0]:ran_idx[0]+1]
#     test, _ = A.predict(test_in)
    
    total_progbar.update(epoch+1, [("G_Style", epoch_t_g_style/steps), ("Val_G_Style", epoch_v_g_style/len(val_low))])
#     clear_output(wait=True)
# #     plt.plot(train_loss['Generator_Style'], '.-')
# #     plt.plot(val_loss['Generator_Style'], '.-')
# #     plt.legend(['Train_Style', 'Validation_Style'], loc=0)
# #     plt.show()
#     plot_fn(train_loss, val_loss, test, train_high[ran_idx[0]:ran_idx[0]+1])
    if (epoch+1)%10 == 0:
        A.save_weights(os.path.join(save_root,'%04d_%.2f_%.2f.h5'%(epoch+1, epoch_t_g_style/steps, epoch_v_g_style/len(val_low))))
        
        

train_df = pd.DataFrame(train_loss)
train_df.to_csv(os.path.join(save_root,'train_loss.csv'))

val_df = pd.DataFrame(val_loss)
val_df.to_csv(os.path.join(save_root, 'val_loss.csv'))

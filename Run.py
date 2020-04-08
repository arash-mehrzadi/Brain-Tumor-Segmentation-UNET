from datetime import datetime
import platform
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import random as r
import math
import glob
from Func import create_data , create_data_onesubject_val , dice_coef , dice_coef_loss , unet_model , crop_tumor_tissue , unet_model_nec3 , paint_color_algo
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

Time = datetime.now().strftime('%Y-%m-%d    Time : %H:%M:%S')
Sysconfig = platform.system()
Processor = platform.processor()
Head = f"""
{'-'*60}

|                                                       
|     Brain Tumor Segmentation System
|     
|     Just Input your file path to segment the tumor  
| 
|     Date : {Time}  
|     
|     {Sysconfig}
|     {Processor}

{'-'*60}
"""
print(Head)
print ("1- Start  2-exit","","-----------------","Please enter your desired operation number:","",sep="\n")
choice = input()
allowed = ['1','2']

while not choice in allowed:
    print("wrong!")
    choice = input()

if choice=="2":
  quit()
else:
  print("enter your File path : ")
  filepath = input()
while not (os.path.exists(filepath)):
  print("your file path does not exist","Try again :",sep="\n")
  filepath = input()
#---------------ch.2---------------------
count = 2
pul_seq = 'flair'
Flair = create_data_onesubject_val(filepath, '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't1ce'
T1c = create_data_onesubject_val(filepath, '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't1'
T1 = create_data_onesubject_val(filepath, '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't2'
T2 = create_data_onesubject_val(filepath, '**/*{}.nii.gz'.format(pul_seq), count, label=False)
label_num = 5
Label_full = create_data_onesubject_val(filepath, '**/*seg.nii.gz', count, label=True)
label_num = 2
Label_core = create_data_onesubject_val(filepath, '**/*seg.nii.gz', count, label=True)
label_num = 4
Label_ET = create_data_onesubject_val(filepath, '**/*seg.nii.gz', count, label=True)
label_num = 3
Label_all = create_data_onesubject_val(filepath, '**/*seg.nii.gz', count, label=True)

model = unet_model()
model.load_weights('/content/drive/My Drive/Brain Tumor Unet/Weights/weights-full-best.h5')

x = np.zeros((1,2,240,240),np.float32)
x[:,:1,:,:] = Flair[89:90,:,:,:]   
x[:,1:,:,:] = T2[89:90,:,:,:] 

pred_full = model.predict(x)

crop , li = crop_tumor_tissue(T1c[90,:,:,:],pred_full[0,:,:,:],64)
crop.shape[0]

model_core = unet_model_nec3()
model_core.load_weights('/content/drive/My Drive/Brain Tumor Unet/Weights/weights-core-best.h5')
model_ET = unet_model_nec3()
model_ET.load_weights('/content/drive/My Drive/Brain Tumor Unet/Weights/weights-ET-best.h5')

pred_core = model_core.predict(crop)
pred_ET = model_ET.predict(crop)

tmp = paint_color_algo(pred_full[0,:,:,:], pred_core, pred_ET, li)

core = np.zeros((1,240,240),np.float32)
ET = np.zeros((1,240,240),np.float32)
core[:,:,:] = tmp[:,:,:]
ET[:,:,:] = tmp[:,:,:]
core[core == 4] = 1
core[core != 1] = 0
ET[ET != 4] = 0

plt.figure(figsize=(15,10))


plt.subplot(345)
plt.title('Prediction Tumor')
plt.axis('off')
plt.imshow(pred_full[0, 0, :, :],cmap='Reds')

plt.subplot(346)
plt.title('Tumor Core')
plt.axis('off')
plt.imshow(core[0, :, :],cmap='Reds')

plt.subplot(347)
plt.title('Tumor ET')
plt.axis('off')
plt.imshow(ET[0, :, :],cmap='Reds')

plt.subplot(348)
plt.title('Tumor All sections')
plt.axis('off')
plt.imshow(tmp[0, :, :],cmap='Reds')

plt.subplot(341)
plt.title('Prediction Tumor')
plt.axis('off')
plt.imshow(np.squeeze(T1c[90, 0, :, :]),cmap='gray')
plt.imshow(np.squeeze(pred_full[0, 0, :, :]),alpha=0.3,cmap='Reds')

plt.subplot(342)
plt.title('Tumor Core')
plt.axis('off')
plt.imshow(np.squeeze(T1c[90, 0, :, :]),cmap='gray')
plt.imshow(np.squeeze(core[0, :, :]),alpha=0.3,cmap='Reds')

plt.subplot(343)
plt.title('Tumor ET')
plt.axis('off')
plt.imshow(np.squeeze(T1c[90, 0, :, :]),cmap='gray')
plt.imshow(np.squeeze(ET[0, :, :]),alpha=0.3,cmap='Reds')

plt.subplot(344)
plt.title('Tumor All sections')
plt.axis('off')
plt.imshow(np.squeeze(T1c[90, 0, :, :]),cmap='gray')
plt.imshow(np.squeeze(tmp[0, :, :]),alpha=0.3,cmap='Reds')

plt.show()
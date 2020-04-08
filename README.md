# Brain-Tumor-Segmentation-UNET

<img src="https://github.com/arash-mehrzadi/Brain-Tumor-Segmentation-UNET/blob/master/Sample/predict%20output.png" width="whatever" height="whatever">

# Description
The process of segmenting tumor from MRI image of a brain is one of the highly focused areas in the community of medical science as MRI is noninvasive imaging. Therefore, I decided to create Brain Tumor Segmentation Launch File that you can easily use the capabilities of this powerful tool. I am currently in the early stages of designing this Launch File and will be constantly updating these files to get to a usable version.
# Unet Architecture

<img src="https://github.com/arash-mehrzadi/Brain-Tumor-Segmentation-UNET/blob/master/_/u-net-architecture.png" width="whatever" height="whatever">

The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin.
architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.
I used this Architecture to build Launch File and in this version i download trained model weights from [This_Link](https://drive.google.com/file/d/1hE9It0ZOOeIuSFvt6GdiR_0cq9inWdTy/view?usp=sharing).

# Data

<img src="https://www.med.upenn.edu/sbia/assets/user-content/BRATS_banner_noCaption.png" width="whatever" height="whatever">
I use BraTS_13 dataset as network input to evaluate network performance 
you can take the MRI image as desired in nii format and give the address of the storage location of these images to the Lunch File . Just note that these images include T1-weighted MRI (T1), T1-weighted MRI with contrast enhancement (T1c), T2-weighted MRI (T2) and T2-weighted MRI with fluid attenuated inversion recovery (T2-Flair)

# Requirements 
This codes is built using Python 3.7.5, and utilizes the following packages : 
- TensorFlow 1.15.2
- Keras 2.2.5
- Matplotlib 3.2.1
- NumPy 1.18.2
- scikit-image 0.16.2

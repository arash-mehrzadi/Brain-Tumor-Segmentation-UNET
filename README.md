# Brain-Tumor-Segmentation-UNET

<img src="https://github.com/arash-mehrzadi/Brain-Tumor-Segmentation-UNET/blob/master/Sample/predict%20output.png" width="whatever" height="whatever">

# Description
The process of segmenting tumor from MRI image of a brain is one of the highly focused areas in the community of medical science as MRI is noninvasive imaging. Therefore, I decided to create Brain Tumor Segmentation Launch File that you can easily use the capabilities of this powerful tool. I am currently in the early stages of designing this Launch File and will be constantly updating these files to get to a usable version.
# Unet Architecture

<img src="https://github.com/arash-mehrzadi/Brain-Tumor-Segmentation-UNET/blob/master/_/u-net-architecture.png" width="whatever" height="whatever">

The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin.
architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.
I used this Architecture to build Launch File and downlo d trained model weights 

# Data

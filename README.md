# RFFNet: Refine, Fuse and Focus for RGB-D Salient Object Detection

### Contents
Requirements
Dataset preparation
How to run

## Requirements
Python 3.6, Pytorch 1.4, Cuda 10.0, opencv-python, scipy 1.5.4, Apex(for larger batchsize)

## Dataset preparation
Download the raw data of NJU2K, NLPR, STERE, DES, and SIP dataset. The train/test division is same to DMRA.
Besides, we use minmax function to normalize the depth maps. In our paper, larger number represents the closer distance. 
The directory structure is as following:
        -dataset\ 
          -RGBD_for_train\  
              -RGB
              -depth
              -GT
          -RGBD_for_test\
          -RGBD_for_val\

## How to run
- How to train RFFNet:

    `python RFFNet_train.py --batchsize 16 --gpu_id 0 --save_path ./cpts/RFFNet_cpts/`

- How to test RFFNet:

    `python RFFNet_test.py --gpu_id 0 --test_model ./cpts/RFFNet_cpts/epoch_best.pth --save_path ./result/RFFNet/`
    
- How to Evaluate the result maps:
  You can evaluate the result maps using the tool in [here](http://dpfan.net/d3netbenchmark/).

the pretrained model will upload soon.
needs update~



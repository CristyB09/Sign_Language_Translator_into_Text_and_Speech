 # Sign Language Translator into text and speech
   
https://user-images.githubusercontent.com/58666940/191355191-1e548fbc-4bd9-4f8f-811e-9f7d8c8c9107.mp4

https://user-images.githubusercontent.com/58666940/191355211-76be2e11-9e5d-4ca1-b103-865ddeaeef6e.mp4

 ## PROJECT AIMS
 
 #### • The aim of this study is to find the possibilities of using Deep Neural Network (DNN) for translating sign languages into speech and text format through human       hand gestures.
 
 #### • The main objective of this study is to explore the feasibility of introducing a vision-based application, for the translation of the sign language to eradicate communication barriers between the hearing blind and deaf communities.
 #### • Sign language is translated into a text and speech format by recognizing hand signs.
 #### • The system has been developed following the training of a pre-trained SSD-Network using the captured dataset.
 #### • This system could be researched further and analyze the different types of sign language vocabulary and improve the interface aspect.

 ## PROJECT OBJECTIVES

 #### • To provide and establish pursuit gesture communication with deaf and blind people by converting hand gestures from sign language into digital text.
 #### • To evaluate the performance of the deep neural network and then import the data collected from human hand gestures into Convolutional Neural Network (CNN).
 #### • To achieve an overall training accuracy score of 90% or higher and more reliable performance metrics that can be used to validate and predict the output data of sign language hand gesture.
 #### • To develop and test a trained neural network model to predict sign language hand gestures with a high level of accuracy for both the audience and the deaf            signer.  
 #### • To discuss the results and to make recommendations for the future improvements.

# System Design and Development

    The below figure illustrates the structure of the conceptual design framework of the proposed system.

![Picture1](https://user-images.githubusercontent.com/58666940/168078886-63cb904d-e574-40b3-a7fb-43a99961a7af.png)
     
# Pretrained Model Selection

This project was chosen to use SSD-Mobile-V2 as the pretrained model due to its lightweight nature, which makes it well suited for embedded vision applications and mobile environments. MobileNet can be employed in the detection of objects, classification, and recognition of signs gestures. It is very fast in its performance, as well as more accurate when it comes to the recognition of sign language. According to this project, the goal was to achieve 99 % accuracy to detect sign language detection.

# System Development

In accordance with Figure below, the entire process of recognizing sign language is based upon three stages. As the initial step, training datasets need to be captured. The images used in this study were captured using a web camera. It was necessary to train the datasets. This was accomplished by training an algorithm on a Jetson Nano developer kit using the transfer learning technique. As a result of the classification step, each of the output predictions has been displayed as words using the trained model.

  
![Picture2](https://user-images.githubusercontent.com/58666940/168081719-2dca3e4f-3c41-4abd-8bed-c3ecbe36445b.png)

    Flow Chart of the Software System.
    
# System Design

The present section discusses the experimental platform for the proposed real time sign language detection. The hardware used for the proposed system is discussed specifically, as well as the process of installing all the necessary software.


![Picture3](https://user-images.githubusercontent.com/58666940/168082908-47c655fa-5024-4297-8ec2-bbfa7e453e00.png)

    Block Diagram of System Design 

# Hardware Selection

| Property   |  Specification | 
|----------------|:---------:|
| Graphical Processing Unit-GPU     |  128-core NVIDIA Maxwell | 
| Processor   | Quad-core ARM A57 (1.43 GHz) | 
| OS type     |    64 bit  | 
| Memory (RAM) |     2 GB LPDDR4 memory     | 
| Operating System     |     Ubuntu    | 
| Storage    |     SSD 240GB SATA 3     | 
| Image capturing device |     IMX219-77 Camera for NVIDIA Jetson Nano Sensor 3280 × 2464 Resolution 8 Megapixels 160 Degree Angle of View   | 
| Display    |     7inch HDMI LCD (H) capacitive touch display    | 
| Audio Speaker    |     Trust Avora 2.1 PC Speaker Set with Subwoofer    | 
   
![Picture4](https://user-images.githubusercontent.com/58666940/168084247-908d648d-033f-4442-b2c9-2ccd718d7b31.png)

    Block Diagram of Hardware Design
    
## Jetson Nano Developer Kit

The system has connection to the monitor display, audio speaker, mouse, keyboard, SSD-hard drive, Wi-Fi and power supply.

![Picture5](https://user-images.githubusercontent.com/58666940/168084695-1b2a6aec-1573-472b-bdef-f58ec8247e94.jpg)

Below is a description of the features included in the NVIDIA Jetpack SDK. The Jetpack SDK includes libraries, OS images, APIs, documentation, samples, and developer tools. Additionally, it is offering the following modules:

    •	TensorRT - Is a software development kit featuring high-performance and high-level deep learning inference designed for the classification, segmentation, and detection of objects in images.

    •	cuDNN - A graphics acceleration library for models based on deep neural networks.   
    •	CUDA-   This toolkit provides a complete development environment for C++ or C developers creating GPU-accelerated applications.  
    •	Multimedia API - The decoding and encoding of video. 
    •	Developer Tools - It includes tools for debugging, Nsight Eclipse Edition, and profiling with Nsight Compute, as well as a tool chain for cross-compiling applications.  
    •	Computer Vision - Toolkit for computer vision and vision processing. 

# Software Selection

Various software’s libraries have been adopted in the purpose of developing this sign language detection system. PyTorch is one of the most popular and widely used deep learning frameworks. It provides facilities for training models with reduced precision, models which can then be exported to be optimized in TensorRT. It is estimated that TensorRT can deliver performance advantages of 40X compared to CPU-only platforms during inference, and it is built using trained neural network models. To develop this system, Python was chosen as the programming language.

## Software used for this project:

    Nvidia Jetpack 	4.6
    Ubuntu	18.04 LTS
    Visual Studio Code (CODE OSS)	1.32.1
    PyTorch	1.6.0
    OpenCV	4.1.1
    Numpy	1.13.3
    gTTS	2.2.4
    playsound	20.1
    mpg123	1.25.10-1
    
 ## Establishing the Environment 
 
The JetPack SDK from NVIDIA is the most effective solution for creating AI applications. This release includes the latest OS images for Jetson products as well as libraries and APIs, samples, developer tools, and documentation.  As part of the JetPack package, a reference file system based on Ubuntu 18.04 is included. The Jetpack can be downloaded from the NVIDIA Jetpack official website on a computer. The operating system images can then be loaded into the memory card. An SSD hard drive was used for booting the operating system image to provide faster loading times.

 ## Setting up Jetson Inference for Object Detection 
 
The Jetson Inference Is a library of TensorRT-accelerated deep learning networks for image recognition, object detection with localization (i.e., bounding boxes), and semantic segmentation. This library can be run on both C++ and Python platforms. Several pre-trained DNN models are automatically downloaded to get you started quickly. 
 
    $ sudo apt-get update
    $ sudo apt-get install git cmake libpython3-dev python3-numpy
    $ git clone --recursive https://github.com/dusty-nv/jetson-inference
    $ cd jetson-inference
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make -j$(nproc)
    $ sudo make install
    $ sudo ldconfig
 
Compiling can take significant time on the Jetson. The process took approximately 45 minutes. 

## Install Media packages

Installing (v4l-utils) packages on Ubuntu. This package contains a number of utilities for handling media such as webcams.

    $ sudo apt-get install v4l-utils
    
## Downloading the SSD-Mobilenet-v2 Model    

Jetson Inference includes many pre-trained networks for image recognition, object detection, and semantic segmentation all available for download and installation through the Model Downloader tool.

When initially configuring the project, cmake will automatically run the downloader tool for you: 

![ssd](https://user-images.githubusercontent.com/58666940/168477162-7edf653d-dc4b-45fe-a02c-6084984c38a1.png)

## Installing PyTorch

Pytorch offers the ability to retrain the existing networks and allow custom object detection and to retrain specific applications.

![PYTORCH](https://user-images.githubusercontent.com/58666940/168477309-2ae68d7a-0e67-45e7-ad5c-56460c5f3381.png)

The following line are responsible for installing some build dependencies that will 
enable OpenCV to accept a variety of video and image formats.   

## Installation of video and image processing packages

    $ sudo apt-get install libjpeg-dev lybpython3-dev libavcodec-dev libavformat-dev libswcale-dev
    
## Installing the Audio Packages (MP3 open-source audio player)

    $ sudo apt install mpg123
    
## Install the Google Text to Speech (gTTS)

    $ sudo pip3 install gTTS
    
## Install Play sound to play audio files

    $ sudo pip3 install playsound
    
# Collecting the Dataset before Training   

A collection of data was collected via the (camera capture) tool. The tool creates datasets in (Pascal VOC) format for supporting during the training. Each 6 classes predict the bounding boxes of each object of that class in a test image, with associated real-valued confidence.

## Launching the Tool

Below are some example commands for launching the tool:

    $ camera-capture csi://0       # using default MIPI CSI camera
    $ camera-capture /dev/video0   # using V4L2 camera /dev/video0

Below is the Data Capture Control window, after the Dataset Type drop-down has been set to Detection mode (do this first).

![cammm](https://user-images.githubusercontent.com/58666940/168478204-9e7df527-964c-45eb-9c47-a244fc6711a6.png)

The data collected for the purpose this program is divided into two main subsets: training/validation data (trainval), and test data (test). Trainval data has been further divided into suggested training (train) and validation (val) sets for the convenience of classes. To be able to train the system effectively, the input dataset should include internal variations such as variations in shape, rotation, and orientation. In the dataset, signers rotated gestures and oriented their hands differently during data collection when they were capturing images corresponding to the same data class.  

Additionally, varying the background of the sign gesture and lighting the background of the sign gesture will improve the accuracy of the overall performance of the sign language detection.  To collect the dataset for sign language detection, the dataset has been gathered under lighting conditions, rotations of the sign, obstacles in the background, as well as variations in the shape of the sign.

## Capturing the Images 

![euuuu](https://user-images.githubusercontent.com/58666940/168478622-2cb6491f-a03f-46b1-b989-a8253f3b7604.jpg)

There have been 1,067 photographs collected for this training. The pictures are 1280 pixels wide and 720 pixels high. The images for each class have been collected in the following manner: 120 pictures for training; 40 pictures for validation, and 40 pictures for testing.

![100](https://user-images.githubusercontent.com/58666940/168478696-06ab1be6-cceb-4dda-afcf-0313e7e1df43.png)

## The obtained Dataset

![1](https://user-images.githubusercontent.com/58666940/168478739-15dc10c9-7470-4e8d-9014-a69f69f2c21d.png)

# Prepare the Data

A dataset is created using the tool in the format PASCAL (Pattern Analysis, Statistical Modeling, and Computational Learning) VOC XML format. Annotation folder stores all captured images. The figure below illustrates the data set exported in XML format.

The xml code of ‘Gun sign’   

![2](https://user-images.githubusercontent.com/58666940/168478954-6dcf3b0e-f37d-4931-9ba7-4bec3b037d51.png)

This (ImageSets) folder contains the id of each individual capture as well as 'test.txt', 'train', 'trainval.txt', and 'val.txt' files as shown in the figures below.

![3](https://user-images.githubusercontent.com/58666940/168479111-55cc7533-df41-4c4b-8a20-90f8c7def57d.png)

The data in the (JPEGImages) folder has the original 1,067 captured images, and the file called (labels.txt). 
This label file typically contains one class label per line, for example:

## Class labels

    Gun
    Hello
    Love
    No
    Peace
    Yes
    
The label file contains one class name per line and is alphabetized (this is to ensure that the ordering of classes in the label file reflects that of the related subdirectories on disk. From the label file, the tool automatically populated the necessary subdirectories for each class.  

# Training the Dataset

Through the training program, which is developed in Python, the dataset has been fed into the system, the data has been pre-processed, the features extracted, and the final training of the CNN has taken place.  A real-time application is being developed to convert the gesture signs into written language because of this system. The training process was conducted over 24 hours. The dataset has been trained using SSD-mobileV2 pretrained network model with 35 epochs of training network with the default batch size of 4 and two workers using the following command line. 

The command line of training the model:

    $ cd jetson-inference/python/training/detection/ssd
    $ python3 train_ssd.py --dataset-type=voc --data=data/gesture_recognize --model-dir=models/gesture_recognize --batch-size=4 --workers=2 --epochs=35
    
 ## Training Option


 Here are some common options that you can run the training script with. The used training option for this project are:

| Argument       |  Default  | Description                                                |
|----------------|:---------:|------------------------------------------------------------|
| `--data`       |  `data/`  | the location of the dataset                                |
| `--model-dir`  | `models/` | output directory for the trained model         |
| `--resume`     |    None   | path to an existing checkpoint to resume training from     |
| `--batch-size` |     4     | try increasing depending on available memory               |
| `--epochs`     |     35    | cycle of training data set    |
| `--workers`    |     2     | number of data loader threads (0 = disable multithreading) |
 
 ## Export Model to ONNX format
 
ONNX is an open format for models relating to machine learning and deep learning. This tool enables the conversion of deep learning and machine learning models from a variety of frameworks, including TensorFlow, PyTorch, MATLAB, Caffe, and Keras, into a single format. To be able to utilize the trained model with TensorRT, the trained model has been converted from PyTorch to ONNX using (onnx_export.py). 

Next we need to convert our trained model from PyTorch to ONNX, so that we can load it with TensorRT:
The following command line is used to export the model to Open Neural Network Exchange.

    $ python3 onnx_export.py --model-dir=models/gesture_recognize
    
 This will save a model called gesture_recognize.onnx under jetson-inference/python/training/detection/ssd/models/gesture_recognize
 
 # Testing The Trained Model
  
Following a processing period of nearly 24 hours for the training of 1,067 numbers of images data through Google MobileNetV2 architecture in a Convolution neural network (CNN), the trained system was able to convert signed hand gestures into text format. A training system has been developed to translate the 6 basic words of English into sign language, and then it has been enhanced to translate sentences instead of individual words using dual hand gestures. Tests of the trained sign language model have been performed using the (detectnet.py) Pytorch script. To load the SSD-Mobilenet ONNX model, the command line from figure below has been added to detectnet (or detectnet.py).
    
 Load and process the model to TensorRT:    

    detectnet --model=models/gesture_recognize/ssd-mobilenet.onnx --labels=models/gesture_recognize/labels.txt 
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes 
            /dev/video1
            
 ![Picture10](https://user-images.githubusercontent.com/58666940/168602404-6589c3b1-a78b-49bf-8471-05e604e832f2.png)
 
 ## Testing at Dark Background
 
 Testing during the dark has been process with an accuracy between 91-99 percent. In a night vision test, a double sign is examined
 
 ![Picture11](https://user-images.githubusercontent.com/58666940/168603063-9e9db69f-7e00-4a42-bb7f-0d2a0f9cd2b6.png)

 Result of SLD Test during Light and Dark Background
 
 | Name of Sign       |  Light Background  | Dark Background                                               |
|----------------|:---------:|------------------------------------------------------------|
| Hello       |  99.9 %  | 95.3 %                            |
| Peace  | 99.0 % | 98.4 %         |
| Love     |    99.2 %   | 99.2 %     |
| Yes |     99.9 %     | 99.4 %              |
| No    |     99.1 %   | 86.6 %    |
| Gun   |     98.3 %    | 98.8 %|

# Project Objectives

 To reach the project objectives, all requirements and objectives have been met. The previous chapter explains how this was accomplished. A Convolution Neural Network is capable of detecting Sign Language with an accuracy of 99.9% in the final test. An analysis of 20 epochs of CNN data was conducted to determine the accuracy of this test. Several challenges have been overcome to reach this result.

The results of this project were obtained with the help of a computer vision system developed using the CNN architecture and a Jetson Nano kit as the hardware. An implementation made with Python programming languages was employed to create the system.  As a result of retraining the SSD-MobileNetV2 model, 99.9% of accuracy of test was achieved at both light background and dark background.  Obtaining a higher level of accuracy can justify the test method most importantly. This implies that the system can detect the given sign gestures to which it has been trained. The results of the project indicate that gesture signs of dual-hand sign language can be transferred into text format and displayed as worlds and even as full sentences. 

 # Further Development
 
To recommend further development and commercial opportunity that should be made for this convolutional neural network-based sign language detection project, it is necessary to identify some limitations. The project should be extended to more proficient and fluent users since the goal of the study is to facilitate the communication of deaf persons. It will be possible to increase the number of gestures by introducing additional classes and, using the evaluation methods, determine the best approach. Deaf people and businesses and service providers can both benefit from an expanded translation service that is useful and convenient.

The application can be further developed by extending the project to multiple devices including smartphones, tablets, laptops, and desktops. It can be extended to translate in real-time and interpret any sign language, as well as integrated into many products, such as video chat applications, AI assistants, etc. As an extension of the project, a pocket interpreter for the deaf community could be developed, with translation taking place on the cloud. The device should have a camera facing the signing person and should be connected to the internet.
 
# References

    SDK, OS, and IDE:
    JETPACK SDK - Https://developer.nvidia.com/embedded/jetpack 
    
    Flash OS Images - https://www.balena.io/etcher 
    Visual Studio (Code OSS) repository for Ubuntu - https://github.com/toolboc/vscode
    
    Nvidia Library: 
    DNN Library for NVIDIA:
    https://github.com/dusty-nv/jetson-inference#system-setup
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md
    
    Deep Neural Networks Models:
    https://github.com/dusty-nv/jetson-inference/releases/tag/model-mirror-190618 

    Code for training support and export the model:
    https://github.com/dusty-nv/pytorch-ssd/tree/3f9ba554e33260c8c493a927d7c4fdaa3f388e72

 

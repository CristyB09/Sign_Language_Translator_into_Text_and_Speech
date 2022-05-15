   # Sign Language Translator into text and speech
    
    PROJECT AIMS
 
    • The aim of this study is to find the possibilities of using Deep Neural Network (DNN) for translating sign languages into speech and text format through human       hand gestures.
 
    • The main objective of this study is to explore the feasibility of introducing a vision-based application,
    for the translation of the sign language to eradicate communication barriers between the hearing
    blind and deaf communities.
    • Sign language is translated into a text and speech format by recognizing hand signs.
    • The system has been developed following the training of a pre-trained SSD-Network using the captured dataset.
    • This system could be researched further and analyze the different types of sign language vocabulary and improve the interface aspect.

    PROJECT OBJECTIVES

     • To provide and establish pursuit gesture communication with deaf and blind people by converting hand gestures from sign language into digital text.
     • To evaluate the performance of the deep neural network and then import the data collected from human hand gestures into Convolutional Neural Network (CNN).
     • To achieve an overall training accuracy score of 90% or higher and more reliable performance metrics that can be used to validate and predict the output data of      sign language hand gesture.
     • To develop and test a trained neural network model to predict sign language hand gestures with a high level of accuracy for both the audience and the deaf            signer.  
     • To discuss the results and to make recommendations for the future improvements.

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

![CaAAAAAAAAAA](https://user-images.githubusercontent.com/58666940/168083805-43f83933-4a50-4c86-9a0b-5ca3952d5122.PNG)

   
![Picture4](https://user-images.githubusercontent.com/58666940/168084247-908d648d-033f-4442-b2c9-2ccd718d7b31.png)

    Block Diagram of Hardware Design
    
# Jetson Nano Developer Kit

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

Software used for this project:

    Nvidia Jetpack 	4.6
    Ubuntu	18.04 LTS
    Visual Studio Code (CODE OSS)	1.32.1
    PyTorch	1.6.0
    OpenCV	4.1.1
    Numpy	1.13.3
    gTTS	2.2.4
    playsound	20.1
    mpg123	1.25.10-1
    
 # Establishing the Environment 
 
The JetPack SDK from NVIDIA is the most effective solution for creating AI applications. This release includes the latest OS images for Jetson products as well as libraries and APIs, samples, developer tools, and documentation.  As part of the JetPack package, a reference file system based on Ubuntu 18.04 is included. The Jetpack can be downloaded from the NVIDIA Jetpack official website on a computer. The operating system images can then be loaded into the memory card. An SSD hard drive was used for booting the operating system image to provide faster loading times.

 # Setting up Jetson Inference for Object Detection 
 
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

# Install Media packages

Installing (v4l-utils) packages on Ubuntu. This package contains a number of utilities for handling media such as webcams.

    $ sudo apt-get install v4l-utils
    
# Downloading the SSD-Mobilenet-v2 Model    

Jetson Inference includes many pre-trained networks for image recognition, object detection, and semantic segmentation all available for download and installation through the Model Downloader tool.

When initially configuring the project, cmake will automatically run the downloader tool for you: 

![ssd](https://user-images.githubusercontent.com/58666940/168477162-7edf653d-dc4b-45fe-a02c-6084984c38a1.png)






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

    Block Diagram of System Design 

![Picture3](https://user-images.githubusercontent.com/58666940/168082908-47c655fa-5024-4297-8ec2-bbfa7e453e00.png)

# Hardware Selection

![CaAAAAAAAAAA](https://user-images.githubusercontent.com/58666940/168083805-43f83933-4a50-4c86-9a0b-5ca3952d5122.PNG)

    Block Diagram of Hardware Design

![Picture4](https://user-images.githubusercontent.com/58666940/168084247-908d648d-033f-4442-b2c9-2ccd718d7b31.png)



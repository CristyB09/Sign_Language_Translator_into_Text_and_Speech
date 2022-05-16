#Import the NVIDIA jetson Object Detection libraries
import jetson.inference
#Import jetson utility in order to work with camera 
import jetson.utils
#Import time, for frames per second
import time
#Import Open Cv library
import cv2
#Import NumPy is a library for the Python programming language. It is support a large, multi-dimensional arrays and matrices.
import numpy as np 
#Import threading for a separate flow of execution
import threading
#Import (Google Text-to-Speech) a Python library 
from gtts import gTTS
#The playsound module contains the function playsound and it requires one argument
#The path to the file with the sound to play
from playsound import playsound
# Import Operating System 
import os
# Define translating in english speech 
language='en'
#Define de Width and Height size of camera frame
dispW=1280
dispH=720
#Define Welcome speech
speak=True
item='Welcome to signs languade detection'
confidence=0
itemOld=''
# Setuo and open the camera and display the defined width and height frame
cam=cv2.VideoCapture('/dev/video1')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
#Load the trained customize ssd-mobilenet-v2 network
net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=models/gesture_recognize/ssd-mobilenet.onnx', '--labels=models/gesture_recognize/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

flip=2
font=cv2.FONT_HERSHEY_SIMPLEX
timeStamp=time.time()
fpsFilt=0

# Define the speach function
def sayItem():
    global speak
    global item
    while True:
        if speak ==True:
            output=gTTS(text=item, lang='en',slow=False)
            # save the output speech into mp3
            output.save('output.mp3')
            os.system('mpg123 output.mp3')
            speak=False
x=threading.Thread(target=sayItem, daemon=True)
x.start()

while True:
    # OpenCv comand for reading the camera
    _,img = cam.read()
    height=img.shape[0]
    width=img.shape[1]
    # Grab the frame, detect and then display the image
    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    # take the image then show it into frame (cuda array)
    frame=jetson.utils.cudaFromNumpy(frame)
    detections=net.Detect(frame, width, height)
    # detect the trained signs regarding the ID
    for detect in detections:
        ID=detect.ClassID
        top=int(detect.Top)
        left=int(detect.Left)
        bottom=int(detect.Bottom)
        right=int(detect.Right)
        item=net.GetClassDesc(ID)
        print(item,top,left,bottom,right)
        if item!=itemOld:
                    speech=gTTS(text=item, lang=language, slow=False)
                    speech.save("text.mp3")
                    playsound("text.mp3")
                    itemOld=item
        # Show the box frame for object
        cv2.rectangle(img,(left,top),(right,bottom),(255.0,0),2)
        # Show the label for detecting object
        cv2.putText(img,item,(left,top+20),font,.75,(0,0,255),2)
    # take the measurement, for grabing the time per second   
    dt=time.time()-timeStamp
    # to use the next time stamp true the loop from time stamp from defined time
    timeStamp=time.time()
    # frames per second to go true the loop
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps
    #label will be concatinated with the string value of the rounded value of frames per second filtered 
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
    # show the frame of original image
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)
    # Press 'q' key to close the program 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
   

     
     
    
    
    

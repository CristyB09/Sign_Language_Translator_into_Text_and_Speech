import cv2
dispW=640
dispH=480
flip=2
cam=cv2.VideoCapture(1)
while True:
    ret, frame=cam.read()
    cv2.imshow('piCam', frame)
    if cv2.waitKey(1)==ord('q'):
       break
       cam.release()


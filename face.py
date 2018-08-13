
import cv2
import numpy as np 


front_cascade=cv2.CascadeClassifier(r'C:\Users\Biswajeet\Downloads\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier(r'C:\Users\Biswajeet\Downloads\opencv-master\opencv-master\data\haarcascades\haarcascade_smile.xml')
cap=cv2.VideoCapture(0)

while (True):
   _,image= cap.read()
  # clur=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2GRAY)
   clur=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
   faces= front_cascade.detectMultiScale(clur,1.5,minNeighbors=8,minSize=(55,55))
   smile= smile_cascade.detectMultiScale(clur,1.5,minNeighbors=22,minSize=(25, 25))
  # smile = smile_cascade.detectMultiScale( cv2.roi_gray,scaleFactor= 1.7,minNeighbors=22,minSize=(25, 25),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    



   for (x,y,wi,hei) in faces:
       cv2.rectangle(image,(x,y),(x+wi,y+hei),(0,0,255),3)
       layer2 = clur[y:y+hei, x:x+wi]

   for (a,b,c,d) in smile: 
       cv2.rectangle(image,(a,b),(a+c,b+d),(255,0,0),4)
       layer = clur[b:b+d, a:a+c]
        
   cv2.imshow('img',image)
   if cv2.waitKey(10) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()

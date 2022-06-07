import cv2
from cv2 import destroyAllWindows
from cv2 import imread
import pytesseract
import csv
from datetime import date
import time
#import imutils
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def detect_number():
    img = cv2.VideoCapture(0)
    #frame=imread("placa.jpg")
    #img = cv2.VideoCapture('rtsp://root:admin@169.254.57.129//axis-media/media.amp?resolution=1280x720&fps=25&videocodec=h264')
    #img = cv2.VideoCapture('rtsp://test:admin@http://239.219.218.212:0')
    #img = cv2.VideoCapture('rtsp://169.254.29.114')

    text= []
   
    while True: 
        #Lectura de una imagen de un fotograma
        ret, frame = img.read()
        if ret == False:
           break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Eliminar el ruido con blur
        gray = cv2.blur(gray,(3,3))

        #Busca los bordes de la imagen en gris
        canny = cv2.Canny(gray,150,200)
        canny = cv2.dilate(canny,None,iterations=1) 
        #encuentra los contornos y los almacena en un vector 
        cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#OpenCV 4

        #dibujar los contornos (pruebas)
        #cv2.drawContours(frame,cnts,-1,(255,0,0),2)
        cv2.imshow("plate", frame)  

        #Se itera dependiendo a cada contorno hallado
        for c in cnts:
            area = cv2.contourArea(c)  
            x,y,w,h = cv2.boundingRect(c)
            #print("area ", area) 
            #cv2.waitKey(0)

            epsilon = 0.09*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            if len(approx)==4 and area>2000:
                #cv2.drawContours(frame,[approx],0,(255,0,0),3) 
                #print('area=',area)        
                 #hacer rectangulo 
                cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0), 2)
                placa = gray[y:y+h,x:x+w] 
                cv2.putText(frame, str(text[0:7]),(x,y),1,2.2,(0,255,0),3)
                cv2.imshow("plate", frame)

                #cv2.waitKey(0)
                #print(h,w)
                aspect_ratio = float(h)/w
                #print(aspect_ratio)
                #330 mm largo x 160 mm ancho  0.48, nunca puede ser menor, sin embargo puede ser mayor

                if aspect_ratio > 0.46 and aspect_ratio < 0.68:
                #if h>=26 and w>=66:
                    Ptext = pytesseract.image_to_string(placa,config='--psm 7')
                    if len(Ptext) >= 7: 
                        text=Ptext
                        save_csv(Ptext)
                break

            
        t = cv2.waitKey(1) 
        if t == 27:
             break
            
    img.release()
    cv2.destroyAllWindows()

def save_csv(detected_plate):
    with open("placas.csv","a",newline='') as csvfile: 
        writer = csv.writer(csvfile, delimiter=',')
        t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        t=t.split('"')
        hour = t[0]
        writer.writerow([detected_plate , hour])
        csvfile.close()
        

detect_number()

    
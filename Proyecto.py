import cv2
import sys
import numpy as np
import glob
import time

NomP = ["Maguey Morado","Anilillo","Sabila","Planta No Detectada"]

def Contornos(img):
  imgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  gauss = cv2.GaussianBlur(imgb, (11,11),0)
  
  umbral, dst  = cv2.threshold(gauss,0, 255, cv2.THRESH_OTSU)

  edges = cv2.Canny(dst,0,255)
  bordes_gordos = cv2.dilate(edges, None)

  contornos, _ = cv2.findContours(bordes_gordos, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  maxcnt = [max(contornos, key=cv2.contourArea)]

  rect = cv2.minAreaRect(maxcnt[0])
  box = cv2.boxPoints(rect)
  box = np.int0(box)

  im = img.copy()
  im = cv2.drawContours(im,[box],0,(255,0,0), 15)

  xs = box.flatten()[::2]
  ys = box.flatten()[1::2]
  return (im,xs,ys)

def Color(imgr,colorBajo,colorAlto,aread,cc,i):
  frameHSV = cv2.cvtColor(imgr,cv2.COLOR_RGB2HSV)
  mask = cv2.inRange(frameHSV,colorBajo,colorAlto)
  contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  im = imgr.copy()
  for c in contornos:
    area = cv2.contourArea(c)
    if(area>aread):  
      im = cv2.drawContours(im, [c], 0, (cc[0],cc[1],cc[2]),5)
      colores[i] = 1
  return im

cap = cv2.VideoCapture('./Files/plantas.mp4')

time_prev_frame = 0
time_new_frame = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    iframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    nimg,xs,ys = Contornos(iframe)
    height, width,channels = nimg.shape 

    xmin = min(xs)
    if(xmin<0):
      xmin=0
    xmax = max(xs)
    if(xmax>width):
      xmax=width-1
    ymin = min(ys)
    if(ymin<0):
      ymin=0
    ymax = max(ys)
    if(ymax>height):
      ymax=height-1

    imgr = iframe[ymin:ymax,xmin:xmax]
    #print(xmin,xmax,ymin,ymax)
    #plt.imshow(nimg)
    cv2.imshow('Objeto Encontrado',nimg)
    
    planta = 3
    colores = [0,0,0]

    cc = [255, 0, 0]
    colorBajo = np.array([120,100,0], np.uint8)
    colorAlto = np.array([170,255,255], np.uint8)
    imgr = Color(imgr,colorBajo,colorAlto,500,cc,0)
    
    cc = [0, 0, 255]
    colorBajo = np.array([41,100,0], np.uint8)
    colorAlto = np.array([55,255,255], np.uint8)
    imgr = Color(imgr,colorBajo,colorAlto,500,cc,1)
    
    cc = [0, 255, 0]
    colorBajo = np.array([25,100,0], np.uint8)
    colorAlto = np.array([40,255,255], np.uint8)
    imgr = Color(imgr,colorBajo,colorAlto,500,cc,2)
    #plt.imshow(imgr)
    uframe = cv2.cvtColor(imgr, cv2.COLOR_RGB2BGR)
    cv2.imshow('Colores Encontrados',uframe)
    #print(colores)
    if((colores[0]==1)and(colores[1]==1 or colores[2]==1)): planta = 0
    if((colores[1]==1)and(colores[0]!=1)): planta = 1
    if((colores[2]==1)and(colores[0]!=1)and(colores[1]!=1)): planta = 2
    print(NomP[planta])

    
    time_new_frame = time.time()

    fps = 1/(time_new_frame-time_prev_frame)
    print("FPS: ",fps)
    time_prev_frame = time_new_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

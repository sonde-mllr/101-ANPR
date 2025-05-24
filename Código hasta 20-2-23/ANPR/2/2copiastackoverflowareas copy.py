import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import pytesseract
import imutils
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#cam = cv.VideoCapture("http://192.168.226.136:8080/video")
#cam = cv.imread("E:\\vscode\\.vscode\\ANPR\\imagenes\\caja2.png")
cam = cv.VideoCapture(0)


while(True):
    ret, frame = cam.read()
    HW = (640,480)
    frame = cv.resize(frame,HW, interpolation= cv.INTER_LINEAR)
    image_copy = np.array(frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    canny = cv.Canny(frame, 225,250)
    canny = cv.dilate(canny, None, iterations=1)
    canny = cv.erode(canny, None, iterations=1)
    cnts,_ = cv.findContours(canny, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

#Buscas el contorno más grande
    lista_areas = []
    lista_control = []
    for i, contornos in enumerate(cnts):
        for c in enumerate(cnts):
            #if lista_areas == []:
            #    lista_areas = [c] 
            area = cv.contourArea(c)
            lista_areas.append(area)
#Te quedas con el area más grande
    mas_grande = cnts[lista_areas.index(max(lista_areas))]

#Representas el contorno más grande
    area = cv.contourArea(mas_grande)
    x,y,w,h = cv.boundingRect(mas_grande)
    cv.rectangle(image_copy, (x,y), (x+w, y+h), (0,255,0), 2)
    cv.imshow('Calculo de Area y Reconocimiento de imagen', image_copy)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
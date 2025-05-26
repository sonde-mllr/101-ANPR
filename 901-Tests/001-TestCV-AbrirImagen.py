
import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("/home/ale/Documentos/101-Python/002-ANPR/201-Imagenes/matricula.jpg",cv.IMREAD_GRAYSCALE))

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("/home/ale/Documentos/101-Python/002-ANPR/201-Imagenes/matricula.jpg", img)
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread("examples/BackgroundRemoval/background_images/1.jpg")
img_path = "examples/BackgroundRemoval/background_images/"
listImg = os.listdir(img_path)
imgList = []
for imgName in listImg:
    img = cv2.imread(img_path+imgName)
    imgList.append(img)
    
indexImg = 0

while True:
    success, img = cap.read()
    # imgOut = segmentor.removeBG(img, (255, 0, 0), threshold=0.8)
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)
    
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0,0,255))
    # cv2.imshow("Image", img)
    # cv2.imshow("Image Out", imgOut)
    cv2.imshow("Image Stacked", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg>0:
            indexImg -= 1
        else:
            indexImg = len(imgList)-1
        
    elif key == ord('d'):
        if indexImg<len(imgList)-1:
            indexImg += 1
        else:
            indexImg = 0 
    elif key == ord('q'):
        break
    
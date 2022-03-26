import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

# 그레이스케일 영상의 히스토그램
# 영상, 그레이스케일이므로 0, mask None, bins 256, range 0~256 -> 1차원 행렬
hist = cv2.calcHist([src], [0], None, [256], [0,256])

cv2.imshow('src', src)
cv2.waitKey(1)

plt.plot(hist)
plt.show()

src = cv2.imread('images/lenna.bmp')

if src is None:
    print("Image load failed!")
    sys.exit()

colors = ['b', 'g', 'r']
bgr_planes = cv2.split(src)

for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)

cv2.imshow('src', src)
cv2.waitKey(1)

plt.show()


src = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

hist = cv2.calcHist([src], [0], None, [256], [0,256])

# opencv를 활용한 hist image 구하는 방법
def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype=np.uint8)

    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

histImg = getGrayHistImage(hist)
cv2.imshow('histImg', histImg)
cv2.imshow('src', src)
cv2.waitKey()
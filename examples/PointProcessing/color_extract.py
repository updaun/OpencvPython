import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

# 밝은 영상
# src = cv2.imread('images/candies.png')

# 상대적으로 어두운 영상
src = cv2.imread('images/candies2.png')

if src is None:
    print("Image load failed!")
    sys.exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# cv2.inRange(src, lowerb(하한 값), upperb(상한 값), dst=None) -> Mask
dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100))
dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

# Trackbar를 활용한 특정 색 추출
# 빨간색을 추출할 때는 작은 hue값과 큰 hue 값을 or 연산 해야한다.

def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min', 'dst')
    hmax = cv2.getTrackbarPos('H_max', 'dst')

    dst = cv2.inRange(src_hsv, (hmin, 150, 0), (hmax, 255, 255))
    cv2.imshow('dst', dst)    

cv2.namedWindow('dst')

cv2.createTrackbar('H_min', 'dst', 50, 179, on_trackbar)
cv2.createTrackbar('H_max', 'dst', 80, 179, on_trackbar)


# ESC로 종료
if cv2.waitKey() == 27:
    cv2.destroyAllWindows()


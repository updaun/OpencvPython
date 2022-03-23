# saturate 연산 255 이상은 255, 0 미만은 0으로

import cv2
import numpy as np

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)
color_src = cv2.imread('images/lenna.bmp')
 
# dst = cv2.add(src, 100)
# dst = cv2.add(src, (100, 0, 0, 0))
# dst = src + 100 # 255보다 클 때 그 값을 0에 가까운 값으로 바꾼다.
# dst = np.clip(src + 100, 0, 255) # 이것도 제대로 안나온다.
dst = np.clip(src + 100., 0, 255).astype(np.uint8) # 실수로 연산하고 정수로 변환해야한다.

color_dst = cv2.add(color_src, 100) # blue 성분만 100 증가
color_dst = cv2.add(color_src, (100, 100, 100, 0)) # blue 성분만 100 증가

# cv2.imshow('src', src)
# cv2.imshow('dst`', dst)
cv2.imshow('color_src', color_src)
cv2.imshow('color_dst', color_dst)

cv2.waitKey()
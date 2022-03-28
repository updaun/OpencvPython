# 히스토그램 평탄화(Histogram equalization)
# 히스토그램이 그레이스케일 전체 구간에서 균일한 분포로 나타나도록 변경하는 명암비 향상 기법
# 히스토그램 균일화, 균등화, 평탄화

import cv2
import sys
import numpy as np

def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype=np.uint8)

    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

src = cv2.imread('images/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([src], [0], None, [256], [0,256])
histImg_src = getGrayHistImage(hist)

if src is None:
    print("Image load failed!")
    sys.exit()

dst = cv2.equalizeHist(src)
hist_dst = cv2.calcHist([dst], [0], None, [256], [0,256])
histImg_dst = getGrayHistImage(hist_dst)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('histImg_src', histImg_src)
cv2.imshow('histImg_dst', histImg_dst)
if cv2.waitKey() == 27:
    cv2.destroyAllWindows()


# 과도하게 명암비가 높아졌다고 생각될 수 있으므로 맹신하지 않는다.
# 컬러영상에 적용하기 위해서는 color space를 Ycrcb로 변경하여 Y값에 적용한다.
color_src = cv2.imread('images/field.bmp')

if color_src is None:
    print("Image load failed!")
    sys.exit()

src_ycrcb = cv2.cvtColor(color_src, cv2.COLOR_BGR2YCrCb)
planes = list(cv2.split(src_ycrcb))

planes[0] = cv2.equalizeHist(planes[0])

merged_dst = cv2.merge(planes)
bgr_dst = cv2.cvtColor(merged_dst, cv2.COLOR_YCrCb2BGR)

cv2.imshow('color_src', color_src)
cv2.imshow('bgr_dst', bgr_dst)
if cv2.waitKey() == 27:
    cv2.destroyAllWindows()
# 명암비 조절에는 보통 곱셈을 사용한다.
# dst = s * src
# s = 0.5 일 때, 0 ~ 128까지만 출력
# 효과적인 명암비 조절 함수
# (1+a)*src - 128*a
# (128, 128) 점을 지나는 직선의 방정식의 기울기를 조절하는 방식
# opencv에 지원하는 함수가 없으므로 numpy로 제작

import cv2
import sys
from cv2 import waitKey
import numpy as np

# opencv를 활용한 hist image 구하는 방법
def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype=np.uint8)

    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

alpha = 1.0
dst = np.clip((1 + alpha) * src - 128 * alpha, 0, 255).astype(np.uint8)


# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()

# 영상의 자동 명암비 조절(Histogram Straching)
# 입력 영상에서 가장 어두운 픽셀 Gmin
# 입력 영상에서 가장 밝은 필셀 Gmax
# min-max scale을 적용(0~255)

src = cv2.imread('images/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

# opencv normalize
dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)

# numpy로 구현 Gmin, Gmax로 직선의 방정식 구하기
gmin = np.min(src)
gmax = np.max(src)
dst = np.clip((src - gmin) * 255. / (gmax-gmin), 0, 255).astype(np.uint8)

hist = cv2.calcHist([src], [0], None, [256], [0,256])
histImg_src = getGrayHistImage(hist)

hist_dst = cv2.calcHist([dst], [0], None, [256], [0,256])
histImg_dst = getGrayHistImage(hist_dst)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('histImg_src', histImg_src)
cv2.imshow('histImg_dst', histImg_dst)
cv2.waitKey()

if waitKey() == 27:
    cv2.destroyAllWindows()


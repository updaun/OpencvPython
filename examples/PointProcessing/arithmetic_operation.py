import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

src1 = cv2.imread('images/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('images/square.bmp', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print("Image load failed!")
    sys.exit()

# 영상의 산술 연산

dst1 = cv2.add(src1, src2, dtype=cv2.CV_8U) # 더하기
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0) # 가중합 
dst3 = cv2.subtract(src1, src2) # 빼기
dst4 = cv2.absdiff(src1, src2) # 차형상

# 영상의 논리 연산
'''
cv2.bitwise_and(src1, src2, dst=None, mask=None) -> dst
cv2.bitwise_or(src1, src2, dst=None, mask=None) -> dst
cv2.bitwise_xor(src1, src2, dst=None, mask=None) -> dst
cv2.bitwise_not(src1, dst=None, mask=None) -> dst
'''

plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title("src1")
plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title("src2")
plt.subplot(233), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title("add")
plt.subplot(234), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title("addWeighted")
plt.subplot(235), plt.axis('off'), plt.imshow(dst3, 'gray'), plt.title("subtract")
plt.subplot(236), plt.axis('off'), plt.imshow(dst4, 'gray'), plt.title("absdiff")

plt.show()

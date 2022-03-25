import cv2
import sys
import matplotlib.pyplot as plt

src = cv2.imread('images/candies.png', cv2.IMREAD_COLOR)
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

if src is None:
    print("Image load failed!")
    sys.exit()

# 컬러 영상 속성 확인
print('src.shape : ', src.shape)
print('src.dtype : ', src.dtype)

# RGB to gray scale
# 299*R + 0.587*G + 0.114*B
# 0.3 : 0.6 : 0.1
planes = cv2.split(src_rgb)

plt.subplot(221), plt.axis('off'), plt.imshow(src_rgb), plt.title("src")
plt.subplot(222), plt.axis('off'), plt.imshow(planes[0], 'gray'), plt.title("R")
plt.subplot(223), plt.axis('off'), plt.imshow(planes[1], 'gray'), plt.title("G")
plt.subplot(224), plt.axis('off'), plt.imshow(planes[2], 'gray'), plt.title("B")

plt.show()

# HSV
hsv_planes = cv2.split(src_hsv)

# Hue : 색상, 색의 종류 (0~360도) cv2.CV_8U 0 <= H <= 179
# Saturation : 채도, 색의 탁하고 선명한 정도 (0~255) cv2.CV_8U 0 <= S <= 255
# Value : 명도, 빛의 밝기 (0~255) cv2.CV_8U 0 <= V <= 255

plt.subplot(221), plt.axis('off'), plt.imshow(src_rgb), plt.title("src")
plt.subplot(222), plt.axis('off'), plt.imshow(hsv_planes[0], 'gray'), plt.title("H")
plt.subplot(223), plt.axis('off'), plt.imshow(hsv_planes[1], 'gray'), plt.title("S")
plt.subplot(224), plt.axis('off'), plt.imshow(hsv_planes[2], 'gray'), plt.title("V")

plt.show()

# YCrCb 색 공간
# PAL, NTSC, SECAM 등의 컬러 비디오 표준에 사용되는 색공간
# 영상의 밝기 정보와 색상 정보를 따로 분리하여 부호화(흑백 TV 호환)
# Y: 밝기 정보(gray scale과 완전히 동일)
# Cr, Cb: 색차(chroma)

# cv2.CV_8U 0 <= Y <= 255
# cv2.CV_8U 0 <= Cr <= 255
# cv2.CV_8U 0 <= Cb <= 255

ycrcb_planes = cv2.split(src_ycrcb)

plt.subplot(221), plt.axis('off'), plt.imshow(src_rgb), plt.title("src")
plt.subplot(222), plt.axis('off'), plt.imshow(ycrcb_planes[0], 'gray'), plt.title("Y")
plt.subplot(223), plt.axis('off'), plt.imshow(ycrcb_planes[1], 'gray'), plt.title("Cr")
plt.subplot(224), plt.axis('off'), plt.imshow(ycrcb_planes[2], 'gray'), plt.title("Cb")

plt.show()

# cv2.imshow('src', src)
cv2.waitKey()





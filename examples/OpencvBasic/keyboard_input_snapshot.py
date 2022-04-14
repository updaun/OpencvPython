import cv2
import sys

cap = cv2.VideoCapture('images/woman.mp4')

count = 1

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    cv2.imshow('output', frame)

    # 동영상 속도 제어
    cv2.waitKey(10)

    # ESC를 눌러 프로그램 종료
    if cv2.waitKey(1) == 27:
        break
    # SPACE를 눌러 사진 저장(연타 필요)
    elif cv2.waitKey(1) == 32:
        cv2.imwrite(f'images/woman_{count}.png', frame)
        count += 1 
        print("Success Save Picture!")
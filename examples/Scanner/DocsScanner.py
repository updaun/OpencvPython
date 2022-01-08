import cv2
import numpy as np
import utlis


###################################################
webCamFeed = True
pathImage = "examples/Scanner/1.jpg"
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 640
widthImg = 480
###################################################

utlis.initializeTrackbars()
count = 0

while True:
    
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    # if webCamFeed:
    #     success, img = cap.read()
    # else:
    #     img = cv2.imread(pathImage)
    
    img = cv2.imread(pathImage)

    # Image Preprocessing
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img, (5,5), 1)
    thres = utlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    ## FIND ALL COUNTOURS
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FROM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1]]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        #Image Array For Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                    [imgBigContour, imgWarpGray, imgWarpGray, imgAdaptiveThre])
            
    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                    [imgBlank, imgBlank, imgBlank, imgBlank])

    
    # LABELS FOR DISPLAY
    labels = [["Original", "Gray", "Threshold", "Contours"],
                ["Biggest Contour", "Warp Prespective", "Warp Gray","Adaptive Threshold"]]


    stackedImage = utlis.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)


    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("examples/Scanner/Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1


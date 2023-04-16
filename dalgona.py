import cv2
from cvzone import HandTrackingModule, overlayPNG
import numpy as np
import os
import time

folderPath = 'frames'
mylist = os.listdir(folderPath)
graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in mylist]
intro = graphic[0];
kill =graphic[1];
winner =graphic[2];
cam = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector(maxHands=1,detectionCon=0.77)

sqr_img=cv2.imread('img/sqr(2).png')
mlsa=cv2.imread('img/mlsa.png')

cv2.imshow('Cookie Cutter',cv2.resize(intro,(0,0),fx=0.69,fy=0.69))
cv2.waitKey(1)

while True:
    cv2.imshow('Cookie Cutter', cv2.resize(intro, (0,0), fx=0.69, fy=0.69))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
TIMER_MAX = 45
TIMER = TIMER_MAX
maxMove = 6500000
font = cv2.FONT_HERSHEY_SIMPLEX 
cap = cv2.VideoCapture(0)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)


gameOver=False
NotWon=True

while True:
    isTrue, frame = cam.read()
    hands, img = detector.findHands(frame, flipType=True)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

    cv2.imshow('Video', frame)


    if(cv2.waitKey(20) & 0xFF==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()

prev=time.time()
prevHand=prev
showFrame = cv2.resize(sqr_img, (0,0), fx=0.69, fy=0.69)
isblue =True
while cam.isOpened() and TIMER >= 0:
    ret, frame = cap.read()
    
    cv2.putText(showFrame, str(TIMER),
                (50, 50), font,
                1, (0, int(255 * (TIMER) / TIMER_MAX), int(255 * (TIMER_MAX - TIMER) / TIMER_MAX)),
                4, cv2.LINE_AA)
    
    cur = time.time()
    
    if isblue:
        showFrame= cv2.resize(sqr_img, (0,0), fx=0.69, fy=0.69)
        isblue = False
        ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    else:
        showFrame=cv2.resize(kill, (0,0), fx=0.69, fy=0.69)
        isblue = True
        
    if not isblue:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (21, 21), 0)
        frameDelta = cv2.absdiff(ref, gray)
        thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
        change = np.sum(thresh)

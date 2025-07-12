import cv2
import mediapipe as mp
import numpy as np
import autopy
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
        return lmList

    def fingersUp(self, lmList):
        fingers = []

        if len(lmList) == 0:
            return []

        # Thumb
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

###############################
wCam, hCam = 640, 480
frameR = 100

cursorSpeed = 5  # 👈 CHANGE THIS VALUE TO CONTROL SPEED (Higher = Faster)

plocX, plocY = 0, 0
clocX, clocY = 0, 0
###############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()
detector = HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        fingers = detector.fingersUp(lmList)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # Move Mouse
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / cursorSpeed
            clocY = plocY + (y3 - plocY) / cursorSpeed

            autopy.mouse.move(clocX, hScr - clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Click when both fingers are up and close
        if fingers[1] == 1 and fingers[2] == 1:
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 40:
                cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Show speed value on screen
    cv2.putText(img, f"Speed: {cursorSpeed}", (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

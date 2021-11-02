import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.4):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]
        self.tenAveragePos = [0, 0, 0]
        self.variances = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []

        # [id, x, y, z]
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z*1000)
                xList.append(cx)
                yList.append(cy)

                # print(id, cx, cy)
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if self.lmList is not None and len(self.lmList) > 0:
            # Thumb
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def checkTouching(self, img):
        if self.lmList == None:
            return

        self.tenAveragePos[0] += self.lmList[8][1]
        self.tenAveragePos[1] += self.lmList[8][2]
        self.tenAveragePos[2] += self.lmList[8][3]
        self.tenAveragePos[0] *= 0.5
        self.tenAveragePos[1] *= 0.5
        self.tenAveragePos[2] *= 0.5
        # print(self.tenAveragePos[2])

        variancex = abs(self.lmList[8][1]-self.tenAveragePos[0])
        variancey = abs(self.lmList[8][2]-self.tenAveragePos[1])
        variancez = abs(self.lmList[8][3]-self.tenAveragePos[2])
        variance = variancex + variancey + variancez

        print(variance)

        self.variances.append(variance)
        if len(self.variances) > 100:
            del self.variances[0]

        if variance > 15:
            cx = self.lmList[8][1]
            cy = self.lmList[8][2]
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            # print("ten finger contact")

        return True


def twoValueFilter(datas, smoothfactor):
    newdatas = []
    prevdata = 0
    for data in datas:
        newdata = data*smoothfactor+prevdata*(1-smoothfactor)
        newdatas.append(newdata)
        prevdata = data
    return newdatas


def arithmeticMeanFilter(datas, N):
    newdatas = []
    i = 0
    for data in datas:
        datasum = 0
        for k in range(N):
            try:
                datasum += datas[i+k]
            except:
                break
        newdata = datasum/N
        newdatas.append(newdata)
        i += 1

    return newdatas


def visualizeData(inputdata, index):
    # datas = twoValueFilter(inputdata, index)
    datas = arithmeticMeanFilter(inputdata, index)
    dataimg = np.zeros((200, 500, 3), np.uint8)
    dataimg.fill(200)
    i = 1
    for data in datas:
        cv2.circle(dataimg, (i*5, int(data)), 2, (255, 0, 0), cv2.FILLED)
        i += 1

    name = "data with smoothefactor "+str(index)
    cv2.imshow(name, dataimg)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            detector.checkTouching(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)

        datas = detector.variances

        # data monitor
        visualizeData(datas, 20)
        visualizeData(datas, 10)
        visualizeData(datas, 8)
        visualizeData(datas, 5)
        visualizeData(datas, 3)
        visualizeData(datas, 2)
        visualizeData(datas, 1)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()

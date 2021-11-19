import cv2
import mediapipe as mp
import threading
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

imgL = np.zeros((480, 640), np.uint8)
imgR = np.zeros((480, 640), np.uint8)

# threading control
thread_run = True


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cap = cv2.VideoCapture(camID)
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    hands = mp_hands.Hands(min_detection_confidence=0.5,
                           min_tracking_confidence=0.9)
    global thread_run
    while cap.isOpened() and thread_run:
        global imgL, imgR
        start = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        if camID == 0:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if camID == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        img_backup = image

        if camID == 1:
            img_backup = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgL = img_backup
        if camID == 0:
            img_backup = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgR = img_backup
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                # mp_drawing_styles.get_default_hand_landmarks_style(),
                # mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        #print("Time taken : {0} seconds".format(seconds))
        #print("Estimated frames per second : {0}".format(fps))
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(cap.get(cv2.CAP_PROP_FPS)))
        imageshow = cv2.flip(image, 1)
        text = str(round(fps, 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imageshow, text, (0, 30), font, 1, (255, 0, 0), 2)
        #imageshow = cv2.resize(imageshow, (960, 540))

        cv2.imshow(previewName, imageshow)

        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            thread_run = False
            break


def stereo(num, size):
    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15) 56 21
    stereo = cv2.StereoSGBM_create(numDisparities=num, blockSize=size)
    global imgL, imgR, thread_run
    while thread_run:
        try:
            #print(imgL.shape, imgR.shape)
            # imgL = cv2.flip(imgL, 1)
            # imgR = cv2.flip(imgR, 1)
            disparity = stereo.compute(imgL, imgR)
            disparity = cv2.flip(disparity, 1)
            #disparity = cv2.resize(disparity, (0, 0), fx=0.3, fy=0.3,interpolation=cv2.INTER_NEAREST)
            # disparity = cv2.flip(disaprity, 1)
            disparity = cv2.normalize(
                disparity, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            h, w = disparity.shape
            cv2.imshow('disparity '+str(num)+',' + str(size), disparity)
        except:
            #print(imgL.shape, imgR.shape)
            True

        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            thread_run = False
            break
    cv2.destroyWindow('disparity')


def threadinggg(num, size):
    thread = threading.Thread(target=stereo, args=(num, size))
    thread.start()


# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()


threadinggg(100, 13)
threadinggg(100, 17)

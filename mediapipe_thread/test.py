import cv2
import mediapipe as mp
import threading
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


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
    SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.9)
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()
        if camID==1:
            image=fish_eye_fix(image)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

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
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        end = time.time()
        seconds = end - start
        fps  = 1 / seconds
        #print("Time taken : {0} seconds".format(seconds))
        print("fps for camera"+str(camID)+": {0}".format(fps))
        imageshow = cv2.flip(image, 1)
        #imageshow = cv2.resize(imageshow, (960, 540))
        cv2.imshow(previewName, imageshow)
        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            break




# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()

# fix matrix to correct fisheye
DIM = (2560, 1440)
K = np.array([[1694.7801205342007, 0.0, 1363.235424743914], [0.0, 1694.4106664109822, 686.0493220616949], [0.0, 0.0, 1.0]])
D = np.array([[-0.010547721322285927], [0.02576312669611254], [-0.1546810103404629], [0.15853057545495458]])

def fish_eye_fix(image):
    # this program is quoted from others, see more information in fisheye_fix.py
    dim1 = image.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    image = cv2.resize(image,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.6 * Knew[(0,1), (0,1)] #scaling
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x,y,w,h = 475,254,1536,864
    img_valid = undistorted_img[y:y+h, x:x+w]
    img_valid = cv2.resize(img_valid, (1280, 720), interpolation=cv2.INTER_AREA)
    return img_valid
    # I finally dont use this because it takes 0.12s per frame in averge, too havy
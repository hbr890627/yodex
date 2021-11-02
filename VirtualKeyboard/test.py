import cv2
import mediapipe as mp
import threading
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

imgL = np.zeros((480,640), np.uint8)
imgR = np.zeros((480,640), np.uint8)

'''# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))'''

'''# For webcam input:
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.9) as hands:
    while cap.isOpened():
        success, image = cap.read()
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
        imageshow = cv2.flip(image, 1)
        #imageshow = cv2.resize(imageshow, (960, 540))
        cv2.imshow('MediaPipe Hands1', imageshow)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

cap2 = cv2.VideoCapture(2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.9) as hands:
    while cap2.isOpened():
        success2, image2 = cap2.read()
        if not success2:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image2.flags.writeable = False
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results2 = hands.process(image)

        # Draw the hand annotations on the image.
        image2.flags.writeable = True
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results2.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        imageshow2 = cv2.flip(image, 1)
        #imageshow = cv2.resize(imageshow, (960, 540))
        cv2.imshow('MediaPipe Hands2', imageshow2)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap2.release()'''


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

    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.9)
    while cap.isOpened():
        global imgL, imgR
        start = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        img_backup = image
        if camID == 1:
            img_backup = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgL = img_backup
        if camID == 2:
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
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
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
            break

    '''if cam.isOpened():  # try to get the first frame
        rval, image = cam.read()
    else:
        rval = False

    while rval:
        #cv2.imshow(previewName, frame).
        rval, image = cam.read()
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.9)
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
        imageshow = cv2.flip(image, 1)
        #imageshow = cv2.resize(imageshow, (960, 540))
        cv2.imshow(previewName, imageshow)
        #if cv2.waitKey(5) & 0xFF == 27:
        #    break
        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)'''

def stereo():
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
    global imgL, imgR
    while True:
        try:
            #print(imgL.shape, imgR.shape)
            disparity = stereo.compute(imgL, imgR)
            #disparity = cv2.resize(disparity, (0, 0), fx=0.3, fy=0.3,interpolation=cv2.INTER_NEAREST)
            disparity = cv2.normalize(disparity, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            h, w = disparity.shape
            cv2.imshow('disparity',disparity)
        except:
            #print(imgL.shape, imgR.shape)
            True

        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            break
    cv2.destroyWindow('disparity')


# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread3 = camThread("Camera 3", 2)
thread4 = threading.Thread(target = stereo)
thread1.start()
thread2.start()
thread3.start()
thread4.start()
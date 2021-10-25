import cv2
import mediapipe as mp
import threading
import time

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

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.9)
    while cap.isOpened():
        start = time.time()
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

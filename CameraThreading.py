import cv2
import threading
import time


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting the thread : " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    # Camera setting
    cap = cv2.VideoCapture(camID)
    cv2.namedWindow(previewName)
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    while cap.isOpened():
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

        # Flip the image horizontally for a selfie-view display.
        imageshow = cv2.flip(image, 1)

        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        text = str(round(fps, 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imageshow, text, (0, 30), font, 1, (255, 0, 0), 2)

        cv2.imshow(previewName, imageshow)

        key = cv2.waitKey(20)
        if key == 27 & 0xFF == 27:  # exit on ESC
            thread_run = False
            break


def main():
    thread1 = camThread('Camera 0', 0)
    thread2 = camThread('Camera 1', 1)
    thread1.start()
    thread2.start()


if __name__ == '__main__':
    main()

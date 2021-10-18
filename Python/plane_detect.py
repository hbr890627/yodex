import cv2
import numpy as np
import glob
import os


# region 1
def DrawContour():
    img1 = cv2.imread("Datasets/Q1_Image/coin01.jpg")
    img2 = cv2.imread("Datasets/Q1_Image/coin02.jpg")

    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, img1_b) = cv2.threshold(img1_g, 127, 255, cv2.THRESH_BINARY)
    (thresh, img2_b) = cv2.threshold(img2_g, 127, 255, cv2.THRESH_BINARY)

    img1_gaussian = cv2.GaussianBlur(img1_b, (11, 11), 0)
    img2_gaussian = cv2.GaussianBlur(img2_b, (13, 13), 0)

    img1_edge = cv2.Canny(img1_gaussian, 0, 200)
    img2_edge = cv2.Canny(img2_gaussian, 0, 300)

    contours1, hierarchy1 = cv2.findContours(
        img1_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(
        img2_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img1, contours1, -1, (0, 0, 255), 2)
    cv2.drawContours(img2, contours2, -1, (0, 0, 255), 2)

    print("There are "+str(len(contours1))+" in coin01")
    print("There are "+str(len(contours2))+" in coin02")
    global count1, count2
    count1 = len(contours1)
    count2 = len(contours2)

    # cv2.imshow("coin01", img1)
    cv2.imshow("coin01", img1)
    cv2.imshow("coin02", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def CountContour(label1, label2):
    label1.setText("There are "+str(count1)+" coins in coin01.jpg")
    label2.setText("There are "+str(count2)+" coins in coin02.jpg")

# endregion

# region 2


def CornerDetection():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob('Datasets/Q2_Image/*.bmp')
    index = 0
    for fname in images:
        img = cv2.imread(fname)
        # img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
            index += 1
            img = cv2.resize(img, (1024, 1024))
            cv2.imshow('img'+str(index), img)

    global mtx, dist, rvecs, tvecs
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Intrinsic():
    print("Intrinsic Matrix:")
    print(mtx)
    # print("\n")


def Extrinsic(index):
    print("Extrinsic Matrix:")
    rmtx = cv2.Rodrigues(rvecs[index])
    rmtx = rmtx[0]
    emtx = np.column_stack((rmtx, tvecs[index]))
    print(emtx)
    # print("\n")


def Distortion():
    print("Distortion Matrix:")
    print(dist)
    # print("\n")

# endregion

# region 3


def AR():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob('Datasets/Q3_Image/*.bmp')
    for fname in images:
        img = cv2.imread(fname)
        # img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    pyramidpoint = np.float32([[1, 1, 0], [5, 1, 0], [3, 5, 0], [3, 3, -3]])

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        if ret:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(
                pyramidpoint, rvecs, tvecs, mtx, dist)

            img = DrawPyramid(img, imgpts)
            img = cv2.resize(img, (512, 512))
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DrawPyramid(img, imgpts):
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(
        imgpts[1].ravel()), (255, 128, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(
        imgpts[2].ravel()), (255, 128, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(
        imgpts[3].ravel()), (255, 128, 0), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(
        imgpts[2].ravel()), (255, 128, 0), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(
        imgpts[3].ravel()), (255, 128, 0), 5)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(
        imgpts[3].ravel()), (255, 128, 0), 5)
    return img
# endregion


def StereoDisparityMap():
    imgL = cv2.imread("Datasets/Q4_Image/imgL.png", 0)
    imgR = cv2.imread("Datasets/Q4_Image/imgR.png", 0)

    stereo = cv2.StereoBM_create(numDisparities=144, blockSize=35)
    disparity = stereo.compute(imgL, imgR)
    disparity = cv2.resize(disparity, (0, 0), fx=0.5, fy=0.5)
    disparity = cv2.normalize(disparity, None, 255, 0,
                              cv2.NORM_MINMAX, cv2.CV_8UC1)

    h, w = disparity.shape
    cv2.imshow('disparity', disparity)

    def mouse(event, x, y, flags, param):
        d = disparity
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(d, (w-200, 0), (w, 60), (255, 255, 255), -1)
            text1 = 'Disparity : '+str(disparity[y][x])+" pixels"
            cv2.putText(d, text1, (w-200, 20),
                        cv2.FONT_ITALIC, 0.6, (0, 255, 255), 2)
            dep = 178*2826/disparity[y][x]
            text2 = 'Depth : ' + str(int(dep))+' mm'
            cv2.putText(d, text2, (w-200, 50),
                        cv2.FONT_ITALIC, 0.6, (0, 255, 255), 2)
            cv2.imshow('disparity', d)

    cv2.setMouseCallback('disparity', mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    DrawContour()


# CornerDetection()
AR()


# images = glob.glob('Datasets/Q2_Image/*.bmp')
# print(images)

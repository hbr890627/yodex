import cv2
import matplotlib.pyplot as plt
import numpy as np


camera = cv2.VideoCapture(0)

if __name__ == '__main__':
    _, img = camera.read()
    #img = cv2.imread('yodex\\test_table_detect\\test.png', 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
 
    edge = cv2.Canny(img, 40, 180)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    tmp = np.zeros(img.shape, np.uint8)
    draw_cnt = []
    arcs = []
    for c in contours:
        arc = cv2.arcLength(c, False)
        arcs.append(arc)
 
    draw_cnt.append(contours[arcs.index(max(arcs))])
    cv2.drawContours(tmp, draw_cnt, -1, (250, 255, 255), 2) 
    approx = cv2.approxPolyDP(draw_cnt[0], 90, True)  # 近似多边形，10为精度，调这个参数使边数为4
    print(str(approx.size))
    approx_point1 = approx.reshape(4, 2).astype(np.float32)
 
    plane = np.array([[0, 0], [0, 600], [400, 600], [400, 0]], dtype="float32")  # 投影到400*600的面上
 
    M = cv2.getPerspectiveTransform(approx_point1, plane)
    out_img = cv2.warpPerspective(img, M, (400, 600))
    dst = cv2.perspectiveTransform(plane.reshape(1, 4, 2), M)
 
    transformed = cv2.resize(out_img, (400, 600))
 
    cv2.imshow('edge', transformed)
 
cv2.waitKey()

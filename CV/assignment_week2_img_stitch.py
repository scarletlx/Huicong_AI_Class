# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def img_stitch(img1, img2):
    MIN = 10
    starttime = time.time()

    # img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # cv2.xfeatures2d.SURF_create ([hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]])
    # 该函数用于生成一个SURF对象，在使用时，为提高速度，可以适当提高hessianThreshold，以减少检测的关键点的数量，
    # 可以extended=False，只生成64维的描述符而不是128维，令upright=True，不检测关键点的方向。
    surf = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
    # surf = cv2.xfeatures2d.SIFT_create()#可以改为SIFT

    # cv2.SURF.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]])
    # 该函数用于计算图片的关键点和描述符，需要对两幅图都进行计算。
    kp1, descrip1 = surf.detectAndCompute(img1, None)
    kp2, descrip2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    # flann快速匹配器
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(descrip1, descrip2, k=2)

    good = []
    for i, (m, n) in enumerate(match):
        if (m.distance < 0.75 * n.distance):
            good.append(m)

    if len(good) > MIN:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M, mask = cv2.findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask]]])
        # 单应性匹配，返回的M是一个矩阵，即对关键点srcPoints做M变换能变到dstPoints的位置。
        M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # warpImg = cv2.warpPerspective(src, np.linalg.inv(M), dsize[, dst[, flags[, borderMode[, borderValue]]]])
        # 进行透视变换，变换视角。src是要变换的图片，np.linalg.inv(M) 中M的逆矩阵，得到方向一致的图片。
        warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))

        # 使用drawMatches可以画出匹配的好的关键点，matchesMask是比较好的匹配点，之间用绿色线连接起来。
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=2)
        img5 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(cv2.cvtColor(img5,cv2.COLOR_BGR2RGB)), plt.show()

        direct = warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] = img1
        simple = time.time()

        # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        # cv2.imshow("Result",warpImg)
        rows, cols = img1.shape[:2]

        for col in range(0, cols):
            if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
        final = time.time()

        img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
        plt.imshow(img3, ), plt.show()
        img4 = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
        plt.imshow(img4, ), plt.show()

        print("simple stitch cost %f" % (simple - starttime))
        print("total cost %f" % (final - starttime))
        cv2.imwrite("img_stitch/simplepanorma.jpg", direct)
        cv2.imwrite("img_stitch/bestpanorma.jpg", warpImg)

    else:
        print("not enough matches!")


if __name__ == "__main__":
    img1 = cv2.imread('img_stitch/img1.jpg')  # query
    img2 = cv2.imread('img_stitch/img2.jpg')  # train

    img_stitch(img1, img2)
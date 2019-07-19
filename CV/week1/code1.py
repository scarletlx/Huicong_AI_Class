# -*- coding: utf-8 -*-

"""
1. Recode all examples;

2. Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
   Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.
"""

import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt


# show gray image
def img_to_gray(fname, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey", img_gray)
    print(img_gray)
    print(img.shape, img_gray.shape)


################################
# image crop
def img_crop(fname, img):
    img_crop = img[100:300,100:200]
    cv2.imshow("crop", img_crop)

def img_crop(fname, img, crop_size, random_crop=True):
    if random_crop:  # 若随机裁剪
        if img.shape[1] > crop_size:
            sz1 = img.shape[1] // 2
            sz2 = crop_size // 2
            diff = sz1 - sz2
            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
            img = img[v:(v + crop_size), h:(h + crop_size), :]



################################
# color split
def img_color_split(fname, img):
    b, g, r = cv2.split(img)
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)


################################
# change color
def img_random_light_color(fname, img):
    b, g, r = cv2.split(img)

    rand = random.randint(-50, 50)
    for i in [b, g, r]:
        if rand == 0:
            pass
        elif rand > 0:
            lim = 255 - rand
            i[i > lim] = 255
            i[i <= lim] = (rand + i[i <= lim]).astype(img.dtype)
        elif rand < 0:
            lim = 0 - rand
            i[i < lim] = 0
            i[i >= lim] = (rand + i[i >= lim]).astype(img.dtype)

    dst = cv2.merge((b,g,r))
    # cv2.imshow("Light Color", dst)

    outPath = os.path.join(os.path.abspath(os.curdir),"output")
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    outfname = os.path.splitext(fname)[0] + "_colorshift" + os.path.splitext(fname)[1]
    cv2.imwrite(os.path.join(outPath, outfname), dst)


################################
# gamma correction
# gamma > 1 时， 暗处的噪点将会被压缩；亮处的噪点虽然会被放大
# gamma < 1 时正好相反
def img_gamma_corr(fname, img, gamma=1.0):
    inv_gamma = 1/gamma
    dst = ((img / 255.0) ** inv_gamma * 255).astype("uint8")
    # cv2.imshow("Gamma correction " + str(gamma), dst)

    outPath = os.path.join(os.path.abspath(os.curdir),"output")
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    outfname = os.path.splitext(fname)[0] + "_gamma_" + str(gamma) + os.path.splitext(fname)[1]
    cv2.imwrite(os.path.join(outPath, outfname), dst)


################################
# histogram
def img_histogram(fname, img):
    img_brighter = img_gamma_corr(img, 2)
    img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0] * 0.5), int(img_brighter.shape[1] * 0.5)))
    plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
    img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(??????), u&v: ???????
    cv2.imshow('Color input image', img_small_brighter)
    cv2.imshow('Histogram equalized', img_output)


################################
# rotation
def img_rotation(fname, img):
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), -30, 0.8)  # getRotationMatrix2D 用于获得图像绕着 某一点的旋转矩阵
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna', img_rotate)
    M[0][2] = M[1][2] = 0
    img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # cv2.imshow('rotated lenna2', img_rotate2)

    # scale+rotation+translation = similarity transform
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.8)  # center, angle, scale
    img_rotate3 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # cv2.imshow('rotated lenna3', img_rotate3)

    outPath = os.path.join(os.path.abspath(os.curdir),"output")
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    outfname = os.path.splitext(fname)[0] + "_rotation" + os.path.splitext(fname)[1]
    cv2.imwrite(os.path.join(outPath, outfname), img_rotate3)


################################
# Affine Transform
def img_affine_transform(fname, img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow('affine lenna', dst)

    outPath = os.path.join(os.path.abspath(os.curdir),"output")
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    outfname = os.path.splitext(fname)[0] + "_affinetrans" + os.path.splitext(fname)[1]
    cv2.imwrite(os.path.join(outPath, outfname), dst)


############################
# perspective transform
def img_perspective(fname, img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    # cv2.imshow('lenna_warp', img_warp)

    outPath = os.path.join(os.path.abspath(os.curdir),"output")
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    outfname = os.path.splitext(fname)[0] + "_perspective" + os.path.splitext(fname)[1]
    cv2.imwrite(os.path.join(outPath, outfname), img_warp)


if __name__ == "__main__":

    # show original image
    fname = "lenna.jpg"
    img = cv2.imread(fname)

    # img_to_gray(img)

    # img_crop(fname, img)

    # img_color_split(fname, img)


    img_random_light_color(fname, img)

    for i in [0.5, 0.7, 1.5, 2]:
        img_gamma_corr(fname, img, i)
    #
    # img_histogram(fname, img)
    #
    img_rotation(fname, img)

    img_affine_transform(fname, img)

    img_perspective(fname, img)


    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()

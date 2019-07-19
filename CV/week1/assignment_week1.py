# -*- coding: utf-8 -*-

"""
image augmentation
read image files in img_in folder, do some image crop, color shift, rotation and perspective transform 
to create some new image files in img_out folder
"""

import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt


# show gray image
def img_to_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey", img_gray)
    print(img_gray)
    print(img.shape, img_gray.shape)


################################
# image crop
def img_crop(img):
    crop_size = int(img.shape[1] * abs(np.random.random()/2 + 0.5))
    sz1 = img.shape[1] // 2
    sz2 = crop_size // 2
    diff = sz1 - sz2
    (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
    img = img[v:(v + crop_size), h:(h + crop_size), :]

    return img


################################
# color split
def img_color_split(img):
    b, g, r = cv2.split(img)
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)


################################
# change color
def img_random_light_color(img):
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
    return dst


################################
# gamma correction
# gamma > 1 时， 暗处的噪点将会被压缩；亮处的噪点虽然会被放大
# gamma < 1 时正好相反
def img_gamma_corr(img, gamma=1.0):
    inv_gamma = 1/gamma
    dst = ((img / 255.0) ** inv_gamma * 255).astype("uint8")
    # cv2.imshow("Gamma correction " + str(gamma), dst)

    return dst


################################
# histogram
def img_histogram(img):
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
def img_rotation(img):
    angle = np.random.randint(-45, 45)
    RotateMatrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle=angle, scale=0.7)  # getRotationMatrix2D 用于获得图像绕着 某一点的旋转矩阵
    dst = cv2.warpAffine(img, RotateMatrix, (img.shape[1], img.shape[0]))

    return dst


################################
# Affine Transform
def img_affine_transform(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow('affine lenna', dst)

    return dst


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

    return dst


def rad(x):
    return x * np.pi / 180

def img_perspective2(img):
    anglex = np.random.randint(-60,60)
    angley = np.random.randint(-60,60)
    anglez = np.random.randint(-60,60)  # 是旋转
    fov = 42
    r = 0
    w = img.shape[1]
    h = img.shape[0]

    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    M_warp = cv2.getPerspectiveTransform(org, dst)
    img_warp = cv2.warpPerspective(img, M_warp, (w, h))

    return img_warp



if __name__ == "__main__":

    curr_dir = os.getcwd()
    out_path = os.path.join(curr_dir,"img_out")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    input_path = os.path.join(curr_dir,"img_in")
    for root, dir, files in os.walk(input_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                surname = os.path.splitext(file)[0]
                img = cv2.imread(os.path.join(input_path,file))

                dst = img_crop(img)
                outfname = surname + "_crop.jpg"
                cv2.imwrite(os.path.join(out_path, outfname), dst)

                dst = img_random_light_color(img)
                outfname = surname + "_colorshift.jpg"
                cv2.imwrite(os.path.join(out_path, outfname), dst)

                dst = img_rotation(img)
                outfname = surname + "_rotation.jpg"
                cv2.imwrite(os.path.join(out_path, outfname), dst)

                dst = img_perspective2(img)
                outfname = surname + "_persp.jpg"
                cv2.imwrite(os.path.join(out_path, outfname), dst)


    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()


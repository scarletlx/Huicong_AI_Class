# -*- coding: utf-8 -*-

# [Coding]:
#    Finish 2D convolution/filtering by your self.
#    What you are supposed to do can be described as "median blur", which means by using a sliding window
#    on an image, your task is not going to do a normal convolution, but to find the median value within
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:

import numpy as np
import cv2
import random

def medianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string

    W, H = np.array(img).shape
    m, n = np.array(kernel).shape

    padding_size = (m // 2, n // 2)
    if padding_way == 'REPLICA':
        img_pad = np.pad(img, padding_size, 'edge')
    elif padding_way == 'ZERO':
        img_pad = np.pad(img, padding_size, 'constant', constant_values=0)
    else:
        return img

    img_blur = np.zeros((W,H), dtype=np.float)
    for w in range(W):
        for h in range(H):
            img_blur[w, h] = np.median(img_pad[w:w+m, h:h+n])/255

    return img_blur


#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A,
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like:
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#

"""
# pseudo code
def ransacMatching(A, B, threshold, d):
    # A & B: List of List
    # threshold - 阈值:作为判断点满足模型的条件
    # d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    
    iterations = 0
    homography = None  # 后面更新
    max_inliers = 0

    ms1 = np.copy(A)
    ms2 = np.copy(B)
    
    k = 2000  #迭代初始值
    confidence = 0.995
    count = ms1.shape[0] * ms2.shape[1]

    while iterations < k :
    
        # 1. 从样本中随机选取4个
        all_idxs = np.arange(np.array(ms1)) #获取A下标索引
        np.random.shuffle(all_idxs) #打乱下标索引
        inliers_idxs = all_idxs[:4]
        outliers_idxs = all_idxs[4:]
        
        ptsA = ms1[inliers_idxs,:]
        ptsB = ms2[inliers_idxs,:]
        
        # 2. 计算单应矩阵
        homography = 通过给定的4组inliers (ptsA, ptsB) 序列计算出单应矩阵 
        
        # 3. 测试计算的单应矩阵
        also_inliers_idxs = []  # 满足误差要求的样本点,开始置空

        for lier in outliers_idxs :        
            error = checkSubset(ms1[lier], ms2[lier], homography) #用上面得出的单应矩阵计算每个点的误差
            if error < threshold :
                also_inliers_idxs.append(lier)  # 增加一个内点

        # 4. 获取所有的内点
        good_count = len(also_inliers_idxs)
        if good_count > d :
            ms1 = np.concatenate( (inliers_idxs, also_inliers_idxs) ) #样本连接
            ms2 = np.concatenate( (inliers_idxs, also_inliers_idxs) ) #样本连接
            
        if good_count > max_inliers :
            max_inliers = len(also_inliers_idxs)

        ep = (count - good_count) / count
        k = np.log(1 - confidence) / np.log((1 - ep) ** 4)   # 更新迭代次数
        
        iterations += 1

    return homography
"""
#
#
# if __name__ == "__main__":
#     src = cv2.imread("lenna.jpg")
#     B, G, R = cv2.split(src)
#
#     cv2.imshow("lenna B", B)
#     kernel = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
#     src_blur = medianBlur(B, kernel, 'REPLICA')
#     cv2.imshow("lenna blur1", src_blur)
#
#     key = cv2.waitKey()
#     if key == 27:
#         cv2.destroyAllWindows()

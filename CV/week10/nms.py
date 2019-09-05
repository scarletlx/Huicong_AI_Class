# -*- coding: utf-8 -*-

# 给出一张图片和上面许多物体检测的候选框（即每个框可能都代表某种物体），但是这些框很可能有互相重叠的部分，
# 我们要做的就是只保留最优的框。假设有N个框，每个框被分类器计算得到的分数为Si, 1<=i<=N。
#
# 0、建造一个存放待处理候选框的集合H，初始化为包含全部N个框；建造一个存放最优框的集合M，初始化为空集。
# 1、将所有集合 H 中的框进行排序，选出分数最高的框 m，从集合 H 移到集合 M；
# 2、遍历集合 H 中的框，分别与框 m 计算交并比（Interection-over-union，IoU），如果高于某个阈值（一般为0~0.5），
# 则认为此框与 m 重叠，将此框从集合 H 中去除。
# 3、回到第1步进行迭代，直到集合 H 为空。集合 M 中的框为我们所需。

# 需要优化的参数：IoU 的阈值是一个可优化的参数，一般范围为0.3~0.5，可以使用交叉验证来选择最优的参数。

import numpy as np
import matplotlib.pyplot as plt

def plot_boxes(lists, c='k'):
    lists = np.array(lists)
    x1 = lists[:, 0]
    y1 = lists[:, 1]
    x2 = lists[:, 2]
    y2 = lists[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.show()


def nms(lists, thre):
    # lists is a list. lists[0:4]: x1, x2, y1, y2; lists[4]: score
    lists = np.array(lists)

    x1 = lists[:, 0]
    y1 = lists[:, 1]
    x2 = lists[:, 2]
    y2 = lists[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = lists[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while len(index) > 0:
        i = index[0]
        keep.append(lists[i])

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        overlaps = (y22 - y11 + 1) * (x22 - x11 + 1)
        ious = 1.0 * overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thre)[0]
        index = index[idx + 1]
    return keep


def soft_nms(lists, thre):
    pass


if __name__ == '__main__':
    boxes = [[100, 100, 210, 210, 0.72],
             [250, 250, 420, 420, 0.8],
             [220, 220, 320, 330, 0.92],
             [100, 100, 210, 210, 0.72],
             [230, 240, 325, 330, 0.81],
             [220, 230, 315, 340, 0.9]]

    plot_boxes(boxes)
    keep = nms(boxes, thre=0.7)
    plot_boxes(keep, c='r')

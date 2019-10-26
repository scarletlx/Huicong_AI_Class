import os
import numpy as np
import cv2
import random
from random import sample
import matplotlib.pyplot as plt


def remove_invalid_image(lines):
    images = []
    for line in lines:
        name = line.split()[0]
        if os.path.isfile(name):
            images.append(line)
    return images


def load_metadata(folder_list):
    tmp_lines = []
    for folder_name in folder_list:
        folder = os.path.join('../', folder_name)
        metadata_file = os.path.join(folder, 'label.txt')
        with open(metadata_file) as f:
            lines = f.readlines()
        tmp_lines.extend(list(map((folder + '/').__add__, lines)))
    res_lines = remove_invalid_image(tmp_lines)
    return res_lines


def visualize_face(line):
    face_info = line.strip().split()
    img = cv2.imread(face_info[0])
    img_height, img_width = img.shape[:2]
    face_pos = [int(float(num)) if float(num) >=0 else 0 for num in face_info[1:]]

    roi_x1, roi_y1, roi_x2, roi_y2, _, _ = expand_roi(face_pos[0], face_pos[1], face_pos[2],
                                                      face_pos[3], img_width, img_height, ratio=0.25)

    img = cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    img = cv2.rectangle(img, (face_pos[0], face_pos[1]), (face_pos[2], face_pos[3]), (0, 255, 0), 2)

    for i in range(21):
        cv2.circle(img, (face_pos[4 + i*2], face_pos[4 + i*2 + 1]), 1, (0, 0, 255), 2)

    plt.imshow(img)
    plt.show()


# 人脸框标识扩充范围
def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):   # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2

    return roi_x1, roi_y1, roi_x2, roi_y2, roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


def get_faces(lines):
    faces_data = []
    for line in lines:
        line = line.strip().split()
        face_pos = [int(float(num)) if float(num) >= 0 else 0 for num in line[1:]]
        img = cv2.imread(line[0])
        img_height, img_width = img.shape[:2]
        roi_x1, roi_y1, roi_x2, roi_y2, _, _ = expand_roi(face_pos[0], face_pos[1],
                                                          face_pos[2], face_pos[3], img_width, img_height,
                                                          ratio=0.25)
        x = [pos - roi_x1 for pos in face_pos[4::2]]
        y = [pos - roi_y1 for pos in face_pos[5::2]]
        landmark = list(zip(x, y))
        # 1表示是人脸
        new_line = '1 ' + line[0] + ' ' + str(roi_x1) + ' ' + str(roi_y1) + ' ' + str(roi_x2) + ' ' + str(roi_y2)
        for pos in landmark:
            new_line += ' ' + str(pos[0]) + ' ' + str(pos[1])

        faces_data.append(new_line)
    return faces_data


def load_truth(lines):
    truth = {}
    for line in lines:
        line = line.strip().split()
        name = line[0]
        if name not in truth:
            truth[name] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(float, line[5::2]))
        y = list(map(float, line[6::2]))
        landmarks = list(zip(x, y))
        truth[name].append((rect, landmarks))
    return truth


def channel_norm(img):
    img = img.astype('float32')
    m_mean = np.mean(img)
    m_std = np.std(img)

    print('mean: ', m_mean)
    print('std: ', m_std)

    return (img - m_mean) / m_std


def check_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    # print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    # print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou


def get_iou(rect1, rect2):
    overlap_w = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]) \
        if min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]) > 0 else 0
    overlap_h = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) \
        if min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) > 0 else 0
    f = lambda a, b: (a[0] - a[2]) * (a[1] - a[3]) + (b[0] - b[2]) * (b[1] - b[3])
    overlap_area = f(rect1, rect2) - overlap_w * overlap_h
    return overlap_w * overlap_h / overlap_area


def generate_random_crops(shape, rects, random_times, neg_gen_thre, random_border=10):
    neg_gen_cnt = 0
    img_h = shape[0]
    img_w = shape[1]
    rect_wmin = img_w   # + 1
    rect_hmin = img_h   # + 1
    rect_wmax = 0
    rect_hmax = 0
    num_rects = len(rects)
    for rect in rects:
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        if w < rect_wmin:
            rect_wmin = w
        if w > rect_wmax:
            rect_wmax = w
        if h < rect_hmin:
            rect_hmin = h
        if h > rect_hmax:
            rect_hmax = h
    random_rect_cnt = 0
    random_rects = []
    while random_rect_cnt < num_rects * random_times and neg_gen_cnt < neg_gen_thre:
        neg_gen_cnt += 1
        if img_h - rect_hmax - random_border > 0:
            top = np.random.randint(0, img_h - rect_hmax - random_border)
        else:
            top = 0
        if img_w - rect_wmax - random_border > 0:
            left = np.random.randint(0, img_w - rect_wmax - random_border)
        else:
            left = 0
        rect_wh = np.random.randint(min(rect_wmin, rect_hmin), max(rect_wmax, rect_hmax) + 1)
        rect_randw = np.random.randint(-3, 3)
        rect_randh = np.random.randint(-3, 3)
        right = left + rect_wh + rect_randw - 1
        bottom = top + rect_wh + rect_randh - 1

        good_cnt = 0
        negsample_ratio = 0.3
        for rect in rects:
            img_rect = [0, 0, img_w - 1, img_h - 1]
            rect_img_iou = get_iou(rect, img_rect)
            if rect_img_iou > negsample_ratio:
                random_rect_cnt += random_times
                break
            random_rect = [left, top, right, bottom]
            iou = get_iou(random_rect, rect)

            if iou < 0.2:
                # good thing
                good_cnt += 1
            else:
                # bad thing
                break

            if good_cnt == num_rects:
                # print('random rect: ', random_rect, '   rect: ', rect)
                _iou = check_iou(random_rect, rect)

                # print('iou: ', iou, '   check_iou: ', _iou)
                # print('\n')
                random_rect_cnt += 1
                random_rects.append(random_rect)
    return random_rects


def get_neg_sample(lines):
    neg_samples = []
    truth = load_truth(lines)
    for name in truth:
        rects = []
        for face in truth[name]:
            rects.append(face[0])
        img = cv2.imread(name)
        neg_rects = generate_random_crops(img.shape, rects, 100, 2)

        # 0 表示非人脸
        for rect in neg_rects:
            new_line = '0 ' + name + ' ' + str(rect[0]) + ' ' + str(rect[1]) + ' ' + str(rect[2]) + ' ' + str(rect[3])
            neg_samples.append(new_line)
    return neg_samples


def _test_draw_faces(lines):
    for line in lines:
        face_info = line.strip().split()
        img = cv2.imread(face_info[0])
        img_height, img_width = img.shape[:2]
        face_pos = [int(float(num)) if float(num) >= 0 else 0 for num in face_info[1:]]

        roi_x1, roi_y1, roi_x2, roi_y2, _, _ = expand_roi(face_pos[0], face_pos[1], face_pos[2],
                                                          face_pos[3], img_width, img_height, ratio=0.25)

        img = cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        img = cv2.rectangle(img, (face_pos[0], face_pos[1]), (face_pos[2], face_pos[3]), (0, 255, 0), 2)

        for i in range(21):
            cv2.circle(img, (face_pos[4 + i * 2], face_pos[4 + i * 2 + 1]), 1, (0, 0, 255), 2)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    folders = ['I', 'II']
    train_list = 'train.txt'
    test_list = 'test.txt'

    lines = load_metadata(folders)
    # _test_draw_faces(lines)

    # 随机显示一张图
    # idx = random.randint(0, len(lines))
    # visualize_face(lines[idx])

    neg_faces = get_neg_sample(lines)
    random.shuffle(neg_faces)
    faces_data = get_faces(lines)
    random.shuffle(faces_data)
    train_set = faces_data[:int(0.7*len(faces_data))] + neg_faces[:int(0.7*len(neg_faces))]
    test_set = faces_data[int(0.7*len(faces_data)):] + neg_faces[int(0.7*len(neg_faces)):]
    random.shuffle(train_set)
    random.shuffle(test_set)

    with open(train_list, 'w') as f:
        for line in train_set:
            f.write(line + '\n')
        f.close()

    with open(test_list, 'w') as f:
        for line in test_set:
            f.write(line + '\n')
        f.close()


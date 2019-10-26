import numpy as np
import cv2
import math

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import matplotlib.pyplot as plt

folder_list = ['I', 'II']
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    label = np.array(int(line_parts[0]))
    img_name = line_parts[1]
    rect = list(map(int, list(map(float, line_parts[2:6]))))
    landmarks = list(map(float, line_parts[6: len(line_parts)]))
    return label, img_name, rect, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)

        return {'image': image,
                'landmarks': landmarks,
                'label': label
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image/255.0
        image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, axis=0)

        landmarks = landmarks/train_boarder
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks).float(),
                'label': torch.from_numpy(label).long()
                }


class RandomHorizontalFlip(object):
    """
        image augmentation: do some image flip to create some new faces
    """
    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        image = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)

        image_flip = cv2.flip(image, 1)
        image_flip = Image.fromarray(np.uint8(image_flip))

        landmarks_flip = [train_boarder - landmarks[i] if i % 2 == 0 else landmarks[i] for i in range(42)]
        landmarks_flip = np.array(landmarks_flip).astype(np.float32)

        return {'image': image_flip,
                'landmarks': landmarks_flip,
                'label': label
                }


class RandomRotation(object):
    """
        image rotation: do some image rotation and perspective transform to create some new faces
    """
    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        image = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)

        height, width = image.shape[:2]
        degree = np.random.randint(-45, 45)

        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle=degree, scale=1)
        img_rotate = cv2.warpAffine(image, M, (width, height))
        img_rotate = Image.fromarray(np.uint8(img_rotate))

        # Q = np.dot(M, np.array([[P[0]], [P[1]], [1]]))
        landmarks_reshape = landmarks.reshape(-1, 2)
        trans = np.c_[landmarks_reshape, np.ones(len(landmarks_reshape))]
        landmarks_rotate = np.dot(M, trans.T).T
        landmarks_rotate = landmarks_rotate.reshape(-1, )

        return {'image': img_rotate,
                'landmarks': landmarks_rotate,
                'label': label
                }


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, transform=None):
        '''
        :param src_lines: src_lines
        :param phase: whether we are training or not
        :param transform: data transform
        '''

        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        label, img_name, rect, landmarks = parse_line(self.lines[idx])
        img = Image.open(img_name).convert('RGB')
        img_crop = img.crop(tuple(rect))

        if label==1:
            landmarks = np.array(landmarks).astype(np.float32)

            # you should let your landmarks fit to the train_boarder(112)
            # please complete your code under this blank
            # your code:
            for i in range(len(landmarks) // 2):
                landmarks[i * 2 + 0] = int(landmarks[i * 2 + 0] * train_boarder / (rect[2] - rect[0]))
                landmarks[i * 2 + 1] = int(landmarks[i * 2 + 1] * train_boarder / (rect[3] - rect[1]))
        else:
            landmarks = np.zeros((42), dtype=np.float32)

        label = np.array(label)
        label = np.expand_dims(label, axis=0)
        sample = {'image': img_crop, 'landmarks': landmarks, 'label': label}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    tsfm = transforms.Compose([
        # RandomHorizontalFlip(),     # do Horizontal flip
        RandomRotation(),           # do rotation
        Normalize(),                # do channel normalization
        ToTensor()                  # convert to torch type: NxCxHxW
    ])
    if phase == 'Train' or phase == 'train':
        pass
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':

    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(test_set, batch_size=128)
    # for batch_idx, batch in enumerate(train_loader):
    #     img = batch['image']
    #     landmark = batch['landmarks']
    #     label = batch['label']

    for i in range(0, 20):
        # sample = train_set[i]
        sample = train_loader.dataset[i]
        # original img is C x H x W
        img = sample['image'].numpy().transpose(1, 2, 0) * 255
        landmarks = sample['landmarks'] * train_boarder
        label = sample['label']
        print(label)

        # 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank

        # draw original image
        x = list(map(int, landmarks[0: len(landmarks): 2]))
        y = list(map(int, landmarks[1: len(landmarks): 2]))
        landmarks_truth = list(zip(x, y))
        for landmark_truth in landmarks_truth:
            cv2.circle(img, tuple(landmark_truth), 2, (0, 255, 0), -1)
        plt.imshow(img)
        plt.show()

        # # draw flip image
        # img1 = cv2.flip(img, 1)
        # landmarks_flip = [train_boarder - landmarks[i] if i % 2 == 0 else landmarks[i] for i in range(42) ]
        # x = list(map(int, landmarks_flip[0: len(landmarks_flip): 2]))
        # y = list(map(int, landmarks_flip[1: len(landmarks_flip): 2]))
        # landmarks_truth = list(zip(x, y))
        # for landmark_truth in landmarks_truth:
        #     cv2.circle(img1, tuple(landmark_truth), 2, (0, 255, 0), -1)
        # cv2.imshow('flip', img1)
        #
        # # draw rotated image
        # degree = np.random.randint(-45, 45)
        # height, width = img.shape[:2]
        # M = cv2.getRotationMatrix2D((width // 2, height // 2), angle=30, scale=1)
        # img2 = cv2.warpAffine(img, M, (width, height))
        #
        # landmarks_reshape = landmarks.reshape(-1, 2)
        # trans = np.c_[landmarks_reshape, np.ones(len(landmarks_reshape))]
        # landmarks_rotate = np.dot(M, trans.T).T
        # landmarks_rotate = landmarks_rotate.reshape(-1, 1)
        # print(landmarks_rotate)
        #
        # x = list(map(int, landmarks_rotate[0: len(landmarks_rotate): 2]))
        # y = list(map(int, landmarks_rotate[1: len(landmarks_rotate): 2]))
        # landmarks_truth = list(zip(x, y))
        # for landmark_truth in landmarks_truth:
        #     cv2.circle(img2, tuple(landmark_truth), 2, (0, 255, 0), -1)
        # cv2.imshow('rotate', img2)

        # key = cv2.waitKey()
        # if key == 27:
        #     exit(0)
        # cv2.destroyAllWindows()

from __future__ import print_function
import torch
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data import get_train_test_set
import config
from net import Net

# 此部分代码针对stage 1中的predict。 是其配套参考代码
# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。


def predict(model, test_img_name):

    # For single GPU
    use_cuda = torch.cuda.is_available()
    if config.DEVICE:
        device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    else:
        device = torch.device("cpu")

    state = torch.load(os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME), map_location='cpu')
    model.load_state_dict(state['state_dict'])

    model.eval()  # prep model for evaluation

    img_src = Image.open(test_img_name)
    width, height = img_src.size
    img_src = np.asarray(img_src.convert('RGB').resize(config.NET_IMG_SIZE, Image.BILINEAR), dtype=np.float32)
    img = img_src.transpose((2, 0, 1))
    img = img/255.0
    img = torch.from_numpy(img).unsqueeze(0)

    output_pts, output_cls = model(img)
    pred_class = output_cls.argmax(dim=1, keepdim=True).squeeze()
    output_pts = output_pts.squeeze() * config.NET_IMG_SIZE[0]
    print(pred_class)
    print(output_pts)

    if pred_class:
        x = output_pts[::2]
        y = output_pts[1::2]
        img_src = Image.fromarray(img_src.astype('uint8'))
        draw = ImageDraw.Draw(img_src)
        points_zip = list(zip(x, y))

        if len(img_src.getbands()) == 4:
            draw.point(points_zip, (0, 255, 0))
        else:
            draw.point(points_zip, 255)
    plt.imshow(img_src)
    plt.show()


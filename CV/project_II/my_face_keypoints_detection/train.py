from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import os
from PIL import Image, ImageDraw

from data import get_train_test_set
from predict import predict
from net import Net
import config

torch.set_default_tensor_type(torch.FloatTensor)


def train(data_loaders, model, criterion):
    # save model
    if config.MODEL_SAVE_PATH:
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)

    # For single GPU
    use_cuda = torch.cuda.is_available()
    if config.DEVICE:
        device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    else:
        device = torch.device("cpu")
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = model.to(device)

    epoch = config.EPOCH
    pts_criterion = criterion[0]
    cls_criterion = criterion[1]

    lr = config.LEARNING_RATE
    opt_SGD = torch.optim.SGD(model.parameters(), lr=lr)
    opt_Momentum = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
    opt_Adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizers = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}
    optimizer = optimizers[config.OPTIMIRZER]

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        ######################
        # training the model #
        ######################
        for phase in ['train', 'val']:
            if phase == 'train':
                if config.FLAG_RESTORE_MODEL and os.path.exists(config.MODEL_NAME):
                    config.FLAG_RESTORE_MODEL = False

                    # model.load_state_dict(torch.load(config.MODEL_NAME, map_location='cpu'))

                    state = torch.load(os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME), map_location='cpu')
                    # epoch_start = state['epoch']+1
                    model.load_state_dict(state['state_dict'])

                model.train()
            else:
                model.eval()

            for batch_idx, batch in enumerate(data_loaders[phase]):
                img = batch['image'].to(device)
                landmark = batch['landmarks'].to(device)
                label = batch['label'].to(device)

                # ground truth
                target_pts = landmark.to(device)
                target_pts = target_pts.float()

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # get output
                    output_pts, output_cls = model(img)

                    # handle positive samples
                    positive_mask = label == 1
                    positive_mask = np.squeeze(positive_mask)
                    len_true_positive = positive_mask.sum().item()
                    if len_true_positive == 0:
                        loss_positive = 0
                        pred_class_pos_correct = 0
                    else:
                        loss_positive_pts = pts_criterion(output_pts[positive_mask], landmark[positive_mask])
                        loss_positive_cls = cls_criterion(output_cls[positive_mask], label[positive_mask].squeeze(1))

                        loss_positive = 5 * loss_positive_pts + loss_positive_cls

                        positive_pred_class = output_cls[positive_mask].argmax(dim=1, keepdim=True)
                        pred_class_pos_correct = positive_pred_class.eq(label[positive_mask]).sum().item()

                    # handle negative samples
                    negative_mask = label == 0
                    negative_mask = np.squeeze(negative_mask)
                    len_true_negative = negative_mask.sum().item()
                    if len_true_negative == 0:
                        loss_negative = 0;
                        pred_class_neg_correct = 0
                    else:
                        loss_negative_cls = cls_criterion(output_cls[negative_mask], label[negative_mask].squeeze(1))
                        loss_negative = loss_negative_cls

                        negative_pred_cls = output_cls[negative_mask].argmax(dim=1, keepdim=True)
                        pred_class_neg_correct = negative_pred_cls.eq(label[negative_mask]).sum().item()

                    # sum up
                    loss = loss_positive + 3 * loss_negative

                    if phase == 'Train':
                        # do BP automatically
                        loss.backward()
                        optimizer.step()

                # show log info
                if batch_idx % config.LOG_INTERVAL == 0:
                    pred_class = output_cls.argmax(dim=1, keepdim=True)
                    index_img_eval = np.random.randint(0, len(img), size=5)
                    for j in index_img_eval:
                        img_ = img[j, :, :, :] * 255
                        landmark_ = output_pts[j, :] * 112
                        img_ = Image.fromarray(img_.cpu().numpy().transpose((1, 2, 0)).astype('uint8'))
                        if pred_class[j]:
                            draw = ImageDraw.Draw(img_)
                            x = landmark_[::2]
                            y = landmark_[1::2]
                            points_zip = list(zip(x, y))
                            draw.point(points_zip, (255, 0, 0))
                        if not os.path.exists(config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH):
                            os.mkdir(config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH)
                        if not os.path.exists(config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH + '/' + phase):
                            os.mkdir(config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH + '/' + phase)

                        img_.save(config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH
                                  + '/' + phase + '/' + str(epoch_id) + '_' + str(batch_idx) + '_' + str(j) + '.jpg')

                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\t\
                    Loss: {:.6f}\tloss_positive_pts: {:.6f}\t loss_positive_cls: {:.6f}\tloss_negative_cls: {:.6f}\t\
                    {} Pos acc: [{}/{} ({:.2f}%)]\tNeg acc: [{}/{} ({:.2f}%)]\tAcc: [{}/{} ({:.2f}%) {}]'.format(
                        phase,
                        epoch_id,
                        batch_idx * len(img),
                        len(data_loaders[phase].dataset),
                        100. * batch_idx / len(data_loaders[phase]),
                        # training losses: total loss, regression loss, classification loss: positive & negative samples
                        loss,
                        loss_positive_pts,
                        loss_positive_cls,
                        loss_negative_cls,
                        phase,
                        # training accuracy: positive samples in a batch
                        pred_class_pos_correct,
                        len_true_positive,
                        100. * pred_class_pos_correct / (len_true_positive+0.001),
                        # training accuracy: negative samples in a batch
                        pred_class_neg_correct,
                        len_true_negative,
                        100. * pred_class_neg_correct / (len_true_negative+0.001),
                        # training accuracy: total samples in a batch
                        pred_class_pos_correct + pred_class_neg_correct,
                        img.shape[0],
                        100. * (pred_class_pos_correct + pred_class_neg_correct) / img.shape[0],
                        phase)
                        )

                # save model
                if phase == 'train' and epoch_id % config.SAVE_MODEL_INTERVAL == 0:
                    state = {'epoch': epoch_id,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                            }
                    saved_model_name = os.path.join(config.MODEL_SAVE_PATH, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
                    torch.save(state, saved_model_name)
    return loss


def main_test():

    ###################################################################################
    torch.manual_seed(1)

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_TRAIN, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_VAL)
    data_loaders = {'train': train_loader, 'val': valid_loader}

    print('===> Building Model')
    # For single GPU
    model = Net()

    ####################################################################
    criterion_pts = nn.MSELoss()
    weights = [1, 3]
    class_weights = torch.FloatTensor(weights)
    criterion_cls = nn.CrossEntropyLoss()

    ####################################################################
    if config.PHASE == 'Train' or config.PHASE == 'train':
        print('===> Start Training')
        _ = train(data_loaders, model, (criterion_pts, criterion_cls))
        print('====================================================')
    elif config.PHASE == 'Test' or config.PHASE == 'test':
        print('===> Test')
    # how to do test?
    elif config.PHASE == 'Finetune' or config.PHASE == 'finetune':
        print('===> Finetune')
        # how to do finetune?
    elif config.PHASE == 'Predict' or config.PHASE == 'predict':
        print('===> Predict')
        # how to do predict?
        test_img_name = 'test01.jpg'
        predict(model, test_img_name)


if __name__ == '__main__':
    main_test()

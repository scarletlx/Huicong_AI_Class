# It's empty. Surprise!
# Please complete this by yourself.
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #         self.conv1 = nn.Sequential(nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
        #                                    nn.MaxPool2d(stride=2,kernel_size=2),
        # #                                    nn.BatchNorm2d(3),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1),
        # #                                    nn.BatchNorm2d(6),
        #                                    nn.MaxPool2d(stride=2,kernel_size=2),
        #                                    nn.ReLU(inplace=True))

        #         self.dense = nn.Sequential(nn.Linear(125*125*6,1024),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Dropout2d(p=0.5))

        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(150, 3)
        self.softmax1 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(150, 2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        #         x = self.conv1(x)
        #         x = x.view(-1, 6 * 125 * 125)
        #         x = self.dense(x)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)
        x_species = self.softmax1(x_species)

        x_class = self.fc3(x)
        x_class = self.softmax2(x_class)

        return x_species, x_class
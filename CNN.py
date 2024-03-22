import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channel: int, num_classes: int):
        """
        Convolutional Neural Network for classification task.\n
        Parameters
        ----------
            input_channel (int): number of channel in input. (RGB=3, grayscale=1)
            num_classes (int): number of classes in the dataset.
        """
        super(CNN, self).__init__()
        self.input_ch = input_channel
        self.num_classes = num_classes
        self.channels = [32, 32, 64, 64]

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_ch, out_channels=self.channels[0], kernel_size=(11), padding=(3), stride=(1))
        nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=(9), padding=(2), stride=(1))
        nn.init.xavier_normal_(self.conv2.weight)
        self.norm1 = nn.BatchNorm2d(self.channels[1])
        self.conv3 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=(5), padding=(2), stride=(1))
        nn.init.xavier_normal_(self.conv3.weight)
        self.conv4 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=(5), padding=(2), stride=(1))
        nn.init.xavier_normal_(self.conv4.weight)
        self.norm2 = nn.BatchNorm2d(self.channels[3])

        # Flatten layer (from ConvLayer to fully-connected)
        self.flat = nn.Flatten()

        # Fully connected
        #self.fc1 = nn.Linear(4096, 256)
        self.fc1 = nn.Linear(3136, 256)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 32)
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(32, self.num_classes)

    def forward(self, x):
        # CNN phase
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.norm1(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = self.norm2(x)

        x = self.flat(x)  # flat the data to get a vector for FC layers

        # FC phase
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)

        # x = torch.softmax(self.fc3(x),dim=1)
        return x

    # def __init__(self, input_channel: int, num_classes: int):
    #     """
    #     Convolutional Neural Network for classification task.\n
    #     Parameters
    #     ----------
    #         input_channel (int): number of channel in input. (RGB=3, grayscale=1)
    #         num_classes (int): number of classes in the dataset.
    #     """
    #     super(CNN, self).__init__()
    #     self.input_ch = input_channel
    #     self.num_classes = num_classes
    #     self.channels = [32, 32, 64, 64]

    #     # Convolutional layers
    #     self.conv1 = nn.Conv2d(in_channels=self.input_ch, out_channels=self.channels[0], kernel_size=(5), padding=(2), stride=(1))
    #     self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=(5), padding=(2), stride=(1))
    #     self.conv3 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=(3), padding=(1), stride=(1))
    #     self.conv4 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=(3), padding=(1), stride=(1))

    #     # Flatten layer (from ConvLayer to fully-connected)
    #     self.flat = nn.Flatten()

    #     # Fully connected
    #     self.fc1 = nn.Linear(4096, 256)
    #     self.drop = nn.Dropout(p=0.1)
    #     self.fc2 = nn.Linear(256, 32)
    #     self.fc3 = nn.Linear(32, self.num_classes)

    # def forward(self, x):
    #     # CNN phase
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv3(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv4(x), 2))

    #     x = self.flat(x)  # flat the data to get a vector for FC layers

    #     # FC phase
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.drop(x)
    #     x = self.fc3(x)

    #     # x = torch.softmax(self.fc3(x),dim=1)
    #     return x



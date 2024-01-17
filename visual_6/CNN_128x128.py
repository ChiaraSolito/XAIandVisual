import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_128x128(nn.Module):
    def __init__(self,input_channel: int,num_classes: int):
        '''
        Convolutional Neural Network for classification task.\n
        Parameters
        ----------
            input_channel (int): number of channel in input. (RGB=3, grayscale=1)
            num_classes (int): number of classes in the dataset.
        '''
        super(CNN_128x128,self).__init__()
        self.input_ch = input_channel
        self.num_classes = num_classes
        self.channels = [32,32,64,64]

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_ch,out_channels=self.channels[0],kernel_size=(9),stride=(1))
        self.conv2 = nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(9),stride=(1))
        self.conv3 = nn.Conv2d(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=(5),stride=(1))
        self.conv4 = nn.Conv2d(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=(5),stride=(1))
        self.drop1 = nn.Dropout1d(p=0.1)

        # Flatten layer (from ConvLayer to fully-connected)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(576,256)
        self.drop2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256,32)
        self.fc3 = nn.Linear(32,self.num_classes)
        

    def forward(self,x):
        # CNN phase
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))

        x = self.flat(x)    # flat the data to get a vector for FC layers

        # FC phase
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)

        # x = torch.softmax(self.fc3(x),dim=1)
        return x

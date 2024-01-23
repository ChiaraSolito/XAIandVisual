


# import CNN
from visual_6 import CNN_128x128

# Libraries
import os
import glob
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import CustomDataset, compute_metrics, plot_weights, visTensor

import os

if __name__ == "__main__":

    path_train = "./data/train/"
    path_test = "./data/test/"

    n_muffins_train = len(os.listdir(path_train + "muffin")) # 2174
    n_muffins_test =  len(os.listdir(path_test + "muffin")) # 544
    n_chihuahua_train = len(os.listdir(path_train + "chihuahua"))  # 2559
    n_chihuahua_test = len(os.listdir(path_test + "chihuahua"))  # 640

    # Define train and test labels
    # 0 muffins
    # 1 chihuahua
    train_labels = np.zeros(n_muffins_train + n_chihuahua_train)
    train_labels[n_muffins_train:] = 1
    test_labels = np.zeros(n_muffins_test + n_chihuahua_test)
    test_labels[n_muffins_test:] = 1
    train_labels = train_labels.astype('uint8')
    test_labels = test_labels.astype('uint8')

    # Load train set
    train_data = [cv2.imread(file) for file in glob.glob('./data/train/muffin/*.jpg')]
    train_data.extend(cv2.imread(file) for file in glob.glob('./data/train/chihuahua/*.jpg'))

    # Load test set
    test_data = [cv2.imread(file) for file in glob.glob('./data/test/muffin/*.jpg')]
    test_data.extend(cv2.imread(file) for file in glob.glob('./data/test/chihuahua/*.jpg'))


    # Random shuffle train and test set
    train_list = list(zip(train_data, train_labels))
    test_list = list(zip(test_data, test_labels))

    random.shuffle(train_list)
    random.shuffle(test_list)

    train_data, train_labels = zip(*train_list)
    test_data, test_labels = zip(*test_list)


    # Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    IMAGE_SIZE = (28,28)

    data_trasformation = Compose([
        ToPILImage(),
        Resize(IMAGE_SIZE),
        ToTensor()
    ])

    # Create Dataloader with batch size = 64
    train_dataset = CustomDataset(train_data, train_labels, data_trasformation)  # we use a custom dataset defined in utils.py file
    test_dataset = CustomDataset(test_data, test_labels, data_trasformation)  # we use a custom dataset defined in utils.py file

    batch_size = 3

    trainset = DataLoader(train_dataset, batch_size=batch_size,
                          drop_last=True)  # construct the trainset with subjects divided in mini-batch
    testset = DataLoader(test_dataset, batch_size=batch_size,
                         drop_last=True)  # construct the testset with subjects divided in mini-batch


    for d in trainset:
        data = d[0]
        label = d[0]

        sample = torch.tensor(data).permute(1,2,0).numpy()
        plt.imshow(sample)
        plt.show()
        break

        # 'paperino'







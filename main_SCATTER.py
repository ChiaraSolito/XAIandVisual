# Libraries
import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from training_main import training_main, test
from utils import CustomDataset, compute_metrics, plot_kernels, data_loading, normalization

if __name__ == "__main__":

    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")
    data = train_data + test_data

    # NORMALIZATION
    data_transform = normalization(data)

    # TRAINING
    model = training_main(data_transform, train_data, train_labels, 'ScatNet')

    # TESTING
    acc = test(data_transform, test_data, test_labels, model, 'ScatNet', device='cpu')
    print("Accuracy", acc)

    # EXTRACTION OF FILTERS

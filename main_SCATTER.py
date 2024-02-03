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
from training_main import training_main
from utils import CustomDataset, compute_metrics, plot_kernels, data_loading

if __name__ == "__main__":
    device = 'cpu'
    
    #### LOAD DATA ####
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/","./data/test/")
    data = train_data + test_data

    model = training_main(data,train_data, train_labels,'ScatNet')
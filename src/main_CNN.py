# Libraries
import torch
from training_main import training_main, test
import matplotlib as plt
from utils import data_loading, normalization, normalization_aug, plot_results
import cv2

import numpy as np
import pandas as pd


if __name__ == "__main__":\

    model_name = 'CNN'
    num_fold = 10
    num_epochs = 10

    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")
    data = train_data + test_data

    # NORMALIZATION
    data_transform_train = normalization_aug(data)
    data_transform_val = normalization(data)

    # TRAINING
    model = training_main(data_transform_train, data_transform_val, train_data, train_labels, model_name, num_epochs, num_fold)
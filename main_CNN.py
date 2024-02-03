# import CNN
from CNN_128x128 import CNN_128x128

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
from training_main import training_main
from utils import CustomDataset, compute_metrics, plot_weights, visTensor
from datetime import datetime


def data_loading(path_train:str, 
                 path_test:str) -> tuple[list[np.ndarray], #train_images
                        list[str], #train_labels
                        list[np.ndarray], #test_images
                        list[str]]: #test_labels

    n_muffins_train = len(os.listdir(path_train + "muffin")) # 2174
    n_muffins_test =  len(os.listdir(path_test + "muffin")) # 544
    n_chihuahua_train = len(os.listdir(path_train + "chihuahua"))  # 2559
    n_chihuahua_test = len(os.listdir(path_test + "chihuahua"))  # 640

    # Define train and test labels - 0 muffins, 1 chihuahua
    train_labels = np.zeros(n_muffins_train + n_chihuahua_train)
    train_labels[n_muffins_train:] = 1
    test_labels = np.zeros(n_muffins_test + n_chihuahua_test)
    test_labels[n_muffins_test:] = 1
    train_labels = train_labels.astype('uint8')
    test_labels = test_labels.astype('uint8')

    # Load train set
    train_data = [cv2.imread(file) for file in glob.glob(path_train + 'muffin/*.jpg')]
    train_data.extend(cv2.imread(file) for file in glob.glob(path_train + 'chihuahua/*.jpg'))

    # Load test set
    test_data = [cv2.imread(file) for file in glob.glob(path_test + '/muffin/*.jpg')]
    test_data.extend(cv2.imread(file) for file in glob.glob(path_test +'/chihuahua/*.jpg'))

    return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":

    #### LOAD DATA ####
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/","./data/test/")
    data = train_data + test_data

    model = training_main(data,train_data, train_labels,CNN_128x128)

    ### 3 - TESTING

    # to use when model is saved
    model_test = CNN_128x128(input_channel=3, num_classes=n_classes).to(device)  # Initialize a new model
    model_test.load_state_dict(torch.load(f'{models_trained_path}_{date_time}.pt'))  # Load the model
    # to use when model is saved

    pred_label_test = torch.empty((0, n_classes)).to(device)
    true_label_test = torch.empty((0)).to(device)

    with torch.no_grad():
        for data in testset:
            X_te, y_te = data
            X_te = X_te.view(batch_size, 3, 128, 128).float().to(device)
            y_te = y_te.to(device)
            output_test = model_test(X_te)
            pred_label_test = torch.cat((pred_label_test, output_test), dim=0)
            true_label_test = torch.cat((true_label_test, y_te), dim=0)

    compute_metrics(y_true=true_label_test, y_pred=pred_label_test,
                    lab_classes=lab_classes)  # function to compute the metrics (accuracy and confusion matrix)
# Libraries
import torch
from training_main import training_main, test
import matplotlib as plt
from utils import data_loading, normalization, normalization2, plot_results
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
    data_transform = normalization(data)
    data_transform2 = normalization2(data)

    # TRAINING
    model = training_main(data_transform, data_transform, train_data, train_labels, model_name, num_epochs, num_fold)

    # PLOTTING
    val_accuracies = np.zeros([num_fold, num_epochs])
    train_losses = np.zeros([num_fold, num_epochs])
    val_losses = np.zeros([num_fold, num_epochs])
    f1_scores = np.zeros([num_fold, num_epochs])

    for i in range(num_fold):
        results_string = f"./csv/{model_name}/results_df_" + str(i) + ".csv"
        val_accuracies[i] = (pd.read_csv(results_string)["val_acc"]).to_list()
        val_losses[i] = (pd.read_csv(results_string)["val_loss"]).to_list()
        train_losses[i] = (pd.read_csv(results_string)["train_loss"]).to_list()
        f1_scores[i] = (pd.read_csv(results_string)["val_f1"]).to_list()

    plot_results(val_accuracies, train_losses, val_losses, f1_scores, model_name)

    # TESTING
    acc = test(data_transform, test_data, test_labels, model, model_name, device='cpu')
    print("Accuracy", acc)
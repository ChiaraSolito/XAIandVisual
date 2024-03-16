# Libraries
import torch
from src.training_main import training_main, test
import matplotlib as plt
from src.utils import data_loading, normalization, filter_extraction, plot_results
import cv2

import numpy as np
import pandas as pd


if __name__ == "__main__":\

    MODEL_NAME = 'CNN'
    NUM_FOLD = 10
    num_epochs = 20

    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")
    data = train_data + test_data

    # NORMALIZATION
    data_transform = normalization(data)

    # TRAINING
    model = training_main(data_transform, train_data, train_labels, MODEL_NAME)

    # PLOTTING
    val_accuracies = np.zeros([NUM_FOLD, num_epochs])
    train_losses = np.zeros([NUM_FOLD, num_epochs])
    val_losses = np.zeros([NUM_FOLD, num_epochs])
    f1_scores = np.zeros([NUM_FOLD, num_epochs])

    for i in range(NUM_FOLD):
        results_string = f"./csv/{MODEL_NAME}/results_df_" + str(i) + ".csv"
        val_accuracies[i] = (pd.read_csv(results_string)["val_acc"]).to_list()
        val_losses[i] = (pd.read_csv(results_string)["val_loss"]).to_list()
        train_losses[i] = (pd.read_csv(results_string)["train_loss"]).to_list()
        f1_scores[i] = (pd.read_csv(results_string)["val_f1"]).to_list()

    plot_results(val_accuracies, train_losses, val_losses, f1_scores, MODEL_NAME)

    # TESTING
    acc = test(data_transform, test_data, test_labels, model, MODEL_NAME, device='cpu')
    print("Accuracy", acc)

    # EXTRACTION OF FILTERS
    filter_extraction(model, data_transform, MODEL_NAME)
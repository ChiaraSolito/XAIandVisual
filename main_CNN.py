# Libraries
import torch
from training_main import training_main, test
import matplotlib as plt
from utils import data_loading, normalization, filter_extraction
import cv2

if __name__ == "__main__":\

    MODEL_NAME = 'CNN'
    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")
    data = train_data + test_data

    # NORMALIZATION
    data_transform = normalization(data)

    # TRAINING
    model = training_main(data_transform, train_data, train_labels, MODEL_NAME)

    # TESTING
    acc = test(data_transform, test_data, test_labels, model, MODEL_NAME, device='cpu')
    print("Accuracy", acc)

    # EXTRACTION OF FILTERS
    filter_extraction(model, data_transform, MODEL_NAME)
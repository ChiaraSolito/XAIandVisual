




from CNN import CNN

import torch
import numpy as np
import pandas as pd

import cv2

from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ToPILImage

import shap

# SHAP TUTORIAL
# https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html
# SHAP TUTORIAL
from utils import data_loading, normalization


if __name__ == "__main__":

    MODEL_NAME = 'CNN'
    NUM_FOLD = 10

    # LOAD BEST MODEL
    max_val_accuracies = np.zeros([NUM_FOLD, 1])

    for i in range(NUM_FOLD):
        results_string = f"./csv/{MODEL_NAME}/results_df_" + str(i) + ".csv"
        max_val_accuracies[i] = np.max(pd.read_csv(results_string)["val_acc"])

    index = np.argmax(max_val_accuracies)

    model_string = f"./models_trained/{MODEL_NAME}/checkpoint_" + str(index) + ".pth"
    checkpoint = torch.load(model_string, map_location=torch.device("cpu"))
    model_CNN = CNN(input_channel=3, num_classes=2)
    model_CNN.load_state_dict(checkpoint["model_state_dict"])


    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")

    data_transform = normalization(test_data)

    images = torch.empty(len(test_data),3,128,128)

    for i, img in enumerate(test_data):
        images[i] = data_transform(img)


    background =  images[:100]
    test_images = images[100:106]

    e = shap.DeepExplainer(model_CNN, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy)




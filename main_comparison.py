
# Libraries
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from training_main import training_main, test
from utils import data_loading, normalization
import random


def take_subset(images, labels, subset_percentage):
    assert len(images) == len(labels), "Length of images and labels must be the same."

    subset_images = []
    subset_labels = []
    classes = list(set(labels))
    for i in classes:
        indices_class = np.where(labels == i)[0]
        num_indices = int(len(indices_class) * subset_percentage)
        subset_indices = random.sample(list(indices_class), num_indices)

        subset_images.append([images[i] for i in subset_indices])
        subset_labels.append([labels[i] for i in subset_indices])

    subset_images = list(chain(*subset_images))
    subset_labels = list(chain(*subset_labels))

    return subset_images, subset_labels


if __name__ == "__main__":
    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")

    subset_ratios = [1.0, 0.75, 0.5, 0.25]
    cnn_accuracies = []
    scatnet_accuracies = []

    for ratio in subset_ratios:

        subset_train_data, subset_train_labels = take_subset(train_data, train_labels, ratio)
        subset_test_data, subset_test_labels = take_subset(test_data, test_labels, ratio)
        print("Number of train images: " + str(len(subset_train_data)) + ", Number of test images: "
              + str(len(subset_test_data)))
        subset_data = subset_train_data + subset_test_data

        # NORMALIZATION
        data_transform = normalization(subset_data, ratio)

        # CNN
        CNNmodel = training_main(data_transform, subset_train_data, subset_train_labels, 'CNN')

        CNNacc = test(data_transform, test_data, test_labels, CNNmodel, 'CNN', device='cpu')
        print("Accuracy", CNNacc)

        # ScatNet
        # ScatNetmodel = training_main(data_transform, subset_train_data, subset_train_labels, 'ScatNet')
        # ScatNetacc = test(data_transform, test_data, test_labels, ScatNetmodel, 'ScatNet', device='cpu')
        # print("Accuracy", ScatNetacc)


        # Store accuracies
        cnn_accuracies.append(CNNacc)
        # scatnet_accuracies.append(ScatNetacc)

    # Plot the results
    # plt.plot(subset_ratios, cnn_accuracies, label='CNN')
    # plt.plot(subset_ratios, scatnet_accuracies, label='ScatNet')
    # plt.xlabel('Subset Size Ratio')
    # plt.ylabel('Accuracy')
    # plt.title('Comparison of CNN and ScatNet Performance with Controlled Subset Size')
    # plt.legend()
    # plt.show()

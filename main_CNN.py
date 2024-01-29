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
from utils import CustomDataset, compute_metrics, plot_weights, visTensor
from datetime import datetime


if __name__ == "__main__":

    ### 1 - DATA LOADING

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
    train_data = [cv2.imread(file) for file in glob.glob(path_train + '/muffin/*.jpg')]
    train_data.extend(cv2.imread(file) for file in glob.glob(path_train + 'chihuahua/*.jpg'))

    # Load test set
    test_data = [cv2.imread(file) for file in glob.glob(path_test + '/muffin/*.jpg')]
    test_data.extend(cv2.imread(file) for file in glob.glob(path_test +'/chihuahua/*.jpg'))


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

    IMAGE_SIZE = (128,128)

    data_trasformation = Compose([
        ToPILImage(),
        Resize(IMAGE_SIZE),
        ToTensor()
    ])

    # Create Dataloader with batch size = 64
    train_dataset = CustomDataset(train_data, train_labels, data_trasformation)  # we use a custom dataset defined in utils.py file
    test_dataset = CustomDataset(test_data, test_labels, data_trasformation)  # we use a custom dataset defined in utils.py file

    batch_size = 2

    trainset = DataLoader(train_dataset, batch_size=batch_size,
                          drop_last=True)  # construct the trainset with subjects divided in mini-batch
    testset = DataLoader(test_dataset, batch_size=batch_size,
                         drop_last=True)  # construct the testset with subjects divided in mini-batch

    # 2 - TRAINING SETTINGS

    best_acc = 0.0
    num_epochs = 5  # number of epochs
    lr = 0.001  # learning rate
    n_classes = len(np.unique(train_labels))  # number of classes in the dataset
    lab_classes = ['Muffin', 'Chihuahua']

    # Variables to store the resuts
    losses = []
    acc_train = []
    pred_label_train = torch.empty((0))
    true_label_train = torch.empty((0))

    # Model
    model = CNN_128x128(input_channel=3, num_classes=n_classes).to(device)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)  # to choose
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()  # to choose

    for epoch in range(num_epochs):
        model.train()
        for batch in trainset:

            inputs = batch[0].to(device)
            labels = batch[1].to(torch.long).to(device)

            output = model(inputs)

            loss = criterion(output,labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            output_cpu = output.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            pred_label_train = np.concatenate((pred_label_train, output_cpu.argmax(1)), axis=0)
            true_label_train = np.concatenate((true_label_train, labels_cpu), axis=0)

        losses.append(loss.cpu().detach().numpy())
        acc_t = accuracy_score(true_label_train, pred_label_train)
        acc_train.append(acc_t)
        print(f'epoch : {epoch + 1}/{num_epochs}, loss = {loss} - acc = {acc_t}')
        # print("  epoch : {}/{}, loss = {:.4f} - acc = {:.4f}".format(epoch + 1, num_epochs, loss, acc_t))
        if acc_t > best_acc:  # save the best model (the highest accuracy in validation)
            # save the best model in .pt format
            # to do
            best_acc = acc_t

        # Reinitialize the variables to compute accuracy
        pred_label_train = torch.empty((0))
        true_label_train = torch.empty((0))

    date_time = datetime.now().strftime('%d-%m-%Y__%H-%M-%S-%f')[:-3]
    models_trained_path = './models_trained/model_CNN_'
    image_path = './models_trained/images/'

    torch.save(model.state_dict(), f'{models_trained_path}_{date_time}.pt')

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(list(range(num_epochs)), losses)
    plt.title("Learning curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f'{image_path}learning_curve_CNN_{date_time}.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(list(range(num_epochs)), acc_train)
    plt.title("Accuracy curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f'{image_path}accuracy_CNN_{date_time}.png')
    plt.show()

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
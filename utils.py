import pandas as pd
import numpy as np
import os
import cv2
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import utils
from torch.utils.data import Dataset
from datetime import datetime
from colorsys import hls_to_rgb
from scipy.fft import fft2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage, Normalize
from sklearn.metrics import ConfusionMatrixDisplay

# Style for chart
sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)


def data_loading(path_train: str,
                 path_test: str) -> tuple[list[np.ndarray],  # train_images
                                          np.ndarray,  # train_labels
                                          list[np.ndarray],  # test_images
                                          np.ndarray]:  # test_labels

    n_meningiomas_train = len(os.listdir(path_train + "meningioma"))  
    n_meningiomas_test = len(os.listdir(path_test + "meningioma"))  
    n_notumor_train = len(os.listdir(path_train + "notumor"))
    n_notumor_test = len(os.listdir(path_test + "notumor")) 

    # Define train and test labels - 0 meningiomas, 1 notumor
    train_labels = np.zeros(n_meningiomas_train + n_notumor_train)
    train_labels[n_meningiomas_train:] = 1
    test_labels = np.zeros(n_meningiomas_test + n_notumor_test)
    test_labels[n_meningiomas_test:] = 1
    train_labels = train_labels.astype('uint8')
    test_labels = test_labels.astype('uint8')

    # Load train set
    train_data = [cv2.imread(file) for file in glob.glob(path_train + 'meningioma/*.jpg')]
    train_data.extend(cv2.imread(file) for file in glob.glob(path_train + 'notumor/*.jpg'))

    # Load test set
    test_data = [cv2.imread(file) for file in glob.glob(path_test + '/meningioma/*.jpg')]
    test_data.extend(cv2.imread(file) for file in glob.glob(path_test + '/notumor/*.jpg'))

    return train_data, train_labels, test_data, test_labels


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)  # .squeeze(0)
        label = self.labels[idx]
        sample = [data, label]
        return sample


def compute_metrics(y_true, y_pred, classes, model_name, ratio: None):
    """
    Compute the metrics: accuracy, confusion matrix, F1 score.\n
    Args:
        y_true: true labels
        y_pred: predicted probabilities for each class
        classes: list of the classes
        model_name: name of the model
        ratio: percentage of data
    """

    # Accuracy
    acc = accuracy_score(y_true, y_pred.numpy())

    # F1 score
    f1score = f1_score(y_true, y_pred.numpy())

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred.numpy(), labels=list(range(0, len(classes))))
    fig_metrics = plt.figure(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
    disp.plot()
    plt.title('confusion matrix: test set')
    plt.xlabel('predicted')
    plt.ylabel('true')
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if ratio is not None:
        plot_string = f"./models_trained/images/{model_name}_ConfusionMatrix_{ratio}_{dt_string}.png"
    else:
        plot_string = f"./models_trained/images/{model_name}_ConfusionMatrix_{dt_string}.png"
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.grid(False)
    plt.savefig(fname=plot_string)
    plt.close(fig_metrics)

    return acc, f1score


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 / (1.0 + abs(z[idx]) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


# Function to visualize the kernels for the two convolutional layers
def visTensor(tensor, model_name, ch=0, allkernels=False, nrow=4, padding=1):
    n, c, _, _ = tensor.shape

    if c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plot_string = f"./models_trained/images/{model_name}_visTensor_{dt_string}.png"

    if allkernels:
        num_subplots = 3
    else:
        num_subplots = 1

    rows = min((tensor.shape[0] // nrow + 1, 64))
    if num_subplots == 1:
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        fig, ax = plt.subplots(figsize=(nrow, rows))
        ax.imshow(grid.cpu().numpy().transpose((1, 2, 0)), cmap='viridis')
    else:
        fig, axes = plt.subplots(1, num_subplots, figsize=(10, 5))
        for i in range(num_subplots):
            channel_tensor = tensor[:, i, :, :].unsqueeze(dim=1)
            grid = utils.make_grid(channel_tensor, nrow=nrow, normalize=True, padding=padding)

            ax = axes[i]
            ax.imshow(grid.cpu().numpy().transpose((1, 2, 0)), cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Channel {i}", fontsize=14)
            ax.grid(False)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(plot_string)
    plt.close(fig)


def plot_filters_single_channel_big(t,model_name):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
    fig = imgplot.get_figure()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fig_string = f"./models_trained/images/{model_name}_SingleChannelBig_{dt_string}.png"
    fig.savefig(fig_string)
    plt.close()


def plot_filters_single_channel(t,model_name):
    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols
    # convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fig_string = f"./models_trained/images/{model_name}_SingleChannel_{dt_string}.png"
    fig.savefig(fig_string)
    plt.close()


def plot_filters_multi_channel(t,model_name):
    # get the number of kernels
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fig_string = f"./models_trained/images/{model_name}_MultiChannel_{dt_string}.png"
    fig.savefig(fig_string)
    plt.close()



def plot_weights(model_layer, single_channel=True, collated=False, model_name='CNN'):
    # extracting the model features at the particular layer number
    layer = model_layer

    # checking whether the layer is convolution layer or not
    if isinstance(layer, torch.nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = layer.weight.data.cpu()

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor, model_name)
            else:
                plot_filters_single_channel(weight_tensor, model_name)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor, model_name)
            else:
                print("Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")


# visualizzare tutti i filtri che sono presenti negli N banchi di filtri che avete creato
def plot_kernels(J, L, scattering, model_name):
    fig_wavelets, axs = plt.subplots(J, L, sharex=True, sharey=True, figsize=(12, 5))
    i = 0
    for filter in scattering.psi:
        f = filter["levels"][0]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        axs[i // L, i % L].imshow(colorize(filter_c))
        axs[i // L, i % L].axis('off')
        axs[i // L, i % L].set_title("$j = {}$ \n $\\theta={}$".format(i // L, i % L), fontsize=12)
        i = i + 1

    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fig_string = f"./models_trained/images/{model_name}_Kernels_Wavelets_{dt_string}.png"
    fig_wavelets.savefig(fig_string)
    plt.close(fig_wavelets)

    fig_filter, axs = plt.subplots(1, J, figsize=(12, 5))
    fig_filter.suptitle(f'Low-pass filters (scaling functions) with J={J}', fontsize=16)
    for j in range(J):
        f = scattering.phi["levels"][j]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        filter_c = np.abs(filter_c)
        axs[j].imshow(filter_c, cmap='Greys')
        axs[j].axis('on')
        axs[j].grid(False)
        axs[j].set_title("$J = {}$".format(j), fontsize=12)
    
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fig_string = f"./models_trained/images/{model_name}_LowPassFilters_{dt_string}.png"
    fig_filter.savefig(fig_string)
    plt.close(fig_filter)


def get_mean(data: list[np.ndarray], ratio: float) -> str:
    r_mean_arr = []
    g_mean_arr = []
    b_mean_arr = []

    for i in range(0, len(data)):
        img_np = data[i]
        r_mean, g_mean, b_mean = np.mean(img_np, axis=(0, 1))
        r_mean_arr.append(r_mean)
        g_mean_arr.append(g_mean)
        b_mean_arr.append(b_mean)

    R_MEAN = np.mean(r_mean_arr) / 255
    G_MEAN = np.mean(g_mean_arr) / 255
    B_MEAN = np.mean(b_mean_arr) / 255

    RGB_df = pd.DataFrame(columns=["R_MEAN", "G_MEAN", "B_MEAN"])
    RGB_df["R_MEAN"] = [R_MEAN]
    RGB_df["G_MEAN"] = [G_MEAN]
    RGB_df["B_MEAN"] = [B_MEAN]

    df_name = f"./csv/norm/RGB_mean_df_{ratio*100}.csv"
    RGB_df.to_csv(df_name)
    return df_name


def get_std(data: list[np.ndarray], ratio: float) -> str:
    r_std_arr = []
    g_std_arr = []
    b_std_arr = []

    for i in range(0, len(data)):
        img_np = data[i]
        r_std, g_std, b_std = np.std(img_np, axis=(0, 1))
        r_std_arr.append(r_std)
        g_std_arr.append(g_std)
        b_std_arr.append(b_std)

    R_STD = np.mean(r_std_arr) / 255
    G_STD = np.mean(g_std_arr) / 255
    B_STD = np.mean(b_std_arr) / 255

    RGB_std_df = pd.DataFrame(columns=["R_STD", "G_STD", "B_STD"])
    RGB_std_df["R_STD"] = [R_STD]
    RGB_std_df["G_STD"] = [G_STD]
    RGB_std_df["B_STD"] = [B_STD]

    df_name = f"./csv/norm/RGB_std_df_{ratio*100}.csv"
    RGB_std_df.to_csv(df_name)
    return df_name


def normalization(data: list[np.ndarray], ratio: float = 1.0):
    RGB_mean_path = get_mean(data, ratio)
    RGB_mean_df = pd.read_csv(RGB_mean_path)
    print("Red ch mean = ", RGB_mean_df.iloc[0].R_MEAN.item(), "\nGreen ch mean = ", RGB_mean_df.iloc[0].G_MEAN.item(),
          "\nBlue ch mean = ", RGB_mean_df.iloc[0].B_MEAN.item())

    RGB_std_path = get_std(data, ratio)
    RGB_std_df = pd.read_csv(RGB_std_path)
    print("Red ch std = ", RGB_std_df.iloc[0].R_STD.item(), "\nGreen ch std = ", RGB_std_df.iloc[0].G_STD.item(),
          "\nBlue ch std = ", RGB_std_df.iloc[0].B_STD.item())

    R_MEAN = RGB_mean_df.iloc[0].R_MEAN.item()
    G_MEAN = RGB_mean_df.iloc[0].G_MEAN.item()
    B_MEAN = RGB_mean_df.iloc[0].B_MEAN.item()
    R_STD = RGB_std_df.iloc[0].R_STD.item()
    G_STD = RGB_std_df.iloc[0].G_STD.item()
    B_STD = RGB_std_df.iloc[0].B_STD.item()

    data_transform = Compose([
        ToPILImage(),
        Resize(size=(128, 128)),
        ToTensor(),
        Normalize(mean=[R_MEAN, G_MEAN, B_MEAN], std=[R_STD, G_STD, B_STD])
    ])

    return data_transform


def plot_results(train_accuracies, val_accuracies, train_losses, val_losses, f1_scores, model_name, best_fold, show=False):

    # PLOT - VALIDATION ACCURACY
    train_acc_matrix = np.array(train_accuracies)
    train_acc_std = np.std(train_acc_matrix, axis=0)
    train_acc_min = np.min(train_acc_matrix, axis=0)
    train_mean_acc = np.mean(train_acc_matrix, axis=0)
    train_best_acc = train_acc_matrix[best_fold, :]

    val_acc_matrix = np.array(val_accuracies)
    val_acc_std = np.std(val_acc_matrix, axis=0)
    val_mean_acc = np.mean(val_acc_matrix, axis=0)
    val_best_acc = val_acc_matrix[best_fold, :]

    num_epochs = range(1, len(train_mean_acc)+1)
    fig, ax = plt.subplots(figsize=(20, 10))
    #for idx, fold_acc in enumerate(acc_matrix):
        #ax.plot(num_epochs, fold_acc, '--', alpha=0.5, label=f"Fold n.{idx}")
    ax.fill_between(num_epochs, train_mean_acc - train_acc_std, train_mean_acc + train_acc_std, color="red", alpha=0.2)
    ax.plot(num_epochs, train_mean_acc, '-', color="red", label='Training Mean Accuracy')
    ax.plot(num_epochs, train_best_acc, '--', color="red", label='Best fold Training Accuracy')

    ax.fill_between(num_epochs, val_mean_acc - val_acc_std, val_mean_acc + val_acc_std, color="blue", alpha=0.2)
    ax.plot(num_epochs, val_mean_acc, '-', color="blue", label='Validation Mean Accuracy')
    ax.plot(num_epochs, val_best_acc, '--', color="blue", label='Best fold Validation Accuracy')
    
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(1, len(num_epochs)+1, step=1), fontsize=20)
    plt.xlim(1, len(num_epochs))
    plt.ylim([min(train_acc_min) - 0.08, 1.0])
    plt.grid(True)
    plt.legend(fontsize=20,loc='lower right', frameon=True, facecolor='lightgray')
    plt.title("Accuracy Between Folds -Training and Validation", fontsize=30)
    if show:
        plt.show()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    acc_string = f"./models_trained/images/{model_name}_AccuracyFolds_{dt_string}.png"
    fig.savefig(acc_string)
    plt.close()

    # PLOT - LEARNING CURVE 
    train_matrix = np.array(train_losses)
    train_max = np.max(train_matrix, axis=0)
    train_std = np.std(train_matrix, axis=0)
    mean_loss_train = np.mean(train_matrix, axis=0)
    best_loss_train = train_losses[best_fold, :]

    val_matrix = np.array(val_losses)
    val_std = np.std(val_matrix, axis=0)
    mean_loss_val = np.mean(val_matrix, axis=0)
    best_loss_val = val_losses[best_fold, :]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(num_epochs, mean_loss_train, '-', color="red", label='Mean Training Curve')
    ax.plot(num_epochs, best_loss_train, '--', color="red", label='Best fold Training Curve')
    ax.fill_between(num_epochs, mean_loss_train - train_std, mean_loss_train + train_std, color="red", alpha=0.2)

    ax.plot(num_epochs, mean_loss_val, '-', color="blue", label='Mean Validation Curve')
    ax.plot(num_epochs, best_loss_val, '--', color="blue", label='Best fold Validation Curve')
    ax.fill_between(num_epochs, mean_loss_val - val_std, mean_loss_val + val_std, color="blue", alpha=0.2)

    plt.yticks(fontsize=20)
    plt.xticks(np.arange(1, len(num_epochs)+1, step=1),fontsize=20)
    plt.xlim(1, len(num_epochs))
    plt.ylim([0.0, max(train_max) + 0.15])
    plt.grid(True)
    plt.legend(fontsize=20,loc='upper left', frameon=True, facecolor='lightgray')
    plt.title("Learning Curves - Training and Validation", fontsize=30)
    if show:
        plt.show()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    lc_string = f"./models_trained/images/{model_name}_LearningCurve_{dt_string}.png"
    fig.savefig(lc_string)
    plt.close()

    # PLOT - F1
    f1_matrix = np.array(f1_scores)
    f1_min = np.min(f1_matrix, axis=0)
    f1_std = np.std(f1_matrix, axis=0)
    mean_f1 = np.mean(f1_matrix, axis=0)
    best_f1 = f1_matrix[best_fold, :]

    fig, ax = plt.subplots(figsize=(20, 10))
    for idx, fold_f1 in enumerate(f1_matrix):
        ax.plot(num_epochs, fold_f1, '--', alpha=0.5, label=f"Fold n.{idx}")
    ax.fill_between(num_epochs, mean_f1 - f1_std, mean_f1 + f1_std, color="grey", alpha=0.2)
    ax.plot(num_epochs, mean_f1, '-', color="red", label='Mean F1 Score between Folds')
    ax.plot(num_epochs, best_f1, '--', color="blue", label='Best fold F1 Score')
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(1, len(num_epochs)+1, step=1),fontsize=20)
    plt.xlim(1, len(num_epochs))
    plt.ylim([min(f1_min) - 0.1, 1.0])
    plt.grid(True)
    plt.legend(ncol=3,fontsize=20,loc='lower right', frameon=True, facecolor='lightgray')
    plt.title("Validation F1 Score Between Folds", fontsize=30)
    if show:
        plt.show()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    f1_string = f"./models_trained/images/{model_name}_F1Folds_{dt_string}.png"
    fig.savefig(f1_string)
    plt.close()

def filter_extraction(model, model_name, image, single_channel):

    model_weights =[]
    conv_layers = []
    model_children = list(model.children())

    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == torch.nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
    print(f"Total convolution layers: {counter}")
    
    outputs = []
    names = []
    for layer in conv_layers[0:]:   
        image = layer(image)
        outputs.append(image) 
        names.append(str(layer))   

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(20, 40))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i],cmap='Greys')
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    f1_string = f"./models_trained/images/{model_name}_FeatureMap_{dt_string}.png"
    fig.savefig(f1_string)
    plt.close(fig)

    visTensor(model_weights[0], model_name, ch=0, allkernels=True)

    plot_weights(model_children[0], single_channel = single_channel, collated = False, model_name = model_name)
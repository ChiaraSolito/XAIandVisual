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
from pandas import DataFrame
from colorsys import hls_to_rgb
from scipy.fft import fft2
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

##### Style for chart
sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)

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
            data = self.transform(data) #.squeeze(0)
        label = self.labels[idx]
        sample = [data,label]
        return sample


def compute_metrics(y_true,y_pred,lab_classes):
    '''
    Compute the metrics: accuracy, confusion matrix, precision, recall, F1 score and auROC.\n
    Args:
        y_true: true labels
        y_pred: predicted probabilities for each class
        lab_classes: list of the groups in the study
    '''
    y_true, y_pred_prob = y_true.cpu(), y_pred.cpu()
    _, y_pred_lab = y_pred_prob.max(1)
    
    # Accuracy
    acc = accuracy_score(y_true,y_pred_lab)
    print('Accuracy: {:.3f}\n'.format(acc))

    # Confusion matrix
    conf_mat = confusion_matrix(y_true,y_pred_lab,labels=list(range(0,len(lab_classes))))
    conf_mat_df = pd.DataFrame(conf_mat,columns=lab_classes,index=lab_classes)
    plt.figure(figsize=(7,5))
    sns.heatmap(conf_mat_df,annot=True)
    plt.title('confusion matrix: test set')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.show()

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
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
    

def plot_filters_single_channel_big(t):
    
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)


def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
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
    plt.show()


def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()


def plot_weights(model_layer, single_channel = True, collated = False):
  
  #extracting the model features at the particular layer number
  layer = model_layer
  
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, torch.nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = layer.weight.data.cpu()
    
    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)
        
    else:
      if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print("Can only plot weights with three channels with single channel = False")
        
  else:
    print("Can only visualize layers which are convolutional")

def plot_kernels(J,L,scattering):      
    fig, axs = plt.subplots(J, L, sharex=True, sharey=True, )
    fig.set_figheight(5)
    fig.set_figwidth(12)
    i = 0
    for filter in scattering.psi:
        f = filter["levels"][0]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        axs[i // L, i % L].imshow(colorize(filter_c))
        axs[i // L, i % L].axis('off')
        axs[i // L, i % L].set_title("$j = {}$ \n $\\theta={}$".format(i // L, i % L), fontsize=12)
        i = i+1

    #plt.title('Wavelets')
    plt.show()

    f = scattering.phi["levels"][0]
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    filter_c = np.abs(filter_c)

    plt.figure(figsize=(5,5))
    plt.imshow(filter_c, cmap='Greys')
    plt.grid(False)
    plt.title('Low-pass filter (scaling function)')
    plt.imshow(np.log(filter_c), cmap='Greys'); plt.grid(False); plt.title('Low-pass filter (scaling function)')
    #plt.style.use(['no-latex'])
    plt.show()

def get_mean(data:list[np.ndarray],model_name:str) -> str:

    r_mean_arr = []
    g_mean_arr = []
    b_mean_arr = []

    for i in range(0,len(data)):
        img_np = data[i]
        r_mean,g_mean,b_mean = np.mean(img_np,axis=(0,1))
        r_mean_arr.append(r_mean)
        g_mean_arr.append(g_mean)
        b_mean_arr.append(b_mean)

    R_MEAN = np.mean(r_mean_arr) / 255
    G_MEAN = np.mean(g_mean_arr) / 255
    B_MEAN = np.mean(b_mean_arr) / 255

    RGB_df = pd.DataFrame(columns= ["R_MEAN","G_MEAN","B_MEAN"])
    RGB_df["R_MEAN"] = [R_MEAN]
    RGB_df["G_MEAN"] = [G_MEAN]
    RGB_df["B_MEAN"] = [B_MEAN]

    df_name = f"./csv/{model_name}/RGB_mean_df.csv"
    RGB_df.to_csv(df_name)
    return df_name

def get_std(data:list[np.ndarray],model_name:str) -> str:
    r_std_arr = []
    g_std_arr = []
    b_std_arr = []

    for i in range(0,len(data)):
        img_np = data[i]
        r_std,g_std,b_std = np.std(img_np,axis=(0,1))
        r_std_arr.append(r_std)
        g_std_arr.append(g_std)
        b_std_arr.append(b_std)

    R_STD = np.mean(r_std_arr) / 255
    G_STD = np.mean(g_std_arr) / 255
    B_STD = np.mean(b_std_arr) / 255

    RGB_std_df = pd.DataFrame(columns= ["R_STD","G_STD","B_STD"])
    RGB_std_df["R_STD"] = [R_STD]
    RGB_std_df["G_STD"] = [G_STD]
    RGB_std_df["B_STD"] = [B_STD]

    df_name = f"./csv/{model_name}/RGB_std_df.csv"
    RGB_std_df.to_csv(df_name)
    return df_name

    
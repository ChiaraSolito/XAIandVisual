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
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import CustomDataset, compute_metrics, plot_weights, visTensor, get_mean, get_std
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from timeit import default_timer as timer 

### DATA LOADING RETURNS LISTS OF IMAGES AND LABELS FOR TRAIN AND TEST (shuffled)
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

def stratified_kfold(train_data:list[np.ndarray], train_labels:list[str]) -> tuple[pd.DataFrame,pd.DataFrame,list,list]:

    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    skf.get_n_splits(train_data, train_labels)
    
    train_idx = []
    val_idx = []
    for i,(train_fold,val_fold) in enumerate(skf.split(train_data, train_labels)):
        train_idx.append(train_fold)
        val_idx.append(val_fold)
        print("Fold: ",i)
        print("\nTrain: index = ",train_fold)
        print("\nValidation:  index = ",val_fold,"\n")
    
    train_splits = pd.DataFrame(columns=["train_0","train_1","train_2","train_3","train_4","train_5","train_6","train_7","train_8","train_9"])
    val_splits = pd.DataFrame(columns=["val_0","val_1","val_2","val_3","val_4","val_5","val_6","val_7","val_8","val_9"])
    
    for i in range(0,10):
        train_splits["train_"+str(i)] = train_idx[i]
        val_splits["val_"+str(i)] = val_idx[i]

    train_splits.to_csv("./csv/train_splits.csv")
    val_splits.to_csv("./csv/val_splits.csv")

    return train_splits, val_splits, train_idx, val_idx

def train_step(model:torch.nn.Module, 
               dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, 
               optimizer:torch.optim.Optimizer,
               device: str):    
    
    model.train()
    train_loss,train_acc = 0,0
    
    for batch,sample_batched in enumerate(dataloader):

        X = sample_batched["image"].to(device)
        y = sample_batched["label"].to(device)
        y_pred = model(X)
        
        loss = loss_fn(y_pred,y)
        train_loss += loss.item() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss,train_acc

def val_step(model:torch.nn.Module, 
              dataloader:torch.utils.data.DataLoader, 
              loss_fn:torch.nn.Module,
              device: str):    
    
    model.eval() 
    val_loss,val_acc = 0,0
    
    with torch.inference_mode():
        
        for batch,sample_batched in enumerate(dataloader):
            
            X = sample_batched["image"].to(device)
            y = sample_batched["label"].to(device)            
            
            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss,val_acc

def train(model:torch.nn.Module, 
          train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, 
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module = CrossEntropyLoss(),
          epochs:int = 5,
          split:int = 0):

    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val = 0
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss, val_acc = val_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn) 
        # Saving the model obtaining the best validation accuracy through the epochs
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {"model": CNN_128x128(),
                          "state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            checkpoint_name = "checkpoint_"+str(split)+".pth"
            torch.save(checkpoint, checkpoint_name)    

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
    return results

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=torch.device("cpu")) 
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def normalization(data:list[np.ndarray]):

    RGB_mean_path = get_mean(data)
    RGB_mean_df = pd.read_csv(RGB_mean_path)
    print("Red ch mean = ",RGB_mean_df.iloc[0].R_MEAN.item(),"\nGreen ch mean = ",RGB_mean_df.iloc[0].G_MEAN.item(),"\nBlue ch mean = ",RGB_mean_df.iloc[0].B_MEAN.item())

    RGB_std_path = get_std(data)
    RGB_std_df = pd.read_csv(RGB_std_path)
    print("Red ch std = ",RGB_std_df.iloc[0].R_STD.item(),"\nGreen ch std = ",RGB_std_df.iloc[0].G_STD.item(),"\nBlue ch std = ",RGB_std_df.iloc[0].B_STD.item())

    IMAGE_WIDTH=224
    IMAGE_HEIGHT=224
    IMAGE_SIZE=(IMAGE_WIDTH,IMAGE_HEIGHT)
    R_MEAN = RGB_mean_df.iloc[0].R_MEAN.item()
    G_MEAN = RGB_mean_df.iloc[0].G_MEAN.item()
    B_MEAN = RGB_mean_df.iloc[0].B_MEAN.item()
    R_STD = RGB_std_df.iloc[0].R_STD.item()
    G_STD = RGB_std_df.iloc[0].G_STD.item()
    B_STD = RGB_std_df.iloc[0].B_STD.item()

    data_transform = Compose([
        Resize(size=IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=[R_MEAN,G_MEAN,B_MEAN],std=[R_STD,G_STD,B_STD])
    ])

    return data_transform

if __name__ == "__main__": #def function(dati):

    #### LOAD DATA ####
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/","./data/test/")

    data = train_data + test_data
    ##### NORMALIZATION #####
    data_transform = normalization(data)

    #### Stratified K-FOLD ####
    train_splits, val_splits, train_idx, val_idx = stratified_kfold(train_data,train_labels)

    #### TRAINING ####
    best_acc = 0.0
    num_epochs = 5 
    lr = 0.001
    n_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model = CNN_128x128(input_channel=3, num_classes=n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # to choose
    criterion = torch.nn.CrossEntropyLoss()  # to choose

    ###### ARRIVATE QUI ##########
    for i in range(0,10):
        
        train_str = "train_"+str(i)
        val_str = "val_"+str(i)
        train_data = CustomDataset(train_data[train_idx[train_str]],train_labels[train_splits[train_str]],transform=data_transform)
        val_data = CustomDataset(train_data[val_splits[val_str]],train_labels[val_splits[val_str]],transform=data_transform)
        
        trainloader = DataLoader(train_data,batch_size=32)
        validationloader = DataLoader(val_data,batch_size=32)
        
        start_time = timer()
        
        model_results = train(model=model,
                                train_dataloader=trainloader,
                                val_dataloader=validationloader,
                                optimizer=optimizer,
                                loss_fn=criterion,
                                epochs=num_epochs,
                                split=i,
                                device=device)
        
        end_time = timer()
        print(f"Total training time for split {i}: {end_time-start_time:.3f} seconds")
        
        results = dict(list(model_results.items()))
        train_loss = results["train_loss"]
        val_loss = results["val_loss"]
        
        train_acc = results["train_acc"]
        val_acc = results["val_acc"]
        
        results_df = pd.DataFrame(columns= ["train_loss","val_loss","train_acc","val_acc","epochs"])
        results_df["train_loss"] = train_loss
        results_df["val_loss"] = val_loss
        results_df["train_acc"] = train_acc
        results_df["val_acc"] = val_acc
        results_df["epochs"] = num_epochs
        results_df_name = "./csv/results_df_"+str(i)+".csv"
        results_df.to_csv(results_df_name)

    val_accuracies = np.zeros([10,1])
    for i in range(10):
        results_string = "./csv/results_df_"+str(i)+".csv"
        val_accuracies[i] = np.max(pd.read_csv(results_string)["val_acc"])
        index = np.argmax(val_accuracies)

    model_string = "checkpoint_"+str(index)+".pth"
    model_cp = load_checkpoint(model_string)
    print(model_cp)

    # return accuracy


    #main in cui chiami con tutto il dataset
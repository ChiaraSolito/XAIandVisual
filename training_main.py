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
from kymatio.torch import Scattering2D
from ScatNet import ScatNet2D
from CNN_128x128 import CNN_128x128
import torch.nn as nn

def stratified_kfold(train_data:list[np.ndarray], train_labels:list[str],model_name:str) -> tuple[pd.DataFrame,pd.DataFrame,list,list]:

    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    skf.get_n_splits(train_data, train_labels)
    
    train_idx = []
    val_idx = []
    for i,(train_fold,val_fold) in enumerate(skf.split(train_data, train_labels)):
        train_idx.append(train_fold)
        val_idx.append(val_fold)
        print("Fold: ",i)
        #print("\nTrain: index = ",train_fold)
        #print("\nValidation:  index = ",val_fold,"\n")
    
    train_splits = pd.DataFrame(columns=["train_0","train_1","train_2","train_3","train_4","train_5","train_6","train_7","train_8","train_9"])
    val_splits = pd.DataFrame(columns=["val_0","val_1","val_2","val_3","val_4","val_5","val_6","val_7","val_8","val_9"])
    
    for i in range(0,10):
        train_splits["train_"+str(i)] = train_idx[i]
        val_splits["val_"+str(i)] = val_idx[i]

    train_splits.to_csv(f"./csv/{model_name}/train_splits.csv")
    val_splits.to_csv(f"./csv/{model_name}/val_splits.csv")

    return train_splits, val_splits, train_idx, val_idx

def train_step(model:torch.nn.Module, 
               dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, 
               optimizer:torch.optim.Optimizer,
               device: str):    
    
    model.train()
    train_loss,train_acc = 0,0
    
    for batch,sample_batched in enumerate(dataloader):

        X = sample_batched[0].to(device)
        y = sample_batched[1].to(device)
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
            
            X = sample_batched[0].to(device)
            y = sample_batched[1].to(device)            
            
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
          split:int = 0,
          device:str = 'cpu',
          model_name:str = 'CNN'):

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
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = val_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device) 
        
        # Saving the model information, obtaining the best validation accuracy through the epochs
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,}
            checkpoint_name = f"./models_trained/{model_name}/checkpoint_"+str(split)+".pth"    

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

    #for every fold save only the best model over all epochs
    torch.save(checkpoint, checkpoint_name)
    return results

def normalization(data:list[np.ndarray],model_name:str):

    RGB_mean_path = get_mean(data,model_name)
    RGB_mean_df = pd.read_csv(RGB_mean_path)
    print("Red ch mean = ",RGB_mean_df.iloc[0].R_MEAN.item(),"\nGreen ch mean = ",RGB_mean_df.iloc[0].G_MEAN.item(),"\nBlue ch mean = ",RGB_mean_df.iloc[0].B_MEAN.item())

    RGB_std_path = get_std(data,model_name)
    RGB_std_df = pd.read_csv(RGB_std_path)
    print("Red ch std = ",RGB_std_df.iloc[0].R_STD.item(),"\nGreen ch std = ",RGB_std_df.iloc[0].G_STD.item(),"\nBlue ch std = ",RGB_std_df.iloc[0].B_STD.item())

    R_MEAN = RGB_mean_df.iloc[0].R_MEAN.item()
    G_MEAN = RGB_mean_df.iloc[0].G_MEAN.item()
    B_MEAN = RGB_mean_df.iloc[0].B_MEAN.item()
    R_STD = RGB_std_df.iloc[0].R_STD.item()
    G_STD = RGB_std_df.iloc[0].G_STD.item()
    B_STD = RGB_std_df.iloc[0].B_STD.item()

    data_transform = Compose([
        ToPILImage(),
        Resize(size=(128,128)),
        ToTensor(),
        Normalize(mean=[R_MEAN,G_MEAN,B_MEAN],std=[R_STD,G_STD,B_STD])
    ])

    return data_transform

def training_main(data,train_data, train_labels,base_model:str):

    ##### NORMALIZATION #####
    data_transform = normalization(data,base_model)

    #### Stratified K-FOLD ####
    train_splits, val_splits, train_idx, val_idx = stratified_kfold(train_data,train_labels,base_model)

    #### TRAINING ####
    num_epochs = 5 
    n_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)


    if base_model == 'CNN':
        app_model = CNN_128x128(input_channel=3, num_classes=n_classes).to(device)
        lr = 0.001
    elif base_model == 'ScatNet':
        L = 8
        J = 2
        scattering = Scattering2D(J=J, shape=(128, 128), L=L)
        K = 81  # Input channels for the ScatNet
        scattering = scattering.to(device)
        app_model = ScatNet2D(input_channels=K, scattering=scattering).to(device)
        lr = 0.0001
    else:
        print('Model not recognized, terminating program.')
        exit()

    criterion = torch.nn.CrossEntropyLoss()  # to choose

    ###### ARRIVATE QUI ##########
    for i in range(0,10): 
        
        model = app_model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        train_data_fold = CustomDataset([train_data[x] for x in train_idx[i]],[train_labels[x] for x in train_idx[i]],transform=data_transform)
        val_data_fold = CustomDataset([train_data[x] for x in val_idx[i]],[train_labels[x] for x in val_idx[i]],transform=data_transform)
        
        trainloader = DataLoader(train_data_fold,batch_size=6) #da adattare numero batch size
        validationloader = DataLoader(val_data_fold,batch_size=6)
        
        start_time = timer()
        
        model_results = train(model=model,
                                train_dataloader=trainloader,
                                val_dataloader=validationloader,
                                optimizer=optimizer,
                                loss_fn=criterion,
                                epochs=num_epochs,
                                split=i,
                                device=device,
                                model_name=base_model)
        
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
        results_df_name = f"./csv/{base_model}/results_df_"+str(i)+".csv"
        results_df.to_csv(results_df_name)

    max_val_accuracies = np.zeros([10,1])
    val_accuracies = np.zeros([10,num_epochs])
    train_losses = np.zeros([10,num_epochs])

    for i in range(10):
        results_string = f"./csv/{base_model}/results_df_"+str(i)+".csv"
        max_val_accuracies[i] = np.max(pd.read_csv(results_string)["val_acc"])
        val_accuracies[i] = (pd.read_csv(results_string)["val_acc"]).to_list()
        train_losses[i] = (pd.read_csv(results_string)["val_acc"]).to_list()

    index = np.argmax(max_val_accuracies)

    model_string = f"./models_trained/{base_model}/checkpoint_"+str(index)+".pth"
    checkpoint = torch.load(model_string,map_location=torch.device("cpu")) 
    best_model = app_model
    best_model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in best_model.parameters():
        parameter.requires_grad = False

    return best_model
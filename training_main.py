# Libraries
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import CustomDataset, compute_metrics, plot_results
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from timeit import default_timer as timer
from kymatio.torch import Scattering2D
from ScatNet import ScatNet2D
from CNN_128x128 import CNN_128x128
from datetime import datetime
from sklearn.metrics import f1_score

NUM_FOLD = 10
SIZE = (128, 128)


def stratified_kfold(train_data: list[np.ndarray], train_labels: list[str], model_name: str) -> tuple[
    pd.DataFrame, pd.DataFrame, list, list]:
    skf = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=42)
    skf.get_n_splits(train_data, train_labels)

    train_idx = []
    val_idx = []
    for i, (train_fold, val_fold) in enumerate(skf.split(train_data, train_labels)):
        train_idx.append(train_fold)
        val_idx.append(val_fold)
    print("Folding done!")

    train_splits = pd.DataFrame(
        columns=["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7", "train_8",
                 "train_9"])
    val_splits = pd.DataFrame(
        columns=["val_0", "val_1", "val_2", "val_3", "val_4", "val_5", "val_6", "val_7", "val_8", "val_9"])

    for i in range(0, NUM_FOLD):
        train_splits["train_" + str(i)] = train_idx[i]
        val_splits["val_" + str(i)] = val_idx[i]

    train_splits.to_csv(f"./csv/{model_name}/train_splits.csv")
    val_splits.to_csv(f"./csv/{model_name}/val_splits.csv")

    return train_splits, val_splits, train_idx, val_idx


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, sample_batched in enumerate(dataloader):
        X = sample_batched[0].to(device)
        y = sample_batched[1].to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device):
    model.eval()
    val_loss, val_acc, val_f1 = 0, 0, 0

    with torch.inference_mode():
        for batch, sample_batched in enumerate(dataloader):
            X = sample_batched[0].to(device)
            y = sample_batched[1].to(device)

            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))
            val_f1 = f1_score(y.cpu().numpy(), val_pred_labels.cpu().numpy())

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    val_f1 = val_f1 / len(dataloader)
    return val_loss, val_acc, val_f1


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = CrossEntropyLoss(),
          epochs: int = 5,
          split: int = 0,
          device: torch.device = 'cpu',
          model_name: str = 'CNN'):
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": [],
               "val_f1": []
               }

    best_val = 0
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc, val_f1 = val_step(model=model,
                                             dataloader=val_dataloader,
                                             loss_fn=loss_fn,
                                             device=device)

        # Saving the model information, obtaining the best validation accuracy through the epochs
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss': val_loss, }
            checkpoint_name = f"./models_trained/{model_name}/checkpoint_" + str(split) + ".pth"

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)

    # for every fold save only the best model over all epochs
    torch.save(checkpoint, checkpoint_name)
    return results


def training_main(data_transform, train_data, train_labels, base_model: str):
    global app_model, lr

    # Stratified K-FOLD
    train_splits, val_splits, train_idx, val_idx = stratified_kfold(train_data, train_labels, base_model)

    # TRAINING
    num_epochs = 5
    n_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    criterion = torch.nn.CrossEntropyLoss()  # to choose

    for i in range(0, NUM_FOLD):

        # Instance of the model
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

        model = app_model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_data_fold = CustomDataset([train_data[x] for x in train_idx[i]], [train_labels[x] for x in train_idx[i]],
                                        transform=data_transform)
        val_data_fold = CustomDataset([train_data[x] for x in val_idx[i]], [train_labels[x] for x in val_idx[i]],
                                      transform=data_transform)

        trainloader = DataLoader(train_data_fold, batch_size=64)  # da adattare numero batch size
        validationloader = DataLoader(val_data_fold, batch_size=64)

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
        print(f"Total training time for split {i}: {end_time - start_time:.3f} seconds")
        print("------------------------------------")

        results = dict(list(model_results.items()))
        results_df = pd.DataFrame.from_dict(results)
        results_df_name = f"./csv/{base_model}/results_df_" + str(i) + ".csv"
        results_df.to_csv(results_df_name)

    max_val_accuracies = np.zeros([NUM_FOLD, 1])
    val_accuracies = np.zeros([NUM_FOLD, num_epochs])
    train_losses = np.zeros([NUM_FOLD, num_epochs])
    val_losses = np.zeros([NUM_FOLD, num_epochs])
    f1_scores = np.zeros([NUM_FOLD,num_epochs])

    for i in range(NUM_FOLD):
        results_string = f"./csv/{base_model}/results_df_" + str(i) + ".csv"
        max_val_accuracies[i] = np.max(pd.read_csv(results_string)["val_acc"])
        val_accuracies[i] = (pd.read_csv(results_string)["val_acc"]).to_list()
        val_losses[i] = (pd.read_csv(results_string)["val_loss"]).to_list()
        train_losses[i] = (pd.read_csv(results_string)["train_loss"]).to_list()
        f1_scores[i] = (pd.read_csv(results_string)["val_f1"]).to_list()

    plot_results(val_accuracies, train_losses, val_losses, f1_scores, base_model)

    index = np.argmax(max_val_accuracies)

    model_string = f"./models_trained/{base_model}/checkpoint_" + str(index) + ".pth"
    checkpoint = torch.load(model_string, map_location=torch.device("cpu"))
    best_model = app_model
    best_model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in best_model.parameters():
        parameter.requires_grad = False

    return best_model


def test(data_transform, test_data, test_labels, model, model_name, device):
    test_data = CustomDataset(test_data, test_labels, transform=data_transform)
    testloader = DataLoader(test_data, batch_size=64)

    model.eval()
    pred_labels = []
    with torch.no_grad():
        for batch, sample_batched in enumerate(testloader):
            X = sample_batched[0].to(device)
            y = sample_batched[1].to(device)
            model.to(device)
            y_pred = model(X)

            pred_label = y_pred.argmax(dim=1)
            pred_labels.append(pred_label)

    acc, f1_score = compute_metrics(test_labels, pred_labels, classes=['muffin', 'chihuahua'])

    # dd/mm/YY H:M:S
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    results_df = pd.DataFrame(columns=["model", "accuracy", "f1 score", "date", "time"])
    results_df["model"] = model_name
    results_df["accuracy"] = acc
    results_df["f1 score"] = f1_score
    results_df["date"] = datetime.now()
    results_df_name = f"./csv/{model_name}/test_results_df_" + dt_string + ".csv"
    results_df.to_csv(results_df_name)

    return acc

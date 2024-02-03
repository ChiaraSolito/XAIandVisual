
# Libraries
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from training_main import training_main
from utils import data_loading,CustomDataset, compute_metrics, plot_weights, visTensor
from datetime import datetime

if __name__ == "__main__":

    #### LOAD DATA ####
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/","./data/test/")
    data = train_data + test_data

    model = training_main(data,train_data, train_labels,'CNN')

    ### 3 - TESTING

    # to use when model is saved
    #model_test = CNN_128x128(input_channel=3, num_classes=n_classes).to(device)  # Initialize a new model
    #model_test.load_state_dict(torch.load(f'{models_trained_path}_{date_time}.pt'))  # Load the model
    # to use when model is saved

    #pred_label_test = torch.empty((0, n_classes)).to(device)
    #true_label_test = torch.empty((0)).to(device)

    #with torch.no_grad():
    #    for data in testset:
    #        X_te, y_te = data
    #        X_te = X_te.view(batch_size, 3, 128, 128).float().to(device)
    #        y_te = y_te.to(device)
    #        output_test = model_test(X_te)
    #        pred_label_test = torch.cat((pred_label_test, output_test), dim=0)
    #        true_label_test = torch.cat((true_label_test, y_te), dim=0)

    #compute_metrics(y_true=true_label_test, y_pred=pred_label_test,
    #                lab_classes=lab_classes)  # function to compute the metrics (accuracy and confusion matrix)
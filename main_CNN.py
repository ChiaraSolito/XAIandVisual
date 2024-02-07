
# Libraries
from training_main import training_main, test
from utils import data_loading, normalization

if __name__ == "__main__":

    # LOAD DATA
    train_data, train_labels, test_data, test_labels = data_loading("./data/train/", "./data/test/")
    data = train_data + test_data

    # NORMALIZATION
    data_transform = normalization(data)

    # TRAINING
    model = training_main(data_transform, train_data, train_labels, 'CNN')

    # TESTING
    acc = test(data_transform, test_data, test_labels, model, 'CNN', device='cpu')
    print("Accuracy", acc)

    # EXTRACTION OF FILTERS

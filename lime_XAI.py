


import numpy as np
import torch
import pandas as pd
from CNN import CNN

from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Normalize, CenterCrop

import cv2

from lime.lime_image import LimeImageExplainer

from utils import normalization, data_loading

from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def trasform_2D():


    R_MEAN = 0.2082946035683011
    G_MEAN = 0.2082298517705837
    B_MEAN = 0.20822414060704353

    R_STD = 0.21201244177066295
    G_STD = 0.21199000827746403
    B_STD = 0.2119912744425196

    data_transform = Compose([
        ToPILImage(),
        ToTensor(),
        Resize(size=(128, 128)),
        Normalize(mean=[R_MEAN, G_MEAN, B_MEAN], std=[R_STD, G_STD, B_STD]),
    ])


    return data_transform


# Attribution plot
def plot_lime(image, img1, img2):

    # REMOVE WHEN THERE ARE MORE THAN 1 IMAGE --> TO AVOID REMOVE i VARIABLE EVERYWHERE
    image = image.reshape(1, 128, 128, 3)
    img1 = img1.reshape(1, 128, 128, 3)
    img2 = img2.reshape(1, 128, 128, 3)
    # REMOVE WHEN THERE ARE MORE THAN 1 IMAGE  --> TO AVOID REMOVE i VARIABLE EVERYWHERE

    fig, axis = plt.subplots(len(image), 3, figsize=(12, 10))

    for i in range(image.shape[0]):

        axis[0].imshow(image[i])  # .cpu().detach().numpy())
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        # axis[i, 0].set_title(titles[i], fontsize=14)

        # im2 = axis[i, 1].imshow(image)
        im2 = axis[1].imshow(img1[i].reshape(128, 128, 3), cmap='inferno', alpha=0.9)
        axis[1].axis('off')
        axis[1].set_title('Mask Positive', fontsize=14)
        # divider = make_axes_locatable(axis[1])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im2, cax=cax, orientation='vertical')

        # im3 = axis[i, 2].imshow(image)
        im3 = axis[2].imshow(img2[i].reshape(128, 128, 3), cmap='inferno', alpha=0.8)
        # im3 = axis[i, 2].imshow(ig_scratch.squeeze(0).permute(1,2,0).cpu(), cmap='jet', alpha=0.9)
        axis[2].axis('off')
        axis[2].set_title('MAsk Positive and Negative', fontsize=14)

        # divider = make_axes_locatable(axis[2])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im3, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()

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

    data_transform = trasform_2D()

    images = torch.empty(len(test_data), 3, 128, 128)

    for i, img in enumerate(test_data):
        images[i] = data_transform(img)

    background = images[:100]
    test_images = images[100:106]

    lime_explainer = LimeImageExplainer(kernel_width=0.25, #kernel=None,
                                        verbose=False, feature_selection='auto', random_state=None)


    def batch_predict(images):
        model_CNN.eval()
        # batch = torch.stack(tuple(data_transform(i) for i in images), dim=0)
        #
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model_CNN.to(device)
        # batch = batch.to(device)

        logits = model_CNN(torch.tensor(images).view(images.shape[0],images.shape[3],images.shape[1],images.shape[2]))
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


    image_original = np.transpose(np.array(background[0]), (1, 2, 0))

    lime_inst_exp = lime_explainer.explain_instance(image_original, batch_predict, num_features=8)

    temp, mask = lime_inst_exp.get_image_and_mask(lime_inst_exp.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    img1 = mark_boundaries(temp, mask)

    temp, mask = lime_inst_exp.get_image_and_mask(lime_inst_exp.top_labels[0], positive_only=False, num_features=20,
                                                hide_rest=False)

    temp = (temp - temp.min()) / (temp.max() - temp.min())
    img2 = mark_boundaries(temp, mask)

    plot_lime(image_original, img1, img2)

import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from CNN import CNN
from ScatNet import ScatNet2D
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients
from kymatio.torch import Scattering2D

import cv2

import numpy as np


# def plot_images(images, title=None):
#     """
#     Plot a batch of images.
#
#     Parameters:
#     - images (torch.Tensor): Batch of images with shape (batch_size, channels, height, width).
#     - title (str): Title for the plot.
#     """
#     batch_size, channels, height, width = images.shape
#     images = images.permute(0, 2, 3, 1)  # Change the order of dimensions for matplotlib
#
#     # Rescale images to the range [0, 1] if not already in that range
#     if images.min() < 0 or images.max() > 1:
#         images = (images - images.min()) / (images.max() - images.min())
#
#     # Create a grid of images
#     rows = int(batch_size ** 0.5)
#     cols = (batch_size + rows - 1) // rows
#
#     plt.figure(figsize=(10, 10))
#     for i in range(batch_size):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(images[i].cpu().numpy())
#         plt.axis('off')
#
#     if title:
#         plt.suptitle(title)
#
#     plt.show()

def plot_train_label(image, mask):
    fig, axis = plt.subplots(1, 3, figsize=(5, 5))

    # axarr[0].imshow(np.squeezesqueeze(image), cmap='gray')
    axis[0].imshow(image)  #, cmap='gray')
    axis[0].set_ylabel('Axial View', fontsize=14)
    axis[0].set_xticks([])
    axis[0].set_yticks([])
    axis[0].set_title('CT', fontsize=14)

    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list("3colors", ['k', 'r', 'g'], N=3)

    axis[1].imshow(mask, cmap='viridis')  #, cmap='tab20c')
    axis[1].axis('off')
    axis[1].set_title('Mask', fontsize=14)

    im3 = axis[2].imshow(image, cmap='viridis', alpha=0.6)
    im3 = axis[2].imshow(mask, cmap='viridis', alpha=0.8)
    axis[2].axis('off')
    axis[2].set_title('Overlay', fontsize=14)

    divider = make_axes_locatable(axis[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def load_model_and_data(model_type):

    if model_type == 'CNN':
        model = CNN(3, 2)
        model.load_state_dict(torch.load('../models_trained/CNN/checkpoint_2.pth')['model_state_dict'])
    elif model_type == 'ScatNet':

        L = 8
        J = 2
        scattering = Scattering2D(J=J, shape=(128, 128), L=L)
        K = 81  # Input channels for the ScatNet
        scattering = scattering.to('cuda')
        model = ScatNet2D(input_channels=K, scattering=scattering, num_classes=2).to('cuda')
        model.load_state_dict(torch.load('../models_trained/ScatNet/checkpoint_0.pth')["model_state_dict"])


    # NO TUMOR
    # img_name = 'Te-no_0010.jpg'  # notumor
    # img_path = f'../data/test/notumor/{img_name}'

    # TUMOR
    # img_name = 'Te-me_0068.jpg'  # meningioma
    img_name = 'Te-me_0157.jpg'  # meningioma
    img_path = f'../data/test/meningioma/{img_name}'

    model.eval()
    model.cuda()

    # read the image
    # image = Image.open(img_path)
    image = cv2.imread(img_path)

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
        Normalize(mean=[R_MEAN,G_MEAN,B_MEAN],std=[R_STD,G_STD,B_STD])
    ])

    original_img = image
    transformed_img = data_transform(image).unsqueeze(0).to('cuda')
    # plot_images(torch.tensor(img).reshape(1,3,128,128), title='Original Images')

    print(f'CLASS: {"NO TUMOR" if torch.argmax(model(transformed_img)) == 1 else "TUMOR"}')

    return original_img, transformed_img, model

def ig_scratch(original_img, transformed_img, model, baseline, n_alpha):

    img = (torch.tensor(transformed_img)).to('cuda')

    difference_img_baseline = (img - baseline).reshape(3,128,128).to('cuda')

    alphas = torch.linspace(0.1,1,n_alpha) # compute for 10 values of alpha
    gradients_list = torch.empty(0,3,128,128).to('cuda')



    model.eval()
    for alpha in alphas:
        input = (baseline.reshape(3,128,128) + (alpha * difference_img_baseline)).reshape(1,3,128,128).to('cuda').requires_grad_(True)
        output = model(input)
        gradients = torch.autograd.grad(torch.max(output), input)[0] * 0.1
        gradients_list = torch.cat([gradients_list, gradients],dim=0)


    gradients_list = torch.sum(gradients_list, dim=0).reshape(1,3,128,128)

    gradients_list = difference_img_baseline * gradients_list


    # Assuming your tensor of images is named 'image_tensor'
    # Replace 'image_tensor' with the actual name of your tensor
    # plot_images(gradients_list, title='Batch of Images')  # THIS WORKS!!!!!

    # overlay = gradients_list + img.reshape(3, 128, 128).to('cuda')
    # overlay = (img).reshape(1,3, 128, 128).to('cuda')
    # plot_images(overlay, title='Batch of Images')

    gradients_list = (gradients_list - gradients_list.min()) / (gradients_list.max() - gradients_list.min()) * 255

    # transformed image
    # plot_train_label(img.permute(1,2,0).cpu().numpy(), gradients_list[0].permute(1,2,0).cpu().numpy())
    # original image
    plot_train_label(original_img, np.mean(gradients_list[0].permute(1,2,0).cpu().numpy(),axis=2))

def ig_captum(original_img, transformed_img, model, baseline, n_alpha):
    img = transformed_img.to('cuda')

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(img, baseline, target=0, return_convergence_delta=True, n_steps=n_alpha)
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)

    attributions = attributions.squeeze(0)

    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # plot transformed
    plot_train_label(transformed_img.squeeze(0).permute(1, 2, 0).cpu().numpy(), np.mean(attributions.permute(1, 2, 0).cpu().numpy(), axis=2))
    # plot original NO SQUEEZE
    # plot_train_label(np.transpose(original_img, (0,1,2)), np.mean(attributions.permute(1, 2, 0).cpu().numpy(), axis=2))

    return img, model

if __name__ == "__main__":

    n_alpha = 200
    # img, model = load_model_and_data('ScatNet')
    original_img, transformed_img, model = load_model_and_data('CNN')
    transformed_img = transformed_img.reshape(1, 3, 128, 128)
    baseline = torch.randn(1,3, 128, 128).to('cuda')
    # baseline = torch.ones_like(transformed_img).to('cuda')

    # ig_scratch(original_img, transformed_img, model, baseline, n_alpha)
    ig_captum(original_img, transformed_img, model, baseline, n_alpha)




import torch
from CNN import CNN
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients



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
    axis[0].imshow(image, cmap='gray')
    axis[0].set_ylabel('Axial View', fontsize=14)
    axis[0].set_xticks([])
    axis[0].set_yticks([])
    axis[0].set_title('CT', fontsize=14)

    axis[1].imshow(mask, cmap='jet')
    axis[1].axis('off')
    axis[1].set_title('Mask', fontsize=14)

    axis[2].imshow(image, cmap='gray', alpha=1)
    axis[2].imshow(mask, cmap='jet', alpha=0.8)
    axis[2].axis('off')
    axis[2].set_title('Overlay', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def load_model_and_data():
    model = CNN(3, 2)
    model.load_state_dict(torch.load('model_CNN.pt'))

    img_name = '01.jpg'
    # img_path = f'muffin_images/{img_name}'
    img_path = f'chihuahua_images/{img_name}'

    model.eval()
    model.cuda()

    # read the image
    image = Image.open(img_path)

    data_transform = Compose([
        # ToPILImage(),
        ToTensor(),
        Resize(size=(128, 128)),
        # Normalize(mean=[R_MEAN,G_MEAN,B_MEAN],std=[R_STD,G_STD,B_STD])
    ])

    img = data_transform(image)
    # plot_images(torch.tensor(img).reshape(1,3,128,128), title='Original Images')

    return img, model

def ig_scratch(img, model):

    img = (torch.tensor(img)).to('cuda')
    baseline = torch.zeros_like(img).to('cuda')

    difference_img_baseline = (img - baseline).reshape(3,128,128).to('cuda')

    alphas = torch.linspace(0.1,1,10) # compute for 5 values of alpha
    gradients_list = torch.empty(0,3,128,128).to('cuda')

    model.eval()
    for alpha in alphas:
        input = (baseline.reshape(3,128,128) + (alpha * difference_img_baseline)).reshape(1,3,128,128).to('cuda').requires_grad_(True)
        output = model(input)
        gradients = torch.autograd.grad(torch.max(output), input)[0]
        gradients_list = torch.cat([gradients_list, gradients],dim=0)


    gradients_list = torch.sum(gradients_list, dim=0).reshape(1,3,128,128)

    gradients_list = difference_img_baseline * gradients_list


    # Assuming your tensor of images is named 'image_tensor'
    # Replace 'image_tensor' with the actual name of your tensor
    # plot_images(gradients_list, title='Batch of Images')  # THIS WORKS!!!!!

    # overlay = gradients_list + img.reshape(3, 128, 128).to('cuda')
    # overlay = (img).reshape(1,3, 128, 128).to('cuda')
    # plot_images(overlay, title='Batch of Images')

    gradients_list = (gradients_list - gradients_list.min()) / (gradients_list.max() - gradients_list.min())
    plot_train_label(img.permute(1,2,0).cpu().numpy(), gradients_list[0].permute(1,2,0).cpu().numpy())

def ig_captum(img, model):
    img = img.to('cuda').reshape(1, 3, 128, 128)
    baseline = torch.zeros_like(img).to('cuda')

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(img, baseline, target=0, return_convergence_delta=True)
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)

    attributions = attributions.squeeze(0)

    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
    plot_train_label(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), attributions.permute(1, 2, 0).cpu().numpy())

    return img, model

if __name__ == "__main__":

    img, model = load_model_and_data()
    ig_scratch(img, model)
    # ig_captum(img, model)




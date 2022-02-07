"""
Reference:

Created on Tues Mar 10 08:13:15 2020
@author: Alex Stoken - https://github.com/alexstoken

Last tested with torchvision 0.5.0 with image and model on cpu
"""
import os
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from utils import recreate_image, save_image

use_cuda = torch.cuda.is_available()


class Data_impression():
    """
        Produces an image that maximizes a certain class with gradient ascent. Uses Gaussian blur, weight decay, and clipping.
    """

    def __init__(self, model, target_class, img_idx, dirichelt_list, dataset):

        if dataset == 'MNIST':
            img_size = 28
            self.rev_mean = [None]
            self.rev_std = [None]
            img_channel = 1
            self.created_image = np.uint8(np.random.uniform(0, 1, (img_size, img_size, img_channel)) * 255)

        elif dataset == 'CIFAR10':
            img_size = 32
            img_channel = 3
            self.rev_mean = [-0.5, -0.5, -0.5]
            self.rev_std = [1 / 0.5, 1 / 0.5, 1 / 0.5]
            self.created_image = np.uint8(np.random.uniform(0, 255, (img_size, img_size, img_channel)))

        elif dataset == 'CIFAR100':
            img_size = 32
            img_channel = 3
            self.rev_mean = [-0.5, -0.5, -0.5]
            self.rev_std = [1 / 0.5, 1 / 0.5, 1 / 0.5]
            self.created_image = np.uint8(np.random.uniform(0, 255, (img_size, img_size, img_channel)))



        self.dataset = dataset
        self.model = model.cuda() if use_cuda else model
        self.model.eval()
        self.target_class = target_class
        self.img_idx = img_idx
        self.dirichelt_list = dirichelt_list
        # Generate a random image
        # Create the folder to export images if not exists
        if not os.path.exists(f'../generated/class_{self.target_class}'):
            os.makedirs(f'../generated/class_{self.target_class}')

    def generate(self, iterations=150):

        diri_sample = self.dirichelt_list[self.target_class].sample()

        initial_learning_rate = 0.001

        self.processed_image = preprocess_and_blur_image(
            self.created_image, self.rev_mean, self.rev_std, self.dataset, False)

        self.processed_image = self.processed_image.cuda()
        optimizer = Adam([self.processed_image], lr=initial_learning_rate)
        for i in range(1, iterations):
            # Forward
            output = self.model(self.processed_image)
            loss = cross_entropy(output/20, diri_sample)

            total_loss = loss
            self.model.zero_grad()
            total_loss.backward()
            optimizer.step()

        self.created_image = recreate_image(self.processed_image.cpu(), self.rev_mean, self.rev_std)

        # save final image
        im_path = f'synimg/cls{self.target_class}/img{self.img_idx}.png'

        if self.dataset == 'MNIST':
            self.created_image = np.concatenate([self.created_image] * 3, axis=2)
        save_image(self.created_image, im_path)


        return self.processed_image

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def preprocess_and_blur_image(pil_im,  rev_mean, rev_std, dataset, resize_im=True, blur_rad=None,):
    """
        Processes image with optional Gaussian blur for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
        blur_rad (int): Pixel radius for Gaussian blurring (default = None)
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """

    # mean and std list for channels (cifar10)
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2023, 0.1994, 0.2010]

    # mean and std list for channels (cifar100)
    # mean = [0.5071, 0.4867, 0.4408]
    # std = [0.2675, 0.2565, 0.2761]

    # mean and std list for channels (Imagenet)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]


    if dataset == 'MNIST':
        pil_im = np.concatenate([pil_im]*3, axis=2)

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")

    # add gaussin blur to image
    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels

    if dataset != 'MNIST':
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] += rev_mean[channel]
            im_as_arr[channel] *= rev_std[channel]
    else:
        im_as_arr = im_as_arr[[0],...]
        im_as_arr/= 255

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    if use_cuda:
        im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    else:
        im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


if __name__ == '__main__':
    target_class = 1  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    csig = RegularizedClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()

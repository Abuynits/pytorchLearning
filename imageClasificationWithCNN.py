# use CIFAR10 dataset of 60k images 32 x32 color images in 10 classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder  # this help you process these type of folder structures
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid


def show_example(img, label):
    plt.figure(figsize=(10, 10))
    print('Label: ', dataset.classes[label], "(" + str(label) + ")")
    # use the permute to move the channels from the 0 to the last position ( show the last channel )
    plt.imshow(img.permute(1, 2, 0))

    plt.show()


# show a batch of training data
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()

        break


def apply_kernel(image, kernel):
    ri, ci = image.shape  # image dimensions
    rk, ck = kernel.shape  # kernel dimensions
    ro, co = ri - rk + 1, ci - ck + 1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro):  # itterate for the output pixels
        for j in range(
                co):  # teh star is an element wise multiplication, and put it in the top left kernet of the output
            output[i, j] = torch.sum(image[i:i + rk, j:j + ck] * kernel)
    return output



if __name__ == '__main__':

    if False:
        # start by downloading the images in PNG format:
        dataset_url = "https://files.fast.ai/data/examples/cifar10.tgz"
        # tar file used in tar, zip file formats
        download_url(dataset_url, '.')
    if False:
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')

    # Now have 2 folders of train and test
    # train: 50k, test: 10k
    # they are hosten in internahl folders

    data_dir = './data/cifar10'
    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)  # have a folder per class

    # now see what is inside of each class:
    airplane_files = os.listdir(data_dir + "/train/airplane")
    print('num airplane: ', len(airplane_files))
    print(airplane_files[:5])

    dataset = ImageFolder(data_dir + '/train', transform=ToTensor())  # convert all images to tensors
    # each image has the shape: (3,32,32) bc have 3 channels of rgb
    img, label = dataset[0]
    print(img.shape, label)

    print(dataset.classes)  # will give all of the classes of the dataset

    # show_example(img, label)
    # these are low resolution images: need to look at data and see how easy it is for us to do
    # model cannot be better than human, so if its hard for you to identify, then this is a sign that you need better data

    # need to split data into 3 parts: training, validation, test set
    # set 10% of images to be used as validation set
    random_seed = 42  # this helps you create the same validation set over and over again
    val_size = 5000
    train_size = len(dataset) - val_size

    torch.manual_seed(random_seed)

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(len(train_ds), len(val_ds))
    batch_size = 128
    # call the dataloaders on the data
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # shuffle the batches on each training of the models
    # set num workers/ pin memoery: ensure if have more cpu cores taht you load more cores
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

    # no benefit to shuffle

    # take a batch and puts it into a grid
    # show_batch(train_dl)

    # definiing the model (cnn)

    # use nn.conv class from pytorch
    """
    what is a convolution:
    2d convolution: start with a kernel: that has data in it
    place a square in the image, and do an element wise multiplication of the kernet and the values in teh image, then you sum all of them up
    have a small matrix that applies weights to the images, and takes an output. then you slide it to the next kernel
    feed forward nn: assign a weight to the image: cnn makes the weights spacially invarient bc its sliding over all of the data

    """
    sample_image = torch.tensor([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1]
    ], dtype=torch.float32)

    sample_kernel = torch.tensor([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ], dtype=torch.float32)

    print(apply_kernel(sample_image, sample_kernel))

    # this is convolution applied for each channel
    # for multi channels: apply a different type of kernel for each channel

    # can also add white pixels around the edges. they will act as padding and this will prevent you from losing information
    # striding: instead of shifting by 1 pixel, you shift it by 2 pixels: get a smaller output kernel

    # for multiiple channels; get back an output from each channel, then do element wise addition and you get a single result
    # this operation is called 1 filter
    # then do this many times to get multiple outputs

    # where are wieghts of the model? we previously had a matrix of weights, here the wieghts are the kernels
    # each kernel has randomly init weights: apply convolution, get outputs, and the final output is some probabilities

    # when do backward optimization step, the weights of the kernels keep changing and they get better at detecting relat bet inptu and output

    # we see that instead of having a weight for each pixel, we use a smaller set of param
    # we can have many filters

    # sparsity of connections: in images, you are concerned about relationships of pixels to each other, making forward and backward passes fmore efficient
    # parameters sharing and spatial invariance: features learned by a kernel in one part of the image cna be used to detect similar pattern in a differetn part of another image
    # replacin gweight matrix with filters and the kernels applied to each channel

    # we also use a max-pooling layers to progressively descrease the height and width of the output tensors for eac convolutional layer
    # this will progressivley decrase the width and height of the output tensors

    # look how 1 convolution layer followed by max-pooling layer operatoes on the data:

    simple_model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2))
    # using nn.seq: say that when given input, pass through the first input and then to the next
    # passing the conv2d: say that start with 3 channels ( have 3 channels of 32x32 pixels)
    # say that in the output we want 8 channels ( htis is not an image), but we have expanded the scope of our model
    # kernel size is a 3x3 matrix
    # uses a stride of 1: shift by 1 pixel each time
    # uses a padding of 1: add a white pixel outside of the image: the output is the same as the input (32x32)
    # for max pool say for every 2x2 matrix, replace it with the maximum value in it
    for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = simple_model(images)
        print('out.shape:', out.shape)
        break
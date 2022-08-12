import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_url

matplotlib.rcParams['figure.facecolor'] = '#ffffff'


def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    # optimzer: get the curernt rate from optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    # take a max pt that the function should touch
    # then take weight decay amt: how much give to the wieght decay
    # then take a gradient clip: the gradients will be limited to a value
    # take the optimization function
    torch.cuda.empty_cache()  # help to clear stale data on gpu

    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    # use lr_schedule wiht built in once Culce lr; give max pearnign rate, the epochs and number of batches of epcohs
    # then identipy the amount that you need to scheudle
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    # dont want randomization of batch normalization and dropout
    # model.train tell that you are in tainign model.
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)  # get the loss from model
            train_losses.append(loss)
            loss.backward()  # have gradients

            # Gradient clipping
            if grad_clip:  # call nn.utils give parameters and take grad property and limits them in grad clip range
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            # sched.step: modifies LR of the optimizer, changin the lr after every batch
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs  # record list of lrs used during match
        model.epoch_end(epoch, result)  # print the history
        history.append(result)
    return history


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]);
        ax.set_yticks([])
        # denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        plt.show()
        break


class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # create the 3 input and output channels: cannot change number of channels in the input channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # pass it through the layers
        out = self.conv1(x)
        out = self.relu1(out)  # can also do F.relu()
        out = self.conv2(out)
        return self.relu2(out) + x  # here we add in the residual
        # cal also try adding the residual back in  and the applying the relu
        # return self.relu2(out + x)


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              # adds a 2d normalization: does batch normalizatio for a 2d input, which is output of convolution
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]  # simply modifies the input in place
    if pool: layers.append(
        nn.MaxPool2d(2))  # pool: if set it to true, we append max pool 2d : take 4 pixels and replace with 1
    # reduce size of the output feature map into half
    return nn.Sequential(*layers)  # give a list of layers, and it will pass input throug ahll of the layers


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)  # have 3 channels to 64 channel (no pooling)
        # the output is 64x32x32
        self.conv2 = conv_block(64, 128, pool=True)
        # then go to 128 channels with max pooling 128x16x16
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        # conv blocks that dont change the channel size: become 128x16x16
        self.conv3 = conv_block(128, 256, pool=True)
        # go to 256 with pool: 256 x8 x8
        self.conv4 = conv_block(256, 512, pool=True)
        # go to 512 x4 x4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        # not change anything: have 512x4 x4
        self.classifier = nn.Sequential(nn.MaxPool2d(4),  # 512 x1 x1
                                        nn.Flatten(),  # 512 flatten to vecotr of size 512
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))  # put in linear layer and get the 10 classes
        # softmax makes use have only positive values
        # now have maxpool 2d, instead of pooling 4x4 slices, will will pool 4x4: for each channel; pick the largest value

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out  # adding back the output after passing thorugh residual
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # add the previous output
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    data_dir = './data/cifar10'
    if False:
        download_url(dataset_url, '.')  # extrat the data
    if False:
        # Extract from archive
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')

        # Look into the data directory

        print(os.listdir(data_dir))
        classes = os.listdir(data_dir + "/train")
        print(classes)

    # not need the labels bc in the folders we have that images for that class
    print(len(os.listdir(data_dir + '/train/horse')))

    # now have the images, but we need to create pytorch datasets
    """
    at the very end: use the test set as the validation set: not set asside 10% for validation, use the entire test set as teh validation set
    we can do this bc we have the labels for the dataset
    once have validation set and have certain hyperparameters on a fixed validation
    At the end you want to retrain the model on all of the available data
    
    channel-wise data normalization
    we have r-g-b channels: each channel is brought into 0-1 as pytorch tensors
    in the dataset, the reds can have a large range while blues can have a smaller range
    bc all of these are multiplied: when start training: change in red can have higher impact than change in other channels
    get the average of all the pixels of all images in the red channel and then for each image we normalize the channels by subtractive the mean and deviding by sd
    all 3 channels will fall in range -1 to 1 and the mean will be 0
    each channel will vary around -1 to 1
    this prevents 1 channel from disproportionatly telling other channels what happens
    you should always do channel=wwise normalization
    
    randomization data augmentation: we noticed overfitting
    data augmentation: put a padding of 4 pixels around image, and then take random crp of 32 x 32 pixels
    the subject will nto be sampled. this will be applied randomily in each batch
    wtih 50% probability, we will flip the image
    the model will see slightly different images andtherefore it can do it better
    should not do this with validation iamges
    validation set used for evaluation
    the results can be different
    """

    # Data transforms (normalization & data augmentation)
    # these are the (means, standard deviations) for the channels
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # need to use the torch transforms to normalize the image
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                             # add in a padding of 4 and then call the padding mode = reflect
                             # this will fill the padding with the reflection of the pixels around it
                             tt.RandomHorizontalFlip(),  # do a random fliP; taes in a probability with default = 0.5
                             # tt.RandomRotate
                             # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
                             # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                             tt.ToTensor(),
                             tt.Normalize(*stats, inplace=True)])  # do in place to modify existing tensors
    # need to normalize images for the trainign and validation data set
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    # PyTorch datasets
    train_ds = ImageFolder(data_dir + '/train', train_tfms)  # use all set for training
    valid_ds = ImageFolder(data_dir + '/test', valid_tfms)  # use training set for validation

    # create data loaders for having images in batches
    batch_size = 400  # this lets u use for gpu memory and therefore it will go faster
    # if increase, the trianing time can increase to not hurt trianing of data

    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3,
                          pin_memory=True)  # shuffle the training data
    # num workers; use the 3 other cores and backgroudn threads to operate on teh data
    # pin memory is another optimization
    valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers=3, pin_memory=True)
    # use 2x the batch size bc not use the gradient calculation
    # for validation you want to keep the dataset fixed
    if False:
        show_batch(train_dl)
    # rgb are centered at 0, the nubmers are sharper, and it is easier to classify them
    device = get_default_device()
    print(device)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    # moved the data onto the device
    """
    Model with residual blocks and batch normalization
    normally when have cnn, pass it thorugh cnn, apply input and continue
    
    in residual block and after the first 1/2 leayers, we take the input and add it to the output of the first layer
    
    """
    simple_resnet = to_device(SimpleResidualBlock(), device)

    for images, labels in train_dl:
        out = simple_resnet(images)
        print(out.shape)
        break

    del simple_resnet, images, labels
    """
    this improves model's performance
    initial model has to go through the weights and the output
    data can only go through the layer
    residual blocks give an input to the shorter part to the output
    can go through the layers
    rather than transforming the input to the output, the convolution layers only find waht is the different betweenthe input and ouptu
    H(x) = R(x) +x 
    you then get the residual
    R(x) = output - input = H(x) - x
    
    another technique:
    batch normalization
    similar to data normalization
    some outputs might become to large and others can become too small
    after layer you get activation/ output
    one particular activation after a layer can become too large, while others can be normal
    this will start affecting the rest of the things disproportionally
    batch normalization: take output of a layer and does normalizaiton there
    (h_ij - myu_j ) / O_j
    have gamma and beta parameters that work
    normalize outputs as we normalize input layer

    also has dropout We will use resnet( archtecture: have 9 layers
    input -> conv (increase num output channels from 3 ->64), batch norm, relu
    conv (increase by 2) , bn, relu, max pool (take 32 and shrink to 16x16)
    Then ahve one path whih is conv, bn, relu x2 and the other which is the residual
    
    take output from pooling layer and add it back
    at the end, we pool, flatten to layer, then take 10 outputs and then have the predictions
    """

    # training the model;
    """
    instead of using fixed learnign rate: change the learning rate after evry batch of training
    init, you want to have a large learning rate and do it in several steps
    
    we have different policies: have 1cycle policy: best way to increase learning rate
    start with low learning rate, increasing it to a large one for 30-40% of the epochs
    then you start decreasing it all the way back for the low value
    
    init have small learnign rate: find way in which decrease the value
    the nfor the remaining part, you can start taking smaller steps and this will converge to minimum
    
    finally have a very small boost at a very small learnign rate
    
    weight decay: first normalize the inputs, then normalize the outputs from every layers, then weight decay: any weights should not become too large
    the weight might disprop affect loss and gradients
    add a term to the loss: we add weight_factor * sum (w^2) and therefore want to keep the weights low
    
    final: gradinet clipping: make sure that none of the gradients gets too large: gradients get large with back propagation 
    
    if have a very steep slope and have a large gradient, the model might jump to a different location
    dont want to use ver ylarge gradients: will use gradient and if its large than a value, you will threshold it
    clip it to -0.1 to +0.1
    """
    # instead of using sgd, use adam optimizer
    # sgb only subtract amount for each gradient
    # also has other optimzier in the torch.nn
    # can track the momentum from previous batches so that you dont boucne arou d a lot
    # then have adaptive learing rates: each learning rate have factor multipleied by learning rate

    # adam does a better job than sgd
    epochs = 8
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

    """
    Epoch [0], last_lr: 0.00393, train_loss: 1.5018, val_loss: 2.3667, val_acc: 0.4839
Epoch [1], last_lr: 0.00935, train_loss: 1.0881, val_loss: 1.8056, val_acc: 0.5457
Epoch [2], last_lr: 0.00972, train_loss: 0.8391, val_loss: 1.1632, val_acc: 0.6179
Epoch [3], last_lr: 0.00812, train_loss: 0.6418, val_loss: 0.7527, val_acc: 0.7377
Epoch [4], last_lr: 0.00556, train_loss: 0.5059, val_loss: 0.5230, val_acc: 0.8203
Epoch [5], last_lr: 0.00283, train_loss: 0.4056, val_loss: 0.4061, val_acc: 0.8620
Epoch [6], last_lr: 0.00077, train_loss: 0.2901, val_loss: 0.3132, val_acc: 0.8925
Epoch [7], last_lr: 0.00000, train_loss: 0.2227, val_loss: 0.2730, val_acc: 0.9055
CPU times: user 37.5 s, sys: 16 s, total: 53.5 s
Wall time: 4min 24s
    
    """
    # when learnign rate is at max, there is bouncing around

    #
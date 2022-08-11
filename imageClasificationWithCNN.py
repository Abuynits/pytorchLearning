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


# make training better and better
@torch.no_grad()  # tel when function being used, dont track the gradients
def evaluate(model, val_loader):
    model.eval()  # have certain layers only turn on in trianing and hidden in evaluation:
    # tell the model is in training mode and to not use those levels
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()  # tell the model to use the training labels
        train_losses = []  # now we will also finding the training loss
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        # take the values for each loss and get the output from it
        model.epoch_end(epoch, result)
        history.append(result)
    return history


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


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


input_size = 784
hidden_size = 32  # this is the power: you can change this value
num_classes = 10


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
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze[0],get_default_device())

    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


"""
image cnn notes:
1 channel; the term filter and kernel are interchaneable, general case: they are different
eah filter is a colelction of kernels witehre there being one kernel for every input channel to the layer and each kernel being unique

process:
for each input kernel, the filter process them and produces an output for each
then each of the per-channel processed versions are then summed together to form one channel
then add a bias term to the output kernel

"""


class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # sequential allows you to pass the inputs from the previous outputs
        # want to increase the number of channels as you decrease the size of the image
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # images are 3 channels of 32: take them and convert it to 32 channels
            # take 3 channel of 32x32 and convert it to 32 channels
            # image size stays the smame
            nn.ReLU(),  # activation function
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # tale 32 channels and convert to 64
            nn.ReLU(),  # the conv2d will reduce the mage size in half, but therefore
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16, previously they were 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # increase the channels again
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # keep number of channels the same
            # each time add linear layer, you increase the power of the model ( makes it go deeper)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            # takes the 256 x 4 x 4 and converts them to a single layer
            nn.Flatten(),
            # each image is down to a single vector of size 256*4*4
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        # then bring it down to 10 outputs

    def forward(self, xb):
        # seq model and pass in the batch of data
        return self.network(xb)


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
    """
    will get:
    images.shape: torch.Size([128, 3, 32, 32])
    out.shape: torch.Size([128, 8, 16, 16])
    
    we go 3 channel of 32 x 32 pixels to 8 channel of 16x16 pixels
    To each one of the 3 channels, we apply a kernel, and then the outputs are added together
    
    we decrease the size of the image (feature map), but increase the number of channels: where logic takes place
    
    """

    # continue adding conv and maxpool: need non linear actiavtion after convolution
    # have input image ( decrease the size of the image but increase the number of channels)
    # tjuhen do pooling which wiill decrease the image size
    # will continue to do this by alternative convolution layer and poolinng until you will geta fully connected layer
    # have a feature map of size 1x1, and then we flatten it out with a fully connected layer
    model = Cifar10CnnModel()
    for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]', out[0])
        break

    # use 10 epocs
    num_epochs = 10
    # this is a new optimizer: stochastic gradient descent subtracts quantity
    # adam has momentum, weight decay: adam does better for plain stochastic gradinet descent
    # call torch.optim to see all hyper parameters
    opt_func = torch.optim.Adam
    lr = 0.001

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print(history)
    plot_accuracies(history)
    # see that the validation loss starts to rise
    # see the trianing loss continues to decrease while the validation loss starts to licnrease
    # over fitting: machine learning problems give bad results on real data
    # as model grow bigger and bigger, the model has power and use power in long way
    # model start to learn patterns unique to the training data
    # this is what happens:
    # see that htere are 55k images in trainings: if memorize 5-10 images, can increase the pattern
    # model is overfitting to trainign data: the validation set say that somethign is wrong with the data
    # the model starts to specialize for that trainign data set

    # how avoid over fitting:
    # need to gather more training data:
    # need to go through roe data for training
    # can also generate data via data augmentation
    # crop, zoom in , zoom out: same set of 45k images, but see something different
    # batch normalization and dropout (from a feature map, randomly set certain number of outputs to 0)
    # can do nn.Dropout() from the amount of things that you want to drop out.
    # nn.Dropout(0.4) -> good technique to set things to 0
    # should only be applied during training
    # during validation, we should nto drop i
    #
    img, label = test_dataset[0]
    # plt.imshow(img.permute(1, 2, 0))
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
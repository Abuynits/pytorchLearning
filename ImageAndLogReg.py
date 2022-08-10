# linear regression model: each output is a sum of linear weights
# need to convert to tensors to process the numpy arrays
# talk about image classification for images with pytorch

# MNIST handwritten digit database: consists of 28px by 28px grey scale images of handwritten digits 0 to 9
#

# begin by exploring the data:
import torch
import torchvision  # utils for working with image data, and hyperclasses to download and import popular datasets
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as tranforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# download the training dataset
dataset = MNIST(root="./trainingData",
                download=True)  # give the location of the download, and tell it to download the data

# not worry about loading the files rn: have datset object
print(dataset)
# can also run the len on the datset:
len(dataset)

# also have test data:
test_dataset = MNIST(root="./trainingData",
                     train=False)  # report test data to report results and not actually download it

print(test_dataset)

# look at element from dataset:
print(dataset[0])  # the inputs that we will pass in to the tensor dataset class
# has size 28x28 and a number 5 as the label for the image

# supervised learning modeL: have the labels: need to process the image and find the label for the image
# this is object of pil library: part of imaging library Pillow

image, label = dataset[10]
plt.imshow(image, cmap='gray')  # plot the image and tell that its greyscale
print('Label', label)
# pytorch not know how to work with images: use transforms to convert images to tensors
# dont set download to true, as already downloaded
dataset = MNIST(root="./trainingData", train=True, transform=tranforms.ToTensor())
# set transform bc tell that when data is loaded, we will look at the tensor of the data
img_tensor, label = dataset[5]
print(img_tensor.shape, label)  # 1 by 28 x 28
# first dim track color channel: only 1 channel bc its greyscale
# otherwise, we will have rgn channels

# look at some sample values inthe tensor:
print(img_tensor[:, 10:15, 10:15])  # look at the first channel and we will look at the data in teh box
print(torch.max(img_tensor), torch.min(img_tensor))  # print the min and max of the tensor
# can also plot this portion of the tensor:
plt.imshow(img_tensor[0, 10:15, 10:15], cmap='gray')
# each of image tensors has 1x28x28

# TRAINING and VALIDATION
# 1: traning set: used to train the model (compute and adjust the wieghts)
# 2: validation set: used toevaluate the model while training adjust hyuperparameters (learnign rate, etc)
# 3: test set: used to compare different models or different types of modeling approaches
# have 60k training and 10k testing
# test set is standardized so that diff ppl can report the results
# we need to split the training set from 60k to validation
train_ds, val_ds = random_split(dataset, [50000, 10000])
print(len(train_ds), len(val_ds))

# need to choose random sample for creating validation set: bc training data is ordere dby the target labels;
# if pick last 20%, we will only have 8 and 9, and not other numbers

# need data to be in batches
batch_size = 128
train_loader = DataLoader(val_ds, batch_size,
                          shuffle=True)  # each time create batches, do a shuffle of the data (get different batches)
val_loader = DataLoader(val_ds,
                        batch_size)  # not shuffle it: not training on it: not use for computing gradients: just for reporting/ evaluating the model

# MODEL:

# logistic regression model is almost identical to a linear regression mode: have weights and bias and the output is pred = x @ w.t() + b
# we can also do linear regression and use nn.Linear to create teh model instead of defining and initializing the matrices
# need to flatten the training example to vea vector, from 1x28x28 to a size of 874 before passing into the model
# the output for each image is a vector of size 10 with each element of the vector signifying the probability a particualr targe label ie 0 to 9
# have 784 inputs ( each individual pixels ), and have 10 ouptuts: for each image show the probability that the image represents a target label


input_size = 28 * 28
num_classes = 10

# logistic regression model:
model = nn.Linear(input_size, num_classes)

# a lot larger than our previous model in terms of number of parameters:
# take a look at weights and bias:
print(model.weight.shape)
print(model.bias.shape)


# use model for predictions
# for images, labels in train_loader:
#     print(labels)
#     print(images.shape)
#     output = model(images)
#     # first dimension is the batch of 128 images
#     # each element is a 1x28x28 tensor
#     # size_missmatch error: model expects an input vector of size 784: we gave it an entire input tensor (3d)
#     # we expect a batch of vectors
#     # need to flatten images to vectors:
#     # define own custon model:
#     break


# print labels and shape of batch and pass images into the model
class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        # the constructor
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # executed when pass inputs to the model
        # can also flatten this way:
        xb = xb.view(xb.size(0), -1)
        # xb = xb.reshape(-1,
        #               784)  # use same data but lay out differently: retain batch dimension, but rest can be flattened
        # puting a -1: telling pytorch for it to figure out what it should be; we are not hard coding the batch size
        out = self.linear1(xb)  # pass in the flatten out vectors to the input and it gets the output
        # can do self.linear1= nn.Linear(28x28,16) # have 16 values now
        # can do out=self.linear2(16,10) # take 16 values and tur ninto 10 values
        # out = self.linear1(xb)
        # out = self.linear2(out)
        # all matrix multiplication
        # issue: when expand self.lienar2(self.linear1)
        # we do self.lienar2(xb@w1 +b)
        # and therefore (xb @w1 + b1) @ w2 +b2 = xb @ (w1 @ w2) + xb @ ( b1 @ w1) + b2
        # take the xb out:
        # xb @ ( (w1(w2) + ( b1 @ w1 )) + b2 = xb @ w3 + b2
        # we not achieve anything from this
        # we just rearanged the packets and its as good as only multiplying thme together

        out = F.relu(out)
        # in betwee thne two linear levels
        # now you can no longer simplify the equation
        # we will apply relu to the output
        # first layer will transform the input matrix to a ban intermediate output matrix: batch size x hidden_size
        # hidden size is a preconfigured paramter eg 32 or 64
        # then pass the intermediate outputs throug ha non-linear activation function
        # then the result of the activation function is passed to the ouptu layer which gives us the size that we want
        # we use the Rectified Linear Unit or ReLU function; relux (x) = max(0,x)
        #
        out = self.linear2(out)

        return out

    # from the batch, get images and training out

    def training_step(self, batch):
        images, labels = batch
        # self(images) passes in the images to the model and calls the forward function
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss Trianign step only returns the loss
        return loss

    # takes images nad lables and gives out an output
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy (for particualr batch of training data)
        return {'val_loss': loss.detach(), 'val_acc': acc}  # detach drops references to all things

    def validation_epoch_end(self, outputs):  # end of validation
        batch_losses = [x['val_loss'] for x in
                        outputs]  # have list of outputs and extract all loses and accuracies nad get the avergae value
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}  # return the metrics from the batches

    # end of every epoch, log epoch num, validation loss, value accuracy
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


input_size = 784
hidden_size = 32  # this is the power: you can change this value
num_classes = 10

for param in model.parameters():
    print(param)

model = MnistModel(input_size, hidden_size, num_classes)
for images, labels in train_loader:
    output = model(images)
    loss = F.cross_entropy(output, labels)
    print('loss: ', loss.item())
    break

print('outputs.shape : ', output.shape)
print('Sample outputs :\n', output[:2].data)


# HOW TO USE GPU's
# check if have nvidia CUDA drivers installed with torch.cuda.is_available

# need to have nvidia gpu for training
# need to have nvidia drivers installed on gpu: nvidia cuda drivers
# gpu has own ram and cpu available for it

def get_training_device():
    if torch.cuda.is_available():
        # the device checks if have cuda ( has nvidia gpu)
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_training_device()
print(device)


# Can also move data and model to a device:
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        # operate on tensors: have a to device on each list of tensor
        return [to_device(x, device) for x in data]
    # .to: accepts a device and the data is coppied from cpu to the gpu
    # non_blocking: does not block you from executing the next steps
    return data.to(device, non_blocking=True)


# this will accept either a model or a tensor bc you are claling the to_device on each tensor in the data
for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images)
    # will show that it is on the device (cpu in my case)
    break


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# devine a Device Data loader class to wrap exsiting data loaders and move data to the selected device as batches are accessed
# dont need to extend existing class, just need __iter__ method to retrieve batches of data nad a __len__ method to get number of batches
class DeviceDataLoader():
    def __init__(self, dl, device):
        # take the data loader and the device
        self.dl = dl
        self.device = device

    def __iter__(self):
        # have to call the yield method: yield batches of data fro mthe iterator
        for b in self.dl:
            # yield: in the first loop, the iter method is called, and once have yield, the method returns and pauses ther
            # next loop: gets the next batch of data
            # yield works with generators: like iterators: you can only iterate over once - not store all valeus in memory ,generate on the fly ( use yield)
            # iterables ( lists ): store the value in memory ( use return )
            yield to_device(b, self.device)

    def __len__(self):
        # need to return the length of batches
        return len(self.dl)


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
# why moove in batches: not all of data can fit on GPU: need to have all of the model nad all of the data on the gpu
for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb', yb)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)  # creates optimization functions ( uses torch.optim.sgd by default
    history = []  # for recording epoch-wise results

    for epoch in range(epochs):  # iterate over epoch

        # Training Phase
        for batch in train_loader:  # iterate ove rbatches in data loader
            loss = model.training_step(batch)  # processes the batches
            loss.backward()  # call to compute gradient descent now have 2 layers: the loss will put last gradients linear2
            # and then bc the inputs to 2nd are from the outputs from thee first 21, it will go back and calcualte the ones from the first
            # this is called back propagation
            optimizer.step()  # take gradients and subtracts the gradients, and multiplying with the learning rate
            optimizer.zero_grad()  # reset the gradients back to 0

        # Validation phase
        result = evaluate(model, val_loader)  # evaluate model with validation data loader
        model.epoch_end(epoch, result)  # call it to end the epoch and print the result
        history.append(result)  # collect the history of the validation loss and accuracy after every epoch

    return history


def evaluate(model, val_loader):
    # take model and validation loader: for each batch in the loader, it calls the validation step and gets back validation loss and accuracy in object
    # this is combined in a list of object. then call the end to the validation epoch
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# need to make sure the data and the model's parameters (weights and biases) are oon the same device (CPU or GPU)
model = MnistModel(input_size, hidden_size, num_classes)
print(to_device(model,device))
# see how the model performs on the validation set with the intial set of weights and biases:
history = [evaluate(model,val_loader)] # with jsut initial set weights and biases
history += fit(5,0.5,model,train_loader,val_loader) # using a higher learning rate that depends on the model
print(history)

exit(1)
# need to output probabilities and elements of the output row have to lie between 0 and 1

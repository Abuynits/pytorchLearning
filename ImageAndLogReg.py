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
    def __init__(self, input_size, num_classes):
        # the constructor
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        # executed when pass inputs to the model
        xb = xb.reshape(-1,
                        784)  # use same data but lay out differently: retain batch dimension, but rest can be flattened
        # puting a -1: telling pytorch for it to figure out what it should be; we are not hard coding the batch size
        out = self.linear(xb)  # pass in the flatten out vectors to the input and it gets the output
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
        return {'val_loss': loss, 'val_acc': acc}

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


model = MnistModel(input_size, num_classes)
for images, labels in train_loader:
    print('output.shape:', images.shape)
    output = model(images)
    break

print('outputs.shape : ', output.shape)
print('Sample outputs :\n', output[:2].data)

# need to output probabilities and elements of the output row have to lie between 0 and 1

# will use the softmax function
print(torch.exp(output[0]))  # raise each outut to e to the power of the value in that output
exps = torch.exp(output[0])
# all numbers are now positive
# still between 0 and 1, image can be of certain class

# can take the exponenets and devide them by the sum of the exponents (get probls)
probs = exps / torch.sum(exps)
print(probs)
# this is called the softmax function

# you have the functional package and requires package and requires to use a specify a dimensiosn along which the softmax
# this also forces the model to choose a specific number value
probs = F.softmax(output, dim=1)
# apply on the first dimension bc 0 dimension is on the batches
# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())

# got 128 sets of probabilities each: pick the element of the highest value for the predicted value
# if do torch.max, you will get the highes value and its index
max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)


# see that the probabilities are not that high
# compares bad with the labels as the model is randomized
# need to evaluate how model perform: check what percentage matched in the model
# do element wise comparison with the labels
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


acc = accuracy(output, labels)
print(acc)
# acc help us evaluate the model
# cannot use as loss function for optimization
# it is not a differentiable function
# does not take into account the actual probabilities predicted by the model so it cant provide sufficient feedback
# if do 1 with 12% - not look at what it predicted for the correct value
# if model still predictive 1 but has higher probability for 3. accuracy not capture that. not look how do for correct value
# cannot give feedback to training process

# commonly used: cross entropy: have certain predictions, and have the actual labels
# ignore the predictions for anything but the actual label, just take the prediction, and take a logorithm of 0.5
# now the model should be as close to 1, bc log of 1 is 0 where loss is 0. if predict closs to 0, log(0.02) is a large negative value

# then apply a negative on top of the log that will make it a large positive value
# when close to it, you will have a low loss,
# this is differentiable; treat the label not as a number, convert it to a vector that has all 0's but at the wanted position

# take log of all prediction and do dot product with the actual target vector

loss_fn = F.cross_entropy

loss = loss_fn(output, labels)  # pass in the outputs as it has softmax built into it ( perform cross entropy afterward)
# takes the labels which are numbers and converts it to hot num encoded vectors ( all 0's and 1 one)
print(loss)


# bc this is the negative log of the predicted probability of the correct label averaged over all training samples
# you can look at the number 2.23 as e^-2.23 which is 0.1 of correct data

# TRAINING THE MODEL

# # identical to linear regression but have a validation phase.
# for epoch in range(num_epochs):
#     # Training phase
#     for batch in train_loader:
#          # Generate predictions
#          # Calculate loss
#          # Compute gradients
#          # Update weights
#          # Reset gradients
#
#     # Validation phase
#     for batch in val_loader:
#          # Generate predictions
#          # Calculate loss
#          # Calculate metrics (accuracy etc.)
# # Calculate average validation loss & metrics
#
# # Log epoch, loss & metrics for inspection

# we can extend the model class to hold more functions and have the problem specific parts:

# introduce fit and evaluation function:
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)  # creates optimization functions ( uses torch.optim.sgd by default
    history = []  # for recording epoch-wise results

    for epoch in range(epochs):  # iterate over epoch

        # Training Phase
        for batch in train_loader:  # iterate ove rbatches in data loader
            loss = model.training_step(batch)  # call the training step and get back the loss
            loss.backward()  # call to compute gradient descent
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


# can run on validation data loader;
res0 = evaluate(model, val_loader)

history1 = fit(5, 0.001, model, train_loader, val_loader)

# logistic regression hits the limits bc we assume a linear relationship, but it cannot be linear
# this might not be a fair assumption to do

# TESTING with individual images:

# definie testing dataset:
test_dataset = MNIST(root="./trainingData",
                     train=False, transform=tranforms.ToTensor())
img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('shape:', img.shape)
print('label:', label)


# this is a sample where do prediction
def predict_image(img, model):
    # we already accept a batch: call unsqueeeze method to add another dimension to make a batch of 1 image
    xb = img.unsqueeze(0)
    yb = model(xb)  # the prediction is just hte image of the probabilityy
    _, preds = torch.max(yb, dim=1)  # just return the highest probability
    return preds[0].item()


print('label:', label, ', predicted: ', predict_image(img, model))

# look at overall loss and accuracy of the model on the test set:
test_loader = DataLoader(test_dataset,batch_size=256)
# can also do: DataLoader(train_ds, batch_size, shuffle=True, num+workers=4, pin_memory=True)
# the num_workers are for parallelization
# the pin memory is for using memory
result = evaluate(model, test_loader)
print(result) # need to report test accuracy in the findings


# SAVING the model
# all training weights are in the state_dict()
torch.save(model.state_dict(), 'mnist-logistic.pth')
# to load it back:
model2 = MnistModel(input_size,num_classes)
model2.load_state_dict(torch.load("mnist-logistic.pth"))
print(model2.state_dict()) # dont need to train them again

# WHERE MAKE MODEL A neural network with 1 hidden layer



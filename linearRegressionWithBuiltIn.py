import torch.nn as nn
import torch
# allows access to rows from inputs and targets as tuples and provides standard api for working with many diff types of datasets
from torch.utils.data import TensorDataset
# split data into batches of a predfined size during training
from torch.utils.data import DataLoader
# import nn.functional
# package containing the loss functions and other stuff for evaluation
import torch.nn.functional as F

import numpy as np

# used for nn
inputs = np.array([[73., 67., 43.]])
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# take 15 elements and use 5 elments at a time
train_ds = TensorDataset(inputs, targets)
# print(train_ds[0:3])  # get first 3 inputs and first 3 targets
# need batches bc you can deal with a lot of data and it cannot fit into memory / make modeling process very slow
# define data loader:
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)  # tell it to shuffle up the rows before using the data
# the batches are created with different set of elements
for xb, yb in train_dl:
    # for a batch of x, y in the batches, print them
    print('batch:')
    print(xb)
    print(yb)

# ====nn.Linear====
# Instead of init weights and biases manually, define model using nn.Linear
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)
# also have a helpful .parameters method which returns list containing all weights and bias amtrices present in the mdoel
# for linear reg model, we have 1 weight matrix and one bias matrix

# paramters: IMP: will show all param that are present in the model
print(list((model.parameters())))

# generate predictions:
preds = model(inputs)

# LOSS FUNCTION
loss_fn=F.mse_loss
loss = loss_fn(model(inputs), targets)

print(loss)

# OPTIMIZER
# instead of manually manipulating weights and biases, we use otim.SGD (SGD means stochastic gradient descent)
# the data is in batches, isstead of training on all of the data, bc stochastic menas samples are sleected in batches instead of as a single group
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# need the parameters from the model to know hwat to pudate
# also need the lr as the learnign rate

# now ready to train modeL
def fit(num_epochs, model, loss_fn, opt,train_dl):
    # repeat for given number of epochs
    for epoch in range(num_epochs):
        # train with batches of data
        # say that
        for xb, yb in train_dl:
            # 1 generate predicitons
            pred = model(xb)
            # 2 calculate loss: with the targets and predictions
            loss = loss_fn(pred, yb)
            #3 compute gradients with respect to the weights
            loss.backward()
            # update paramters using gradients: multiplies gradient with learning rate and subtracts it
            opt.step()
            # reset the gradinets to zero
            opt.zero_grad()
        # print thr progress
        if (epoch+1)%10 ==0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(100,model,loss_fn,opt,train_dl)

# generate predictions:
preds = model(inputs)
print(preds)
print(targets)

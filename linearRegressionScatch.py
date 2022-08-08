# linear regression: creatae model to predict crop yield in a region from temperation and rainfall
import numpy as np
import torch


# set the grad set to true as we will calculate derivatives with respect to the weights and biases
# model: performs matrix multiplication of inputs and the wieghts transform
# inputs * weights + bias
def model(x):
    # model is simply a function that takes in input matrix/tensor and then uses @ operator
    # @ operator show matrix multiplication in python
    # need to transpose the weights as they are currently 2x3, while we need them to be in 3x2
    # then we need to add in bias - takes in input x, multiples the weihts and adds in the bias
    return x @ weights.t() + bias


def mse(t1, t2):
    # compare the model's predictions with the actual targets:
    dif = (t1 - t2)
    dif = dif * dif  # this is an element wise multiplication: now only dealing with positive values
    # want to get a  single number to evaluate the model; add up all of the elements, and get average
    ave = torch.sum(dif * dif) / dif.numel()  # dif.numel() give the number of elements in the tensor
    # this is the mean square error
    # 1. calculate the diff between two matrices
    # 2. square all elements of the difference to remove negative alus
    # 3. calculate the average of the elements in the resulting matrix
    # this is the mean squared eror (MSE)
    return ave


# in inear regression; assume each target varaible is estimated to a weighted sum of the input variables offset by some constant knownw as a bias
# yield-apple: wl1*temp +wl2*rainfall +wl3*humidity + b3
# you need a bias that is not dependent on any of the input variables
# this means that the yield of apples is a function of humidity, rainfall and temperature

# learning part of linear regression: called linear bc linear sum, regression bc output we predict is a continuous number

# learning part of lin reg is to figure out a set of wieghts wl1, wl2  and by looking at the training data to make accurate predictions from new data

# start with random weights, and adjust the weights many times to make slightly better predictions and use gradient descent for it

# weights: the number that are multiplied with numbers to get the yield of numbers

# create the training data
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
# create the output data for the training for apples and oranges
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')
# we have seperated the input and target bariables bc we'll operate on them seperately

# normally start with csv file, and will use pandas to read the csv file
# set them to float 32 bc not make prediction in ints but in floating point numbers

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# now create the linear regression model from scratch: have 3 weights for apples/oranges and 1 bias for each
# the wights for a matrix wile the variables form another vector and the bias is just added on
weights = torch.randn(2, 3,
                      requires_grad=True)  # randn: create the matrix of the dimension and fills in the tensor filled with random numbers of mean 0 and standard deviation of 1

bias = torch.randn(2, requires_grad=True)

print(weights)
print(bias)

# get predictions for the model
preds = model(inputs)
print(preds)

# compare with the targets:
print(targets)

# LOSS FUNCTION
# way to evaluate how wellmodel is performing
loss = mse(preds, targets)
print(loss)
# interpret:
# on average, each element in the prediction differs from the actually target by about the square root of the loss
# called the loss bc it indicates how bad the model is at predicting the targe tvariables

# COMPUTE GRADIENTS
# use gradient descent for improving the modesl
# the loss is a quadratic function of the wieghts and the biases bc we have a squaring in the function
# the lower the loss, the closer the model is to the truth

# compute derivatives ( gradients)
# not set requires_grad to true for inputs or targets, only for weights and biases
loss.backward()
print(weights)
print(weights.grad)
# the first element is the partial derivative of the loss with respect to the loss function

# loss is quadratic function of wiehgts and biases: we need to find set of weights where loss is the lowest
# as we chagne a wieght element, the loss and the gradient will change

# as the wieght becomes too large/ small, the weight will become too large as you increase or decrease it
# the derivative represents the slope of the loss curve
# if the derivative is positive: increasing weights = increase loss
# if the derivative is negative, the slope is decreasing: decreasing weights will increase loss

# we move in direction oposite of the derivative
# this is basis for gradient descent optimization function
# we need to reset the gradients back to 0 on the gradients:

# need to set the grad to 0 after do a calcualation, but the weights will stay the saem
# weights.grad.zero_()
# bias.grad.zero_()
print(weights.grad)
print(bias.grad)
# OPTIMIZATION algorithm
"""
reproduce the loss and improve model with gradient descent optimization algorith
1. genearate predictions from the model ( take inputs, put into model)
preds=model(inputs)
print(preds)
2. calculate the loss (via mse)
loss = mse(preds, targets)
print(loss)
3. compute gradients with respect to the the weights and biases w.grad and b.grad
loss.backward()
print(w.grad)
print(b.grand) 
4. adjust the weights by subtracting a small quantity proportional to the gradient (most important)
with torch.no_grad():
    # dont want torch to not track the calculations to work with the gradients
    weights -= weights.grad * 1e-5
    bias -= bias.grad * 1e-5
    # reset the gradients after operating on them
    weights.grad.zero_()
    bias.grad.zero_()
5. Reset the gradients to zero
"""

# adjust weights and reset gradients:
with torch.no_grad():
    # dont want torch to not track the calculations to work with the gradients so set no_grad()
    # this is the update step (subtract a small proportion of the gradients form the wiehgts and biases
    # always subtract bc of diff equ and the way in which we want to move
    weights -= weights.grad * 0.000000001
    bias -= bias.grad * 0.000000001
    # if gradient is negative, increasing the value will decrase the loss
    # if gradient is positive, decreasing the value will icrease the decrease the loss

    # reset the gradients after operating on them

    # reset the weights and biases bc we are done with them
    weights.grad.zero_()
    bias.grad.zero_()

    # 1e-5 is called the learning rate

    # the problem: the gradient can be quite large: if just subtract the gradient, we will move far away: we subtract a small quantity proportionate to the gradient
    # keep taking small takes until we get to a point where the loss is teh lowest
    # set require grad to true for the wieghts and biases, not the inputs or outputs

print("here")
print(weights)
print(bias)
# with the new weights, we shold have a lower loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

# TRAINING FOR MULTIPLE EPOCHS
# to reduce loss further, we repeat process of adjusting the wieghts and biases multiple times
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        weights -= weights.grad * 0.00000001
        bias -= bias.grad * 0.00000001
        bias.grad.zero_()
        weights.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print(loss)
print(weights)
print(bias)

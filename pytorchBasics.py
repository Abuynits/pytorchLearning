"""
ML: take data and answers and define rules
normal programming: take the rules and data nad gives naswers
Ml: once have rules for the data, we can use it in the real work with new datapoints
jupiter notebooks: access them online
"""
# tensors:
# pytorch is lib for processing tensors: tensor is a number ,vector, matrix or any n-dim array
import torch
import numpy as np

# 4. is shorthand for 4.0 - indicate to python and pytoch that you want to createa floating pont number
t1 = torch.tensor(4.)
# can check the type of the tensor via
print(t1.dtype)
# can also create vectors:
t2 = torch.tensor([1., 2, 3, 4])
# all numbers will be converted to float points: tensors require the same datatype for all tensors
print(t2)

# here we have a 3d tensor
# tensor require the data to be in a regular shape: every element in the outer list need to have same number of elements
# every element in a list, you need to have the same number of elements
t3 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[16, 9, 13],
     [13, 14, 55.]]])
# can inspect the shape of the data:
# if not in a regular shape, the tensor will not be created
print(t3.shape)

# TENSOR OPERATIONS
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

print(x, w, b)

# do as if they are numbers - dont have to be scalars of any shape
y = w * x + b

print(y)
# helps bc you can compute derivative on any result with respect to inputs of any inputs
# do it with respect to the variables

# compute derivatives:
y.backward()  # look back at inputs to tensor y, and looks at the inputs that have require_grad set to true
# whenever this happens, it inserts the gradient to y with respect to the inputs
# dy/dw will be present in w.grad
# dy/dx will not be present in x.grad
# dy/db will be present in b.grad
print('dy/dw:', w.grad)  # derivative of y=w*x+b with respect to w is x
print('dy/dx:', x.grad)  # derivative dne bc x not specified as requires gradient
print('dy/db:', b.grad)  # derivative of y with respect to B is just 1

# can do more complex arithmetic operations
# .grad stands for gradient which is used when dealing with matricies and tensors

# what is difference between tensor and matrix
# matrix: special type of tensor that have 2 dimensions
# tensor can be a 3d array with regular shape, a number etc

# INTEROPERABILITY with NUMPY
# popular open source library used for math and science
# help with operations on large arrays
# have matplotlib for ploting, opencv for images, pandas for files I/O and data analysis

nx = np.array([[1, 2], [3, 4.]])
# convert the numpy array:
ny = torch.from_numpy(nx)
# check that they have the same type:
print(nx.dtype, ny.dtype)

# convert a torch tensor back to numpy:
nz = ny.numpy()
print(nz)
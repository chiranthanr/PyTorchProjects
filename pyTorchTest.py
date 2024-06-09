import torch
import pandas as pd
import numpy as np
print(torch.__version__)
'''
Scalar is has no dimensions. In the case the scalar variable is a tensor with integer
value of 7.
'''
scalar = torch.tensor(7)
print (scalar.item())

'''
A Vector can have any number of dimensions. In the case the vector variable is a tensor
with multiple dimensions.
Note: A Matrix is a tensor with 2 dimensions.
'''
vector = torch.tensor([[7,7],[8,8]])
print(vector)
print(vector.shape)

'''
Random tensors are used in deep learning. Neural networks start with tensors full of random numbers,
then look at data and then update the random numbers, then again look at data and so on.
'''
random_tensor = torch.rand(3,4)
print(random_tensor)
'''
For images, the random tensor used is of shape (224,224,3) which stands for height, width and
 color channel respectively.
'''
random_tensor_image = torch.rand(224,224,3)
print(random_tensor_image)
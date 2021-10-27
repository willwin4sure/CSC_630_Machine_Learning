# Standard imports

import torch
import numpy as np
import os

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''Tensors are a data structure similar to arrays/matrices. 
They can be created directly from data in nested lists, np arrays, other tensors, or random/constant values'''

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
# print(x_data)

x_np_array = np.array(data)
x_np = torch.from_numpy(x_np_array)
# print(x_np)

x_ones = torch.ones_like(x_data)
# print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f'Random Tensor: \n {x_rand} \n')

tensor = torch.rand(3,4)

'''Tensors are normally created on the CPU, and need to explictly be moved to the GPU using the `.to` method'''
if torch.cuda.is_available():
    print('switching to cuda:')
    tensor = tensor.to('cuda')
print(tensor)

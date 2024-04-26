import torch
import numpy as np

# Create tensor from data
x_data = torch.tensor([[1, 2], [3, 4]])

# Create tensor from numpy array
x_np = torch.from_numpy(np.array([[1, 2], [3, 4]]))

# Create tensor from tensor (use input tensor properties)
x_ones = torch.ones_like(x_data) # filled with 1
x_rand = torch.rand_like(x_data, dtype=float) # filled with random floats

# Create tensor from tuple of tensor dimensions
shape = (2,3)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

# Get tensor's attributes
get_shape = rand_tensor.shape # [2,3]
get_datatype = rand_tensor.dtype # int/float/...
get_device = rand_tensor.device # cpu/gpu/...

# Indexing and slicing tensors
tensor = torch.ones(4, 4)
get_first_row = tensor[0]
get_first_column = tensor[:, 0]
get_last_column = tensor[..., -1]
tensor[:, 1] = 0 # set first column values to 0

# Concatenate tensors
cat_tensor = torch.cat([tensor, tensor], dim=1) # existing dimension
stack_tensor = torch.stack([tensor, tensor], dim=1) # new dimension

# Single element tensor
agg = tensor.sum()
agg_item = agg.item()

# In place operations (discouraged)
tensor.add_(5)

# Tensor to numpy array
np_array = tensor.numpy() # tensor as reference

# Numpy array to tensor
new_tensor = tensor.from_numpy(np_array) # array as reference


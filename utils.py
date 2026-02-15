import numpy as np
from torchvision import utils


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    """Convert a batch tensor to a numpy grid for visualization"""
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))

    # convert single-channel to 3-channel for visualization
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)

    return vis


def de_norm(tensor_data):
    """De-normalize tensor from [-1, 1] to [0, 1]"""
    return tensor_data * 0.5 + 0.5



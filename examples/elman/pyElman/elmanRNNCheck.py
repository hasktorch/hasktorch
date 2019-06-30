import torch.nn as nn
import torch


def read_params(filename):

    s = ""
    with open(filename, "r+") as f:
        s = f.readlines()

    return s


tensors = read_params("out")
for tensor in tensors:
    print(tensor)

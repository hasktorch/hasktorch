# Untyped MNIST Example

This is a simple MLP implementation of an MNIST classifier using untyped tensors.

Before running this example, from the linked `datasets/` directory, run:

`./download-mnist.sh` 

This downloads the mnist dataset into `datasets/mnist/`. Then run:

`stack run mnist-mlp`

The code trains a model then prints some example images to the terminal as 
ascii, showing model outputs and ground truth for the number shown.

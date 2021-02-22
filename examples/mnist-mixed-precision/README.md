# Untyped Mixed Precision Example

This is a simple MLP implementation of an MNIST classifier using untyped tensors.

Before running this example, from the linked `datasets/` directory, run:

`./download-mnist.sh` 

This downloads the mnist dataset into `datasets/mnist/`. Then run:

```sh
$ export DEVICE="cuda:0"          # Set device to CUDA
$ export DTYPE="half"             # Set data type to half-precision floating point
$ stack run mnist-mixed-precision # Run
```
or just run:

```sh
$ stack run mnist-mixed-precision
```

The code trains a model then prints some example images to the terminal as 
ascii, showing model outputs and ground truth for the number shown.

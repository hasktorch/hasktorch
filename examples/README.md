# Examples

Examples are split into dynamic and typed tensors. 

Typed tensors track dimensions of computations and ensure that 
dimension invariants of the computation while dynamic tensors
treat tensors as an opaque type similar to PyTorch.

If you are new to hasktorch, we recommend starting with a
simple example such as the [regression](regression)
or [xor-mlp](xor-mlp) to get familiar with basic
mechanics of computation and training a model.

## Dynamic Tensor Examples

These examples do not attempt to type-check tensor dimensions.

- gaussian_process - basic gaussian process implementation
- gd-field - visualize autodiff gradients of a mathematical function
- image-processing - small test of convolution ops
- minimal-text-example - "hello" string test of `rnn` modules, uses `rnn` as dependency
- load-torchscript - load a serialized PyTorch model using torchscript (WIP)
- matrix-factorization - recommender system example using matrix factorization fitted with stochastic gradient descent
- optimizers - test of gradient-based optimizers using test functions
- regression - linear regression example
- rnn - implementations of Elman, LSTM, and GRU RNN layers
- serialization - test serialization / deserialization of model state
- vae - variational autoencoder
- xor-mlp - an XOR multilayer perceptron
- autograd - the dataflow through autograd in Hasktorch
- alexNet - feature-extraction based on pretrained AlexNet for non-trivial data-sets

## Typed Tensor Examples

Some examples demonstrate typed tensor functionality. 

- static-mnist-cnn - a convolutional neural network mnist classifier
- static-mnist-mlp - a mlp neural network mnist classifier
- static-mnist - shared mnist functions
- static-xor-mlp - an XOR multilayer perceptron
- typed-transformer - transformer with attention implementation

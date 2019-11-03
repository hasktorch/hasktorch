# Initializers Test for Untyped Tensors

See [pytorch initializer implementaitons](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/init.py) and [libtorch initializer implementations](https://github.com/pytorch/pytorch/blob/dd277e90869cabfaca1e2f36464480935682b281/torch/csrc/api/src/nn/init.cpp)

## Linear layer

Weights are initialized with kaiming uniform, a=sqrt(0.5), mode="fan in" (default), nonlinearity="leaky_relu" (default)
[linear initialization](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/linear.py#L79)

## CNN 1d, 2d, 3d
[cnn initialization](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/conv.py#L48)

## LSTM, GRU
[rnn initialization](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/rnn.py#L136)

## Transformer
[transformer initialization](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/transformer.py#L132)

# Useful Reviews

- [Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)
- [Understand Kaiming Initialization and Implementation Detail in PyTorch](https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138)

## Xavier Initialization

$$
std = \sqrt{\frac{2.0}{fan_{in} + fan_{out}}}
$$

## Kaiming Initialization

$$
std = \sqrt{\frac{2}{(1 + a^2) * fan}}
$$

With some variations depending on whether the input or output dimensions are used for the van value

For default values, $a = 0$ corresponding to recommended gain for a relu non-linearity, and fan mode is $fan_{in}$:

$$
std = \sqrt{\frac{2}{fan_{in}}}
$$

## Uniform vs. Normal Versions

Scale for both Kaiming and Xavier are scaled according to the standard deviation.

For normal distributions, the distribution is simply scaled by $std$.

For uniform distributions, the standard deviation is:

$$
\frac{1}{2 \sqrt{3}} width = std
$$

the half-width is:

$$
\frac{1}{2} width = \sqrt{3} std
$$

which is assigned to the `bound` variable (following the naming convention of [pytorch]((https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/init.py))

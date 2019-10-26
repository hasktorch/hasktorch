# Initializers Test for Untyped Tensors

See [pytorch initializer implementaitons](https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/init.py)

## Linear layer

Weights are initialized with kaiming uniform, a=sqrt(0.5), mode="fan in" (default), nonlinearity="leaky_relu" (default)
https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/linear.py#L79

## CNN 1d, 2d, 3d
https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/conv.py#L48

## LSTM, GRU
https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/rnn.py#L136

## Transformer
https://github.com/pytorch/pytorch/blob/6301d62e0b205c53a445ffb87ce7be1ac52d9cb9/torch/nn/modules/transformer.py#L132

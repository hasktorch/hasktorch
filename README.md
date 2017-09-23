# HaskTorch

A Haskell library for tensors and dynamic neural networks using
[Torch](https://github.com/torch/torch7).

*This library is in early development and is not recommended for use except by
project contributors*

## Installation Instructions

From within the `vendor`, directory, run:

```
build-torch-core.sh
```

Then build using stack:

```
stack build
```

If everything built, you should be able to run tests successfully:

```
stack test torch-tests
```

## Acknowledgements

Thanks to Adam Paszke for his writeup on the [torch
internals](https://apaszke.github.io/torch-internals.html) and to Edward Z. Yang
for his example writeup on [Haskell/Torch binding using
backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).

---
title: Walkthrough
---

To get the most out of the walkthrough, we encourage you to follow along with the examples in a GHCi session.

To start, import `Torch`{.haskell .identifier}:

```haskell top
import Torch
```

```haskell top hide
import Inliterate.Import (AskInliterate)
instance AskInliterate Tensor
```

## Tensors

A `Tensor` in Hasktorch is multidimensional array with a fixed shape and element type.

For example, we can initialize a tensor with shape `[3, 4]` and filled with zeros using


```haskell eval
Torch.zeros' ([3, 4] :: [Int])
```

We can also initialize a tensor from a Haskell list using `Torch.Tensor.asTensor`:

```haskell eval
asTensor ([[4, 3], [2, 1]] :: [[Float]])
```

Note that the numerical type of the tensor is inferred from the types of the values in the list.

Scalar values are represented in Hasktorch as tensors with shape `[]`:

```haskell eval
asTensor (3.5 :: Float)
```

We can get the scalar value back out using `Torch.Tensor.asValue`:

```haskell eval
asValue (asTensor (3.5 :: Float)) :: Float
```

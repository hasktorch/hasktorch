---
title: Tensors
---

# Tensors

To get the most out of the walkthrough, we encourage you to follow
along with the examples in a GHCi session.

To start, import `Torch`:

```haskell top
import Torch
```

```haskell top hide
import Inliterate.Import (AskInliterate)
instance AskInliterate Tensor
```

A `Tensor` in Hasktorch is multidimensional array with a fixed shape
and element type.

For example, we can initialize a tensor with shape `[3, 4]` and filled
with zeros using

```haskell eval
zeros' ([3, 4] :: [Int])
```

We can also initialize a tensor from a Haskell list using `asTensor`:

```haskell eval
asTensor ([[4, 3], [2, 1]] :: [[Float]])
```

Note that the numerical type of the tensor is inferred from the types
of the values in the list.

Scalar values are represented in Hasktorch as tensors with shape `[]`:

```haskell eval
asTensor (3.5 :: Float)
```

We can get the scalar value back out using `asValue`:

```haskell eval
asValue (asTensor (3.5 :: Float)) :: Float
```

## Specifying Tensor parameters

In the previous section we initialized a tensor filled with zeros using
`zeros'` (note the prime suffix). Hasktorch functions use a convention
where default versions of functions use a prime suffix. The unprimed
versions of these functions expect an additional parameter specifying
tensor parameters. For example:

```haskell
zeros :: [Int] -> Torch.TensorOptions -> Tensor
```

`TensorOptions` are typically specified by starting with
`Torch.TensorOptions.defaultOpts` and modifying using one or more of
the following:

- `withDType` configures the data type of the
    elements

- `withDevice` configures on which device the
    tensor is to be used

- others (see `Torch.TensorOptions`)

For example, to construct a matrix filled with zeros of dtype `Int64`:

```haskell eval
zeros [4, 4] (withDType Int64 defaultOpts)
```

## Tensor factories

Hasktorch comes with many "factory" functions similar to `zeros` and
`zeros'` useful for initializing common kinds of tensors. For example,
`ones`, `full`, `eye`, and the primed versions of these. See the
`Torch.TensorFactories` module for a complete list.

One useful class of factory functions are those suffixed with "-like"
(e.g. `onesLike`), which initialize a tensor
with the same dimensions as their argument. For example:

```haskell eval
let x = zeros' [3, 2] in onesLike x
```

## Operations

Most operations are pure functions, similar to Haskell standard
library math operations.

Tensors implement the `Num` typeclass, meaning they support the usual
arithmetic operations:

```haskell eval
let x = ones' [4] in x + x
```

Some operations transform a tensor:

```haskell eval
relu (asTensor ([-1.0, -0.5, 0.5, 1] :: [Float]))
```

`Torch.Tensor.select` slices out a selection by specifying a dimension
and index:

```haskell do
let x = asTensor ([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]] :: [[[Int]]])
```

```haskell eval
shape x
```

```haskell eval
select 2 1 x
```

```haskell do
let y = asTensor ([1, 2, 3] :: [Int])
```

```haskell eval
select 0 1 y
```

Values can be extracted from a tensor using `asValue` so long as the
dtype matches the Haskell type:

```haskell eval
let x = asTensor ([2] :: [Int]); y = asValue x :: Int in y
```


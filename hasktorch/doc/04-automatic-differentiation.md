---
title: Automatic Differentiation
---

```haskell top hide
import Torch
import Inliterate.Import (AskInliterate)
instance AskInliterate Bool
instance AskInliterate Tensor
```

# Automatic Differentiation

Automatic differentiation is achieved through the use of two primary
functions in the `Torch.Autograd` module: `makeIndependent` and
`grad`.

## Independent Tensors

`makeIndependent` is used to instantiate an independent tensor variable
from which a compute graph is constructed for differentiation, while
`grad` uses compute graph to compute gradients.

`makeIndependent` takes a tensor as input and returns an IO action which
produces an `Torch.Autograd.IndependentTensor`{.haskell .identifier}:

```haskell
makeIndependent :: Tensor -> IO IndependentTensor
```

What is the definition of the `IndependentTensor` type produced by the
`makeIndependent` action? It's defined in the Hasktorch library as:

```haskell
newtype IndependentTensor = IndependentTensor {toDependent :: Tensor} deriving (Show)
```

Thus `IndependentTensor` is simply a wrapper around the underlying
Tensor that is passed in as the argument to
`makeIndependent`. Building up computations using ops applied to the
`toDependent` tensor of an `IndependentTensor` will implicitly
construct a compute graph to which `grad` can be applied.

All tensors have an underlying property that can be retrieved using
the `Torch.Autograd.requiresGrad` function which indicates whether
they are a differentiable value in a compute graph.[^requires-grad]

```haskell do
let x = asTensor ([1, 2, 3] :: [Float])
y <- makeIndependent (asTensor ([4, 5, 6] :: [Float]))
let y' = toDependent y
let z = x + y'
```

```haskell eval
requiresGrad x
```

```haskell eval
requiresGrad y'
```

```haskell eval
requiresGrad z
```

In summary, tensors that are computations of values derived from
tensor constructors (e.g. `ones`, `zeros`, `fill`, `randIO` etc.)
outside the context of a `IndependentTensor` are not
differentiable. Tensors that are derived from computations on the
`toDependent` value of an `IndependentTensor` are differentiable, as
the above example illustrates.

## Gradients

Once a computation graph is constructed by applying ops and computing
derived quantities stemming from a `toDependent` value of an
`IndependentTensor`, a gradient can be taken by using the `grad`
function specifying in the first argument tensor corresponding to
function value of interest and a list of `Independent` tensor
variables that the the derivative is taken with respect to:

```haskell
grad :: Tensor -> [IndependentTensor] -> [Tensor]
```

Let's demonstrate this with a concrete example. We create a tensor and
derive an `IndependentTensor` from it:

```haskell do
a <- makeIndependent (ones' [2, 2])
let a' = toDependent a
```

```haskell eval
a'
```

Now do some computations on the dependent tensor:

```haskell do
let b = a' + 2
```

```haskell eval
b
```

Since `b` is dependent on the independent tensor `a`, it is differentiable:

```haskell eval
requiresGrad b
```

Applying more operations:

```haskell do
let c = b * b * 3
let out = mean c
```

```haskell eval
c
```

Now retrieve the gradient:

```haskell eval
grad out [a]
```

[^requires-grad]: PyTorch users will be familiar with this as the `requires_grad`
    member variable for the PyTorch tensor type. The Hasktorch
    mechanism is distinct from PyTorch's mechanism - by only allowing
    gradients to be applied in the context of a set of
    `IndependentTensor` variables, it allows ops to be semantically
    pure and preserve referential transparency.

---
title: Linear Regression
---

# Linear Regression

Let's start with a simple example of linear regression. Here we
generate random data with an underlying affine relationship between
the inputs and outputs, then fit a linear regression to reproduce that
relationship.

This example is adapted from
<https://github.com/hasktorch/hasktorch/tree/master/examples/regression>.

In a standard supervised learning model, the neural network is
initialized using a randomized initialization scheme. An iterative
optimization is performed such that at each iteration a batch.

Here is a simple end-to-end example in Hasktorch:

```haskell
module Main where

import Control.Monad (when)
import Torch

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

main :: IO ()
main = do
    init <- sample $ LinearSpec{in_features = numFeatures, out_features = 1}
    randGen <- mkGenerator (Device CPU 0) 12345
    (trained, _) <- foldLoop (init, randGen) 2000 $ \(state, randGen) i -> do
        let (input, randGen') = randn' [batchSize, numFeatures] randGen
            (y, y') = (groundTruth input, model state input)
            loss = mseLoss y y'
        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        (state', _) <- runStep state GD loss 5e-3
        pure (state', randGen')
    pure ()
  where
    batchSize = 4
    numFeatures = 3
```

Let's break this down in pieces:

1. The `init` variable is initialized as a `Linear` type (defined in
   `Torch.NN`) using `sample` which randomly initializes a `Linear`
   value. Initialization is discussed in more detail in the following
   section
1. `Linear` is a built-in algebraic data type (ADT) implementing the
   `Parameterized` typeclass and representing a fully connected linear
   layer, equivalent to linear regression when no hidden layers are
   present.
1. `init` is passed into the `Torch.Optim.foldLoop` as the state
   variable. Note that `foldLoop` is just a convenience function
   defined using `foldM`:

    ```haskell
    foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
    foldLoop init count body = foldM body init [1 .. count]
    ```

1. At each optimization step, `Torch.Optim.runStep` computes an
   updated model state given the current state, optimizer, loss
   function, and learning rate.

Note the expression of the architecture in the `Torch.NN.linear`
function (a single linear layer, or alternatively a neural network
with zero hidden layers) does not require an explicit representation
of the compute graph, but is simply a composition of tensor
ops. Because of the autodiff mechanism described in the previous
section, the graph is constructed automatically as pure functional ops
are applied, given a context of a set of independent variables.

## Weight Initialization

Random initialization of weights is not a pure function since two
random initializations return different values. Initialization occurs
by calling the `Torch.NN.sample` function for an ADT implementing the
`Torch.NN.Randomizable` typeclass:

```haskell
class Randomizable spec f | spec -> f where
  sample :: spec -> IO f
```

In a typical (but not required) usage, `f` is an ADT that implements
the `Parameterized` typeclass, so that there's a pair of types—a
specification type implementing the `spec` input to `sample` and a
type implementing `Parameterizable` representing the model state.

For example, a linear fully connected layer is provided by the
`Torch.NN` module and defined therein as:

```haskell
data Linear = Linear {weight :: Parameter, bias :: Parameter} deriving (Show, Generic)
```

and is typically used with a specification type:

```haskell
data LinearSpec = LinearSpec {in_features :: Int, out_features :: Int}
  deriving (Show, Eq)
```

Putting this together, in untyped tensor usage, the user can implement
custom models or layers implementing the `Parameterizable` typeclass
built up from other ADTs implementing `Parameterizable`. The shape of
the data required for initialization is described by a type
implementing `Randomizable`'s `spec` parameter, and the `sample`
implementation specifies the default weight initialization.

Note this initialization approach is specific to untyped tensors. One
consequence of using typed tensors is that the information in these
`spec` types is reflected in the type itself and thus are not needed.

What if you want to use a custom initialization that differs from the
default? You can define an alternative function with the same
signature `spec -> IO f` and use the alternative function instead of
`sample`.

### Optimizers

Optimization implementations are functions that take as input the
current parameter values of a model, parameter gradient estimates of
the loss function at those parameters for a single batch, and a
characteristic learning describing how large a perturbation to make to
the parameters in order to reduce the loss. Given those inputs, they
output a new set of parameters.

In the simple case of stochastic gradient descent, the function to
output a new set of parameters is to subtract from the current
parameter (θ), the gradient of the loss ∇ J scaled by the learning
rate η:

$$\theta_{i+1} = \theta_i - \eta \nabla J(\theta)$$

While stochastic gradient descent is a stateless function of the
parameters, loss, and gradient, some optimizers have a notion of
internal state that is propagated from one step to the step, for
example, retaining and updating momentum between steps:

$$\Delta \theta_i = \alpha \Delta \theta_{i-1} - \eta \nabla J(\theta)$$
$$\theta_{i+1} = \theta_i + \Delta \theta_i$$

In this case, the momentum term Δ θᵢ is carried forward as internal
state of the optimizer that is propagated to the next step. α is an
optimizer parameter which determines a weighting on the momentum term
relative to the gradient.

Implementation of an optimizer consists of defining an ADT describing
the optimizer state and a `step` function that implements a single
step perturbation given the learning rate, loss gradients, current
parameters, and optimizer state.

This function interface is described in the `Torch.Optim.Optimizer`
typeclass interface:

```haskell
class Optimizer o where
    step :: LearningRate -> Gradients -> [Tensor] -> o -> ([Tensor], o)
```

`Gradients` is a newtype wrapper around a list of tensors to make
intent explicit: `newtype Gradients = Gradients [Tensor]`.

Hasktorch provides built-in optimizer implementations in
`Torch.Optim`.  Some illustrative example implementations follow.

Being stateless, stochastic gradient descent has an ADT that has only
one constructor value:

```haskell
data GD = GD
```

and implements the step function as:

```haskell
instance Optimizer GD where
  step lr gradients depParameters dummy = (gd lr gradients depParameters, dummy)
    where
      step p dp = p - (lr * dp)
      gd lr (Gradients gradients) parameters = zipWith step parameters gradients
```

The use of an optimizer was illustrated in the linear regression example
using the function `runStep`

```haskell
(state', _) <- runStep state GD loss 5e-3
```

In this case the new optimizer state returned is ignored (as `_`)
since gradient descent does not have any internal state. Under the
hood, `runStep` does a little bookkeeping making independent variables
from a model, computing gradients, and passing values to the `step`
function.  Usually a user can ignore the details and just pass model
parameters and the optimizer to `runStep` as an abstracted interface
which takes parameter values, the optimizer value, loss (a tensor),
and learning rate as input and returns updated model and optimizer
values.

```haskell
runStep ::
  (Parameterized model, Optimizer optimizer) =>
  model ->
  optimizer ->
  Tensor ->
  LearningRate ->
  IO (model, optimizer)
```

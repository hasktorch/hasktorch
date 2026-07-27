---
title: Optimizers
---

# Optimizers

Every training loop in this tutorial ends in the same line:

```haskell
(model', optim') <- runStep model optim loss learningRate
```

This chapter is about what stands behind that line. Hasktorch ships
two optimizer families with the same interface: purely functional
optimizers written in Haskell, whose state is an ordinary value, and
bindings to libtorch's C++ optimizers, which update parameters
in place. They are interchangeable in the loop — we will run the
exact same code with both — but they make opposite trade-offs.

```haskell top hide
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
import Inliterate.Import (AskInliterate)
```

```haskell top
import Data.Default.Class (def)
import GHC.Generics (Generic)
import Torch
import Torch.Optim.CppOptim
```

```haskell top hide
instance AskInliterate Tensor
```

## The interface

```haskell
class Optimizer optimizer where
  step :: LearningRate -> Gradients -> [Tensor] -> optimizer -> ([Tensor], optimizer)
  runStep :: Parameterized model
          => model -> optimizer -> Loss -> LearningRate -> IO (model, optimizer)
```

`step` is the mathematical content: given gradients and parameters,
produce new parameters and the optimizer's next state. `runStep`
wraps it with the autograd call (`grad'`) and the flattening of the
model record into a parameter list — that is the line the training
loops use.

A test problem hard enough to tell optimizers apart — the Rosenbrock
banana function, whose minimum at \((1, 1)\) hides at the end of a
long, curved, nearly-flat valley:

```haskell top
data XY = XY {x :: Parameter, y :: Parameter}
  deriving (Generic, Show, Parameterized)

rosenbrock :: XY -> Tensor
rosenbrock XY {..} = (1 - x') ^ 2 + 100 * (y' - x' * x') ^ 2
  where
    x' = toDependent x
    y' = toDependent y

optimize :: Optimizer o => o -> LearningRate -> Int -> XY -> IO (XY, o)
optimize opt lr steps p0 = foldLoop (p0, opt) steps $
  \(p, o) _ -> runStep p o (rosenbrock p) lr

result :: XY -> Tensor
result XY {..} = stack (Dim 0) [toDependent x, toDependent y]
```

## Optimizer state is a value

The Haskell-side optimizers live in `Torch.Optim`: `GD`, `GDM`
(momentum), `Adagrad`, `Adam`, `AdamW`. Their state is a plain
record — `GD` has none, `GDM` carries momentum tensors, `Adam`
carries its two moment estimates and a step counter (built with
`mkAdam`, which needs the parameter list to size those tensors):

```haskell do
p0 <- XY <$> makeIndependent (asTensor (-1.2 :: Float))
         <*> makeIndependent (asTensor (1.0 :: Float))
(gdP, _) <- optimize GD 1e-3 2000 p0
(adamP, adamState) <- optimize (mkAdam 0 0.9 0.999 (flattenParameters p0)) 2e-2 2000 p0
```

Starting from the classic \((-1.2, 1)\), after 2000 steps each —
gradient descent is still crawling along the valley floor:

```haskell eval
result gdP
```

while Adam's per-coordinate step sizes have carried it to within a
few percent of the minimum:

```haskell eval
result adamP
```

Because the optimizer state is a value, there is nothing hidden to
manage: checkpointing an optimizer is saving a record of tensors,
inspecting it is pattern matching. Adam's internal step counter, for
instance, is right there:

```haskell eval
iter adamState
```

The price is allocation: every step builds fresh parameter and state
tensors and the old ones become garbage.

## The C++ optimizers update in place

`Torch.Optim.CppOptim` binds libtorch's native optimizers — SGD,
Adagrad, RMSprop, Adam, AdamW, and L-BFGS, which has no Haskell-side
counterpart. Each is configured by an options record with a
`Data.Default` instance, and `initOptimizer` registers the model's
parameters with a C++ optimizer object:

```haskell do
p1 <- XY <$> makeIndependent (asTensor (-1.2 :: Float))
         <*> makeIndependent (asTensor (1.0 :: Float))
cppAdam <- initOptimizer (def {adamLr = 2e-2} :: AdamOptions) p1
(cppP, _) <- optimize cppAdam 0 2000 p1
```

```haskell eval
result cppP
```

It lands on the same numbers as the pure Adam run above, up to
floating-point noise — same algorithm, same trajectory.

Note what did *not* change: `optimize` is the same function, because
`CppOptimizerState` has an `Optimizer` instance. Two differences hide
in that call, though. The learning rate argument is ignored — the `0`
above is a dummy, because a C++ optimizer's hyperparameters live in
its options (change them mid-run with `cppOptimizerSetLr`).
And the step is genuinely in-place: the parameter tensors now live
inside the C++ optimizer, which overwrites them on every step. The
model returned by `runStep` is rebuilt around those same tensors, so
the *old* model value silently changes too — the underlying binding
is honestly named `unsafeStep`. Keep only the newest model and this
never bites; keep a history of models for comparison and it will.
In exchange, a step allocates nothing new for parameters or state,
which for large models means less garbage-collector pressure and a
smaller resident set. (`runStep` also nudges the collector each step
— `performGC` plus a `malloc_trim` — so that memory held by dead
foreign tensors actually returns to the OS during training.)

## Parameter groups

PyTorch training recipes routinely split parameters into groups —
the classic case is AdamW with weight decay on the weights but *not*
on biases and normalization parameters. The AdamW binding exposes
this: `initAdamwWithGroups` takes the decayed and undecayed parameter
lists separately, and per-group learning rates can be scheduled
afterwards:

```haskell
optim <- initAdamwWithGroups (def {adamwPgWeightDecay = 1e-2}) decayed undecayed
cppOptimizerSetGroupLr optim 0 (lr * warmupFactor)
```

This is the piece you need to port a modern transformer training
recipe faithfully; the pure optimizers have no equivalent yet.

## Choosing

Use the **pure optimizers** when you want the state visible — for
teaching, for reproducibility experiments, for checkpointing as
ordinary values, or with the typed API, where `Torch.Typed.Optim`
provides typed `GD`/`GDM`/`Adam`. Use the **C++ optimizers** when
throughput and memory matter, when you need L-BFGS or parameter
groups, or when you are matching a PyTorch training run
step-for-step (`Torch.Typed.Optim.CppOptim` wraps them for the typed
API too). The training loop does not care which you picked — which is
the point of `runStep` being a typeclass method.

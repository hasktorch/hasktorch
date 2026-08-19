---
title: Networks as Arrows
---

# Networks as Arrows

Composing layers with `.` works until the architecture stops being a
straight line: skip connections, parallel branches, multi-input
blocks. Those are exactly what Haskell's `Arrow` vocabulary is for,
and `Torch.Typed.NN.Arrow` makes networks arrows: `>>>` chains, `&&&`
fans out, `***` runs in parallel, and `proc` notation wires
arbitrary graphs.

```haskell top hide
{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
import Inliterate.Import (AskInliterate)
```

```haskell top
import Control.Arrow
import Control.Category (id, (>>>))
import GHC.Generics (Generic)
import Torch (Tensor)
import qualified Torch.Functional as UF
import qualified Torch.Tensor as UT
import qualified Torch.Typed as T
import Torch.Typed.NN.Arrow
import Torch.Typed.NN.BatchNorm
import Prelude hiding (id)
```

```haskell top hide
instance AskInliterate Tensor
```

## A network is a stochastic function

```haskell
newtype Net x y = Net { runNet :: x -> IO y }
```

`Net` has real `Category` and `Arrow` instances. Pure typed
operations lift with `arr`, whole modules with `layer` (which uses
their stochastic `forwardStoch`), and `IO` layers embed directly.
Because composition is *value level* — the middle type of
`f >>> g` is fixed by the values `f` and `g` — chains of convolutions
type-check without any partial type signatures: each combinator's
output shape is computed from its input shape by the same type
families the rest of the typed API uses.

## Skip connections

A ResNet residual is the arrow idiom:

```haskell
residual f = (id &&& f) >>> arr (uncurry (+))
```

```haskell do
skip <- pure (residual (arr (T.mulScalar (2 :: Float))) :: Net (T.Tensor '( 'T.CPU, 0) 'T.Float '[2, 3]) (T.Tensor '( 'T.CPU, 0) 'T.Float '[2, 3]))
out1 <- runNet skip T.ones
```

The result is `2x + x = 3` everywhere:

```haskell eval
T.toDynamic out1
```

`residualWith` takes a projection for the skip path, which is how
ResNet's stride-2 downsampling blocks are wired (see
`test/Torch/Typed/ResNetSpec.hs` for a complete miniature ResNet).

## proc notation

Multi-branch blocks read like their architecture diagrams. An
Inception-style block runs parallel convolutions at different
receptive fields and concatenates the channels; here is its
one-dimensional cousin, live:

```haskell do
let incept :: Net (T.Tensor '( 'T.CPU, 0) 'T.Float '[2]) (T.Tensor '( 'T.CPU, 0) 'T.Float '[2])
    incept = proc x -> do
      a <- arr (T.mulScalar (10 :: Float)) -< x
      b <- residual id -< x
      returnA -< a + b
out2 <- runNet incept T.ones
```

`a` contributes `10x`, `b` contributes `x + x`, so:

```haskell eval
T.toDynamic out2
```

The branches `a` and `b` both consume `x`; the desugarer builds the
`&&&`/`***` plumbing.

## Batch normalization: honestly stateful

Batch norm updates its running statistics *in place* during training
— it never was a pure function, and `Net`'s `IO` makes that explicit
instead of hiding it. `Torch.Typed.NN.BatchNorm` keeps weight and
bias as typed `Parameter`s and the running statistics as mutable
buffers:

```haskell do
bn <- T.sample (BatchNorm2dSpec :: BatchNorm2dSpec 3 'T.Float '( 'T.CPU, 0))
x3 <- T.randn :: IO (T.Tensor '( 'T.CPU, 0) 'T.Float '[4, 3, 8, 8])
before <- UF.clone (case bnRunningMean bn of UT.MutableTensor t -> t)
_ <- runNet (Net (batchNorm2dForward bn True)) x3
after <- UF.clone (case bnRunningMean bn of UT.MutableTensor t -> t)
```

The running mean starts at zero and moves after one training-mode
forward pass — same record, mutated buffer:

```haskell eval
before
```

```haskell eval
after
```

In evaluation mode (`batchNorm2dForward bn False`) the statistics are
used but not touched.

## Where the parameters live

Deliberately *not* inside the arrow. A model is an ordinary record —

```haskell
data Block c = Block
  { k1 :: Conv2d c c 3 3 'T.Float device, n1 :: BatchNorm2d c 'T.Float device, ... }
  deriving (Generic, Parameterized)
```

— with derived `Parameterized` and the usual `runStep` training loop,
and a plain function turns the record into wiring:
`identityBlock train Block{..} = residual (arr (conv2dForward ... k1) >>> Net (batchNorm2dForward n1 train) >>> ...)`.
The earlier `feature/arrow` experiment stored the composition in the
*type* (`Conv2d ... :>>> MaxPool :>>> ...`) and paid for it: every
composition point was an ambiguous type for the compiler, `deriving
Parameterized` broke, and examples needed `PartialTypeSignatures`.
Records for state, arrows for wiring — each side does what it is good
at.

## Folding a stack of layers

`Net` is a real `Category`, and that pays off when a model is a
*stack* of same-shaped blocks: composition has an identity, so a whole
list of layers folds into one network.

```haskell do
let blocks = map (\k -> residual (arr (T.mulScalar (k :: Float))))
             [1, 2, 3] :: [Net (T.Tensor '( 'T.CPU, 0) 'T.Float '[2]) (T.Tensor '( 'T.CPU, 0) 'T.Float '[2])]
    stack = foldr (>>>) id blocks
out3 <- runNet stack T.ones
```

Each block computes `x + k*x = (1+k)x`, so the folded stack multiplies
by `2 * 3 * 4`:

```haskell eval
T.toDynamic out3
```

The same fold works when every layer has its own parameter record in
an `HList` — the seed is still `id`, the combining step is still
`>>>`, just expressed with `hfoldr` and an `Apply'` instance.

## In the wild

For a full-scale example, see
[gpt2-haskell](https://github.com/collinarnett/gpt2-haskell) by Collin
Arnett, whose GPT-2 decoder stack is wired exactly in this style —

```haskell
residual (layer ln >>> selfAttention mha mask) >>> transformerMLP
```

— with the causal mask written as its `tabulate` formula and typed
slicing (`getSlice`) replacing runtime shape proofs that previously
needed `unsafeCoerce`.  Its migration PR is a good study in porting an
existing model to these APIs.

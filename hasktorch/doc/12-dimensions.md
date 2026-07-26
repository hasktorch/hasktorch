---
title: Moving Dimensions
---

# Moving Dimensions

A dimension of a tensor can live in two places. Inside the tensor, it
is fast: one fused kernel processes every position at once. Outside
the tensor — as an actual Haskell functor — it has *structure*:
positions with names you can pattern-match, containers you can fold,
recursion. `dimUp` and `dimDown` from `Torch.Typed.Representable`
move the outermost dimension between the two worlds:

```haskell
dimUp   :: NamedTensor device dtype (f ': shape) -> f (NamedTensor device dtype shape)
dimDown :: f (NamedTensor device dtype shape) -> NamedTensor device dtype (f ': shape)
```

Think of them as a valve: keep a dimension on the tensor side while
the computation is uniform, lift it out the moment you need to treat
positions differently.

```haskell top hide
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
import Inliterate.Import (AskInliterate)
import Torch (Tensor)
```

```haskell top
import Data.Default.Class (Default)
import Data.Vector.Sized (Vector)
import GHC.Generics (Generic)
import Torch.HList
import qualified Torch.Typed as T
import Torch.Typed.Representable
```

```haskell top hide
instance AskInliterate Tensor
```

## Channels by name

```haskell top
data RGB a = RGB { r :: a, g :: a, b :: a } deriving (Show, Generic, Default)
```

```haskell do
let img = tabulate (\(c :. j :. HNil) -> fromIntegral (fromEnum c) * 10 + fromIntegral (fromEnum j))
            :: T.NamedTensor '( 'T.CPU, 0) 'T.Float '[RGB, Vector 4]
    chans = dimUp img :: RGB (T.NamedTensor '( 'T.CPU, 0) 'T.Float '[Vector 4])
```

The channel dimension became a record; each field is a smaller
tensor, addressed by name rather than by `select 0 2`:

```haskell eval
T.toDynamic (r chans)
```

```haskell eval
T.toDynamic (b chans)
```

`dimDown` puts the dimension back — it is the exact inverse:

```haskell eval
T.toDynamic (dimDown chans)
```

## One matmul, four gates

The idiom that makes this more than a convenience. An LSTM computes
four gates; the standard trick is to compute all of them with a
*single* fused matrix multiplication into a `4·H`-row result and then
split. In index notation, the split is slice arithmetic
(`pre[0:H]`, `pre[H:2H]`, …) — easy to get silently wrong. Here the
gate axis is a record, and `dimUp` names the split:

```haskell top
data Gates a = Gates { gi :: a, gf :: a, go :: a, gu :: a }
  deriving (Show, Generic, Default)
```

```haskell do
let pre = tabulate (\(gate :. j :. HNil) -> fromIntegral (fromEnum gate) * 100 + fromIntegral (fromEnum j))
            :: T.NamedTensor '( 'T.CPU, 0) 'T.Float '[Gates, Vector 8]
    Gates {..} = dimUp pre
```

`gi` is rows 0, `gf` rows 1, and so on — in declaration order, checked
by the same `ToNat`/`Generic` machinery that sized the axis:

```haskell eval
T.toDynamic gi
```

```haskell eval
T.toDynamic gu
```

After the split, each gate takes its *own* nonlinearity —
`sigmoid gi`, `tanh gu` — selected by field name, not by offset.

## Trees: the dimension that cannot be an axis

Some structure never fits a tensor dimension at all. A parse tree, an
abstract syntax tree, a molecule: every input has a different shape,
so the recursion over it stays in Haskell — and around that recursion,
`dimUp` and `dimDown` do real work. `test/Torch/Typed/TreeLSTMSpec.hs`
implements a Child-Sum Tree-LSTM:

```haskell
data Tree = Leaf Int | Node Tree Tree   -- a runtime value; every input differs

treeLSTM :: Params -> Tree -> (T '[H], T '[H])
treeLSTM p (Node l r) =
  let children = V.fromTuple (treeLSTM p l, treeLSTM p r)
      -- dimDown: the children's states arrive as Haskell structure;
      -- pack them into an axis so per-child math runs in bulk
      hs = toUnnamed (dimDown (fmap (fromUnnamed . fst) children)) :: T '[2, H]
      hsum = sumDim @0 hs
      -- one fused matmul computes all four gates, and dimUp names them
      Gates {..} = dimUp (fromUnnamed (reshape @'[4, H] (uG `matmul` hsum)))
      -- per-child forget gates, batched over the packed axis
      fs = sigmoid (hs `matmul` uF + expand @'[2, H] False (toUnnamed gf))
      c  = sigmoid (toUnnamed gi) * tanh (toUnnamed gu) + sumDim @0 (fs * cs)
   in (sigmoid (toUnnamed go) * tanh c, c)
```

Reading it as a diagram of the valve: the tree recursion produces
*structure*; `dimDown` packs the children into a tensor axis the
moment the math becomes uniform (forget gates, `Σ fₖ⊙cₖ`); the fused
gate result flows the other way through `dimUp` the moment positions
need different treatment.

## Couldn't a matrix do this?

Yes — and at scale it often should. Mainstream graph and tree
learning encodes structure as data: adjacency matrices, parent-index
vectors, padding and masks, with node types one-hot encoded and
per-type computation done by gather/scatter. That wins on batching:
siblings and whole trees process in parallel on the GPU.

What the structural side buys is correctness and expressiveness where
nodes are *heterogeneous*: an AST with twenty constructors of
different arities becomes pattern matching with typed payloads
instead of index bookkeeping whose off-by-ones fail silently at
training time. The two are ends of a spectrum, and the valve is what
lets you pick a point on it — including the middle: group nodes by
depth, `dimDown` each level into a batch axis, and process level by
level, which recovers most of the batching without giving up the
typed structure.

## Lineage

This chapter's operations continue
[hasktorch-naperian](https://github.com/jasigal/hasktorch-naperian)
by Jesse Sigal (GSoC 2019) — whose `Dim ns fs` type pioneered
dimensions migrating between tensor and structure, and whose Tree-LSTM
was the original showcase — itself based on Jeremy Gibbons's
*APLicative Programming with Naperian Functors*: a Naperian functor
is a container that is a function from a fixed set of positions,
which is exactly the `Representable` view of chapter
[8](08-named-tensors.html).

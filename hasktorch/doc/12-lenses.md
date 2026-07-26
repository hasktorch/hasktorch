---
title: Lenses
---

# Lenses

Hasktorch uses lenses in two directions: *inward*, addressing parts
of a single tensor (slices, named fields), and *outward*, traversing
every tensor inside an arbitrary structure. Both come in untyped and
typed flavors. This chapter walks through all four quadrants with
executable examples.

```haskell top hide
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
import Inliterate.Import (AskInliterate)
import qualified Torch.DType
```

```haskell top
import Torch
import Torch.Index (lslice)
import Torch.Lens (types, flattenValues, replaceValues)
import Lens.Family
import GHC.Generics (Generic)
import qualified Torch.Typed as T
import qualified Torch.Typed.Lens as TL
import Data.Vector.Sized (Vector)
```

```haskell top hide
instance AskInliterate Tensor
instance AskInliterate Torch.DType.DType
```

## Slicing lenses (untyped)

The `lslice` quasiquoter turns PyTorch indexing syntax into a *lens*
into a tensor: something you can read through, but also write
through.

```haskell do
let t = asTensor ([[0, 1, 2], [3, 4, 5]] :: [[Float]])
```

Reading a slice with `^.`:

```haskell eval
t ^. [lslice|1|] :: Tensor
```

```haskell eval
t ^. [lslice|:, ::2|] :: Tensor
```

Writing through the same lens with `.~` returns an updated tensor,
with everything outside the slice untouched:

```haskell eval
t & [lslice|0|] .~ zeros' [3]
```

And `%~` modifies the focused slice with a function:

```haskell eval
t & [lslice|:, 1|] %~ ((* 100) :: Tensor -> Tensor)
```

This is the lens counterpart of `t[0] = ...` in Python — but pure:
the original `t` is unchanged.

## Structure traversals (untyped)

`Torch.Lens` provides `types`, a generic traversal of every value of
a given type inside a structure. Any record with a `Generic` instance
works.

```haskell top
data TwoLayer = TwoLayer
  { weight1 :: Tensor
  , weight2 :: Tensor
  , steps   :: Int
  } deriving (Generic, Show)
```

```haskell do
let m = TwoLayer (ones' [2, 3]) (ones' [3]) 0
```

Collect every tensor:

```haskell eval
map shape (flattenValues (types @Tensor) m)
```

Rewrite every tensor — this is exactly how `toType` and `toDevice`
convert whole models; here we switch the model to half precision:

```haskell eval
map dtype (flattenValues (types @Tensor) (toType Half m))
```

Note the `Int` field rides along untouched; only the tensors are
visited.

## Field and shape lenses (typed)

In the typed API, dimensions can be records, and record fields become
lenses. `field @"r"` reads or writes the `r` component of an RGB
dimension, and its very existence is checked at compile time —
`field @"q"` would not compile.

```haskell top
data RGB a = RGB { r :: a, g :: a, b :: a } deriving (Generic, Show)
```

```haskell do
let img = T.fromUnnamed T.ones :: T.NamedTensor '( 'CPU, 0) 'Float '[Vector 2, RGB]
```

```haskell eval
T.toDynamic (img ^. TL.field @"r")
```

```haskell eval
T.toDynamic (img & TL.field @"g" .~ T.fromUnnamed T.zeros)
```

The outward traversal also exists in typed form, where it becomes
*shape-selective*: `types` at a typed tensor target visits only the
tensors of exactly that device, dtype and shape.

```haskell top
data TypedNet device = TypedNet
  { l1 :: T.Tensor device 'Float '[2, 3]
  , l2 :: T.Tensor device 'Float '[3, 4]
  , l3 :: T.Tensor device 'Float '[2, 3]
  } deriving Generic
```

```haskell do
let net = TypedNet T.ones T.ones T.zeros :: TypedNet '( 'CPU, 0)
```

Only the two `'[2, 3]` layers are visited; the `'[3, 4]` one is not:

```haskell eval
length (flattenValues (types @(T.Tensor '( 'CPU, 0) 'Float '[2, 3])) net)
```

```haskell eval
T.toDynamic (l3 (over (types @(T.Tensor '( 'CPU, 0) 'Float '[2, 3])) (+ 1) net))
```

`replaceValues` swaps specific structures wholesale — replace the two
`'[2, 3]` tensors and leave everything else:

```haskell eval
T.toDynamic (l1 (replaceValues (types @(T.Tensor '( 'CPU, 0) 'Float '[2, 3])) net [T.zeros, T.ones]))
```

## Retyping conversions (typed)

Finally, the typed counterparts of `toType`/`toDevice` change the
*type* along with the values. `Torch.Typed.DType.toDType` converts
every tensor in a model and rewrites the model's dtype parameter, so
after converting to `'Half` a stray `'Float` batch is a compile-time
error rather than a silent upcast:

```haskell
model                                :: Linear 10 1 'Float '( 'CPU, 0)
toDType @'Half @'Float model         :: Linear 10 1 'Half  '( 'CPU, 0)
toDevice @'( 'CUDA, 0) @'( 'CPU, 0) model
                                     :: Linear 10 1 'Float '( 'CUDA, 0)
```

See [Named Tensors and Lenses](08-named-tensors.html) for the
reference treatment of these conversions.

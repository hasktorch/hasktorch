---
title: Indexing and Slicing
---

# Indexing and Slicing

PyTorch programs lean heavily on `t[1, :, 1:3:2]`-style indexing.
Hasktorch supports this in both APIs: dynamically in the untyped API,
and with compile-time shape checking in the typed API.

## Untyped: `!` and the `slice` quasiquoter

The untyped `Torch.Tensor` module provides `(!)` together with a
family of index values (`None`, `Ellipsis`, `Slice`, booleans,
integers, tensors), and `Torch.Index` provides a quasiquoter for
PyTorch's textual syntax:

```haskell
import Torch
import Torch.Index

u = t ! (1 :: Int)               -- select
v = t ! [slice| 1, :, 1:3:2 |]   -- pytorch syntax, parsed at compile time
```

The syntax is parsed at compile time, but the *semantics* are dynamic:
shapes and bounds are only known when the program runs, so a mistake
surfaces as a runtime error, exactly as in Python.

## Typed: `getSlice` and the `slice` quasiquoter

In the typed API (`Torch.Typed.Index`) the index is a type-level
list, and the result shape is computed by the `IndexedShape` type
family:

```haskell
import Torch.Typed.Index

t :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 3, 4]

getSlice @'[SliceAt 1] t                   :: Tensor _ _ '[3, 4]
getSlice @'[SliceAll, SliceAt 0] t         :: Tensor _ _ '[2, 4]
getSlice @'[NewAxis, SliceFromUpTo 1 3] t  :: Tensor _ _ '[1, 2, 3, 4]
```

The index constructors follow the naming of the gradually-typed API's
indexing (PR #613): `SliceAt`, `SliceAll`, `NewAxis`, `SliceFrom`,
`SliceUpTo`, `SliceFromUpTo`, and `WithStep` variants; step lengths
are computed by ceiling division. Dimensions beyond the given indices
are kept unchanged, as in PyTorch.

Because the index is a type, the *same* `slice` quasiquoter that the
untyped API uses in expression position works here in type position —
one syntax for both APIs:

```haskell
getSlice @[slice| 1, :, 1:3:2 |] t
-- ≡ getSlice @'[SliceAt 1, SliceAll, SliceFromUpToWithStep 1 3 2] t

getSlice @[slice| None, :, 1:3 |] t   -- insert an axis, keep, then slice
getSlice @[slice| :, ::2 |] t         -- every second element of dimension 1
```

`setSlice` is the writing counterpart; the value's shape is forced to
match what `getSlice` with the same indices would produce:

```haskell
u = setSlice @'[SliceAt 0] t zeros
```

## What the types catch

Out-of-bounds indices, oversized slices, inverted ranges and zero
steps are all compile-time errors with readable messages:

```
• Index 5 is out of bounds for a dimension of size 2.
• Slice end 3 is out of bounds for a dimension of size 2.
• Slice step must be positive.
```

These checks compose with the quasiquoter. During development of the
test suite, `getSlice @[slice| None, 1:3 |]` applied to a `'[2, 3, 4]`
tensor was rejected at compile time — after `None` inserts an axis,
the `1:3` lands on the dimension of size 2 — a mistake that in Python
would only surface (or worse, silently clamp) at runtime.

## Limitations

Indices in the typed API are type-level naturals, so indices computed
at runtime do not fit this interface (use the untyped `!` for those,
or `Torch.Typed.Representable.index` for bounds-checked single-element
reads with `Finite` indices). Negative indices, `Ellipsis`, and
boolean or tensor ("fancy") indexing are not supported in the typed
layer — the latter two because their result shapes are not functions
of the types.

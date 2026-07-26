---
title: Graded and Staged Tensor Programs
---

# Graded and Staged Tensor Programs

This chapter covers three modules that grew out of one question: can
tensor programs be written element-by-element — the way the math is
written — without giving up whole-tensor execution?

## Why `Tensor` is not a `Monad`

A `Monad` instance must keep the type constructor fixed while the
element type changes. Tensor computations do the opposite: binding a
`'[Batch 2]` computation to a per-element continuation that produces
`'[RGB]` values yields a `'[Batch 2, RGB]` tensor — the *shape*
changes, and the element type (fixed by the dtype) does not. The
structure this actually is, is a monad *graded* by the monoid of
shapes under concatenation:

```haskell
-- Torch.Typed.Graded
greturn :: Grade m -> m '[]
gbind   :: m s -> (Grade m -> m t) -> m (s ++ t)
```

`'[]` is the unit of `++` and `++` is associative, so the graded
monad laws hold — and because they hold *definitionally* at the type
level, GHC accepts both sides of the associativity law at the same
type without any coercions. `QualifiedDo` gives back `do`-notation:

```haskell
{-# LANGUAGE QualifiedDo #-}
import qualified Torch.Typed.Graded as G

rgbOf :: TensorMonad '( 'D.CPU, 0) 'D.Float '[Batch 2, RGB]
rgbOf = G.do
  x <- batch          -- '[Batch 2]
  channelwise x       -- each element expands to '[RGB]
```

All three laws are verified against real tensors in the test suite.
`gbind` evaluates its continuation once per element, which makes it
the *reference semantics* — correct, slow, and the oracle that the
fast path below is tested against.

## Staged element-wise code

The fast path is Coyoneda-shaped: write the element function once,
*polymorphically*, and reinterpret it.

```haskell
-- Torch.Typed.Staged
f :: (Floating a, Cond a) => a -> a
f x = whereE (gtE x 0) (sin x * 10) 0
```

The same `f` runs under two instantiations:

- `a = Float` — per-element reference semantics;
- `a = Tensor` — every operation is a whole-tensor ATen call, so the
  body executes *once* regardless of tensor size, no per-element FFI
  happens, and autograd sees every step.

This works because the untyped `Tensor` has `Num`, `Fractional` and
`Floating` instances, and because the rank-2 type means the only
operations available inside `f` are class methods — parametricity
guarantees the function is a pointwise expression, so reinterpreting
it is sound. An already-monomorphic `Float -> Float` cannot be
vectorized this way; the polymorphic type is what keeps the code
inspectable.

Value-dependent control flow cannot use `if` (there is no `Bool` to
inspect at `a = Tensor`), so it goes through the `Cond` class:
`whereE` compiles to `torch.where`, comparisons return 0/1 masks, and
`maxE`/`minE` are overridden with native ATen calls in the `Tensor`
instance.

The entry points:

```haskell
emap     :: (forall a. (Floating a, Cond a) => a -> a)      -> NamedTensor d t s -> NamedTensor d t s
ezipWith :: (forall a. (Floating a, Cond a) => a -> a -> a) -> ...
gbindV   :: TensorMonad d dt s
         -> (forall a. (Floating a, Cond a) => a -> HList (ToFinites t) -> a)
         -> TensorMonad d dt (s ++ t)
```

`gbindV` satisfies `gbindV m k = gbind m (fromNamed . tabulate . k)`
— an equation the test suite checks with `gbind` as the oracle. The
two instantiations of a formula built from `+`, `*`, comparisons and
`whereE` agree bit-for-bit (IEEE basic operations are deterministic);
only transcendental functions differ within float tolerance.

## Case study: non-maximum suppression

`Torch.Typed.Vision` implements NMS in this style. The whole
algorithmic content of the pairwise step is one scalar formula:

```haskell
iou :: (Fractional a, Cond a) => Box a -> Box a -> a
iou a b = inter / (area a + area b - inter)
  where
    iw = maxE 0 (minE (x2 a) (x2 b) - maxE (x1 a) (x1 b))
    ih = maxE 0 (minE (y2 a) (y2 b) - maxE (y1 a) (y1 b))
    inter = iw * ih
    area v = (x2 v - x1 v) * (y2 v - y1 v)
```

`boxIou` instantiates it at `a = Tensor` with fields shaped `[n,1]`
and `[1,n]`, so broadcasting produces the whole matrix from one
evaluation; the tests instantiate the same code at `a = Float` as the
reference. The greedy suppression itself is three lines of list
recursion — the sequential part of the algorithm is where plain
Haskell is already the clearest notation, and no tensor machinery is
forced onto it. `nms` computes IoU rows lazily, only for boxes that
are actually kept, so memory stays `O(n)`.

Indicative CPU timings (FHD-scale boxes, `n` = pre-NMS top-k):
`n`=1000 ≈ 8 ms, `n`=3000 ≈ 22 ms, `n`=6000 ≈ 48 ms. A native C++
kernel is still a few times faster at the top end — the remaining gap
is eager per-op dispatch, discussed in
[TorchScript and the JIT](11-torchscript-and-jit.html).

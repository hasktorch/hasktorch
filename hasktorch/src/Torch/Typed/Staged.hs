{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Staged element-wise operations: write per-element code once, run it
-- vectorized.
--
-- The trick is Coyoneda-shaped, with the function space restricted to what can
-- be reinterpreted: element functions are written polymorphically,
--
-- > f :: (Floating a, Cond a) => a -> a
-- > f x = whereE (gtE x 0) (sin x * 10) 0
--
-- and the /same/ code runs under two instantiations:
--
-- * @a = Float@ — reference semantics, one call per element (what
--   'Torch.Typed.Representable.tabulate' and 'Torch.Typed.Graded.gbind' do);
-- * @a = Tensor@ — each operation is a whole-tensor ATen call, so the function
--   body executes /once/ regardless of tensor size, autograd sees every step,
--   and no per-element FFI happens.
--
-- Because the argument type is universally quantified, the only operations
-- available inside @f@ are the class methods — parametricity guarantees the
-- function is a pointwise arithmetic expression, so reinterpreting it at
-- @Tensor@ is sound.
--
-- An opaque, already-monomorphic @Float -> Float@ cannot be vectorized this
-- way (GHC has no runtime code inspection); the polymorphic type is what keeps
-- the code inspectable.
module Torch.Typed.Staged
  ( -- * Value-dependent control flow
    Cond (..),
    maxE,
    minE,

    -- * Vectorized element-wise operations
    emap,
    ezipWith,

    -- * Staged graded bind
    gbindV,
  )
where

import Data.Proxy (Proxy (..))
import qualified Torch.DType as DT
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import Torch.HList (HList, type (++))
import qualified Torch.Tensor as D
import Torch.Typed.Graded (TensorMonad, fromNamed, toNamed)
import Torch.Typed.Representable
import Torch.Typed.Tensor

-- | Branching and comparisons for staged element code.
--
-- @if@\/pattern matching on the element value cannot be reinterpreted at
-- @Tensor@ (there is no @Bool@ to inspect), so value-dependent control flow
-- goes through 'whereE', which compiles to @torch.where@.  Comparisons return
-- @0@\/@1@ masks in the same type @a@ so they compose with arithmetic.
class Cond a where
  -- | @whereE c t e@ is elementwise @if c /= 0 then t else e@.
  whereE :: a -> a -> a -> a

  ltE :: a -> a -> a
  leE :: a -> a -> a
  gtE :: a -> a -> a
  geE :: a -> a -> a
  eqE :: a -> a -> a
  neE :: a -> a -> a

instance Cond Float where
  whereE c t e = if c /= 0 then t else e
  ltE a b = if a < b then 1 else 0
  leE a b = if a <= b then 1 else 0
  gtE a b = if a > b then 1 else 0
  geE a b = if a >= b then 1 else 0
  eqE a b = if a == b then 1 else 0
  neE a b = if a /= b then 1 else 0

instance Cond Double where
  whereE c t e = if c /= 0 then t else e
  ltE a b = if a < b then 1 else 0
  leE a b = if a <= b then 1 else 0
  gtE a b = if a > b then 1 else 0
  geE a b = if a >= b then 1 else 0
  eqE a b = if a == b then 1 else 0
  neE a b = if a /= b then 1 else 0

-- | Elementwise maximum, via 'whereE'.
maxE :: Cond a => a -> a -> a
maxE a b = whereE (gtE a b) a b

-- | Elementwise minimum, via 'whereE'.
minE :: Cond a => a -> a -> a
minE a b = whereE (ltE a b) a b

instance Cond D.Tensor where
  whereE c t e = I.where' (F.toDType DT.Bool c) t e
  ltE a b = F.toDType (D.dtype a) (F.lt a b)
  leE a b = F.toDType (D.dtype a) (F.le a b)
  gtE a b = F.toDType (D.dtype a) (F.gt a b)
  geE a b = F.toDType (D.dtype a) (F.ge a b)
  eqE a b = F.toDType (D.dtype a) (F.eq a b)
  neE a b = F.toDType (D.dtype a) (F.ne a b)

-- | Element-wise map, executed as whole-tensor operations.
--
-- Semantically @emap f t = tabulate (f . index t)@, but @f@ runs once at
-- @a = Tensor@ instead of once per element.
emap ::
  forall shape dtype device.
  (forall a. (Floating a, Cond a) => a -> a) ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape
emap f = fromUnnamed . UnsafeMkTensor . f . toDynamic

-- | Element-wise combination of two same-shaped tensors, executed as
-- whole-tensor operations.
ezipWith ::
  forall shape dtype device.
  (forall a. (Floating a, Cond a) => a -> a -> a) ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape ->
  NamedTensor device dtype shape
ezipWith f x y = fromUnnamed . UnsafeMkTensor $ f (toDynamic x) (toDynamic y)

-- | Staged version of 'Torch.Typed.Graded.gbind'.  Definitionally,
--
-- > gbindV m k = gbind m (\x -> fromNamed (tabulate (k x)))
--
-- (this equation is checked in the test suite, with
-- 'Torch.Typed.Graded.gbind' as the oracle), but where 'Torch.Typed.Graded.gbind'
-- evaluates @k@ per element, 'gbindV' evaluates it once per /inner index/ on
-- the whole outer tensor: the cost is @numel t@ chains of ATen calls instead
-- of @numel s * numel t@ per-element FFI round trips, and gradients flow
-- through every step.
--
-- The inner indices are enumerated concretely, so this suits small inner
-- dimensions (channels, coordinates, classes).  A large inner dimension would
-- need a symbolic index (@arange@-based) instead.
gbindV ::
  forall t s device dtype.
  (HasIndex s, HasIndex t) =>
  TensorMonad device dtype s ->
  (forall a. (Floating a, Cond a) => a -> HList (ToFinites t) -> a) ->
  TensorMonad device dtype (s ++ t)
gbindV m k =
  fromNamed . fromUnnamed . UnsafeMkTensor $
    D.reshape (outerDims ++ innerDims) stacked
  where
    x = toDynamic (toNamed m)
    outerDims = dims (Proxy @s)
    innerDims = dims (Proxy @t)
    -- one whole-tensor evaluation of k per inner index, row-major, matching
    -- the element order of the reference gbind
    stacked =
      F.stack
        (F.Dim (length outerDims))
        [k x (fromInts @t ix) | ix <- allIndices innerDims]

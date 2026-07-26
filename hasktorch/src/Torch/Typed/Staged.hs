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
    --
    -- | Re-exported from "Torch.Elementwise", their untyped home: the same
    -- classes drive element-wise code over untyped tensors.
    Cond (..),

    -- * Vectorized element-wise operations
    emap,
    ezipWith,

    -- * Staged graded bind
    gbindV,
  )
where

import Data.Proxy (Proxy (..))
import Torch.Elementwise (Cond (..))
import qualified Torch.Functional as F
import Torch.HList (HList, type (++))
import qualified Torch.Tensor as D
import Torch.Typed.Graded (TensorMonad, fromNamed, toNamed)
import Torch.Typed.Representable
import Torch.Typed.Tensor

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

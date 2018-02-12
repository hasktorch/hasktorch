-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.Tensor.Static
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Tensors with dimensional phantom types.
--
-- Be aware of https://ghc.haskell.org/trac/ghc/wiki/Roles but since Dynamic
-- and static tensors are the same (minus the dimension operators in the
-- phantom type), I (@stites) don't think we need to be too concerned.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-orphans #-}
module Torch.Core.Tensor.Static
  ( ByteTensor
  , ShortTensor
  , IntTensor
  , LongTensor
  , FloatTensor
  , DoubleTensor
  -- helper constraints
  , StaticConstraint
  , StaticConstraint2

  -- experimental helper function (potentially delete)
  , withInplace

  -- generalized static functions
  , fromList
  , resizeAs
  , isSameSizeAs

  -- specialized static functions
  , fromList1d
  , newTranspose2d
  , expand2d

  -- reexports
  , IsStatic(..)
  ) where

import THTypes
import Foreign
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import Data.Proxy
import Control.Exception.Safe
import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num

import Torch.Class.C.Tensor.Static (IsStatic(..))
import qualified Torch.Core.Tensor.Dynamic as Class (IsTensor)

import qualified Torch.Core.ByteTensor.Static as B
import qualified Torch.Core.ShortTensor.Static as S
import qualified Torch.Core.IntTensor.Static as I
import qualified Torch.Core.LongTensor.Static as L
import qualified Torch.Core.FloatTensor.Static as F
import qualified Torch.Core.DoubleTensor.Static as D

import qualified Torch.Core.Storage as Storage
import qualified Torch.Core.LongStorage as L

import qualified Torch.Core.Tensor.Dynamic as Dynamic

type ByteTensor = B.Tensor
-- type CharTensor = C.Tensor
type ShortTensor = S.Tensor
type IntTensor = I.Tensor
type LongTensor = L.Tensor
-- type HalfTensor = H.Tensor
type FloatTensor = F.Tensor
type DoubleTensor = D.Tensor
type LongStorage = L.Storage

-- TODO: Slowly remove these generalized newtype instances as we get better static
-- checks
instance Class.IsTensor (ByteTensor   (d::[Nat]))
instance Class.IsTensor (ShortTensor  (d::[Nat]))
instance Class.IsTensor (IntTensor    (d::[Nat]))
instance Class.IsTensor (LongTensor   (d::[Nat]))
instance Class.IsTensor (FloatTensor  (d::[Nat]))
instance Class.IsTensor (DoubleTensor (d::[Nat]))

-- These instances can be derived
instance Dynamic.TensorCopy (ByteTensor   (d::[Nat]))
instance Dynamic.TensorCopy (ShortTensor  (d::[Nat]))
instance Dynamic.TensorCopy (IntTensor    (d::[Nat]))
instance Dynamic.TensorCopy (LongTensor   (d::[Nat]))
instance Dynamic.TensorCopy (FloatTensor  (d::[Nat]))
instance Dynamic.TensorCopy (DoubleTensor (d::[Nat]))

-- These might require changing
instance Dynamic.TensorConv (ByteTensor   (d::[Nat]))
instance Dynamic.TensorConv (ShortTensor  (d::[Nat]))
instance Dynamic.TensorConv (IntTensor    (d::[Nat]))
instance Dynamic.TensorConv (LongTensor   (d::[Nat]))
instance Dynamic.TensorConv (FloatTensor  (d::[Nat]))
instance Dynamic.TensorConv (DoubleTensor (d::[Nat]))

-- Some of these are dimension-specific. See 'Torch.Core.Tensor.Static.Random'
-- instance Dynamic.TensorRandom (ByteTensor   (d::[Nat]))
-- instance Dynamic.TensorRandom (ShortTensor  (d::[Nat]))
-- instance Dynamic.TensorRandom (IntTensor    (d::[Nat]))
-- instance Dynamic.TensorRandom (LongTensor   (d::[Nat]))
-- instance Dynamic.TensorRandom (FloatTensor  (d::[Nat]))
-- instance Dynamic.TensorRandom (DoubleTensor (d::[Nat]))

-- ========================================================================= --

-- Constraints that will be garunteed for every static tensor. Only 'Dynamic.IsTensor'
-- because we require downcasting for a lot of operations
type StaticConstraint t = (IsStatic t, HsReal t ~ HsReal (AsDynamic t), Dynamic.IsTensor (AsDynamic t), Num (HsReal t))

-- Constraints used on two static tensors. Essentially that both static tensors have
-- the same internal tensor representations.
type StaticConstraint2 t0 t1 = (StaticConstraint t0, StaticConstraint t1, AsDynamic t0 ~ AsDynamic t1)

-------------------------------------------------------------------------------

withInplace :: forall t d . (Dimensions d, StaticConstraint (t d)) => (AsDynamic (t d) -> IO ()) -> IO (t d)
withInplace op = do
  res <- Dynamic.newWithDim (dim :: Dim d)
  op res
  pure (asStatic res)

-------------------------------------------------------------------------------

-- | 'Dynamic.isSameSizeAs' without calling down through the FFI since we have
-- this information
isSameSizeAs
  :: forall t d d' . (IsStatic (t d), IsStatic (t d'), Dimensions d', Dimensions d)
  => t d -> t d' -> Bool
isSameSizeAs _ _ = dimVals (dim :: Dim d) == dimVals (dim :: Dim d')

-- | pure 'Dynamic.resizeAs'
resizeAs
  :: forall t d d' . StaticConstraint2 (t d) (t d')
  => (Dimensions d', Dimensions d)
  => t d -> IO (t d')
resizeAs src = do
  dummy :: AsDynamic (t d') <- Dynamic.new
  asStatic <$> Dynamic.resizeAs (asDynamic src) dummy


-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
fromList1d
  :: forall t n . (KnownNat n, Dynamic.IsTensor (t '[n]))
  => [HsReal (t '[n])] -> IO (t '[n])
fromList1d l
  | fromIntegral (natVal (Proxy :: Proxy n)) /= length l =
    throwString "List length does not match tensor dimensions"
  | otherwise = do
    res :: t '[n] <- Dynamic.new
    mapM_  (upd res) (zip [0..length l - 1] l)
    pure res
  where
    upd :: t '[n] -> (Int, HsReal (t '[n])) -> IO ()
    upd t (idx, v) = someDimsM [idx] >>= \sd -> Dynamic.setDim' t sd v

-- TODO: Potentially just use newWithData from Storage
fromList
  :: forall t d . (Dimensions d, StaticConstraint (t d))
  => [HsReal (t d)] -> IO (t d)
fromList l
  | product (dimVals d) /= length l =
    throwString "List length does not match tensor dimensions"
  | otherwise = do
    res :: AsDynamic (t d) <- Dynamic.newWithDim d
    mapM_  (upd res) (zip [0..length l - 1] l)
    pure (asStatic res)
  where
    d :: Dim d
    d = dim

    upd :: AsDynamic (t d) -> (Int, HsReal (t d)) -> IO ()
    upd t (idx, v) = someDimsM [idx] >>= \sd -> Dynamic.setDim' t sd v

newTranspose2d
  :: forall t r c . (StaticConstraint2 (t '[r, c]) (t '[c, r]))
  => t '[r, c] -> IO (t '[c, r])
newTranspose2d t =
  asStatic <$> Dynamic.newTranspose (asDynamic t) 1 0

-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
expand2d
  :: forall t d1 d2 . (KnownNatDim2 d1 d2)
  => StaticConstraint2 (t '[d2, d1]) (t '[d1])
  => Dynamic.TensorMath (AsDynamic (t '[d1])) -- for 'Dynamic.constant' which uses 'Torch.Class.C.Tensor.Math.fill'
  => t '[d1] -> IO (t '[d2, d1])
expand2d t = do
  res :: AsDynamic (t '[d2, d1]) <- Dynamic.constant 0
  s :: LongStorage <- Storage.newWithSize2 s2 s1
  Dynamic.expand res (asDynamic t) s
  pure (asStatic res)
  where
    s1, s2 :: Integer
    s1 = natVal (Proxy :: Proxy d1)
    s2 = natVal (Proxy :: Proxy d2)




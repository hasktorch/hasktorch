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
  , new

  -- reexports
  , IsStatic(..)
  , module X
  ) where

import THTypes
import Foreign ()
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import Data.Proxy
import Control.Exception.Safe
import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num

import qualified Torch.Core.Tensor.Dynamic as Dynamic

import qualified Torch.Core.Storage as Storage
import qualified Torch.Core.LongStorage as L

import Torch.Class.C.Tensor.Static (IsStatic(..))
import qualified Torch.Core.Tensor.Dynamic as Class (IsTensor)

-- import qualified Torch.Core.ByteTensor.Static as B
-- import qualified Torch.Core.ShortTensor.Static as S
-- import qualified Torch.Core.IntTensor.Static as I
-- import qualified Torch.Core.LongTensor.Static as L
-- import qualified Torch.Core.FloatTensor.Static as F
-- import qualified Torch.Core.DoubleTensor.Static as D
--
-- ========================================================================= --
-- re-export all SigTypes so that Aliases propogate
import qualified THByteTypes   as B
import qualified THShortTypes  as S
import qualified THIntTypes    as I
import qualified THLongTypes   as L
import qualified THFloatTypes  as F
import qualified THDoubleTypes as D


-- ========================================================================= --
-- re-export all IsTensor functions --
import Torch.Class.C.IsTensor as X hiding (resizeAs, isSameSizeAs, new) -- except for ones not specialized for static
import Torch.Core.ByteTensor.Static.IsTensor ()
import Torch.Core.ShortTensor.Static.IsTensor ()
import Torch.Core.IntTensor.Static.IsTensor ()
import Torch.Core.LongTensor.Static.IsTensor ()
import Torch.Core.FloatTensor.Static.IsTensor ()
import Torch.Core.DoubleTensor.Static.IsTensor ()

-- ========================================================================= --
-- re-export all Random functions --
import Torch.Class.C.Tensor.Random as X

import Torch.Core.ByteTensor.Static.Random   ()
import Torch.Core.ShortTensor.Static.Random  ()
import Torch.Core.IntTensor.Static.Random    ()
import Torch.Core.LongTensor.Static.Random   ()
import Torch.Core.FloatTensor.Static.Random  ()
import Torch.Core.DoubleTensor.Static.Random ()

import Torch.Core.FloatTensor.Static.Random.Floating  ()
import Torch.Core.DoubleTensor.Static.Random.Floating ()

-- ========================================================================= --
-- re-export all TensorCopy functions (for dynamic copies)
import Torch.Class.C.Tensor.Copy as X

import Torch.Core.ByteTensor.Static.Copy   ()
import Torch.Core.ShortTensor.Static.Copy  ()
import Torch.Core.IntTensor.Static.Copy    ()
import Torch.Core.LongTensor.Static.Copy   ()
import Torch.Core.FloatTensor.Static.Copy  ()
import Torch.Core.DoubleTensor.Static.Copy ()

-------------------------------------------------------------------------------

type ByteTensor   = B.Tensor
type ShortTensor  = S.Tensor
type IntTensor    = I.Tensor
type LongTensor   = L.Tensor
type FloatTensor  = F.Tensor
type DoubleTensor = D.Tensor
type LongStorage  = L.Storage


-- -- These might require changing
-- instance Dynamic.TensorConv (ByteTensor   (d::[Nat]))
-- instance Dynamic.TensorConv (ShortTensor  (d::[Nat]))
-- instance Dynamic.TensorConv (IntTensor    (d::[Nat]))
-- instance Dynamic.TensorConv (LongTensor   (d::[Nat]))
-- instance Dynamic.TensorConv (FloatTensor  (d::[Nat]))
-- instance Dynamic.TensorConv (DoubleTensor (d::[Nat]))

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
  res <- Dynamic.new (dim :: Dim d)
  op res
  pure (asStatic res)

-------------------------------------------------------------------------------

new :: forall t d . (Dimensions d, StaticConstraint (t d)) => IO (t d)
new = asStatic <$> Dynamic.new (dim :: Dim d)

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
  dummy :: AsDynamic (t d') <- Dynamic.new (dim :: Dim d')
  asStatic <$> Dynamic.resizeAs (asDynamic src) dummy


-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
fromList1d
  :: forall t n . (KnownNatDim n, Dynamic.IsTensor (t '[n]))
  => [HsReal (t '[n])] -> IO (t '[n])
fromList1d l
  | fromIntegral (natVal (Proxy :: Proxy n)) /= length l =
    throwString "List length does not match tensor dimensions"
  | otherwise = do
    res :: t '[n] <- Dynamic.new (dim :: Dim '[n])
    mapM_  (upd res) (zip [0..length l - 1] l)
    pure res
  where
    upd :: t '[n] -> (Int, HsReal (t '[n])) -> IO ()
    upd t (idx, v) = someDimsM [idx] >>= \sd -> Dynamic.setDim'_ t sd v

-- TODO: Potentially just use newWithData from Storage
fromList
  :: forall t d . (Dimensions d, StaticConstraint (t d))
  => [HsReal (t d)] -> IO (t d)
fromList l
  | product (dimVals d) /= length l =
    throwString "List length does not match tensor dimensions"
  | otherwise = do
    res :: AsDynamic (t d) <- Dynamic.new d
    mapM_  (upd res) (zip [0..length l - 1] l)
    pure (asStatic res)
  where
    d :: Dim d
    d = dim

    upd :: AsDynamic (t d) -> (Int, HsReal (t d)) -> IO ()
    upd t (idx, v) = someDimsM [idx] >>= \sd -> Dynamic.setDim'_ t sd v

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
  Dynamic.expand_ res (asDynamic t) s
  pure (asStatic res)
  where
    s1, s2 :: Integer
    s1 = natVal (Proxy :: Proxy d1)
    s2 = natVal (Proxy :: Proxy d2)

{-
-- retrieves a single row
tds_getRow :: forall n m . (KnownNatDim n, KnownNatDim m) => TDS '[n, m] -> Integer -> TDS '[1, m]
tds_getRow t r =
  if r >= 0 && r < ( round ((realToFrac $ natVal (Proxy :: Proxy n)) :: Double) :: Integer ) then
    unsafePerformIO $ do
    let res = tds_new
        indices_ :: TLS '[1] = tls_fromList [ r ]
    runManaged $ do
        tPtr <- managed $ withForeignPtr (getForeign t)
        resPtr <- managed $ withForeignPtr (getForeign res)
        iPtr <- managed $ withForeignPtr (tlsTensor indices_)
        liftIO $ c_THDoubleTensor_indexSelect resPtr tPtr 0 iPtr
    pure res
  else
    error "Row out of bounds"

tds_getColumn :: forall n m . (KnownNatDim n, KnownNatDim m) => TDS '[n, m] -> Integer -> TDS '[n, 1]
tds_getColumn t r =
  if r >= 0 && r < ( round ((realToFrac $ natVal (Proxy :: Proxy n)) :: Double) :: Integer ) then
    unsafePerformIO $ do
    let res = tds_new
        indices_ :: TLS '[1] = tls_fromList [ r ]
    runManaged $ do
        tPtr <- managed $ withForeignPtr (getForeign t)
        resPtr <- managed $ withForeignPtr (getForeign res)
        iPtr <- managed $ withForeignPtr (tlsTensor indices_)
        liftIO $ c_THDoubleTensor_indexSelect resPtr tPtr 1 iPtr
    pure res
  else
    error "Column out of bounds"

tds_getElem :: forall n m . (KnownNatDim n, KnownNatDim m) => TDS '[n, m] -> Int -> Int -> Double
tds_getElem t r c =
  if r >= 0 && r < ( round ((realToFrac $ natVal (Proxy :: Proxy n)) :: Double) :: Int ) &&
     c >= 0 && c < ( round ((realToFrac $ natVal (Proxy :: Proxy m)) :: Double) :: Int ) then
    unsafePerformIO $ do
    e <- withForeignPtr (tdsTensor t) (\t_ ->
                                              pure $
                                                  c_THDoubleTensor_get2d
                                                  t_
                                                  (fromIntegral r)
                                                  (fromIntegral c))
    pure $ realToFrac e
  else
    error "Indices out of bounds"

tds_setElem :: forall n m . (KnownNatDim n, KnownNatDim m) => TDS '[n, m] -> Int -> Int -> Double -> TDS '[n, m]
tds_setElem t r c v = apply1__ tSet t
  where tSet r_ = c_THDoubleTensor_set2d r_ (fromIntegral r) (fromIntegral c) (realToFrac v)

-}

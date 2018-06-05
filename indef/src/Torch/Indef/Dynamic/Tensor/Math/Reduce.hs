-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Reduce
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Dynamic.Tensor.Math.Reduce
  ( minall
  , maxall
  , medianall
  , sumall
  , prodall
  , _max
  , _min
  , _median
  , _sum
  , _prod
  , Torch.Indef.Dynamic.Tensor.Math.Reduce.max
  , Torch.Indef.Dynamic.Tensor.Math.Reduce.min
  , median
  ) where

import System.IO.Unsafe
import Numeric.Dimensions

import Torch.Indef.Types

import Torch.Indef.Dynamic.Tensor
import qualified Torch.Indef.Index as Ix
import qualified Torch.Sig.Tensor.Math.Reduce as Sig

-- | get the minima of a tensor's elements
minall :: Dynamic -> HsReal
minall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_minall)

-- | get the maxima of a tensor's elements
maxall :: Dynamic -> HsReal
maxall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_maxall)

-- | get the median value of a tensor's elements
medianall :: Dynamic -> HsReal
medianall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_medianall)

-- | get the sum of a tensor's elements
sumall :: Dynamic -> HsAccReal
sumall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsAccReal .: Sig.c_sumall)

-- | get the product of a tensor's elements
prodall :: Dynamic -> HsAccReal
prodall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsAccReal .: Sig.c_prodall)

-- | get the maximal value in the specified dimension and a corresponding index tensor of the maximum value's index.
--
-- Inplace and C-Style mutation
_max :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_max (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_max s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

-- | get the minimal value in the specified dimension and a corresponding index tensor of the minimum value's index.
--
-- Inplace and C-Style mutation
_min :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_min (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_min s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

-- | get the median value in the specified dimension and a corresponding index tensor of the median value's index.
--
-- Inplace and C-Style mutation
_median :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_median (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_median s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

-- | sum the tensor in the specified dimension.
--
-- Inplace and C-Style mutation
_sum :: Dynamic -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_sum t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_sum (fromIntegral i0) (fromKeepDim i1)

-- | take the product of the tensor in the specified dimension.
--
-- Inplace and C-Style mutation
_prod :: Dynamic -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_prod t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_prod (fromIntegral i0) (fromKeepDim i1)

withKeepDim
  :: ((Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ())
  -> Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
withKeepDim _fn t d k = do
  tdim@(SomeDims d')<- getDims t
  let (i:_) = listDims d'
  ret :: Dynamic      <- new' tdim
  let ix = Ix.newIxDyn i
  _fn (ret, ix) t d k
  pure (ret, maybe (Just ix) (pure Nothing) k)

-- | get the maximum value in the specified dimension and return an optional corresponding index tensor of the maximum value's index.
max :: Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
max = withKeepDim _max

-- | get the minimum value in the specified dimension and return an optional corresponding index tensor of the minimum value's index.
min :: Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
min = withKeepDim _min

-- | get the median value in the specified dimension and return an optional corresponding index tensor of the median value's index.
median :: Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
median = withKeepDim _median


-- * not in THC.BYte
-- c_renorm :: Ptr CState -> t -> t -> HsReal t -> CInt -> HsReal t -> IO ()
-- c_std :: Ptr CState -> t -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
-- c_stdall :: Ptr CState -> Ptr CTensor -> CInt -> HsReal t
-- c_var :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
-- c_varall :: Ptr CState -> Ptr CTensor -> CInt -> HsReal t
-- c_dist :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> HsReal t

-- * not in TH.Byte
-- c_norm :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> CInt -> CInt -> IO ()
-- c_normall :: Ptr CState -> Ptr CTensor -> HsReal t -> HsReal t
-- c_mean :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
-- c_meanall :: Ptr CState -> Ptr CTensor -> HsReal t

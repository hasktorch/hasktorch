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
{-# OPTIONS_GHC -fno-cse #-}
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

import Control.Monad.Managed
import Foreign (withForeignPtr)
import System.IO.Unsafe
import Numeric.Dimensions

import Torch.Indef.Types

import Torch.Indef.Dynamic.Tensor
import qualified Torch.Indef.Index as Ix
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.Math.Reduce as Sig

-- | get the minima of a tensor's elements
minall :: Dynamic -> HsReal
minall t = unsafeDupablePerformIO . flip with (pure . c2hsReal) . (liftIO =<<) $ Sig.c_minall
  <$> managedState
  <*> managedTensor t
{-# NOINLINE minall #-}

-- | get the maxima of a tensor's elements
maxall :: Dynamic -> HsReal
maxall t = unsafeDupablePerformIO . flip with (pure . c2hsReal) . (liftIO =<<) $ Sig.c_maxall
  <$> managedState
  <*> managedTensor t
{-# NOINLINE maxall #-}

-- | get the median value of a tensor's elements
medianall :: Dynamic -> HsReal
medianall t = unsafeDupablePerformIO . flip with (pure . c2hsReal) . (liftIO =<<) $ Sig.c_medianall
  <$> managedState
  <*> managedTensor t
{-# NOINLINE medianall #-}

-- | get the sum of a tensor's elements
sumall :: Dynamic -> HsAccReal
sumall t = unsafeDupablePerformIO . flip with (pure . c2hsAccReal) . (liftIO =<<) $ Sig.c_sumall
  <$> managedState
  <*> managedTensor t
{-# NOINLINE sumall #-}

-- | get the product of a tensor's elements
prodall :: Dynamic -> HsAccReal
prodall t = unsafeDupablePerformIO . flip with (pure . c2hsAccReal) . (liftIO =<<) $ Sig.c_prodall
  <$> managedState
  <*> managedTensor t
{-# NOINLINE prodall #-}

-- | get the maximal value in the specified dimension and a corresponding index tensor of the maximum value's index.
--
-- Inplace and C-Style mutation
_max
  :: (Dynamic, IndexDynamic) -> Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim -> IO ()
_max (t0, ix) t1 i0 i1 = withLift $ Sig.c_max
  <$> managedState
  <*> managedTensor t0
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromKeepDim i1)
{-# NOINLINE _max #-}

-- | get the minimal value in the specified dimension and a corresponding index tensor of the minimum value's index.
--
-- Inplace and C-Style mutation
_min :: (Dynamic, IndexDynamic) -> Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim -> IO ()
_min (t0, ix) t1 i0 i1  = withLift $ Sig.c_min
  <$> managedState
  <*> managedTensor t0
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromKeepDim i1)
{-# NOINLINE _min #-}

-- | get the median value in the specified dimension and a corresponding index tensor of the median value's index.
--
-- Inplace and C-Style mutation
_median :: (Dynamic, IndexDynamic) -> Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim -> IO ()
_median (t0, ix) t1 i0 i1 = withLift $ Sig.c_median
  <$> managedState
  <*> managedTensor t0
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromKeepDim i1)
{-# NOINLINE _median #-}

-- | sum the tensor in the specified dimension.
--
-- Inplace and C-Style mutation
_sum :: Dynamic -> Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim -> IO ()
_sum t0 t1 i0 i1 = withLift $ Sig.c_sum
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromKeepDim i1)
{-# NOINLINE _sum #-}

-- | take the product of the tensor in the specified dimension.
--
-- Inplace and C-Style mutation
_prod
  :: Dynamic -> Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim -> IO ()
_prod t0 t1 i0 i1 = withLift $ Sig.c_prod
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromKeepDim i1)
{-# NOINLINE _prod #-}


withKeepDim
  :: ((Dynamic, IndexDynamic) -> Dynamic -> Word -> Maybe KeepDim -> IO ())
  -> Dynamic -> Word -> Maybe KeepDim -> (Dynamic, Maybe (IndexDynamic))
withKeepDim _fn t d k = unsafeDupablePerformIO $ do
  _fn (ret, ix) t d k
  pure (ret, maybe Nothing (\(KeepDim b) -> if b then Just ix else Nothing) k)
  where
    tdim = getSomeDims t
    (i:_) = shape t
    ret :: Dynamic = new' tdim
    ix = Ix.newIxDyn [i]
{-# NOINLINE withKeepDim #-}

-- | get the maximum value in the specified dimension and return an optional corresponding index tensor of the maximum value's index.
max
  :: Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim
  -> (Dynamic, Maybe (IndexDynamic))
max = withKeepDim _max

-- | get the minimum value in the specified dimension and return an optional corresponding index tensor of the minimum value's index.
min
  :: Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim
  -> (Dynamic, Maybe (IndexDynamic))
min = withKeepDim _min

-- | get the median value in the specified dimension and return an optional corresponding index tensor of the median value's index.
median
  :: Dynamic
  -> Word -- ^ dimension to operate over
  -> Maybe KeepDim
  -> (Dynamic, Maybe (IndexDynamic))
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

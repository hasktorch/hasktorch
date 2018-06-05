{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Dynamic.Tensor.Math.Reduce where

import System.IO.Unsafe
import Numeric.Dimensions

import Torch.Indef.Types

import Torch.Indef.Dynamic.Tensor
import qualified Torch.Indef.Index as Ix
import qualified Torch.Sig.Tensor.Math.Reduce as Sig

minall :: Dynamic -> HsReal
minall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_minall)

maxall :: Dynamic -> HsReal
maxall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_maxall)

medianall :: Dynamic -> HsReal
medianall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsReal .: Sig.c_medianall)

sumall :: Dynamic -> HsAccReal
sumall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsAccReal .: Sig.c_sumall)

prodall :: Dynamic -> HsAccReal
prodall = unsafeDupablePerformIO . flip withDynamicState (fmap c2hsAccReal .: Sig.c_prodall)

_max :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_max (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_max s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_min :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_min (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_min s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_median :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_median (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_median s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_sum :: Dynamic -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_sum t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_sum (fromIntegral i0) (fromKeepDim i1)

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

max, min, median
  :: Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
max = withKeepDim _max
min = withKeepDim _min
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

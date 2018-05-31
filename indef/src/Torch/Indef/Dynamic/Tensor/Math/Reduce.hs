{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Dynamic.Tensor.Math.Reduce where

import Torch.Indef.Types
import Torch.Dimensions

import Torch.Indef.Dynamic.Tensor
import Torch.Indef.Index
import qualified Torch.Sig.Tensor.Math.Reduce as Sig

minall :: Dynamic -> IO HsReal
minall = flip withDynamicState (fmap c2hsReal .: Sig.c_minall)

maxall :: Dynamic -> IO HsReal
maxall = flip withDynamicState (fmap c2hsReal .: Sig.c_maxall)

medianall :: Dynamic -> IO HsReal
medianall = flip withDynamicState (fmap c2hsReal .: Sig.c_medianall)

sumall :: Dynamic -> IO HsAccReal
sumall = flip withDynamicState (fmap c2hsAccReal .: Sig.c_sumall)

prodall :: Dynamic -> IO HsAccReal
prodall = flip withDynamicState (fmap c2hsAccReal .: Sig.c_prodall)

_max :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_max (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  withIx ix $ \ix' ->
    Sig.c_max s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_min :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_min (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  withIx ix $ \ix' ->
    Sig.c_min s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_median :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_median (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  withIx ix $ \ix' ->
    Sig.c_median s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)

_sum :: Dynamic -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_sum t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_sum (fromIntegral i0) (fromKeepDim i1)

_prod :: Dynamic -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_prod t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_prod (fromIntegral i0) (fromKeepDim i1)

withKeepDim
  :: ((Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ())
  -> Dynamic -> DimVal -> Maybe KeepDim -> IO (Dynamic, Maybe (IndexDynamic))
withKeepDim _fn t d k = do
  tdim <- getDims t
  let (i:_) = dimVals' tdim
  ret :: Dynamic      <- new' tdim
  ix  :: IndexDynamic <- newIxDyn i
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
-- c_stdall :: Ptr CState -> Ptr CTensor -> CInt -> IO HsReal t
-- c_var :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
-- c_varall :: Ptr CState -> Ptr CTensor -> CInt -> IO HsReal t
-- c_dist :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> IO HsReal t

-- * not in TH.Byte
-- c_norm :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> CInt -> CInt -> IO ()
-- c_normall :: Ptr CState -> Ptr CTensor -> HsReal t -> IO HsReal t
-- c_mean :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
-- c_meanall :: Ptr CState -> Ptr CTensor -> IO HsReal t

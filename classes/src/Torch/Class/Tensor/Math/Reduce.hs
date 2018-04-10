{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Class.Tensor.Math.Reduce where

import Torch.Class.Types
import Torch.Class.Tensor
import Data.Word
import Data.Int
import Torch.Dimensions
-- import Torch.Types.TH hiding (IndexDynamic)

class TensorMathReduce t where
  minall       :: t -> IO (HsReal t)
  maxall       :: t -> IO (HsReal t)
  medianall    :: t -> IO (HsReal t)
  sumall       :: t -> IO (HsAccReal t)
  prodall      :: t -> IO (HsAccReal t)
  _max         :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  _min         :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  _median      :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  _sum         :: t -> t -> DimVal -> Maybe KeepDim -> IO ()
  _prod        :: t -> t -> DimVal -> Maybe KeepDim -> IO ()


withKeepDim
  :: (TensorMathReduce t, IsTensor t, IsTensor (IndexDynamic t))
  => ((t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ())
  -> t -> DimVal -> Maybe KeepDim -> IO (t, Maybe (IndexDynamic t))
withKeepDim _fn t d k = do
  tdim <- getDims t
  ret :: t              <- new' tdim
  ix  :: IndexDynamic t <- new' tdim
  _fn (ret, ix) t d k
  pure (ret, maybe (Just ix) (pure Nothing) k)

max, min, median
  :: (TensorMathReduce t, IsTensor t, IsTensor (IndexDynamic t))
  => t -> DimVal -> Maybe KeepDim -> IO (t, Maybe (IndexDynamic t))
max = withKeepDim _max
min = withKeepDim _min
median = withKeepDim _median

class TensorMathReduceFloating t where
  _mean         :: t -> t -> Int -> Int -> IO ()
  _std          :: t -> t -> Int -> Int -> Int -> IO ()
  _var          :: t -> t -> Int -> Int -> Int -> IO ()
  _norm         :: t -> t -> HsReal t -> Int -> Int -> IO ()
  _renorm       :: t -> t -> HsReal t -> Int -> HsReal t -> IO ()
  dist          :: t -> t -> HsReal t -> IO (HsAccReal t)
  meanall       :: t -> IO (HsAccReal t)
  varall        :: t -> Int -> IO (HsAccReal t)
  stdall        :: t -> Int -> IO (HsAccReal t)
  normall       :: t -> HsReal t -> IO (HsAccReal t)


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

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
  max_         :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  min_         :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  median_      :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()
  sum_         :: t -> t -> DimVal -> Maybe KeepDim -> IO ()
  prod_        :: t -> t -> DimVal -> Maybe KeepDim -> IO ()


withKeepDim
  :: (TensorMathReduce t, Tensor t, Tensor (IndexDynamic t))
  => ((t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ())
  -> t -> DimVal -> Maybe KeepDim -> IO (t, Maybe (IndexDynamic t))
withKeepDim fn_ t d k = do
  tdim <- getDims t
  ret :: t              <- new' tdim
  ix  :: IndexDynamic t <- new' tdim
  fn_ (ret, ix) t d k
  pure (ret, maybe (Just ix) (pure Nothing) k)

max, min, median
  :: (TensorMathReduce t, Tensor t, Tensor (IndexDynamic t))
  => t -> DimVal -> Maybe KeepDim -> IO (t, Maybe (IndexDynamic t))
max = withKeepDim max_
min = withKeepDim min_
median = withKeepDim median_

{-
class TensorMathReduceFloating t where
  dist    :: t -> t -> HsReal t -> IO (HsAccReal t)
  var     :: t -> t -> CInt -> CInt -> CInt -> IO ()
  varall  :: t -> CInt -> IO (HsAccReal t)
  std     :: t -> t -> CInt -> CInt -> CInt -> IO ()
  stdall  :: t -> CInt -> IO (HsAccReal t)
  renorm  :: t -> t -> HsReal t -> CInt -> HsReal t -> IO ()
  norm    :: t -> t -> HsReal t -> CInt -> CInt -> IO ()
  normall :: t -> HsReal t -> IO (HsAccReal t)
  mean    :: t -> t -> CInt -> CInt -> IO ()
  meanall :: t -> IO (HsAccReal t)
-}

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

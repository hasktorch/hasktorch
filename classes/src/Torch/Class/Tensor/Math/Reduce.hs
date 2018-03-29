module Torch.Class.Tensor.Math.Reduce where

import Foreign
import Foreign.C.Types
import Torch.Class.Types
import Data.Word
import Data.Int
import Torch.Types.TH hiding (IndexTensor)

class TensorMathReduce t where
  minall       :: t -> IO (HsReal t)
  maxall       :: t -> IO (HsReal t)
  medianall    :: t -> IO (HsReal t)
  sumall       :: t -> IO (HsAccReal t)
  prodall      :: t -> IO (HsAccReal t)
  max_         :: (t, IndexTensor t) -> t -> Int -> Int -> IO ()
  min_         :: (t, IndexTensor t) -> t -> Int -> Int -> IO ()
  median_      :: (t, IndexTensor t) -> t -> Int -> Int -> IO ()
  sum_         :: t -> t -> Int -> Int -> IO ()
  prod_        :: t -> t -> Int -> Int -> IO ()

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

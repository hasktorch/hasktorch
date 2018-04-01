module Torch.Class.Tensor.Math.Lapack.Static where

import GHC.Int
import Torch.Class.Types
import Torch.Dimensions

class TensorMathLapack t where
  getri_     :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  potri_     :: (Dimensions d, Dimensions d') => t d -> t d' -> [Int8] -> IO ()
  potrf_     :: (Dimensions d, Dimensions d') => t d -> t d' -> [Int8] -> IO ()
  geqrf_     :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> IO ()
  qr_        :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> IO ()
  geev_      :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> [Int8] -> IO ()
  potrs_     :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> [Int8] -> IO ()
  syev_      :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> [Int8] -> [Int8] -> IO ()
  gesv_      :: (Dimensions d, Dimensions d', Dimensions d'', Dimensions d''') => t d -> t d' -> t d'' -> t d''' -> IO ()
  gels_      :: (Dimensions d, Dimensions d', Dimensions d'', Dimensions d''') => t d -> t d' -> t d'' -> t d''' -> IO ()
  gesvd_     :: (Dimensions d, Dimensions d', Dimensions d'', Dimensions d''') => t d -> t d' -> t d'' -> t d''' -> [Int8] -> IO ()
  gesvd2_    :: (Dimensions d, Dimensions d', Dimensions d'', Dimensions d''', Dimensions d'''') => t d -> t d' -> t d'' -> t d''' -> t d'''' -> [Int8] -> IO ()

-- class CPUTensorMathLapack t where
--   trtrs_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
--   orgqr_     :: t -> t -> t -> IO ()
--   ormqr_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
--   pstrf_     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
--   btrifact_  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
--   btrisolve_ :: t -> t -> t -> Int.DynTensor -> IO ()

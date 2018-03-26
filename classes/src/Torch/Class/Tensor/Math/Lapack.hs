module Torch.Class.Tensor.Math.Lapack where

import GHC.Int
import Torch.Class.Types
import qualified Torch.Types.TH.Int as Int

class TensorLapack t where
  gesv_      :: t -> t -> t -> t -> IO ()
  gels_      :: t -> t -> t -> t -> IO ()
  syev_      :: t -> t -> t -> [Int8] -> [Int8] -> IO ()
  geev_      :: t -> t -> t -> [Int8] -> IO ()
  gesvd_     :: t -> t -> t -> t -> [Int8] -> IO ()
  gesvd2_    :: t -> t -> t -> t -> t -> [Int8] -> IO ()
  getri_     :: t -> t -> IO ()
  potrf_     :: t -> t -> [Int8] -> IO ()
  potrs_     :: t -> t -> t -> [Int8] -> IO ()
  potri_     :: t -> t -> [Int8] -> IO ()
  geqrf_     :: t -> t -> t -> IO ()
  qr_        :: t -> t -> t -> IO ()

class CPUTensorMathLapack t where
  trtrs_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
  orgqr_     :: t -> t -> t -> IO ()
  ormqr_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
  pstrf_     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
  btrifact_  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
  btrisolve_ :: t -> t -> t -> Int.DynTensor -> IO ()

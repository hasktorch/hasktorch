module Torch.Class.Tensor.Lapack where

import Torch.Types.TH
import Foreign
import Foreign.C.Types
import GHC.Int
import Torch.Class.Internal
import qualified Torch.Types.TH.Int as Int

class TensorLapack t where
  gesv_      :: t -> t -> t -> t -> IO ()
  trtrs_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
  gels_      :: t -> t -> t -> t -> IO ()
  syev_      :: t -> t -> t -> [Int8] -> [Int8] -> IO ()
  geev_      :: t -> t -> t -> [Int8] -> IO ()
  gesvd_     :: t -> t -> t -> t -> [Int8] -> IO ()
  gesvd2_    :: t -> t -> t -> t -> t -> [Int8] -> IO ()
  getri_     :: t -> t -> IO ()
  potrf_     :: t -> t -> [Int8] -> IO ()
  potrs_     :: t -> t -> t -> [Int8] -> IO ()
  potri_     :: t -> t -> [Int8] -> IO ()
  qr_        :: t -> t -> t -> IO ()
  geqrf_     :: t -> t -> t -> IO ()
  orgqr_     :: t -> t -> t -> IO ()
  ormqr_     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
  pstrf_     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
  btrifact_  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
  btrisolve_ :: t -> t -> t -> Int.DynTensor -> IO ()

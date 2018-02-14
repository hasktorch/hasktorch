module Torch.Class.C.Tensor.Lapack where

import THTypes
import Foreign
import Foreign.C.Types
import GHC.Int
import Torch.Class.C.Internal

class TensorLapack t where
  gesv      :: t -> t -> t -> t -> IO ()
  trtrs     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
  gels      :: t -> t -> t -> t -> IO ()
  syev      :: t -> t -> t -> [Int8] -> [Int8] -> IO ()
  geev      :: t -> t -> t -> [Int8] -> IO ()
  gesvd     :: t -> t -> t -> t -> [Int8] -> IO ()
  gesvd2    :: t -> t -> t -> t -> t -> [Int8] -> IO ()
  getri     :: t -> t -> IO ()
  potrf     :: t -> t -> [Int8] -> IO ()
  potrs     :: t -> t -> t -> [Int8] -> IO ()
  potri     :: t -> t -> [Int8] -> IO ()
  qr        :: t -> t -> t -> IO ()
  geqrf     :: t -> t -> t -> IO ()
  orgqr     :: t -> t -> t -> IO ()
  ormqr     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
  pstrf     :: t -> Ptr CTHIntTensor -> t -> [Int8] -> HsReal t -> IO ()
  btrifact  :: t -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> Int32 -> t -> IO ()
  btrisolve :: t -> t -> t -> Ptr CTHIntTensor -> IO ()

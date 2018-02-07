module Torch.Class.Tensor.Lapack where

import THTypes
import Foreign
import Foreign.C.Types
import GHC.Int
import Torch.Class.Internal

class TensorLapack t where
  gesv      :: t -> t -> t -> t -> IO ()
  trtrs     :: t -> t -> t -> t -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()
  gels      :: t -> t -> t -> t -> IO ()
  syev      :: t -> t -> t -> Ptr CChar -> Ptr CChar -> IO ()
  geev      :: t -> t -> t -> Ptr CChar -> IO ()
  gesvd     :: t -> t -> t -> t -> Ptr CChar -> IO ()
  gesvd2    :: t -> t -> t -> t -> t -> Ptr CChar -> IO ()
  getri     :: t -> t -> IO ()
  potrf     :: t -> t -> Ptr CChar -> IO ()
  potrs     :: t -> t -> t -> Ptr CChar -> IO ()
  potri     :: t -> t -> Ptr CChar -> IO ()
  qr        :: t -> t -> t -> IO ()
  geqrf     :: t -> t -> t -> IO ()
  orgqr     :: t -> t -> t -> IO ()
  ormqr     :: t -> t -> t -> t -> Ptr CChar -> Ptr CChar -> IO ()
  pstrf     :: t -> Ptr CTHIntTensor -> t -> Ptr CChar -> HsReal t -> IO ()
  btrifact  :: t -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> Int32 -> t -> IO ()
  btrisolve :: t -> t -> t -> Ptr CTHIntTensor -> IO ()

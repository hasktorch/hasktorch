module Torch.Class.Tensor.Math.Lapack where

import GHC.Int
import Torch.Class.Types
import qualified Torch.Types.TH.Int as Int

class TensorMathLapack t where
  _getri     :: t -> t -> IO ()
  _potrf     :: t -> t -> [Int8] -> IO ()
  _potri     :: t -> t -> [Int8] -> IO ()
  _potrs     :: t -> t -> t -> [Int8] -> IO ()
  _geqrf     :: t -> t -> t -> IO ()
  _qr        :: t -> t -> t -> IO ()
  _geev      :: t -> t -> t -> [Int8] -> IO ()
  _syev      :: t -> t -> t -> [Int8] -> [Int8] -> IO ()
  _gesv      :: t -> t -> t -> t -> IO ()
  _gels      :: t -> t -> t -> t -> IO ()
  _gesvd     :: t -> t -> t -> t -> [Int8] -> IO ()
  _gesvd2    :: t -> t -> t -> t -> t -> [Int8] -> IO ()

{-
class CPUTensorMathLapack t where
  _trtrs     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
  _orgqr     :: t -> t -> t -> IO ()
  _ormqr     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
  _pstrf     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
  _btrifact  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
  _btrisolve :: t -> t -> t -> Int.DynTensor -> IO ()
-}

module Torch.Class.Tensor.Math.Lapack.Static where

import GHC.Int
import Torch.Class.Types
import Torch.Dimensions

class TensorMathLapack t where
  _getri     :: (Dimensions2 d d') => t d -> t d' -> IO ()
  _potri     :: (Dimensions2 d d') => t d -> t d' -> [Int8] -> IO ()
  _potrf     :: (Dimensions2 d d') => t d -> t d' -> [Int8] -> IO ()
  _geqrf     :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> IO ()
  _qr        :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> IO ()
  _geev      :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> [Int8] -> IO ()
  _potrs     :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> [Int8] -> IO ()
  _syev      :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> [Int8] -> [Int8] -> IO ()
  _gesv      :: (Dimensions4 d d' d'' d''') => t d -> t d' -> t d'' -> t d''' -> IO ()
  _gels      :: (Dimensions4 d d' d'' d''') => t d -> t d' -> t d'' -> t d''' -> IO ()
  _gesvd     :: (Dimensions4 d d' d'' d''') => t d -> t d' -> t d'' -> t d''' -> [Int8] -> IO ()
  _gesvd2    :: (Dimensions4 d d' d'' d''', Dimensions d'''') => t d -> t d' -> t d'' -> t d''' -> t d'''' -> [Int8] -> IO ()

-- class CPUTensorMathLapack t where
--   _trtrs     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
--   _orgqr     :: t -> t -> t -> IO ()
--   _ormqr     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
--   _pstrf     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
--   _btrifact  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
--   _btrisolve :: t -> t -> t -> Int.DynTensor -> IO ()

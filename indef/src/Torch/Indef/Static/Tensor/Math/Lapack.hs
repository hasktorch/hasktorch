module Torch.Indef.Static.Tensor.Math.Lapack where

import GHC.Int
import Torch.Dimensions

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Lapack as Dynamic

_getri :: Tensor d -> Tensor d' -> IO ()
_getri a b = Dynamic._getri (asDynamic a) (asDynamic b)

_potri :: Tensor d -> Tensor d' -> [Int8] -> IO ()
_potri a b = Dynamic._potri (asDynamic a) (asDynamic b)

_potrf :: Tensor d -> Tensor d' -> [Int8] -> IO ()
_potrf a b = Dynamic._potrf (asDynamic a) (asDynamic b)

_geqrf :: Tensor d -> Tensor d' -> Tensor d'' -> IO ()
_geqrf a b c = Dynamic._geqrf (asDynamic a) (asDynamic b) (asDynamic c)

_qr :: Tensor d -> Tensor d' -> Tensor d'' -> IO ()
_qr a b c = Dynamic._qr (asDynamic a) (asDynamic b) (asDynamic c)

_geev :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> IO ()
_geev a b c = Dynamic._geev (asDynamic a) (asDynamic b) (asDynamic c)

_potrs :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> IO ()
_potrs a b c = Dynamic._potrs (asDynamic a) (asDynamic b) (asDynamic c)

_syev :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> [Int8] -> IO ()
_syev a b c = Dynamic._syev (asDynamic a) (asDynamic b) (asDynamic c)

_gesv :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> IO ()
_gesv a b c d = Dynamic._gesv (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

_gels :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> IO ()
_gels a b c d = Dynamic._gels (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

_gesvd :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> [Int8] -> IO ()
_gesvd a b c d = Dynamic._gesvd (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

_gesvd2 :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d'''' -> [Int8] -> IO ()
_gesvd2 a b c d e = Dynamic._gesvd2 (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d) (asDynamic e)

-- class CPUTensorMathLapack t where
--   _trtrs     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()
--   _orgqr     :: t -> t -> t -> IO ()
--   _ormqr     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
--   _pstrf     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
--   _btrifact  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
--   _btrisolve :: t -> t -> t -> Int.DynTensor -> IO ()

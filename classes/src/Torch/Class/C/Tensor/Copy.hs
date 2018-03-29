module Torch.Class.C.Tensor.Copy where

import THTypes
import Foreign (Ptr)
import qualified THByteTypes    as B
import qualified THShortTypes   as S
import qualified THIntTypes     as I
import qualified THLongTypes    as L
import qualified THFloatTypes   as F
import qualified THDoubleTypes  as D


class TensorCopy t where
  copy       :: t -> IO t
  copyByte   :: t -> IO B.DynTensor
  -- copyChar   :: t -> IO .DynTensor
  copyShort  :: t -> IO S.DynTensor
  copyInt    :: t -> IO I.DynTensor
  copyLong   :: t -> IO L.DynTensor
  -- copyHalf   :: t -> IO .DynTensor
  copyFloat  :: t -> IO F.DynTensor
  copyDouble :: t -> IO D.DynTensor

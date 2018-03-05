module Torch.Class.Tensor.Copy where

import Torch.Types.TH
import Foreign (Ptr)
import qualified Torch.Types.TH.Byte    as B
import qualified Torch.Types.TH.Short   as S
import qualified Torch.Types.TH.Int     as I
import qualified Torch.Types.TH.Long    as L
import qualified Torch.Types.TH.Float   as F
import qualified Torch.Types.TH.Double  as D


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

module Torch.Class.Tensor.Masked.Static where

import Torch.Class.Types
import GHC.TypeLits
import Torch.Dimensions

class TensorMasked t where
  maskedFill_   :: Dimensions d => t d -> MaskTensor (t d) '[(n::Nat)] -> HsReal (t d) -> IO ()
  maskedCopy_   :: Dimensions d => t d -> MaskTensor (t d) '[(n::Nat)] -> t d -> IO ()
  maskedSelect_ :: Dimensions d => t d -> t d -> MaskTensor (t d) '[(n::Nat)] -> IO ()

-- class GPUTensorMasked t where
--   maskedFillByte_   :: t -> MaskDynamic t -> HsReal t -> io ()
--   maskedCopyByte_   :: t -> MaskDynamic t -> t -> io ()
--   maskedSelectByte_ :: t -> t -> MaskDynamic t -> io ()


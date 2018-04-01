module Torch.Class.Tensor.Masked where

import Torch.Class.Types

class TensorMasked t where
  maskedFill_   :: t -> MaskDynamic t -> HsReal t -> IO ()
  maskedCopy_   :: t -> MaskDynamic t -> t -> IO ()
  maskedSelect_ :: t -> t -> MaskDynamic t -> IO ()

class GPUTensorMasked t where
  maskedFillByte_   :: t -> MaskDynamic t -> HsReal t -> io ()
  maskedCopyByte_   :: t -> MaskDynamic t -> t -> io ()
  maskedSelectByte_ :: t -> t -> MaskDynamic t -> io ()


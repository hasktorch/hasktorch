module Torch.Class.Tensor.Masked where

import Torch.Class.Types

class TensorMasked t where
  maskedFill_   :: t -> MaskTensor t -> HsReal t -> IO ()
  maskedCopy_   :: t -> MaskTensor t -> t -> IO ()
  maskedSelect_ :: t -> t -> MaskTensor t -> IO ()

class GPUTensorMasked t where
  maskedFillByte_   :: t -> MaskTensor t -> HsReal t -> io ()
  maskedCopyByte_   :: t -> MaskTensor t -> t -> io ()
  maskedSelectByte_ :: t -> t -> MaskTensor t -> io ()


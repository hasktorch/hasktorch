module Torch.Class.Tensor.Masked where

import Torch.Class.Types

class TensorMasked t where
  maskedFill   :: t -> MaskTensor t -> HsReal t -> IO ()
  maskedCopy   :: t -> MaskTensor t -> t -> IO ()
  maskedSelect :: t -> t -> MaskTensor t -> IO ()

class GPUTensorMasked t where
  maskedFillByte   :: t -> MaskTensor t -> HsReal t -> io ()
  maskedCopyByte   :: t -> MaskTensor t -> t -> io ()
  maskedSelectByte :: t -> t -> MaskTensor t -> io ()


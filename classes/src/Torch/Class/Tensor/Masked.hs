module Torch.Class.Tensor.Masked where

import Torch.Class.Types

class TensorMasked t where
  _maskedFill   :: t -> MaskDynamic t -> HsReal t -> IO ()
  _maskedCopy   :: t -> MaskDynamic t -> t -> IO ()
  _maskedSelect :: t -> t -> MaskDynamic t -> IO ()

class GPUTensorMasked t where
  _maskedFillByte   :: t -> MaskDynamic t -> HsReal t -> io ()
  _maskedCopyByte   :: t -> MaskDynamic t -> t -> io ()
  _maskedSelectByte :: t -> t -> MaskDynamic t -> io ()


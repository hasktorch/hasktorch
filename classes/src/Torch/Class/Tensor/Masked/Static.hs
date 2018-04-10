module Torch.Class.Tensor.Masked.Static where

import Torch.Class.Types
import GHC.TypeLits
import Torch.Dimensions

class TensorMasked t where
  _maskedFill   :: Dimensions d => t d -> MaskTensor (t d) '[(n::Nat)] -> HsReal (t d) -> IO ()
  _maskedCopy   :: Dimensions d => t d -> MaskTensor (t d) '[(n::Nat)] -> t d -> IO ()
  _maskedSelect :: Dimensions d => t d -> t d -> MaskTensor (t d) '[(n::Nat)] -> IO ()

-- class GPUTensorMasked t where
--   _maskedFillByte   :: t -> MaskDynamic t -> HsReal t -> io ()
--   _maskedCopyByte   :: t -> MaskDynamic t -> t -> io ()
--   _maskedSelectByte :: t -> t -> MaskDynamic t -> io ()


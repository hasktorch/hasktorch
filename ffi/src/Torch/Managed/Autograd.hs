
module Torch.Managed.Autograd where

import Foreign.ForeignPtr

import qualified Torch.Unmanaged.Autograd as Unmanaged
import qualified ATen.Unmanaged.Type.Tensor
import qualified ATen.Unmanaged.Type.TensorList
import ATen.Type
import ATen.Class
import ATen.Cast


grad :: ForeignPtr Tensor -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
grad = cast2 Unmanaged.grad


makeIndependent :: ForeignPtr Tensor -> IO (ForeignPtr Tensor)
makeIndependent = cast1 Unmanaged.makeIndependent

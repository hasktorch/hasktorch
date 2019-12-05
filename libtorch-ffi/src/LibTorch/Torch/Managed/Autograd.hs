
module LibTorch.Torch.Managed.Autograd where

import Foreign.ForeignPtr

import qualified LibTorch.Torch.Unmanaged.Autograd as Unmanaged
import qualified LibTorch.ATen.Unmanaged.Type.Tensor
import qualified LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Type
import LibTorch.ATen.Class
import LibTorch.ATen.Cast


grad :: ForeignPtr Tensor -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
grad = cast2 Unmanaged.grad


makeIndependent :: ForeignPtr Tensor -> IO (ForeignPtr Tensor)
makeIndependent = cast1 Unmanaged.makeIndependent

dropVariable :: ForeignPtr Tensor -> IO (ForeignPtr Tensor)
dropVariable = cast1 Unmanaged.dropVariable

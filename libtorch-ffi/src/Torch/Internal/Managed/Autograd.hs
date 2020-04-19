module Torch.Internal.Managed.Autograd where

import Foreign.ForeignPtr
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Autograd as Unmanaged
import qualified Torch.Internal.Unmanaged.Type.Tensor
import qualified Torch.Internal.Unmanaged.Type.TensorList

grad :: ForeignPtr Tensor -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
grad = cast2 Unmanaged.grad

makeIndependent :: ForeignPtr Tensor -> IO (ForeignPtr Tensor)
makeIndependent = cast1 Unmanaged.makeIndependent

dropVariable :: ForeignPtr Tensor -> IO (ForeignPtr Tensor)
dropVariable = cast1 Unmanaged.dropVariable

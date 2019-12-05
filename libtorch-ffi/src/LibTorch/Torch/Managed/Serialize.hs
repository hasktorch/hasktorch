
module LibTorch.Torch.Managed.Serialize where

import Foreign.ForeignPtr

import qualified LibTorch.Torch.Unmanaged.Serialize as Unmanaged
import qualified LibTorch.ATen.Unmanaged.Type.Tensor
import qualified LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Type
import LibTorch.ATen.Class
import LibTorch.ATen.Cast


save :: ForeignPtr TensorList -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr TensorList)
load = cast1 Unmanaged.load

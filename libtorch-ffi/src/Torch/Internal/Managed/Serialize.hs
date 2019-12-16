
module Torch.Managed.Serialize where

import Foreign.ForeignPtr

import qualified Torch.Unmanaged.Serialize as Unmanaged
import qualified ATen.Unmanaged.Type.Tensor
import qualified ATen.Unmanaged.Type.TensorList
import ATen.Type
import ATen.Class
import ATen.Cast


save :: ForeignPtr TensorList -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr TensorList)
load = cast1 Unmanaged.load

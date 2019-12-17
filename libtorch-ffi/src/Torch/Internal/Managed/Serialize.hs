
module Torch.Internal.Managed.Serialize where

import Foreign.ForeignPtr

import qualified Torch.Internal.Unmanaged.Serialize as Unmanaged
import qualified Torch.Internal.Unmanaged.Type.Tensor
import qualified Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast


save :: ForeignPtr TensorList -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr TensorList)
load = cast1 Unmanaged.load

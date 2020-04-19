module Torch.Internal.Managed.Serialize where

import Foreign.ForeignPtr
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Serialize as Unmanaged
import qualified Torch.Internal.Unmanaged.Type.Tensor
import qualified Torch.Internal.Unmanaged.Type.TensorList

save :: ForeignPtr TensorList -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr TensorList)
load = cast1 Unmanaged.load

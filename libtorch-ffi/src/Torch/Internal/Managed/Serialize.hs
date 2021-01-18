
module Torch.Internal.Managed.Serialize where

import Foreign.ForeignPtr

import qualified Torch.Internal.Unmanaged.Serialize as Unmanaged
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects


save :: ForeignPtr TensorList -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr TensorList)
load = cast1 Unmanaged.load

pickleSave :: ForeignPtr IValue -> FilePath -> IO ()
pickleSave = cast2 Unmanaged.pickleSave

pickleLoad :: FilePath -> IO (ForeignPtr IValue)
pickleLoad = cast1 Unmanaged.pickleLoad

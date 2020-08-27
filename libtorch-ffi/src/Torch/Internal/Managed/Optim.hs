
module Torch.Internal.Managed.Optim where

import qualified Torch.Internal.Unmanaged.Optim as Unmanaged

import Foreign.C.String
import Foreign.C.Types
import Foreign
import Foreign.ForeignPtr.Unsafe
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

optimizerWithAdam
  :: Unmanaged.AdamParams
  -> ForeignPtr TensorList
  -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor))
  -> Int
  -> IO (ForeignPtr TensorList)
optimizerWithAdam optimizerParams initParams loss numIter = cast2 (\i n -> Unmanaged.optimizerWithAdam optimizerParams i (trans loss) n) initParams numIter 
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> Ptr TensorList -> IO (Ptr Tensor)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret

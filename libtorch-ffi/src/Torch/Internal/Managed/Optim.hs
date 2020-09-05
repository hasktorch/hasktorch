module Torch.Internal.Managed.Optim where

import Foreign
import Foreign.C.String
import Foreign.C.Types
import Foreign.ForeignPtr.Unsafe
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Objects
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Optim as Unmanaged

optimizerWithAdam
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor))
  -> Int
  -> IO (ForeignPtr TensorList)
optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams loss numIter = cast2 (\i n -> Unmanaged.optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad i (trans loss) n) initParams numIter
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> Ptr TensorList -> IO (Ptr Tensor)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret

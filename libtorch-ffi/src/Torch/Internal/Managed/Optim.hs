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

-- optimizerWithAdam
--   :: CDouble
--   -> CDouble
--   -> CDouble
--   -> CDouble
--   -> CDouble
--   -> CBool
--   -> ForeignPtr TensorList
--   -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor))
--   -> Int
--   -> IO (ForeignPtr TensorList)
-- optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams loss numIter = cast2 (\i n -> Unmanaged.optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad i (trans loss) n) initParams numIter
--   where
--     trans :: (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> Ptr TensorList -> IO (Ptr Tensor)
--     trans func inputs = do
--       inputs' <- newForeignPtr_ inputs
--       ret <- func inputs'
--       return $ unsafeForeignPtrToPtr ret

adagrad
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
adagrad = cast6 Unmanaged.adagrad

rmsprop
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
rmsprop = cast7 Unmanaged.rmsprop

sgd
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
sgd = cast6 Unmanaged.sgd

adam
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
adam = cast7 Unmanaged.adam

adamw
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
adamw = cast7 Unmanaged.adamw

lbfgs
  :: CDouble
  -> CInt
  -> CInt
  -> CDouble
  -> CDouble
  -> CInt
  -> Maybe (ForeignPtr StdString)
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
lbfgs = cast8 Unmanaged.lbfgs

getParams :: ForeignPtr Optimizer -> IO (ForeignPtr TensorList) 
getParams = cast1 Unmanaged.getParams

step :: ForeignPtr Optimizer -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> IO (ForeignPtr Tensor)
step optimizer loss = cast1 (\opt -> Unmanaged.step opt (trans loss)) optimizer
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> Ptr TensorList -> IO (Ptr Tensor)
    trans func inputs = do
      inputs' <- newForeignPtr_ inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret

unsafeStep :: ForeignPtr Optimizer -> ForeignPtr TensorList -> ForeignPtr Tensor -> IO (ForeignPtr TensorList)
unsafeStep = cast3 Unmanaged.unsafeStep

save :: ForeignPtr Optimizer -> ForeignPtr StdString -> IO ()
save = cast2 Unmanaged.save

load :: ForeignPtr Optimizer -> ForeignPtr StdString -> IO ()
load = cast2 Unmanaged.load

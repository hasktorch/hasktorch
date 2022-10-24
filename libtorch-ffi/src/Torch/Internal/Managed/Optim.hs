{-# LANGUAGE DataKinds #-}
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
import Control.Concurrent.MVar (MVar(..), newEmptyMVar, putMVar, takeMVar)

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
-- optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams loss numIter = _cast2 (\i n -> Unmanaged.optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad i (trans loss) n) initParams numIter
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
adagrad = _cast6 Unmanaged.adagrad

rmsprop
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
rmsprop = _cast7 Unmanaged.rmsprop

sgd
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
sgd = _cast6 Unmanaged.sgd

adam
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
adam = _cast7 Unmanaged.adam

adamw
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Optimizer)
adamw = _cast7 Unmanaged.adamw

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
lbfgs = _cast8 Unmanaged.lbfgs

getParams :: ForeignPtr Optimizer -> IO (ForeignPtr TensorList) 
getParams = _cast1 Unmanaged.getParams

step :: ForeignPtr Optimizer -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> IO (ForeignPtr Tensor)
step optimizer loss = do
  ref <- newEmptyMVar
  ret <- cast1 (\opt -> Unmanaged.step opt (trans ref loss)) optimizer
  v <- takeMVar ref
  touchForeignPtr v
  return ret
  where
    trans :: MVar (ForeignPtr Tensor) -> (ForeignPtr TensorList -> IO (ForeignPtr Tensor)) -> Ptr TensorList -> IO (Ptr Tensor)
    trans ref func inputs = do
      inputs' <- newForeignPtr_ inputs
      ret <- func inputs'
      putMVar ref ret
      return $ unsafeForeignPtrToPtr ret

stepWithGenerator
  :: ForeignPtr Optimizer
  -> ForeignPtr Generator
  -> (ForeignPtr TensorList -> ForeignPtr Generator -> IO (ForeignPtr (StdTuple '(Tensor,Generator))))
  -> IO (ForeignPtr (StdTuple '(Tensor,Generator)))
stepWithGenerator optimizer generator loss = do
  ref <- newEmptyMVar
  ret <- cast2 (\opt gen -> Unmanaged.stepWithGenerator opt gen (trans ref loss)) optimizer generator
  v <- takeMVar ref
  touchForeignPtr v
  return ret
  where
    trans
      :: MVar (ForeignPtr (StdTuple '(Tensor,Generator)))
      -> (ForeignPtr TensorList -> ForeignPtr Generator -> IO (ForeignPtr (StdTuple '(Tensor,Generator))))
      -> Ptr TensorList
      -> Ptr Generator
      -> IO (Ptr (StdTuple '(Tensor,Generator)))
    trans ref func inputs generator = do
      inputs' <- newForeignPtr_ inputs
      generator' <- newForeignPtr_ generator
      ret <- func inputs' generator'
      putMVar ref ret
      return $ unsafeForeignPtrToPtr ret


unsafeStep :: ForeignPtr Optimizer -> ForeignPtr Tensor -> IO (ForeignPtr TensorList)
unsafeStep = _cast2 Unmanaged.unsafeStep

save :: ForeignPtr Optimizer -> ForeignPtr StdString -> IO ()
save = _cast2 Unmanaged.save

load :: ForeignPtr Optimizer -> ForeignPtr StdString -> IO ()
load = _cast2 Unmanaged.load

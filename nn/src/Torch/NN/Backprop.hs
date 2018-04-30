{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.NN.Backprop where

import Foreign.C.Types
import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad.Trans.Class
import Control.Monad.IO.Class

import Torch.Class.NN.Static

newtype TorchT m a = TorchT { unTorch :: m a }
  deriving (Functor, Applicative, Monad, MonadIO)

instance MonadTrans TorchT where
  lift = TorchT

type Torch = TorchT IO

io :: MonadIO io => IO x -> io x
io = liftIO

type NNOp t m d = (NN t, Dimensions d, MonadIO m)

liftOp :: NNOp t m d => (t d -> IO ()) -> TorchT m (t d)
liftOp op = io new >>= \t -> io (op t) >> pure t

-- abs :: NNOp t m d => t d -> TorchT m (t d)
-- abs inp = liftOp (`abs_updateOutput` inp)
-- 
-- abs' :: (Dimensions d', NNOp t m d) => t d -> TorchT m (t d')
-- abs' inp = do
--   gradOut <- io new
--   gradIn  <- io new
--   io $ abs_updateGradInput inp gradOut gradIn
--   pure gradOut
-- 
-- 

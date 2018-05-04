{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

-- import Numeric.Backprop
import Data.Singletons.Prelude.Num
-- import Control.Monad
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Cuda as Torch

type Tensor = DoubleTensor
type Batch = 2
type KW = 2
type DW = 2
type Input = 5
type Output = 2
type Sequence1 = 7
type Sequence2 = 13

params :: Proxy '[Input, Output, KW, DW]
params = Proxy 

main :: IO ()
main = do
  conv <- initConv1d params

  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT getCosVec
  o1 <- conv1d_forward input conv
  shape o1 >>= print

  gw :: Tensor '[Sequence2, Output] <- constant 1
  shape gw >>= print

  o1' :: Tensor '[13, 5] <-
    conv1d_backward input gw (weights conv) (Proxy :: Proxy '(KW, DW))
  shape o1' >>= print

  Just (binput :: Tensor '[Batch,Sequence1, Input]) <- runMaybeT getCosVec
  o2 <- conv1d_forwardBatch binput conv
  shape o2 >>= print

  gw :: Tensor '[Batch, Sequence1, Output] <- constant 1
  shape gw >>= print

  o2' :: Tensor '[2,7,5] <-
    conv1d_backwardBatch binput gw (weights conv) (Proxy :: Proxy '(KW, DW))
  shape o2' >>= print

 where
  getCosVec :: forall d . Dimensions d => KnownNatDim (Product d) => MaybeT IO (Tensor d)
  getCosVec = do
    t <- MaybeT . pure . fromList $ [1..(fromIntegral $ natVal (Proxy :: Proxy (Product d)))]
    lift $ Torch.cos t

  -- Perhaps these should be moved into the nn package...
  initConv1d
    :: KnownNatDim3 s o (s * kW)
    => Proxy '[s,o,kW,dW]
    -> IO (Conv1d Tensor s o kW dW)
  initConv1d _ =
    fmap Conv1d $ (,)
      <$> Torch.uniform (-10::Double) 10
      <*> Torch.constant 1

-- conv1d_backward
--   :: forall t s f kW dW o b
--   .  KnownNat5 s f o kW dW
--   => TemporalConvC t s f kW dW o
--   => t '[s, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
--   -> t '[s, o]                         -- ^ grad output
--   -> t '[o, f*kW]                      -- ^ weight
--   -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
--                                        --   dW: The step of the convolution. Default is 1 in C.
--   -> IO (t '[s, f])                    -- ^ output


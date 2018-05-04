{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

-- import Numeric.Backprop
import Data.Singletons.Prelude.Num
-- import Control.Monad
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch
import qualified Torch.Core.Random as RNG

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
  ((weights, bias), _) <- initConv1d params

  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT getCosVec
  o1 <- conv1d_forward input weights bias (Proxy :: Proxy '(KW, DW))
  print o1

  Just (binput :: Tensor '[Batch,Sequence1, Input]) <- runMaybeT getCosVec
  o2 <- conv1d_forwardBatch binput weights bias (Proxy :: Proxy '(KW, DW))
  print o2

 where
  getCosVec :: forall d . Dimensions d => KnownNatDim (Product d) => MaybeT IO (Tensor d)
  getCosVec = do
    t <- MaybeT . pure . fromList $ [1..(fromIntegral $ natVal (Proxy :: Proxy (Product d)))]
    lift $ Torch.cos t

  initConv1d
    :: forall s o kW dW
    .  KnownNatDim3 s o (s*kW)
    => Proxy '[s,o,kW,dW]
    -> IO ((Tensor '[o, s*kW], Tensor '[o]), (Tensor '[o, s*kW], Tensor '[o]))
  initConv1d _ = do
    g <- RNG.new
    weights :: Tensor '[o, s * kW] <- Torch.uniform g (-10::Double) 10
    bias    :: Tensor '[o]    <- Torch.constant 1

    gWeights :: Tensor '[o, s * kW] <- Torch.constant 1
    gBias    :: Tensor '[o]    <- Torch.constant 1
    pure ((weights, bias), (gWeights, gBias))


-- conv1d inp out kw dw

{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Conv1d where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import qualified Utils
import qualified Torch.Double.NN.Conv1d as NN1
import qualified Torch.Double.NN.Conv2d as NN2

type Batch = 2
type KW = 2
type DW = 2
type Input = 5
type Output = 2
type Sequence1 = 7
type Sequence2 = 13

params :: Dims '[Input, Output, KW, DW]
params = dims

main :: IO ()
main = do
  usingBackpack1d
  directFunctionCalls1d

usingBackpack1d :: IO ()
usingBackpack1d = Utils.section "Using Backpack" $ do
  g <- newRNG
  manualSeed g 1
  c <- initConv1d g params
  Utils.printFullConv1d "initial conv1d state" c

  -- do a forward pass
  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT Utils.mkCosineTensor

  -- backprop manages our forward and backward passes
  let (o1, (c1', g1)) = backprop2 (conv1d 0.5) c input
  shape o1 >>= print
  shape g1 >>= print
  Utils.printFullConv1d "Unbatched convolution layer after backprop" c1'

  putStrLn "======================================="

  Utils.printFullConv1d "Ensure that the last weight update is pure" c

  -- do a backprop pass with batch data
  Just (binput :: Tensor '[Batch, Sequence1, Input]) <- runMaybeT Utils.mkCosineTensor
  let (o2, (c2', g2)) = backprop2 (conv1dBatch 0.5) c binput
  shape o2 >>= print
  shape g2 >>= print
  Utils.printFullConv1d "Batched convolution layer update" c2'


-- -- ========================================================================= --
-- -- Example directly using functions

directFunctionCalls1d :: IO ()
directFunctionCalls1d = Utils.section "Directly calling functions" $ do
  g <- newRNG
  manualSeed g 1
  conv <- initConv1d g params
  Utils.printFullConv1d "initial conv1d state" conv

  -- do a forward pass
  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT Utils.mkCosineTensor
  o1 <- conv1d_forward conv input
  shape o1 >>= print

  -- initialize a gradient output
  let gout :: Tensor '[Sequence2, Output] = constant 1

  -- do a backward pass for grad input
  o1' :: Tensor '[13, 5] <- conv1d_backwardGradInput conv input gout
  print o1'

  -- do a backward pass for grad weights
  conv' <- conv1d_updGradParams conv input gout 0.5
  print conv'
  print (NN1.weights conv')
  print (NN1.bias conv')

  putStrLn "======================================="

  Utils.printFullConv1d "ensure that the last weight update is pure" conv

  -- do a forward pass with batch data
  Just (binput :: Tensor '[Batch, Sequence1, Input]) <- runMaybeT Utils.mkCosineTensor
  o2 <- conv1d_forwardBatch conv binput
  shape o2 >>= print

  -- initialize a gradient input
  let bgout :: Tensor '[Batch, Sequence1, Output] = constant 1
  shape bgout >>= print

  -- do a backwards pass with batch data
  o2' :: Tensor '[2,7,5] <- conv1d_backwardGradInputBatch conv binput bgout
  shape o2' >>= print

  -- do a backward pass for grad weights
  conv'' <- conv1d_updGradParamsBatch conv binput bgout 0.5

  Utils.printFullConv1d "ensure that the last weight update is pure" conv''

-- ========================================================================= --
-- utility functions for this exercise

-- simple initilizer
initConv1d
  :: KnownDim3 s o (s * kW)
  => Generator -> Dims '[s,o,kW,dW]
  -> IO (Conv1d s o kW dW)
initConv1d g _ =
  (Conv1d . (,Torch.constant 1))
    <$> Torch.uniform g (-10::Double) 10



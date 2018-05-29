{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TupleSections #-}
module Conv2d where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import qualified Utils
import qualified Torch.Double.NN.Conv2d as NN

type Ch = 3
type H = 8
type Wid = 15

type D = 1
type Pad = 1
type KW = 3
type KH = 4
type F = 3
type O = 2
type InputDims = '[Ch, H, Wid]
type BatchSize = 3

type OH = ((Div (H   + (2*Pad) - KH) D) + 1)
type OW = ((Div (Wid + (2*Pad) - KW) D) + 1)

steps = Step2d    :: Step2d D D
pad   = Padding2d :: Padding2d Pad Pad

main :: IO ()
main = do
  usingBackpack2d
  directFunctionCalls2d

usingBackpack2d :: IO ()
usingBackpack2d = Utils.section "Using Backpack" $ do
  g <- newRNG
  manualSeed g 1
  c :: Conv2d F O KH KW <- initConv2d g
  Utils.printFullConv2d "initial conv1d state" c

  -- do a forward pass
  Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor

  -- backprop manages our forward and backward passes
  let (o1, (c1', g1)) = backprop2 (NN.conv2dMM steps pad 0.5) c input
  shape o1 >>= print
  shape g1 >>= print
  Utils.printFullConv2d "Unbatched convolution layer after backprop" c1'

  putStrLn "======================================="

  Utils.printFullConv2d "Ensure that the last weight update is pure" c

  -- do a backprop pass with batch data
  Just (binput :: Tensor (BatchSize:+InputDims)) <- runMaybeT Utils.mkCosineTensor
  let (o2, (c2', g2)) = backprop2 (conv2dMMBatch steps pad 0.5) c binput
  shape o2 >>= print
  shape g2 >>= print
  Utils.printFullConv2d "Batched convolution layer update" c2'


-- ========================================================================= --
-- Example directly using functions

directFunctionCalls2d :: IO ()
directFunctionCalls2d = do
  g <- newRNG
  manualSeed g 1
  conv :: Conv2d F O KH KW <- initConv2d g

  -- do a forward pass
  Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor

  let o1 = conv2dMM_forward steps pad conv input :: Tensor '[O,OH,OW]

  shape o1 >>= print

  let gout = constant 10
  -- do a backward pass
  let o1' = conv2dMM_updGradInput steps pad conv input gout
  shape o1' >>= print

  let conv' = conv2dMM_updGradParameters steps pad 0.5 conv input gout

  Utils.printFullConv2d "Ensure that the last weight update is pure" conv'

  -- do a forward pass with a batch dimension
  Just (binput :: Tensor (BatchSize:+InputDims)) <- runMaybeT Utils.mkCosineTensor

  let o2 = conv2dMM_forwardBatch (Step2d :: Step2d 1 1) (Padding2d :: Padding2d 2 2) conv binput
  putStrLn "============="
  shape o2 >>= print

  let gout = constant 10
  -- do a backward pass
  putStrLn "============="
  let o1' = conv2dMM_updGradInputBatch steps pad conv binput gout
  shape o1' >>= print
  putStrLn "============="

initConv2d :: KnownNatDim4 f o kW kH => Generator -> IO (Conv2d f o kW kH)
initConv2d g =
  (Conv2d . (,Torch.constant 1))
    <$> Torch.uniform g (-10::Double) 10


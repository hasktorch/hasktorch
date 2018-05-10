{-# LANGUAGE TypeOperators #-}
module Conv2d where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import qualified Utils
import qualified Torch.Double.NN.Conv2d as NN

type InputDims = '[2, 7, 13]
type BatchSize = 3

main :: IO ()
main = do
  -- usingBackpack2d
  directFunctionCalls2d

-- usingBackpack2d :: IO ()
-- usingBackpack2d = Utils.section "Using Backpack" $ do
--   g <- newRNG
--   manualSeed g 1
--   c <- initConv2d g params
--   Utils.printFullConv1d "initial conv1d state" c
-- 
--   -- do a forward pass
--   Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor
-- 
--   -- backprop manages our forward and backward passes
--   let (o1, (c1', g1)) = backprop2 (conv1d 0.5) c input
--   shape o1 >>= print
--   shape g1 >>= print
--   Utils.printFullConv1d "Unbatched convolution layer after backprop" c1'
-- 
--   putStrLn "======================================="
-- 
--   Utils.printFullConv1d "Ensure that the last weight update is pure" c
-- 
--   -- do a backprop pass with batch data
--   Just (binput :: Tensor (BatchSize+:InputDims)) <- runMaybeT Utils.mkCosineTensor
--   let (o2, (c2', g2)) = backprop2 (conv1dBatch 0.5) c binput
--   shape o2 >>= print
--   shape g2 >>= print
--   Utils.printFullConv1d "Batched convolution layer update" c2'


-- ========================================================================= --
-- Example directly using functions

directFunctionCalls2d :: IO ()
directFunctionCalls2d = do
  g <- newRNG
  manualSeed g 1
  conv :: Conv2d 2 3 3 3 <- initConv2d g
  let steps = Param2d :: Param2d 1 1
      pad   = Param2d :: Param2d 2 2

  -- do a forward pass
  Just (input :: Tensor '[2, 7, 13]) <- runMaybeT Utils.mkCosineTensor

  o1 <- conv2dMM_forward input conv steps pad
  shape o1 >>= print

  gout <- constant 10
  -- do a backward pass
  o1' <- conv2dMM_backwardGradInput input gout conv steps pad
  shape o1' >>= print

  show <$> shape (NN.weights conv) >>= putStrLn . ("weights: " ++)
  show <$> shape (NN.bias conv)    >>= putStrLn . ("bias   : " ++)
-- 
--   -- do a forward pass with a batch dimension
--   Just (binput :: Tensor '[5, 2, 7, 13]) <- runMaybeT getCosVec
-- 
--   (o2, finput2, fgradInput2) <-
--     conv2dMM_forwardBatch binput conv (Param2d :: Param2d 1 1) (Param2d :: Param2d 2 2)
--   (,,) <$> shape o2 <*> shape finput2 <*> shape fgradInput2 >>= print
--   print (finput2 Torch.!! 1 :: Tensor '[])
--   print fgradInput2
-- 
-- -- conv2dMM_backward
-- --   :: (KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH)
-- --   => Tensor inp        -- ^ input
-- --   -> Tensor gout       -- ^ gradOutput
-- --   -> Conv2d f o kW kH  -- ^ conv2d state
-- --   -> Param2d dW dH     -- ^ (dW, dH) step of the convolution in width and height dimensions
-- --   -> Param2d pW pH     -- ^ (pW, pH) zero padding to the input plane for width and height.
-- --   -> IO (Tensor d, Tensor d')
-- -- conv2dMM_backward = _conv2dMM_backward


initConv2d :: KnownNatDim4 f o kW kH => Generator -> IO (Conv2d f o kW kH)
initConv2d g =
  fmap Conv2d $ (,)
    <$> Torch.uniform g (-10::Double) 10
    <*> Torch.constant 1



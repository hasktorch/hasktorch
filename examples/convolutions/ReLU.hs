{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module ReLU where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import Conv2d
import qualified Utils
import qualified Torch.Double.NN.Conv2d     as NN
import qualified Torch.Double.NN.Activation as NN


reluLayer
  :: Reifies s W
  => Step2d D D
  -> Padding2d Pad Pad
  -> Double
  -> BVar s (Conv2d F O KW KH)
  -> BVar s (Tensor '[F, H, Wid])
  -> BVar s (Tensor '[O, 7, 15])
reluLayer steps pad lr conv inp = relu (NN.conv2dMM steps pad lr conv inp)

main :: IO ()
main = Utils.section "Using Backpack" $ do
  g <- newRNG
  manualSeed g 1
  c :: Conv2d F O KW KH <- initConv2d g
  Utils.printFullConv2d "initial conv1d state" c

  -- do a forward pass
  Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor

  -- backprop manages our forward and backward passes
  let (o1, (c1', g1)) = backprop2 (reluLayer steps pad 0.5) c input
  shape o1 >>= print
  shape g1 >>= print
  Utils.printFullConv2d "Unbatched convolution layer after backprop" c1'



{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module MaxPooling where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import Conv2d
import qualified ReLU
import qualified Torch.Long as Ix
import qualified Utils
import qualified Torch.Double.NN.Conv2d     as NN
import qualified Torch.Double.NN.Activation as NN


lenetLayer
  :: forall s . Reifies s W
  => Double
  -> BVar s (Conv2d F O KH KW)
  -> BVar s (Tensor '[F, H, Wid])
  -> BVar s (Tensor '[O, 6, 14])
lenetLayer lr conv
  = maxPooling2d
      (Kernel2d :: Kernel2d 2 2)
      (Step2d :: Step2d 1 1)
      (Padding2d :: Padding2d 0 0)
      (sing :: SBool 'True)
  . relu
  . NN.conv2dMM
      (Step2d :: Step2d D D)
      (Padding2d :: Padding2d Pad Pad)
      lr conv


main :: IO ()
main = Utils.section "Using Backpack" $ do
  g <- newRNG
  manualSeed g 1
  c :: Conv2d F O KH KW <- initConv2d g
  Utils.printFullConv2d "initial conv1d state" c

  -- do a forward pass
  Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor

  -- backprop manages our forward and backward passes
  let (o1, (c1', g1)) = backprop2 (ReLU.reluLayer steps pad 0.5) c input
  shape o1 >>= print
  let o1' = evalBP myMaxPool o1
  shape o1' >>= print

  let gin = gradBP myMaxPool o1
  shape gin >>= print
  print gin
  where
    myMaxPool
      :: Reifies s W
      => BVar s (Tensor '[2, 7, 15])
      -> BVar s (Tensor '[2, 4, 8])
    myMaxPool = maxPooling2d
      (Kernel2d :: Kernel2d 4 4) (Step2d :: Step2d 2 2)
      (Padding2d :: Padding2d 1 1) (sing :: SBool 'True)




{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module LeNet where

import Numeric.Backprop

import Torch.Double as Torch
import qualified ReLU
import qualified Torch.Long as Ix
import qualified Utils
import qualified Torch.Double.NN.Conv2d     as NN
import qualified Torch.Double.NN.Activation as NN

type DStep = 1
type DPad  = 0

defaultStep2d :: Step2d DStep DStep
defaultStep2d = Step2d

defaultPadding2d :: Padding2d DPad DPad
defaultPadding2d =  Padding2d

-- in ceiling mode for dimensions in maxPooling2d
defaultCeilingMode :: SBool 'True
defaultCeilingMode = sing

lenetLayer
  :: forall inp h w ker ow oh s out
  -- constraint for backprop to hold the wengert tape
  .  Reifies s W

  -- leave input, output and square kernel size variable so that we
  -- can reuse the layer...
  => KnownNatDim3 inp out ker

  -- ...this means we need the constraints for conv2dMM and maxPooling2d
  => SpatialConvolutionC inp h w ker ker DStep DStep DPad DPad ow oh

  -- GHC will tell you this other constraint:
  => ow ~ (1 + (ow - 2))
  => oh ~ (1 + (oh - 2))

  => Double                            -- ^ learning rate
  -> BVar s (Conv2d inp out ker ker)   -- ^ convolutional layer
  -> BVar s (Tensor '[inp, h, w])      -- ^ input
  -> BVar s (Tensor '[out, oh, ow])    -- ^ output
lenetLayer lr conv
  = maxPooling2d
      (Kernel2d :: Kernel2d 2 2)
      defaultStep2d
      defaultPadding2d
      defaultCeilingMode
  . relu
  . NN.conv2dMM
      defaultStep2d
      defaultPadding2d
      lr conv

main :: IO ()
main = undefined
  -- where
  --   lenetConvLayers
  --     :: Double
  --     -> BVar s (Tensor '[inp, h, w])      -- ^ input
  --     -> BVar s (Tensor '[out, oh, ow])    -- ^ output
  --   lenetConvLayers lr =
  --     lenetLayer (undefined :: BVar s (Conv2d 1  6 5 5))
  --     lenetLayer (undefined :: BVar s (Conv2d 6 16 5 5))


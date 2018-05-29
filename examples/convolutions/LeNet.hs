{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module LeNet where

import Data.Function ((&))
import Numeric.Backprop

import Torch.Double as Torch
import qualified ReLU
import qualified Torch.Long as Ix
import qualified Utils
import qualified Torch.Double.NN.Conv2d     as NN
import qualified Torch.Double.NN.Layers     as NN
import qualified Torch.Double.NN.Activation as NN


main :: IO ()
main = undefined

lenet
  :: forall s
  .  Reifies s W
  => Double
  -> BVar s (Tensor '[1,32,32])      -- ^ input
  -> BVar s (Tensor '[10])           -- ^ output
lenet lr inp
  = lenetLayer lr (undefined :: BVar s (Conv2d 1  6 5 5)) inp
  & lenetLayer lr (undefined :: BVar s (Conv2d 6 16 5 5))

  & flattenBP

  -- start fully connected network
  & relu . linear (undefined :: BVar s (Linear (16*5*5) 120))
  & relu . linear (undefined :: BVar s (Linear      120  84))
  &        linear (undefined :: BVar s (Linear       84  10))

{- what each layer's type looks like (unused)

lenetLayer1
  :: Reifies s W
  => Double                         -- ^ learning rate
  -> BVar s (Conv2d 1 6 5 5)        -- ^ convolutional layer
  -> BVar s (Tensor '[1, 32, 32])   -- ^ input
  -> BVar s (Tensor '[6, 14, 14])   -- ^ output
lenetLayer1 = lenetLayer

lenetLayer2
  :: Reifies s W
  => Double                          -- ^ learning rate
  -> BVar s (Conv2d 6 16 5 5)        -- ^ convolutional layer
  -> BVar s (Tensor '[ 6, 14, 14])   -- ^ input
  -> BVar s (Tensor '[16,  5,  5])   -- ^ output
lenetLayer2 = lenetLayer
-}


lenetLayer
  :: forall inp h w ker ow oh s out mow moh

  -- backprop constraint to hold the wengert tape
  .  Reifies s W

  -- leave input, output and square kernel size variable so that we
  -- can reuse the layer...
  => KnownNatDim3 inp out ker

  -- ...this means we need the constraints for conv2dMM and maxPooling2d
  -- Note that oh and ow are then used as input to the maxPooling2d constraint.
  => SpatialConvolutionC inp h  w ker ker DStep DStep DPad DPad  oh  ow
  => SpatialDilationC       oh ow   2   2     2     2 DPad DPad mow moh 1 1 'True

  -- Start withe parameters
  => Double                            -- ^ learning rate for convolution layer
  -> BVar s (Conv2d inp out ker ker)   -- ^ convolutional layer
  -> BVar s (Tensor '[inp,   h,   w])  -- ^ input
  -> BVar s (Tensor '[out, moh, mow])  -- ^ output
lenetLayer lr conv
  = maxPooling2d
      (Kernel2d :: Kernel2d 2 2)
      (Step2d :: Step2d 2 2)
      defaultPadding2d
      defaultCeilingMode
  . relu
  . NN.conv2dMM
      defaultStep2d
      defaultPadding2d
      lr conv

type DStep = 1
type DPad  = 0

defaultStep2d :: Step2d DStep DStep
defaultStep2d = Step2d

defaultPadding2d :: Padding2d DPad DPad
defaultPadding2d =  Padding2d

-- in ceiling mode for dimensions in maxPooling2d
defaultCeilingMode :: SBool 'True
defaultCeilingMode = sing



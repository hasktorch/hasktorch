{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Models.LeNet where

import Data.Function ((&))
import GHC.Generics
import Numeric.Backprop
import Prelude as P
import GHC.TypeLits (KnownNat)
import Data.Singletons.Prelude hiding (type (*), All)

import Torch.Double as Torch
import qualified Torch.Double.NN.Conv2d as NN
import Torch.Double.NN.Linear (Linear(..), linear)
import Lens.Micro.TH

import Torch.Models.Internal

data LeNet ch = LeNet
  { _conv1 :: !(Conv2d ch 6 '(5,5))
  , _conv2 :: !(Conv2d 6 16 '(5,5))
  , _fc1   :: !(Linear  (16*5*5) 120)
  , _fc2   :: !(Linear       120  84)
  , _fc3   :: !(Linear        84  10)
  } deriving (Show, Generic)

makeLenses ''LeNet
instance KnownNat ch => Backprop (LeNet ch)


-------------------------------------------------------------------------------

main :: IO ()
main = do
  net <- newLeNet @3
  print net

newLeNet :: KnownDim ch => IO (LeNet ch)
newLeNet = LeNet
  <$> newConv2d
  <*> newConv2d
  <*> newLinear
  <*> newLinear
  <*> newLinear

lenet
  :: forall s ch h w o
  .  Reifies s W
  => All KnownDim '[ch,h,w,o,ch*5*5]
  => All KnownNat '[ch,h,w,o,ch*5*5]
  => o ~ 10
  => h ~ 32
  => w ~ 32
  => Double
  -> BVar s (LeNet ch)               -- ^ lenet architecture
  -> BVar s (Tensor '[ch,h,w])      -- ^ input
  -> BVar s (Tensor '[o])           -- ^ output
lenet lr arch inp
  = lenetLayer lr (arch ^^. conv1) inp
  & lenetLayer lr (arch ^^. conv2)

  & flattenBP

  -- start fully connected network
  & relu . linear lr (arch ^^. fc1)
  & relu . linear lr (arch ^^. fc2)
  &        linear lr (arch ^^. fc3)

-- Optionally, we can remove the explicit type and everything would be fine.
-- Including it is quite a bit of work and requires pulling in the correct
-- constraints
lenetLayer
  :: forall inp h w ker ow oh s out mow moh step pad

  -- backprop constraint to hold the wengert tape
  .  Reifies s W

  -- leave input, output and square kernel size variable so that we
  -- can reuse the layer...
  => All KnownDim '[inp,out,ker]

  -- FIXME: derive these from the signature (maybe assign them as args)
  => pad ~ 0   --  default padding size
  => step ~ 1  --  default step size for Conv2d

  -- ...this means we need the constraints for conv2dMM and maxPooling2d
  -- Note that oh and ow are then used as input to the maxPooling2d constraint.
  => SpatialConvolutionC inp h  w ker ker step step pad pad  oh  ow
  => SpatialDilationC       oh ow   2   2    2    2 pad pad mow moh 1 1 'True

  -- Start withe parameters
  => Double                            -- ^ learning rate for convolution layer
  -> BVar s (Conv2d inp out '(ker,ker))   -- ^ convolutional layer
  -> BVar s (Tensor '[inp,   h,   w])  -- ^ input
  -> BVar s (Tensor '[out, moh, mow])  -- ^ output
lenetLayer lr conv inp
  = NN.conv2dMM
      (Step2d    :: Step2d '(1,1))
      (Padding2d :: Padding2d '(0,0))
      lr conv inp
  & relu
  & maxPooling2d
      (Kernel2d  :: Kernel2d '(2,2))
      (Step2d    :: Step2d '(2,2))
      (Padding2d :: Padding2d '(0,0))
      (sing      :: SBool 'True)

{- Here is what each layer's intermediate type would like (unused)
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



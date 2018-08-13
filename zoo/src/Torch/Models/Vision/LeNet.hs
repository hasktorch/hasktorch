{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Models.Vision.LeNet where

import Data.Function ((&))
import Data.List (intercalate)
import Numeric.Backprop as Bp
import Prelude as P
import Lens.Micro.TH
import Data.Singletons.Prelude hiding (type (*), All)

#ifdef CUDA
import Numeric.Dimensions
import Torch.Cuda.Double as Torch
import Torch.Cuda.Double.NN.Linear -- (Linear(..), linear)
import qualified Torch.Cuda.Double.NN.Conv2d as NN
#else
import Torch.Double as Torch
import Torch.Double.NN.Linear -- (Linear(..), linear)
import qualified Torch.Double.NN.Conv2d as NN
#endif

import Torch.Models.Internal

data LeNet ch step = LeNet
  { _conv1 :: !(Conv2d ch 6 '(step,step))
  , _conv2 :: !(Conv2d 6 16 '(step,step))

  , _fc1   :: !(Linear  (16*step*step) 120)
  , _fc2   :: !(Linear       120  84)
  , _fc3   :: !(Linear        84  10)
  }

instance (KnownDim (16*step*step), KnownDim ch, KnownDim step) => Show (LeNet ch step) where
  show (LeNet c1 c2 f1 f2 f3) = intercalate "\n"
#ifdef CUDA
    [ "CudaLeNet {"
#else
    [ "LeNet {"
#endif
    , "  conv1 :: " ++ show c1
    , "  conv2 :: " ++ show c2
    , "  fc1   :: " ++ show f1
    , "  fc2   :: " ++ show f2
    , "  fc3   :: " ++ show f3
    , "}"
    ]

makeLenses ''LeNet

instance (KnownDim (16*step*step), KnownDim ch, KnownDim step) => Backprop (LeNet ch step) where
  add a b = LeNet
    (Bp.add (_conv1 a) (_conv1 b))
    (Bp.add (_conv2 a) (_conv2 b))
    (Bp.add (_fc1 a) (_fc1 b))
    (Bp.add (_fc2 a) (_fc2 b))
    (Bp.add (_fc3 a) (_fc3 b))

  one _ = LeNet
    (Bp.one undefined)
    (Bp.one undefined)
    (Bp.one undefined)
    (Bp.one undefined)
    (Bp.one undefined)

  zero _ = LeNet
    (Bp.zero undefined)
    (Bp.zero undefined)
    (Bp.zero undefined)
    (Bp.zero undefined)
    (Bp.zero undefined)

-------------------------------------------------------------------------------

#ifdef DEBUG
main :: IO ()
main = do
  net <- newLeNet @3 @5
  print net
#endif

newLeNet :: All KnownDim '[ch,step,16*step*step] => IO (LeNet ch step)
newLeNet = LeNet
  <$> newConv2d
  <*> newConv2d
  <*> newLinear
  <*> newLinear
  <*> newLinear

-- lenet
--   :: forall s ch h w o step pad -- ker moh mow
--   .  Reifies s W
--   => All KnownNat '[ch,h,w,o,step]
--   => All KnownDim '[ch,h,w,o,ch*step*step] -- , (16*step*step)]
--   => o ~ 10
--   => h ~ 32
--   => w ~ 32
--   => pad ~ 0
--   -- => SpatialConvolutionC ch h w ker ker step step pad pad (16*step*step) mow
--   -- => SpatialConvolutionC ch moh mow ker ker step step pad pad moh mow
--
--   => Double
--
--   -> BVar s (LeNet ch step)         -- ^ lenet architecture
--   -> BVar s (Tensor '[ch,h,w])      -- ^ input
--   -> BVar s (Tensor '[o])           -- ^ output
lenet lr arch inp
  = lenetLayer lr (arch ^^. conv1) inp
  & lenetLayer lr (arch ^^. conv2)

  & flattenBP

  -- start fully connected network
  & relu . linear lr (arch ^^. fc1)
  & relu . linear lr (arch ^^. fc2)
  &        linear lr (arch ^^. fc3)
  -- & logSoftMax
  & softmax

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
  -> BVar s (Conv2d 1 6 '(5,5) )        -- ^ convolutional layer
  -> BVar s (Tensor '[1, 32, 32])   -- ^ input
  -> BVar s (Tensor '[6, 14, 14])   -- ^ output
lenetLayer1 = lenetLayer

lenetLayer2
  :: Reifies s W
  => Double                          -- ^ learning rate
  -> BVar s (Conv2d 6 16 '(5,5) )        -- ^ convolutional layer
  -> BVar s (Tensor '[ 6, 14, 14])   -- ^ input
  -> BVar s (Tensor '[16,  5,  5])   -- ^ output
lenetLayer2 = lenetLayer
-}

lenetBatch lr arch inp
  = lenetLayerBatch lr (arch ^^. conv1) inp
  & lenetLayerBatch lr (arch ^^. conv2)

  & flattenBPBatch

  -- start fully connected network
  & relu . linearBatch lr (arch ^^. fc1)
  & relu . linearBatch lr (arch ^^. fc2)
  &        linearBatch lr (arch ^^. fc3)
  -- & logSoftMax
  & softmaxN (dim :: Dim 1)


lenetLayerBatch
  :: forall inp h w ker ow oh s out mow moh step pad batch

  -- backprop constraint to hold the wengert tape
  .  Reifies s W

  -- leave input, output and square kernel size variable so that we
  -- can reuse the layer...
  => All KnownDim '[batch,inp,out,ker]

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
  -> BVar s (Tensor '[batch, inp,   h,   w])  -- ^ input
  -> BVar s (Tensor '[batch, out, moh, mow])  -- ^ output
lenetLayerBatch lr conv inp
  = NN.conv2dMMBatch
      (Step2d    :: Step2d '(1,1))
      (Padding2d :: Padding2d '(0,0))
      lr conv inp
  & relu
  & maxPooling2dBatch
      (Kernel2d  :: Kernel2d '(2,2))
      (Step2d    :: Step2d '(2,2))
      (Padding2d :: Padding2d '(0,0))
      (sing      :: SBool 'True)



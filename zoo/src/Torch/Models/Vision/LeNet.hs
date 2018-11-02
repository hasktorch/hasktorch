{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE RankNTypes #-}

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Models.Vision.LeNet where

import Data.Function ((&))
import Data.Generics.Product.Fields (field)
import Data.Generics.Product.Typed (typed)
import Data.List (intercalate)
import Data.Singletons.Prelude (SBool, sing)
import GHC.Generics (Generic)
import Lens.Micro (Lens', (^.))
import Numeric.Backprop (Backprop, BVar, Reifies, W, (^^.))
import GHC.TypeLits (KnownNat)
import qualified Numeric.Backprop as Bp
import qualified GHC.TypeLits

#ifdef CUDA
import Numeric.Dimensions
import Torch.Cuda.Double as Torch
import Torch.Cuda.Double.NN.Linear -- (Linear(..), linear)
import qualified Torch.Cuda.Double.NN.Conv2d as Conv2d
import qualified Torch.Cuda.Double.NN.Linear as Linear
#else
import Torch.Double as Torch
import Torch.Double.NN.Linear -- (Linear(..), linear)
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double.NN.Linear as Linear
#endif

import Torch.Initialization

type Flattened ker = (16*ker*ker)
data LeNet ch ker = LeNet
  { _conv1 :: !(Conv2d ch 6 '(ker, ker))
  , _conv2 :: !(Conv2d 6 16 '(ker,ker))

  , _fc1   :: !(Linear  (Flattened ker) 120)
  , _fc2   :: !(Linear       120  84)
  , _fc3   :: !(Linear        84  10)
  } deriving (Generic)

conv1 :: Lens' (LeNet ch ker) (Conv2d ch 6 '(ker, ker))
conv1 = field @"_conv1"

conv2 :: Lens' (LeNet ch ker) (Conv2d 6 16 '(ker,ker))
conv2 = field @"_conv2"

fc1 :: forall ch ker . Lens' (LeNet ch ker) (Linear (Flattened ker) 120)
fc1 = typed @(Linear (Flattened ker) 120)

fc2 :: Lens' (LeNet ch ker) (Linear 120  84)
fc2 = field @"_fc2"

fc3 :: Lens' (LeNet ch ker) (Linear 84  10)
fc3 = field @"_fc3"

instance (KnownDim (Flattened ker), KnownDim ch, KnownDim ker) => Show (LeNet ch ker) where
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

instance (KnownDim (Flattened ker), KnownDim ch, KnownDim ker) => Backprop (LeNet ch ker) where
  add a b = LeNet
    (Bp.add (_conv1 a) (_conv1 b))
    (Bp.add (_conv2 a) (_conv2 b))
    (Bp.add (_fc1 a) (_fc1 b))
    (Bp.add (_fc2 a) (_fc2 b))
    (Bp.add (_fc3 a) (_fc3 b))

  one net = LeNet
    (Bp.one (net^.conv1))
    (Bp.one (net^.conv2))
    (Bp.one (net^.fc1)  )
    (Bp.one (net^.fc2)  )
    (Bp.one (net^.fc3)  )

  zero net = LeNet
    (Bp.zero (net^.conv1))
    (Bp.zero (net^.conv2))
    (Bp.zero (net^.fc1)  )
    (Bp.zero (net^.fc2)  )
    (Bp.zero (net^.fc3)  )




-------------------------------------------------------------------------------

newLeNet :: All KnownDim '[ch,ker,Flattened ker, ker*ker] => Generator -> IO (LeNet ch ker)
newLeNet g = LeNet
  <$> newConv2d g
  <*> newConv2d g
  <*> newLinear g
  <*> newLinear g
  <*> newLinear g

-- | update a LeNet network
update net lr grad = LeNet
  (Conv2d.update (net^.conv1) lr (grad^.conv1))
  (Conv2d.update (net^.conv2) lr (grad^.conv2))
  (Linear.update (net^.fc1)   lr (grad^.fc1))
  (Linear.update (net^.fc2)   lr (grad^.fc2))
  (Linear.update (net^.fc3)   lr (grad^.fc3))

-- | update a LeNet network inplace
update_ net lr grad = do
  (Conv2d.update_ (net^.conv1) lr (grad^.conv1))
  (Conv2d.update_ (net^.conv2) lr (grad^.conv2))
  (Linear.update_ (net^.fc1)   lr (grad^.fc1))
  (Linear.update_ (net^.fc2)   lr (grad^.fc2))
  (Linear.update_ (net^.fc3)   lr (grad^.fc3))


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
  & relu . linear (arch ^^. fc1)
  & relu . linear (arch ^^. fc2)
  &        linear (arch ^^. fc3)
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
  => All KnownDim '[inp,out,ker,(ker*ker)*inp]

  -- FIXME: derive these from the signature (maybe assign them as args)
  => pad ~ 0   --  default padding size
  => step ~ 1  --  default step size for Conv2d

  -- ...this means we need the constraints for conv2d and maxPooling2d
  -- Note that oh and ow are then used as input to the maxPooling2d constraint.
  => SpatialConvolutionC inp h  w ker ker step step pad pad  oh  ow
  => SpatialDilationC       oh ow   2   2    2    2 pad pad mow moh 1 1 'True

  -- Start withe parameters
  => Double                            -- ^ learning rate for convolution layer
  -> BVar s (Conv2d inp out '(ker,ker))   -- ^ convolutional layer
  -> BVar s (Tensor '[inp,   h,   w])  -- ^ input
  -> BVar s (Tensor '[out, moh, mow])  -- ^ output
lenetLayer lr conv inp
  = Conv2d.conv2d
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
  & relu . linearBatch (arch ^^. fc1)
  & relu . linearBatch (arch ^^. fc2)
  &        linearBatch (arch ^^. fc3)
  -- & logSoftMax
  & softmaxN (dim :: Dim 1)


lenetLayerBatch
  :: forall inp h w ker ow oh s out mow moh step pad batch

  -- backprop constraint to hold the wengert tape
  .  Reifies s W

  -- leave input, output and square kernel size variable so that we
  -- can reuse the layer...
  => All KnownDim '[batch,inp,out,ker,(ker*ker)*inp]

  -- FIXME: derive these from the signature (maybe assign them as args)
  => pad ~ 0   --  default padding size
  => step ~ 1  --  default step size for Conv2d

  -- ...this means we need the constraints for conv2d and maxPooling2d
  -- Note that oh and ow are then used as input to the maxPooling2d constraint.
  => SpatialConvolutionC inp h  w ker ker step step pad pad  oh  ow
  => SpatialDilationC       oh ow   2   2    2    2 pad pad mow moh 1 1 'True

  -- Start withe parameters
  => Double                            -- ^ learning rate for convolution layer
  -> BVar s (Conv2d inp out '(ker,ker))   -- ^ convolutional layer
  -> BVar s (Tensor '[batch, inp,   h,   w])  -- ^ input
  -> BVar s (Tensor '[batch, out, moh, mow])  -- ^ output
lenetLayerBatch lr conv inp
  = Conv2d.conv2dBatch
      (Step2d    :: Step2d '(1,1))
      (Padding2d :: Padding2d '(0,0))
      lr conv inp
  & relu
  & maxPooling2dBatch
      (Kernel2d  :: Kernel2d '(2,2))
      (Step2d    :: Step2d '(2,2))
      (Padding2d :: Padding2d '(0,0))
      (sing      :: SBool 'True)


-- -- lenet
-- --   :: forall s ch h w o step pad -- ker moh mow
-- --   .  Reifies s W
-- --   => All KnownNat '[ch,h,w,o,step]
-- --   => All KnownDim '[ch,h,w,o,ch*step*step] -- , (16*step*step)]
-- --   => o ~ 10
-- --   => h ~ 32
-- --   => w ~ 32
-- --   => pad ~ 0
-- --   -- => SpatialConvolutionC ch h w ker ker step step pad pad (16*step*step) mow
-- --   -- => SpatialConvolutionC ch moh mow ker ker step step pad pad moh mow
-- --
-- --   => Double
-- --
-- --   -> BVar s (LeNet ch step)         -- ^ lenet architecture
-- --   -> BVar s (Tensor '[ch,h,w])      -- ^ input
-- --   -> BVar s (Tensor '[o])           -- ^ output
-- lenet lr arch inp
--   = lenetLayer lr (arch ^^. conv1) inp
--   & lenetLayer lr (arch ^^. conv2)
--
--   & flattenBP
--
--   -- start fully connected network
--   & relu . linear lr (arch ^^. fc1)
--   & relu . linear lr (arch ^^. fc2)
--   &        linear lr (arch ^^. fc3)
--   -- & logSoftMax
--   & softmax

-- lenetBatch lr arch inp
--   = lenetLayerBatch lr (arch ^^. conv1) inp
--   & lenetLayerBatch lr (arch ^^. conv2)
--
--   & flattenBPBatch
--
--   -- start fully connected network
--   & relu . linearBatch lr (arch ^^. fc1)
--   & relu . linearBatch lr (arch ^^. fc2)
--   &        linearBatch lr (arch ^^. fc3)
--   -- & logSoftMax
--   & softmaxN (dim :: Dim 1)


-- -- FIXME: Move this to ST
-- lenetLayerBatch_
--   :: forall inp h w ker ow oh s out mow moh step pad batch
--
--   -- backprop constraint to hold the wengert tape
--   .  Reifies s W
--
--   -- leave input, output and square kernel size variable so that we
--   -- can reuse the layer...
--   => All KnownDim '[batch,inp,out,ker]
--
--   -- FIXME: derive these from the signature (maybe assign them as args)
--   => pad ~ 0   --  default padding size
--   => step ~ 1  --  default step size for Conv2d
--
--   -- ...this means we need the constraints for conv2d and maxPooling2d
--   -- Note that oh and ow are then used as input to the maxPooling2d constraint.
--   => SpatialConvolutionC inp h  w ker ker step step pad pad  oh  ow
--   => SpatialDilationC       oh ow   2   2    2    2 pad pad mow moh 1 1 'True
--
--   -- Start withe parameters
--   => Tensor '[batch, out, moh, mow]    -- ^ output to mutate
--   -> Double                            -- ^ learning rate for convolution layer
--   -> Conv2d inp out '(ker,ker)         -- ^ convolutional layer
--   -> Tensor '[batch, inp,   h,   w]    -- ^ input
--   -> IO ()                             -- ^ output
-- lenetLayerBatch_ lr conv inp = do
-- Conv2d.conv2dBatch
--       (Step2d    :: Step2d '(1,1))
--       (Padding2d :: Padding2d '(0,0))
--       lr conv inp
--   & relu
--   & maxPooling2dBatch
--       (Kernel2d  :: Kernel2d '(2,2))
--       (Step2d    :: Step2d '(2,2))
--       (Padding2d :: Padding2d '(0,0))
--       (sing      :: SBool 'True)
--
--


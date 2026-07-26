{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

-- | Typed 2d batch normalization.
--
-- Batch normalization is honestly stateful: in training mode the forward pass
-- updates the running mean and variance in place.  The forward function here
-- is therefore in 'IO' — which composes directly into the arrow layer
-- ("Torch.Typed.NN.Arrow", whose 'Torch.Typed.NN.Arrow.Net' is a stochastic
-- function) instead of pretending to be pure.  The running statistics are
-- untyped 'MutableTensor' buffers, excluded from 'Parameterized' (they are
-- not optimized), while weight and bias are ordinary typed 'Parameter's.
module Torch.Typed.NN.BatchNorm
  ( BatchNorm2dSpec (..),
    BatchNorm2d (..),
    batchNorm2dForward,
  )
where

import GHC.Generics (Generic)
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as F
import Torch.NN (Randomizable (..))
import Torch.Tensor (MutableTensor (..))
import Torch.Typed.Factories
import Torch.Typed.Parameter
import Torch.Typed.Tensor

data BatchNorm2dSpec (channels :: Nat) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) = BatchNorm2dSpec
  deriving (Show, Eq)

data BatchNorm2d (channels :: Nat) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) = BatchNorm2d
  { bnWeight :: Parameter device dtype '[channels],
    bnBias :: Parameter device dtype '[channels],
    bnRunningMean :: MutableTensor,
    bnRunningVar :: MutableTensor
  }
  deriving (Generic, Parameterized)

instance
  (TensorOptions '[channels] dtype device) =>
  Randomizable (BatchNorm2dSpec channels dtype device) (BatchNorm2d channels dtype device)
  where
  sample BatchNorm2dSpec = do
    w <- makeIndependent (ones :: Tensor device dtype '[channels])
    b <- makeIndependent (zeros :: Tensor device dtype '[channels])
    -- cloned: these buffers are mutated in place during training, so they
    -- must not share storage with anything (in particular not with the
    -- CAF-like typed `zeros`/`ones`)
    mean <- MutableTensor <$> F.clone (toDynamic (zeros :: Tensor device dtype '[channels]))
    var <- MutableTensor <$> F.clone (toDynamic (ones :: Tensor device dtype '[channels]))
    pure (BatchNorm2d w b mean var)

-- | The forward pass.  In training mode (first argument @True@) the running
-- statistics are updated in place; in evaluation mode they are used as is.
-- Momentum and epsilon are the PyTorch defaults (0.1, 1e-5).
batchNorm2dForward ::
  BatchNorm2d channels dtype device ->
  -- | training mode
  Bool ->
  Tensor device dtype '[batchSize, channels, h, w] ->
  IO (Tensor device dtype '[batchSize, channels, h, w])
batchNorm2dForward BatchNorm2d {..} train input =
  UnsafeMkTensor
    <$> F.batchNormIO
      (toDynamic (toDependent bnWeight))
      (toDynamic (toDependent bnBias))
      bnRunningMean
      bnRunningVar
      train
      0.1
      1e-5
      (toDynamic input)

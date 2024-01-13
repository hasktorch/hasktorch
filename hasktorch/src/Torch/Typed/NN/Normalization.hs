{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.NN.Normalization where

import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.NN (HasForward (..), Randomizable (..))
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor

data LayerNormSpec (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) where
  LayerNormSpec ::
    forall normalizedShape dtype device.
    {layerNormEpsSpec :: Double} ->
    LayerNormSpec normalizedShape dtype device
  deriving (Show, Eq)

data LayerNorm (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) where
  LayerNorm ::
    { layerNormWeight :: Parameter device dtype normalizedShape,
      layerNormBias :: Parameter device dtype normalizedShape,
      layerNormEps :: Double
    } ->
    LayerNorm normalizedShape dtype device
  deriving (Show, Generic, Parameterized)

layerNormForward ::
  forall normalizedShape shape dtype device.
  ( IsSuffixOf normalizedShape shape,
    KnownShape normalizedShape
  ) =>
  LayerNorm normalizedShape dtype device ->
  Tensor device dtype shape ->
  Tensor device dtype shape
layerNormForward LayerNorm {..} =
  layerNorm @normalizedShape
    (toDependent layerNormWeight)
    (toDependent layerNormBias)
    layerNormEps

instance
  ( IsSuffixOf normalizedShape shape,
    KnownShape normalizedShape
  ) =>
  HasForward (LayerNorm normalizedShape dtype device) (Tensor device dtype shape) (Tensor device dtype shape)
  where
  forward = layerNormForward
  forwardStoch = (pure .) . forward

instance
  ( TensorOptions normalizedShape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (LayerNormSpec normalizedShape dtype device)
    (LayerNorm normalizedShape dtype device)
  where
  sample LayerNormSpec {..} =
    LayerNorm
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> pure layerNormEpsSpec

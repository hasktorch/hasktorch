{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN.Linear where

import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.NN (HasForward (..), Randomizable (..))
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor

data
  LinearSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = LinearSpec
  deriving (Show, Eq)

data
  Linear
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  Linear ::
    forall inputFeatures outputFeatures dtype device.
    { weight :: Parameter device dtype '[outputFeatures, inputFeatures],
      bias :: Parameter device dtype '[outputFeatures]
    } ->
    Linear inputFeatures outputFeatures dtype device
  deriving (Show, Generic, Parameterized)

-- | linear
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
linearForward ::
  _ =>
  Linear _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
linearForward Linear {..} input = linear' (toDependent weight) (toDependent bias) input

instance
  ( shape'' ~ MatMul shape '[inputFeatures, outputFeatures],
    shape' ~ Broadcast shape'' shape''
  ) =>
  HasForward (Linear inputFeatures outputFeatures dtype device) (Tensor device dtype shape) (Tensor device dtype shape')
  where
  forward = linearForward
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputFeatures,
    KnownNat outputFeatures,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (LinearSpec inputFeatures outputFeatures dtype device)
    (Linear inputFeatures outputFeatures dtype device)
  where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

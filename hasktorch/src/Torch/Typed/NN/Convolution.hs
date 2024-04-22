{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN.Convolution where

import Data.Proxy
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

data
  Conv1dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = Conv1dSpec
  deriving (Show, Eq)

data
  Conv1d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  Conv1d ::
    forall inputChannelSize outputChannelSize kernelSize dtype device.
    { weight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    Conv1d
      inputChannelSize
      outputChannelSize
      kernelSize
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | conv1d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv1dForward ::
  forall stride padding.
  _ =>
  Conv1d _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
conv1dForward Conv1d {..} input =
  conv1d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ stride,
         padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize,
         inputSize,
         batchSize,
         outputSize
       ],
    ConvSideCheck inputSize kernelSize stride padding outputSize
  ) =>
  HasForward (Conv1d inputChannelSize outputChannelSize kernelSize dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize])
  where
  forward model (input, Proxy, Proxy) = conv1dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (Conv1dSpec inputChannelSize outputChannelSize kernelSize dtype device)
    (Conv1d inputChannelSize outputChannelSize kernelSize dtype device)
  where
  sample Conv1dSpec =
    Conv1d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data
  Conv2dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = Conv2dSpec
  deriving (Show, Eq)

data
  Conv2d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  Conv2d ::
    forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device.
    { weight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    Conv2d
      inputChannelSize
      outputChannelSize
      kernelSize0
      kernelSize1
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | conv2d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv2dForward ::
  forall stride padding.
  _ =>
  Conv2d _ _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
conv2dForward Conv2d {..} input =
  conv2d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ Torch.Typed.Auxiliary.Fst stride,
         Torch.Typed.Auxiliary.Snd stride,
         Torch.Typed.Auxiliary.Fst padding,
         Torch.Typed.Auxiliary.Snd padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize0,
         kernelSize1,
         inputSize0,
         inputSize1,
         batchSize,
         outputSize0,
         outputSize1
       ],
    ConvSideCheck inputSize0 kernelSize0 (Torch.Typed.Auxiliary.Fst stride) (Torch.Typed.Auxiliary.Fst padding) outputSize0,
    ConvSideCheck inputSize1 kernelSize1 (Torch.Typed.Auxiliary.Snd stride) (Torch.Typed.Auxiliary.Snd padding) outputSize1
  ) =>
  HasForward (Conv2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1])
  where
  forward model (input, Proxy, Proxy) = conv2dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize0,
    KnownNat kernelSize1,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (Conv2dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
    (Conv2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
  where
  sample Conv2dSpec =
    Conv2d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data
  Conv3dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (kernelSize2 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = Conv3dSpec
  deriving (Show, Eq)

data
  Conv3d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (kernelSize2 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  Conv3d ::
    forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device.
    { weight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1, kernelSize2],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    Conv3d
      inputChannelSize
      outputChannelSize
      kernelSize0
      kernelSize1
      kernelSize2
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | conv3d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv3dForward ::
  forall stride padding.
  _ =>
  Conv3d _ _ _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
conv3dForward Conv3d {..} input =
  conv3d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ Fst3 stride,
         Snd3 stride,
         Trd3 stride,
         Fst3 padding,
         Snd3 padding,
         Trd3 padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize0,
         kernelSize1,
         kernelSize2,
         inputSize0,
         inputSize1,
         inputSize2,
         batchSize
       ],
    ConvSideCheck inputSize0 kernelSize0 (Fst3 stride) (Fst3 padding) outputSize0,
    ConvSideCheck inputSize1 kernelSize1 (Snd3 stride) (Snd3 padding) outputSize1,
    ConvSideCheck inputSize2 kernelSize2 (Trd3 stride) (Trd3 padding) outputSize2
  ) =>
  HasForward (Conv3d inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1, inputSize2], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1, outputSize2])
  where
  forward model (input, Proxy, Proxy) = conv3dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize0,
    KnownNat kernelSize1,
    KnownNat kernelSize2,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (Conv3dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
    (Conv3d inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
  where
  sample Conv3dSpec =
    Conv3d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data
  ConvTranspose1dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = ConvTranspose1dSpec
  deriving (Show, Eq)

data
  ConvTranspose1d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  ConvTranspose1d ::
    forall inputChannelSize outputChannelSize kernelSize dtype device.
    { weight :: Parameter device dtype '[inputChannelSize, outputChannelSize, kernelSize],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    ConvTranspose1d
      inputChannelSize
      outputChannelSize
      kernelSize
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | convTranspose1d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
convTranspose1dForward ::
  forall stride padding.
  _ =>
  ConvTranspose1d _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
convTranspose1dForward ConvTranspose1d {..} input =
  convTranspose1d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ stride,
         padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize,
         inputSize,
         batchSize,
         outputSize
       ],
    ConvSideCheck inputSize kernelSize stride padding outputSize
  ) =>
  HasForward (ConvTranspose1d inputChannelSize outputChannelSize kernelSize dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize])
  where
  forward model (input, Proxy, Proxy) = convTranspose1dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (ConvTranspose1dSpec inputChannelSize outputChannelSize kernelSize dtype device)
    (ConvTranspose1d inputChannelSize outputChannelSize kernelSize dtype device)
  where
  sample ConvTranspose1dSpec =
    ConvTranspose1d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data
  ConvTranspose2dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = ConvTranspose2dSpec
  deriving (Show, Eq)

data
  ConvTranspose2d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  ConvTranspose2d ::
    forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device.
    { weight :: Parameter device dtype '[inputChannelSize, outputChannelSize, kernelSize0, kernelSize1],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    ConvTranspose2d
      inputChannelSize
      outputChannelSize
      kernelSize0
      kernelSize1
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | convTranspose2d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
convTranspose2dForward ::
  forall stride padding.
  _ =>
  ConvTranspose2d _ _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
convTranspose2dForward ConvTranspose2d {..} input =
  convTranspose2d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ Torch.Typed.Auxiliary.Fst stride,
         Torch.Typed.Auxiliary.Snd stride,
         Torch.Typed.Auxiliary.Fst padding,
         Torch.Typed.Auxiliary.Snd padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize0,
         kernelSize1,
         inputSize0,
         inputSize1,
         batchSize,
         outputSize0,
         outputSize1
       ],
    ConvSideCheck inputSize0 kernelSize0 (Torch.Typed.Auxiliary.Fst stride) (Torch.Typed.Auxiliary.Fst padding) outputSize0,
    ConvSideCheck inputSize1 kernelSize1 (Torch.Typed.Auxiliary.Snd stride) (Torch.Typed.Auxiliary.Snd padding) outputSize1
  ) =>
  HasForward (ConvTranspose2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1])
  where
  forward model (input, Proxy, Proxy) = convTranspose2dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize0,
    KnownNat kernelSize1,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (ConvTranspose2dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
    (ConvTranspose2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
  where
  sample ConvTranspose2dSpec =
    ConvTranspose2d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data
  ConvTranspose3dSpec
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (kernelSize2 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = ConvTranspose3dSpec
  deriving (Show, Eq)

data
  ConvTranspose3d
    (inputChannelSize :: Nat)
    (outputChannelSize :: Nat)
    (kernelSize0 :: Nat)
    (kernelSize1 :: Nat)
    (kernelSize2 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  ConvTranspose3d ::
    forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device.
    { weight :: Parameter device dtype '[inputChannelSize, outputChannelSize, kernelSize0, kernelSize1, kernelSize2],
      bias :: Parameter device dtype '[outputChannelSize]
    } ->
    ConvTranspose3d
      inputChannelSize
      outputChannelSize
      kernelSize0
      kernelSize1
      kernelSize2
      dtype
      device
  deriving (Show, Generic, Parameterized)

-- | convTranspose3d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
convTranspose3dForward ::
  forall stride padding.
  _ =>
  ConvTranspose3d _ _ _ _ _ _ _ ->
  Tensor _ _ _ ->
  Tensor _ _ _
convTranspose3dForward ConvTranspose3d {..} input =
  convTranspose3d @stride @padding
    (toDependent weight)
    (toDependent bias)
    input

instance
  ( All
      KnownNat
      '[ Fst3 stride,
         Snd3 stride,
         Trd3 stride,
         Fst3 padding,
         Snd3 padding,
         Trd3 padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize0,
         kernelSize1,
         kernelSize2,
         inputSize0,
         inputSize1,
         inputSize2,
         batchSize
       ],
    ConvSideCheck inputSize0 kernelSize0 (Fst3 stride) (Fst3 padding) outputSize0,
    ConvSideCheck inputSize1 kernelSize1 (Snd3 stride) (Snd3 padding) outputSize1,
    ConvSideCheck inputSize2 kernelSize2 (Trd3 stride) (Trd3 padding) outputSize2
  ) =>
  HasForward (ConvTranspose3d inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device) (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1, inputSize2], Proxy stride, Proxy padding) (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1, outputSize2])
  where
  forward model (input, Proxy, Proxy) = convTranspose3dForward @stride @padding model input
  forwardStoch = (pure .) . forward

instance
  ( KnownNat inputChannelSize,
    KnownNat outputChannelSize,
    KnownNat kernelSize0,
    KnownNat kernelSize1,
    KnownNat kernelSize2,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (ConvTranspose3dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
    (ConvTranspose3d inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
  where
  sample ConvTranspose3dSpec =
    ConvTranspose3d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

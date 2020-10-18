{-# LANGUAGE TypeApplications #-}
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
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Type.Equality (type (==))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), DimType (..), Shape (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data
  Linear
    (device :: Device (DeviceType Nat))
    (dataType :: DataType (DType))
    (inputFeatures :: Dim (DimType Symbol Nat))
    (outputFeatures :: Dim (DimType Symbol Nat))
  where
  Linear ::
    forall device dataType inputFeatures outputFeatures.
    { linearWeight :: Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputFeatures, inputFeatures]),
      linearBias :: Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputFeatures])
    } ->
    Linear device dataType inputFeatures outputFeatures
  deriving (Generic)

instance
  () =>
  HasForward (Linear device dataType inputFeatures outputFeatures) (Tensor requiresGradient layout device dataType shape)
  where
  type ForwardOutput (Linear device dataType inputFeatures outputFeatures) (Tensor requiresGradient layout device dataType shape) = (Tensor requiresGradient layout device dataType shape)
  forward = undefined

instance
  ( WithDeviceC (device == 'UncheckedDevice) device (WithDataTypeF (dataType == 'UncheckedDataType) (WithDimF (inputDim == 'UncheckedDim) (WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device))))),
    WithDataTypeC (dataType == 'UncheckedDataType) dataType (WithDimF (inputDim == 'UncheckedDim) (WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device)))),
    WithDimC (inputDim == 'UncheckedDim) inputDim (WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device))),
    WithDimC (outputDim == 'UncheckedDim) outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device))
  ) =>
  HasInitialize (Linear device dataType inputDim outputDim)
  where
  type
    InitializeF (Linear device dataType inputDim outputDim) =
      ( WithDeviceF
          (device == 'UncheckedDevice)
          ( WithDataTypeF
              (dataType == 'UncheckedDataType)
              ( WithDimF
                  (inputDim == 'UncheckedDim)
                  ( WithDimF
                      (outputDim == 'UncheckedDim)
                      (Generator device -> (Linear device dataType inputDim outputDim, Generator device))
                  )
              )
          )
      )
  initialize =
    withDevice @(device == 'UncheckedDevice) @device @(WithDataTypeF (dataType == 'UncheckedDataType) (WithDimF (inputDim == 'UncheckedDim) (WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device))))) $
      \device ->
        withDataType @(dataType == 'UncheckedDataType) @dataType @(WithDimF (inputDim == 'UncheckedDim) (WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device)))) $
         \dataType ->
           withDim @(inputDim == 'UncheckedDim) @inputDim @(WithDimF (outputDim == 'UncheckedDim) (Generator device -> (Linear device dataType inputDim outputDim, Generator device))) $
            \inputDim ->
              withDim @(outputDim == 'UncheckedDim) @outputDim @(Generator device -> (Linear device dataType inputDim outputDim, Generator device)) $
                \outputDim ->
                  go device dataType inputDim outputDim
    where go device dataType inputDim outputDim = runState $ do
            weight <- state $ undefined
            bias <- state $ undefined
            pure $ Linear weight bias

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.State.Strict (MonadState (state), runState)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Initialization (xavierUniform)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), DimType (..), Shape (..), WidenShapeF, WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (CreateC, unCreate)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Prelude (KnownElem)

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
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device))))),
    WithDataTypeC dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device)))),
    WithDimC inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device))),
    WithDimC outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device)),
    CreateC (Double -> Generator device -> (Tensor 'Independent ( 'Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]), Generator device)) 'Independent ( 'Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]),
    CreateC (Generator device -> (Tensor 'Independent ( 'Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]), Generator device)) 'Independent ( 'Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])
  ) =>
  HasInitialize (Linear device dataType inputDim outputDim)
  where
  type
    InitializeF (Linear device dataType inputDim outputDim) =
      ( WithDeviceF
          device
          ( WithDataTypeF
              dataType
              ( WithDimF
                  inputDim
                  ( WithDimF
                      outputDim
                      (Generator device -> (Linear device dataType inputDim outputDim, Generator device))
                  )
              )
          )
      )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @inputDim $
              \inputDim ->
                withDim @outputDim @(Generator device -> (Linear device dataType inputDim outputDim, Generator device)) $
                  \outputDim ->
                    go deviceType dType inputDim outputDim
    where
      go deviceType dType inputDim outputDim = runState $ do
        weight <-
          state $
            unCreate @_ @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim])
              (xavierUniform @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim]))
              Independent
              Dense
              deviceType
              dType
              [outputDim, inputDim]
              (0.5 :: Double)
        bias <- state $ undefined
        pure $ Linear weight bias

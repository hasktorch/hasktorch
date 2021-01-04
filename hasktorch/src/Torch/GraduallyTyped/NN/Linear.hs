{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.State.Strict (MonadState (state), runState)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF, linear)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), NonLinearity (..), calculateFan, getter, kaimingUniform)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name, Shape (..), Size, WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), randn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  Linear
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat))
  where
  Linear ::
    forall device dataType inputDim outputDim.
    { linearWeight :: Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
      linearBias :: Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim])
    } ->
    Linear device dataType inputDim outputDim
  deriving (Generic)

type HasInitializeLinearC device dataType inputDim outputDim =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device))))),
    WithDataTypeC dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device)))),
    WithDimC inputDim (WithDimF outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device))),
    WithDimC outputDim (Generator device -> (Linear device dataType inputDim outputDim, Generator device)),
    WithCreateC (FanMode -> NonLinearity -> Generator device -> (Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
    WithCreateC (Generator device -> (Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
    WithCreateC (Generator device -> (Tensor 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim]), Generator device)) 'Independent ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim])
  )

instance
  HasInitializeLinearC device dataType inputDim outputDim =>
  HasInitialize (Linear device dataType inputDim outputDim)
  where
  type
    InitializeF (Linear device dataType inputDim outputDim) =
      WithDeviceF
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
            withoutCreate @_ @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim])
              (kaimingUniform @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim]) @device)
              Independent
              Dense
              deviceType
              dType
              [outputDim, inputDim]
              FanIn
              (LeakyRelu . Prelude.sqrt $ 5)
        bias <-
          state $
            withoutCreate @_ @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim])
              (randn @ 'Independent @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim]) @device)
              Independent
              Dense
              deviceType
              dType
              [outputDim]
        let bound :: Float =
              1
                / ( Prelude.sqrt . fromIntegral
                      . getter FanIn
                      . calculateFan
                      $ [outputDim, inputDim]
                  )

        pure $ Linear weight ((bias `mulScalar` (bound * 2)) `subScalar` bound)

instance
  HasForward
    (Linear device dataType inputFeatures outputFeatures)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
  where
  type
    ForwardOutput
      (Linear device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      Tensor
        requiresGradient'
        ('Layout 'Dense <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        (LinearF ( 'Shape '[outputFeatures, inputFeatures]) ( 'Shape '[outputFeatures]) shape')
  type
    ForwardGeneratorOutput
      (Linear device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      generator
  forward Linear {..} input g = (linear linearWeight linearBias input, g)

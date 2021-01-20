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
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), NonLinearity (..), calculateFan, getter, kaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name, Shape (..), Size, WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), randn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  Linear
    (hasBias :: HasBias)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat))
  where
  LinearWithBias ::
    forall device dataType inputDim outputDim.
    { linearWithBiasWeight :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
      linearBias :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim])
    } ->
    Linear 'WithBias device dataType inputDim outputDim
  LinearWithoutBias ::
    forall device dataType inputDim outputDim.
    { linearWithoutBiasWeight :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim])
    } ->
    Linear 'WithoutBias device dataType inputDim outputDim

type HasInitializeLinearWithBiasC device dataType inputDim outputDim =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device))))),
    WithDataTypeC dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device)))),
    WithDimC inputDim (WithDimF outputDim (Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device))),
    WithDimC outputDim (Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device)),
    WithCreateC (FanMode -> NonLinearity -> Generator device -> (Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
    WithCreateC (Generator device -> (Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
    WithCreateC (Generator device -> (Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim]), Generator device)) 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim])
  )

instance
  HasInitializeLinearWithBiasC device dataType inputDim outputDim =>
  HasInitialize (Linear 'WithBias device dataType inputDim outputDim)
  where
  type
    InitializeF (Linear 'WithBias device dataType inputDim outputDim) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                inputDim
                ( WithDimF
                    outputDim
                    (Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device))
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
                withDim @outputDim @(Generator device -> (Linear 'WithBias device dataType inputDim outputDim, Generator device)) $
                  \outputDim ->
                    go deviceType dType inputDim outputDim
    where
      go deviceType dType inputDim outputDim = runState $ do
        weight <-
          state $
            withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim])
              (kaimingUniform @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim]) @device)
              WithGradient
              Dense
              deviceType
              dType
              [outputDim, inputDim]
              FanIn
              (LeakyRelu . Prelude.sqrt $ 5)
        bias <-
          state $
            withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim])
              (randn @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim]) @device)
              WithGradient
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
        pure $ LinearWithBias weight ((bias `mulScalar` (bound * 2)) `subScalar` bound)

instance
  HasForward
    (Linear 'WithBias device dataType inputFeatures outputFeatures)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
  where
  type
    ForwardOutput
      (Linear 'WithBias device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      Tensor
        'WithGradient
        ( 'Layout 'Dense <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        (LinearWithBiasF ( 'Shape '[outputFeatures, inputFeatures]) ( 'Shape '[outputFeatures]) shape')
  type
    ForwardGeneratorOutput
      (Linear 'WithBias device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      generator
  forward LinearWithBias {..} input g = (linearWithBias linearWithBiasWeight linearBias input, g)

type HasInitializeLinearWithoutBiasC device dataType inputDim outputDim =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device))))),
    WithDataTypeC dataType (WithDimF inputDim (WithDimF outputDim (Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device)))),
    WithDimC inputDim (WithDimF outputDim (Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device))),
    WithDimC outputDim (Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device)),
    WithCreateC (FanMode -> NonLinearity -> Generator device -> (Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]),
    WithCreateC (Generator device -> (Tensor 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim]), Generator device)) 'WithGradient ( 'Layout 'Dense) device dataType ( 'Shape '[outputDim, inputDim])
  )

instance
  HasInitializeLinearWithoutBiasC device dataType inputDim outputDim =>
  HasInitialize (Linear 'WithoutBias device dataType inputDim outputDim)
  where
  type
    InitializeF (Linear 'WithoutBias device dataType inputDim outputDim) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                inputDim
                ( WithDimF
                    outputDim
                    (Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device))
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
                withDim @outputDim @(Generator device -> (Linear 'WithoutBias device dataType inputDim outputDim, Generator device)) $
                  \outputDim ->
                    go deviceType dType inputDim outputDim
    where
      go deviceType dType inputDim outputDim = runState $ do
        weight <-
          state $
            withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim])
              (kaimingUniform @ 'WithGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[outputDim, inputDim]) @device)
              WithGradient
              Dense
              deviceType
              dType
              [outputDim, inputDim]
              FanIn
              (LeakyRelu . Prelude.sqrt $ 5)
        pure $ LinearWithoutBias weight

instance
  HasForward
    (Linear 'WithoutBias device dataType inputFeatures outputFeatures)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
  where
  type
    ForwardOutput
      (Linear 'WithoutBias device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      Tensor
        'WithGradient
        ( 'Layout 'Dense <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        (LinearWithoutBiasF ( 'Shape '[outputFeatures, inputFeatures]) shape')
  type
    ForwardGeneratorOutput
      (Linear 'WithoutBias device dataType inputFeatures outputFeatures)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      generator
  forward (LinearWithoutBias linearWeight) input g = (linearWithoutBias linearWeight input, g)

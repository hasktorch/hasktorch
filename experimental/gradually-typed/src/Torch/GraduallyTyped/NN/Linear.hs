{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
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
import Data.Singletons (SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), ForNonLinearity (..), calculateFan, getter, kaimingUniform, sKaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SShape (..), Shape (..), Size (..), WithDimC (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), randn, sRandn)
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
    { linearWithBiasWeight :: Tensor 'WithGradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]),
      linearBias :: Tensor 'WithGradient ('Layout 'Dense) device dataType ('Shape '[outputDim])
    } ->
    Linear 'WithBias device dataType inputDim outputDim
  LinearWithoutBias ::
    forall device dataType inputDim outputDim.
    { linearWithoutBiasWeight :: Tensor 'WithGradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])
    } ->
    Linear 'WithoutBias device dataType inputDim outputDim

-- | TODO: Add 'ForNonLinearity' as parameter.
instance HasInitialize (Linear 'WithBias device dataType inputDim outputDim) where
  type
    InitializeF (Linear 'WithBias device dataType inputDim outputDim) =
      SDevice device ->
      SDataType dataType ->
      SDim inputDim ->
      SDim outputDim ->
      Generator device ->
      (Linear 'WithBias device dataType inputDim outputDim, Generator device)
  initialize device dataType inputDim outputDim =
    runState $ do
      let shape = SShape $ outputDim :|: inputDim :|: SNil
      weight <-
        state $
          sKaimingUniform
            SWithGradient
            (SLayout SDense)
            device
            dataType
            shape
            FanIn
            (ForLeakyRelu . Prelude.sqrt $ 5)
      bias <-
        state $
          sRandn SWithGradient (SLayout SDense) device dataType (SShape $ outputDim :|: SNil)
      let dims =
            fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
              . forgetIsChecked
              . fromSing
              $ shape
          bound :: Float =
            1
              / ( Prelude.sqrt . fromIntegral
                    . getter FanIn
                    . calculateFan
                    $ dims
                )
      pure $ LinearWithBias weight ((bias `mulScalar` (bound * 2)) `subScalar` bound)

instance
  ( output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithBiasF ('Shape '[outputFeatures, inputFeatures]) ('Shape '[outputFeatures]) shape')
  ) =>
  HasForward
    (Linear 'WithBias device dataType inputFeatures outputFeatures)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward LinearWithBias {..} input g = (linearWithBias linearWithBiasWeight linearBias input, g)

instance HasInitialize (Linear 'WithoutBias device dataType inputDim outputDim) where
  type
    InitializeF (Linear 'WithoutBias device dataType inputDim outputDim) =
      SDevice device ->
      SDataType dataType ->
      SDim inputDim ->
      SDim outputDim ->
      Generator device ->
      (Linear 'WithoutBias device dataType inputDim outputDim, Generator device)
  initialize device dataType inputDim outputDim =
    runState $ do
      weight <-
        state $
          sKaimingUniform
            SWithGradient
            (SLayout SDense)
            device
            dataType
            (SShape $ outputDim :|: inputDim :|: SNil)
            FanIn
            (ForLeakyRelu . Prelude.sqrt $ 5)
      pure $ LinearWithoutBias weight

instance
  ( output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithoutBiasF ('Shape '[outputFeatures, inputFeatures]) shape')
  ) =>
  HasForward
    (Linear 'WithoutBias device dataType inputFeatures outputFeatures)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward (LinearWithoutBias linearWeight) input g = (linearWithoutBias linearWeight input, g)

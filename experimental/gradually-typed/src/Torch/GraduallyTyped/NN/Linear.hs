{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Singletons (SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), ForNonLinearity (..), calculateFan, getter, sKaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data
  Linear
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat))
  where
  LinearWithBias ::
    forall gradient device dataType inputDim outputDim.
    { linearWithBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]),
      linearBias :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])
    } ->
    Linear 'WithBias gradient device dataType inputDim outputDim
  LinearWithoutBias ::
    forall gradient device dataType inputDim outputDim.
    { linearWithoutBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])
    } ->
    Linear 'WithoutBias gradient device dataType inputDim outputDim

deriving stock instance Show (Linear hasBias gradient device dataType inputDim outputDim)

data
  LinearSpec
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat))
  where
  LinearSpec ::
    forall hasBias gradient device dataType inputDim outputDim.
    SHasBias hasBias ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim inputDim ->
    SDim outputDim ->
    LinearSpec hasBias gradient device dataType inputDim outputDim

type instance ModelSpec (Linear hasBias gradient device dataType inputDim outputDim) = LinearSpec hasBias gradient device dataType inputDim outputDim

-- | TODO: Add 'ForNonLinearity' as parameter.
instance
  ( output ~ Linear withBias gradient (device <+> generatorDevice) dataType inputDim outputDim,
    generatorOutputDevice ~ (device <+> generatorDevice)
  ) =>
  HasInitialize
    (Linear withBias gradient device dataType inputDim outputDim)
    generatorDevice
    output
    generatorOutputDevice
  where
  initialize (LinearSpec SWithBias gradient device dataType inputDim outputDim) =
    let shape = SShape $ outputDim :|: inputDim :|: SNil
        weight =
          IxStateT $
            sKaimingUniform
              (TensorSpec gradient (SLayout SDense) device dataType shape)
              FanIn
              (ForLeakyRelu . Prelude.sqrt $ 5)
        dims =
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
        bias =
          IxStateT (sRandn (TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: SNil)))
            >>>= ireturn . (\bias' -> (bias' `mulScalar` (bound * 2)) `subScalar` bound)
     in runIxStateT $
          LinearWithBias <<$>> weight <<*>> bias
  initialize (LinearSpec SWithoutBias gradient device dataType inputDim outputDim) =
    let weight =
          IxStateT $
            sKaimingUniform
              (TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: inputDim :|: SNil))
              FanIn
              (ForLeakyRelu . Prelude.sqrt $ 5)
     in runIxStateT $ LinearWithoutBias <<$>> weight

instance
  HasStateDict
    (Linear hasBias gradient dataType inputDim outputDim device)
  where
  fromStateDict (LinearSpec SWithBias gradient device dataType inputDim outputDim) k =
    LinearWithBias
      <$> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: inputDim :|: SNil)) (k <> "weight")
      <*> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: SNil)) (k <> "bias")
  fromStateDict (LinearSpec SWithoutBias gradient device dataType inputDim outputDim) k =
    LinearWithoutBias
      <$> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: inputDim :|: SNil)) (k <> "weight")
  toStateDict k LinearWithBias {..} = do
    toStateDict (k <> "weight") linearWithBiasWeight
    toStateDict (k <> "bias") linearBias
  toStateDict k LinearWithoutBias {..} = do
    toStateDict (k <> "weight") linearWithoutBiasWeight

instance
  ( output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithBiasF ('Shape '[outputFeatures, inputFeatures]) ('Shape '[outputFeatures]) shape')
  ) =>
  HasForward
    (Linear 'WithBias gradient device dataType inputFeatures outputFeatures)
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward LinearWithBias {..} input = pure . (linearWithBias linearWithBiasWeight linearBias input,)

instance
  ( output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithoutBiasF ('Shape '[outputFeatures, inputFeatures]) shape')
  ) =>
  HasForward
    (Linear 'WithoutBias gradient device dataType inputFeatures outputFeatures)
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward (LinearWithoutBias linearWeight) input = pure . (linearWithoutBias linearWeight input,)

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
import Control.Monad.Indexed.State (IxState (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Singletons (SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), ForNonLinearity (..), calculateFan, getter, sKaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data
  Linear
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat))
    (device :: Device (DeviceType Nat))
  where
  LinearWithBias ::
    forall gradient dataType inputDim outputDim device.
    { linearWithBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]),
      linearBias :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])
    } ->
    Linear 'WithBias gradient dataType inputDim outputDim device
  LinearWithoutBias ::
    forall gradient dataType inputDim outputDim device.
    { linearWithoutBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])
    } ->
    Linear 'WithoutBias gradient dataType inputDim outputDim device

deriving stock instance Show (Linear hasBias gradient device dataType inputDim outputDim)

-- | TODO: Add 'ForNonLinearity' as parameter.
instance
  HasInitialize
    (Linear 'WithBias gradient dataType inputDim outputDim)
    ( SGradient gradient,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
    device
    generatorDevice
  where
  initialize device (gradient, dataType, inputDim, outputDim) =
    let shape = SShape $ outputDim :|: inputDim :|: SNil
        weight =
          IxState $
            sKaimingUniform
              gradient
              (SLayout SDense)
              device
              dataType
              shape
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
          IxState (sRandn gradient (SLayout SDense) device dataType (SShape $ outputDim :|: SNil))
            >>>= ireturn . (\bias' -> (bias' `mulScalar` (bound * 2)) `subScalar` bound)
     in runIxState $
          LinearWithBias <<$>> weight <<*>> bias

instance
  HasStateDict
    (Linear 'WithBias gradient dataType inputDim outputDim device)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
  where
  fromStateDict (gradient, device, dataType, inputDim, outputDim) k =
    LinearWithBias
      <$> fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ outputDim :|: inputDim :|: SNil) (k <> "weight")
      <*> fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ outputDim :|: SNil) (k <> "bias")
  toStateDict k LinearWithBias {..} = do
    toStateDict (k <> "weight") linearWithBiasWeight
    toStateDict (k <> "bias") linearBias

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
    (Linear 'WithBias gradient dataType inputFeatures outputFeatures device)
    (Tensor gradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward LinearWithBias {..} input = pure . (linearWithBias linearWithBiasWeight linearBias input,)

instance
  HasInitialize
    (Linear 'WithoutBias gradient dataType inputDim outputDim)
    ( SGradient gradient,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
    device
    generatorDevice
  where
  initialize device (gradient, dataType, inputDim, outputDim) =
    let weight =
          IxState $
            sKaimingUniform
              gradient
              (SLayout SDense)
              device
              dataType
              (SShape $ outputDim :|: inputDim :|: SNil)
              FanIn
              (ForLeakyRelu . Prelude.sqrt $ 5)
     in runIxState $ LinearWithoutBias <<$>> weight

instance
  HasStateDict
    (Linear 'WithoutBias gradient dataType inputDim outputDim device)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
  where
  fromStateDict (gradient, device, dataType, inputDim, outputDim) k =
    LinearWithoutBias
      <$> fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ outputDim :|: inputDim :|: SNil) (k <> "weight")
  toStateDict k LinearWithoutBias {..} = do
    toStateDict (k <> "weight") linearWithoutBiasWeight

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
    (Linear 'WithoutBias gradient dataType inputFeatures outputFeatures device)
    (Tensor gradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward (LinearWithoutBias linearWeight) input = pure . (linearWithoutBias linearWeight input,)

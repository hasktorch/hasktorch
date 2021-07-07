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
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE StandaloneDeriving #-}
module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Singletons (SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), ForNonLinearity (..), calculateFan, getter, sKaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator)
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

-- | TODO: Add 'ForNonLinearity' as parameter.
instance
  ( generator ~ Generator device',
    generator' ~ Generator (device <+> device')
  ) =>
  HasInitialize
    (Linear 'WithBias gradient device dataType inputDim outputDim)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
    generator
    generator'
  where
  initialize (gradient, device, dataType, inputDim, outputDim) =
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
    (Linear 'WithBias gradient device dataType inputDim outputDim)
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
    (Linear 'WithBias gradient device dataType inputFeatures outputFeatures)
    (Tensor gradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward LinearWithBias {..} input = pure . (linearWithBias linearWithBiasWeight linearBias input,)

instance
  ( generator ~ Generator device',
    generator' ~ Generator (device <+> device')
  ) =>
  HasInitialize
    (Linear 'WithoutBias gradient device dataType inputDim outputDim)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SDim inputDim,
      SDim outputDim
    )
    generator
    generator'
  where
  initialize (gradient, device, dataType, inputDim, outputDim) =
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
    (Linear 'WithoutBias gradient device dataType inputDim outputDim)
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
    (Linear 'WithoutBias gradient device dataType inputFeatures outputFeatures)
    (Tensor gradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward (LinearWithoutBias linearWeight) input = pure . (linearWithoutBias linearWeight input,)

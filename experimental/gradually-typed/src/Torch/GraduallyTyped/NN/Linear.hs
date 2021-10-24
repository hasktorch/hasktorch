{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
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
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Linear where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorDims)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF, linearWithBias, linearWithoutBias)
import Torch.GraduallyTyped.NN.Initialization (FanMode (..), ForNonLinearity (..), calculateFan, getter, sKaimingUniform)
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Prelude (pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.Random (SGetGeneratorDevice)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SShape (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic linear model with weight and optional bias.
data
  GLinear
    (weight :: Type)
    (bias :: Type)
  where
  GLinear ::
    forall weight bias.
    { -- | Linear weight
      linearWeight :: weight,
      -- | Linear bias
      linearBias :: bias
    } ->
    GLinear weight bias
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec (GLinear weight bias) = GLinear (ModelSpec weight) (ModelSpec bias)

type family
  GLinearF
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  GLinearF hasBias gradient device dataType inputDim outputDim =
    GLinear
      (NamedModel (LinearWeightF gradient device dataType inputDim outputDim))
      (NamedModel (LinearBiasF hasBias gradient device dataType outputDim))

type family
  LinearWeightF
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputDim :: Dim (Name Symbol) (Size Nat))
    (outputDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LinearWeightF gradient device dataType inputDim outputDim = Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])

type family
  LinearBiasF
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (outputDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LinearBiasF 'WithoutBias _ _ _ _ = ()
  LinearBiasF 'WithBias gradient device dataType outputDim = Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])

linearSpec ::
  forall hasBias gradient device dataType inputDim outputDim.
  SHasBias hasBias ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim inputDim ->
  SDim outputDim ->
  ModelSpec (GLinearF hasBias gradient device dataType inputDim outputDim)
linearSpec hasBias gradient device dataType inputDim outputDim =
  let weightSpec = TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: inputDim :|: SNil)
      biasSpec SWithBias = TensorSpec gradient (SLayout SDense) device dataType (SShape $ outputDim :|: SNil)
      biasSpec SWithoutBias = ()
   in GLinear (NamedModel "weight" weightSpec) (NamedModel "bias" $ biasSpec hasBias)

-- | TODO: Add 'ForNonLinearity' as parameter.
instance
  ( output
      ~ GLinear
          (Tensor gradient ('Layout 'Dense) (device <+> generatorDevice) dataType ('Shape '[outputDim, inputDim]))
          (),
    generatorOutputDevice ~ (device <+> generatorDevice),
    SGetGeneratorDevice generatorDevice
  ) =>
  HasInitialize
    (GLinear (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])) ())
    generatorDevice
    output
    generatorOutputDevice
  where
  initialize GLinear {..} =
    let weight =
          IxStateT $
            sKaimingUniform
              linearWeight
              FanIn
              (ForLeakyRelu . Prelude.sqrt $ 5)
        bias = IxStateT . initialize $ linearBias
     in runIxStateT $ GLinear <<$>> weight <<*>> bias

instance
  ( output
      ~ GLinear
          (Tensor gradient ('Layout 'Dense) (device <+> generatorDevice) dataType ('Shape '[outputDim, inputDim]))
          (Tensor gradient ('Layout 'Dense) (device <+> generatorDevice) dataType ('Shape '[outputDim])),
    generatorOutputDevice ~ (device <+> generatorDevice),
    SGetGeneratorDevice generatorDevice,
    SGetGeneratorDevice generatorOutputDevice
  ) =>
  HasInitialize
    (GLinear (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])) (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])))
    generatorDevice
    output
    generatorOutputDevice
  where
  initialize GLinear {..} =
    let weight =
          IxStateT $
            sKaimingUniform
              linearWeight
              FanIn
              (ForLeakyRelu . Prelude.sqrt $ 5)
        dims = tensorDims . tsShape $ linearWeight
        bound :: Float =
          1
            / ( Prelude.sqrt . fromIntegral
                  . getter FanIn
                  . calculateFan
                  $ dims
              )
        bias =
          IxStateT (sRandn linearBias)
            >>>= ilift
              . ( \bias' -> do
                    x <- bias' `mulScalar` (bound * 2)
                    x `subScalar` bound
                )
     in runIxStateT $ GLinear <<$>> weight <<*>> bias

instance
  HasInitialize
    (GLinear weight bias)
    generatorDevice
    (GLinear weight bias)
    generatorDevice =>
  HasInitialize
    (GLinear (NamedModel weight) (NamedModel bias))
    generatorDevice
    (GLinear (NamedModel weight) (NamedModel bias))
    generatorDevice
  where
  initialize GLinear {..} =
    let NamedModel weightName weightSpec = linearWeight
        NamedModel biasName biasSpec = linearBias
        linear = IxStateT . initialize $ GLinear weightSpec biasSpec
     in runIxStateT $ linear >>>= ireturn . (\(GLinear weight bias) -> GLinear (NamedModel weightName weight) (NamedModel biasName bias))

instance
  ( HasStateDict weight,
    HasStateDict bias
  ) =>
  HasStateDict (GLinear weight bias)
  where
  fromStateDict GLinear {..} k = GLinear <$> fromStateDict linearWeight k <*> fromStateDict linearBias k
  toStateDict k GLinear {..} = do
    toStateDict k linearWeight
    toStateDict k linearBias

instance
  ( output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithoutBiasF ('Shape '[outputDim, inputDim]) shape')
  ) =>
  HasForward
    ( GLinear
        (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]))
        ()
    )
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward GLinear {..} input = pure . (linearWithoutBias linearWeight input,)

instance
  ( output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LinearWithBiasF ('Shape '[outputDim, inputDim]) ('Shape '[outputDim]) shape')
  ) =>
  HasForward
    ( GLinear
        (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim]))
        (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim]))
    )
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward GLinear {..} input = pure . (linearWithBias linearWeight linearBias input,)

instance
  HasForward
    (GLinear weight bias)
    input
    generatorDevice
    output
    generatorDevice =>
  HasForward
    (GLinear (NamedModel weight) (NamedModel bias))
    input
    generatorDevice
    output
    generatorDevice
  where
  forward GLinear {..} input =
    let NamedModel _ weight = linearWeight
        NamedModel _ bias = linearBias
     in forward (GLinear weight bias) input

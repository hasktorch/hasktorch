{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Pooler where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.NN.Activation (Tanh)
import Torch.GraduallyTyped.NN.Class (HasForward (..))
import Torch.GraduallyTyped.NN.Linear (Linear)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient)
import Torch.GraduallyTyped.Shape.Type (Dim, Name, Size)

data
  GPooler
    (dense :: Type)
    (activation :: Type)
  where
  GPooler ::
    forall dense activation.
    { poolerDense :: dense,
      poolerActivation :: activation
    } ->
    GPooler dense activation

newtype
  Pooler
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  where
  Pooler ::
    forall style gradient device dataType inputEmbedDim.
    GPoolerF style gradient device dataType inputEmbedDim ->
    Pooler style gradient device dataType inputEmbedDim

type GPoolerF
  (style :: TransformerStyle)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) =
  GPooler
    (PoolerDenseF style gradient device dataType inputEmbedDim)
    (PoolerActivationF style)

type family
  PoolerDenseF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  PoolerDenseF 'RoBERTa gradient device dataType inputEmbedDim = Linear 'WithBias gradient device dataType inputEmbedDim inputEmbedDim

type family
  PoolerActivationF
    (style :: TransformerStyle) ::
    Type
  where
  PoolerActivationF 'RoBERTa = Tanh

instance
  ( HasForward
      (PoolerDenseF style gradient device dataType inputEmbedDim)
      input
      generator
      denseOutput
      denseGeneratorOutput,
    HasForward
      (PoolerActivationF style)
      denseOutput
      denseGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (Pooler style gradient device dataType inputEmbedDim)
    input
    generator
    output
    generatorOutput
  where
  forward (Pooler GPooler {..}) input =
    runIxState $
      ireturn input
        >>>= IxState . forward poolerDense
        >>>= IxState . forward poolerActivation
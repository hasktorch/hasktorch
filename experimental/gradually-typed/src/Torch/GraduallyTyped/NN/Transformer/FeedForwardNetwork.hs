{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL5C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (sing))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), GeluNew (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, SGetShape, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic transformer feed-forward network.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerFeedForwardNetwork'.
data
  GTransformerFeedForwardNetwork
    (inputWeight1 :: Type)
    (inputWeight2 :: Type)
    (outputWeight :: Type)
    (activation :: Type)
    (activationDropout :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GTransformerFeedForwardNetwork ::
    forall inputWeight1 inputWeight2 outputWeight activation activationDropout layerNorm dropout.
    { -- | first input weight
      ffnInputWeight1 :: inputWeight1,
      -- | second input weight
      ffnInputWeight2 :: inputWeight2,
      -- | output weight
      ffnOutputWeight :: outputWeight,
      -- | activation
      ffnActivation :: activation,
      -- | activation dropout
      ffnActivationDropout :: activationDropout,
      -- | feed-forward layer norm
      ffnLayerNorm :: layerNorm,
      -- | feed-forward dropout
      ffnDropout :: dropout
    } ->
    GTransformerFeedForwardNetwork inputWeight1 inputWeight2 outputWeight activation activationDropout layerNorm dropout

-- | Transformer feed-forward network.
data
  TransformerFeedForwardNetwork
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerFeedForwardNetwork ::
    forall style gradient device dataType queryEmbedDim ffnDim dropoutP.
    GTransformerFeedForwardNetwork
      (FFNInputWeight1F style gradient device dataType queryEmbedDim ffnDim)
      (FFNInputWeight2F style gradient device dataType queryEmbedDim ffnDim)
      (FFNOutputWeightF style gradient device dataType queryEmbedDim ffnDim)
      (FFNActivationF style)
      (FFNActivationDropoutF style dropoutP)
      (FFNLayerNormF style gradient device dataType queryEmbedDim)
      (FFNDropoutF style dropoutP) ->
    TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP

type family
  FFNInputWeight1F
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputWeight1F 'T5 gradient device dataType queryEmbedDim ffnDim = Linear 'WithoutBias gradient device dataType queryEmbedDim ffnDim
  FFNInputWeight1F 'ByT5 gradient device dataType queryEmbedDim ffnDim = FFNInputWeight1F 'T5 gradient device dataType queryEmbedDim ffnDim
  FFNInputWeight1F _ gradient device dataType queryEmbedDim ffnDim = Linear 'WithBias gradient device dataType queryEmbedDim ffnDim

type family
  FFNInputWeight2F
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputWeight2F 'ByT5 gradient device dataType queryEmbedDim ffnDim = Linear 'WithoutBias gradient device dataType queryEmbedDim ffnDim
  FFNInputWeight2F _ gradient device dataType queryEmbedDim ffnDim = ()

type family
  FFNOutputWeightF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNOutputWeightF 'T5 gradient device dataType queryEmbedDim ffnDim = Linear 'WithoutBias gradient device dataType ffnDim queryEmbedDim
  FFNOutputWeightF 'ByT5 gradient device dataType queryEmbedDim ffnDim = FFNOutputWeightF 'T5 gradient device dataType queryEmbedDim ffnDim
  FFNOutputWeightF _ gradient device dataType queryEmbedDim ffnDim = Linear 'WithBias gradient device dataType ffnDim queryEmbedDim

type family
  FFNActivationF
    (style :: TransformerStyle) ::
    Type
  where
  FFNActivationF 'T5 = Relu
  FFNActivationF 'ByT5 = GeluNew
  FFNActivationF 'BART = Gelu
  FFNActivationF 'MBART = Gelu
  FFNActivationF 'Pegasus = Relu
  FFNActivationF 'BERT = Gelu
  FFNActivationF 'RoBERTa = Gelu
  FFNActivationF 'GPT2 = Gelu

type family
  FFNActivationDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNActivationDropoutF 'T5 dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'ByT5 dropoutP = FFNActivationDropoutF 'T5 dropoutP
  FFNActivationDropoutF 'BART dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'MBART dropoutP = FFNActivationDropoutF 'BART dropoutP
  FFNActivationDropoutF 'Pegasus dropoutP = FFNActivationDropoutF 'BART dropoutP
  FFNActivationDropoutF 'BERT _ = ()
  FFNActivationDropoutF 'RoBERTa dropoutP = FFNActivationDropoutF 'BERT dropoutP
  FFNActivationDropoutF 'GPT2 _ = ()

type family
  FFNLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNLayerNormF 'T5 gradient device dataType queryEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim])
  FFNLayerNormF 'ByT5 gradient device dataType queryEmbedDim = FFNLayerNormF 'T5 gradient device dataType queryEmbedDim
  FFNLayerNormF _ gradient device dataType queryEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim])

type family
  FFNDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNDropoutF _ dropoutP = Dropout dropoutP

type family
  HasInitializeFFNInputWeight2InputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeFFNInputWeight2InputF 'T5 _ _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'ByT5 gradient device dataType queryEmbedDim ffnDim = (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim)
  HasInitializeFFNInputWeight2InputF 'BART _ _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'MBART gradient device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BART gradient device dataType queryEmbedDim ffnDim
  HasInitializeFFNInputWeight2InputF 'Pegasus gradient device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BART gradient device dataType queryEmbedDim ffnDim
  HasInitializeFFNInputWeight2InputF 'BERT _ _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'RoBERTa gradient device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BERT gradient device dataType queryEmbedDim ffnDim

type family
  HasInitializeFFNActivationDropoutInputF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  HasInitializeFFNActivationDropoutInputF 'T5 dropoutP = dropoutP
  HasInitializeFFNActivationDropoutInputF 'ByT5 dropoutP = HasInitializeFFNActivationDropoutInputF 'T5 dropoutP
  HasInitializeFFNActivationDropoutInputF 'BART dropoutP = dropoutP
  HasInitializeFFNActivationDropoutInputF 'MBART dropoutP = HasInitializeFFNActivationDropoutInputF 'BART dropoutP
  HasInitializeFFNActivationDropoutInputF 'Pegasus dropoutP = HasInitializeFFNActivationDropoutInputF 'BART dropoutP
  HasInitializeFFNActivationDropoutInputF 'BERT _ = ()
  HasInitializeFFNActivationDropoutInputF 'RoBERTa dropoutP = HasInitializeFFNActivationDropoutInputF 'BERT dropoutP

instance
  ( SingI style,
    inputWeight1 ~ FFNInputWeight1F style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight1 (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim) generator generator',
    inputWeight2 ~ FFNInputWeight2F style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight2 (HasInitializeFFNInputWeight2InputF style gradient device dataType queryEmbedDim ffnDim) generator' generator'',
    outputWeight ~ FFNOutputWeightF style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize outputWeight (SGradient gradient, SDevice device, SDataType dataType, SDim ffnDim, SDim queryEmbedDim) generator'' generator''',
    activation ~ FFNActivationF style,
    HasInitialize activation () generator''' generator''',
    activationDropout ~ FFNActivationDropoutF style dropoutP,
    HasInitialize activationDropout (HasInitializeFFNActivationDropoutInputF style dropoutP) generator''' generator''',
    layerNorm ~ FFNLayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[queryEmbedDim]), Double) generator''' generator''',
    dropout ~ FFNDropoutF style dropoutP,
    HasInitialize dropout dropoutP generator''' generator'''
  ) =>
  HasInitialize
    (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SDim queryEmbedDim,
      SDim ffnDim,
      dropoutP,
      Double
    )
    generator
    generator'''
  where
  initialize (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) =
    let inputWeight1 = IxState . initialize $ (gradient, device, dataType, queryEmbedDim, ffnDim)
        inputWeight2 = IxState . initialize $
          case sing @style of
            ST5 -> ()
            SByT5 -> (gradient, device, dataType, queryEmbedDim, ffnDim)
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        outputWeight = IxState . initialize $ (gradient, device, dataType, ffnDim, queryEmbedDim)
        activation = IxState . initialize $ ()
        activationDropout = IxState . initialize $
          case sing @style of
            ST5 -> dropoutP
            SByT5 -> dropoutP
            SBART -> dropoutP
            SMBART -> dropoutP
            SPegasus -> dropoutP
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        layerNorm = IxState . initialize $ (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps)
        dropout = IxState . initialize $ dropoutP
     in runIxState $
          ( GTransformerFeedForwardNetwork
              <<$>> inputWeight1
              <<*>> inputWeight2
              <<*>> outputWeight
              <<*>> activation
              <<*>> activationDropout
              <<*>> layerNorm
              <<*>> dropout
          )
            >>>= ireturn . TransformerFeedForwardNetwork

instance
  SingI style =>
  HasStateDict
    (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k =
    let inputWeight1 ST5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "DenseReluDense.wi.")
        inputWeight1 SByT5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "DenseReluDense.wi_0.")
        inputWeight1 SBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "fc1.")
        inputWeight1 SMBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "fc1.")
        inputWeight1 SPegasus = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "fc1.")
        inputWeight1 SBERT = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "intermediate.dense.")
        inputWeight1 SRoBERTa = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "intermediate.dense.")
        inputWeight1 SGPT2 = undefined
        inputWeight2 ST5 = fromStateDict () k
        inputWeight2 SByT5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim) (k <> "DenseReluDense.wi_1.")
        inputWeight2 SBART = fromStateDict () k
        inputWeight2 SMBART = fromStateDict () k
        inputWeight2 SPegasus = fromStateDict () k
        inputWeight2 SBERT = fromStateDict () k
        inputWeight2 SRoBERTa = fromStateDict () k
        inputWeight2 SGPT2 = fromStateDict () k
        outputWeight ST5 = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "DenseReluDense.wo.")
        outputWeight SByT5 = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "DenseReluDense.wo.")
        outputWeight SBART = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "fc2.")
        outputWeight SMBART = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "fc2.")
        outputWeight SPegasus = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "fc2.")
        outputWeight SBERT = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "output.dense.")
        outputWeight SRoBERTa = fromStateDict (gradient, device, dataType, ffnDim, queryEmbedDim) (k <> "output.dense.")
        outputWeight SGPT2 = undefined
        activation ST5 = fromStateDict () k
        activation SByT5 = fromStateDict () k
        activation SBART = fromStateDict () k
        activation SMBART = fromStateDict () k
        activation SPegasus = fromStateDict () k
        activation SBERT = fromStateDict () k
        activation SRoBERTa = fromStateDict () k
        activation SGPT2 = undefined
        activationDropout ST5 = fromStateDict dropoutP k
        activationDropout SByT5 = fromStateDict dropoutP k
        activationDropout SBART = fromStateDict dropoutP k
        activationDropout SMBART = fromStateDict dropoutP k
        activationDropout SPegasus = fromStateDict dropoutP k
        activationDropout SBERT = fromStateDict () k
        activationDropout SRoBERTa = fromStateDict () k
        activationDropout SGPT2 = undefined
        layerNorm ST5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SMBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SPegasus = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SBERT = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
     in TransformerFeedForwardNetwork
          <$> ( GTransformerFeedForwardNetwork
                  <$> inputWeight1 (sing @style)
                  <*> inputWeight2 (sing @style)
                  <*> outputWeight (sing @style)
                  <*> activation (sing @style)
                  <*> activationDropout (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
              )
  toStateDict k (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) =
    let inputWeight1 ST5 = toStateDict (k <> "DenseReluDense.wi.")
        inputWeight1 SByT5 = toStateDict (k <> "DenseReluDense.wi_0.")
        inputWeight1 SBART = toStateDict (k <> "fc1.")
        inputWeight1 SMBART = toStateDict (k <> "fc1.")
        inputWeight1 SPegasus = toStateDict (k <> "fc1.")
        inputWeight1 SBERT = toStateDict (k <> "intermediate.dense.")
        inputWeight1 SRoBERTa = toStateDict (k <> "intermediate.dense.")
        inputWeight1 SGPT2 = undefined
        inputWeight2 ST5 = toStateDict k
        inputWeight2 SByT5 = toStateDict (k <> "DenseReluDense.wi_1.")
        inputWeight2 SBART = toStateDict k
        inputWeight2 SMBART = toStateDict k
        inputWeight2 SPegasus = toStateDict k
        inputWeight2 SBERT = toStateDict k
        inputWeight2 SRoBERTa = toStateDict k
        inputWeight2 SGPT2 = undefined
        outputWeight ST5 = toStateDict (k <> "DenseReluDense.wo.")
        outputWeight SByT5 = toStateDict (k <> "DenseReluDense.wo.")
        outputWeight SBART = toStateDict (k <> "fc2.")
        outputWeight SMBART = toStateDict (k <> "fc2.")
        outputWeight SPegasus = toStateDict (k <> "fc2.")
        outputWeight SBERT = toStateDict (k <> "output.dense.")
        outputWeight SRoBERTa = toStateDict (k <> "output.dense.")
        outputWeight SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "layer_norm.")
        layerNorm SBART = toStateDict (k <> "final_layer_norm.")
        layerNorm SMBART = toStateDict (k <> "final_layer_norm.")
        layerNorm SPegasus = toStateDict (k <> "final_layer_norm.")
        layerNorm SBERT = toStateDict (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = toStateDict (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
     in do
          () <- inputWeight1 (sing @style) ffnInputWeight1
          () <- inputWeight2 (sing @style) ffnInputWeight2
          () <- outputWeight (sing @style) ffnOutputWeight
          () <- layerNorm (sing @style) ffnLayerNorm
          pure ()

type family
  FeedForwardNetworkOutputShape
    (style :: TransformerStyle)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryShape :: Shape [Dim (Name Symbol) (Size Nat)]) ::
    Shape [Dim (Name Symbol) (Size Nat)]
  where
  FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape =
    BroadcastShapesF
      queryShape
      ( LinearWithoutBiasF
          ('Shape '[queryEmbedDim, ffnDim])
          ( LinearWithoutBiasF
              ('Shape '[ffnDim, queryEmbedDim])
              ( LayerNormWithoutBiasF
                  ('Shape '[queryEmbedDim])
                  queryShape
              )
          )
      )
  FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape = FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape
  FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape =
    BroadcastShapesF
      queryShape
      ( LinearWithBiasF
          ('Shape '[queryEmbedDim, ffnDim])
          ('Shape '[queryEmbedDim])
          ( LinearWithBiasF
              ('Shape '[ffnDim, queryEmbedDim])
              ('Shape '[ffnDim])
              ( LayerNormWithBiasF
                  ('Shape '[queryEmbedDim])
                  ('Shape '[queryEmbedDim])
                  queryShape
              )
          )
      )
  FeedForwardNetworkOutputShape _ queryEmbedDim ffnDim queryShape =
    LayerNormWithBiasF
      ('Shape '[queryEmbedDim])
      ('Shape '[queryEmbedDim])
      ( BroadcastShapesF
          queryShape
          ( LinearWithBiasF
              ('Shape '[queryEmbedDim, ffnDim])
              ('Shape '[queryEmbedDim])
              ( LinearWithBiasF
                  ('Shape '[ffnDim, queryEmbedDim])
                  ('Shape '[ffnDim])
                  queryShape
              )
          )
      )

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'T5@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'T5 gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnLayerNorm
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'ByT5@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'ByT5 gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    let activate query' =
          ireturn query'
            >>>= IxState . forward ffnInputWeight1
            >>>= IxState . forward ffnActivation
        gate query' = (*) <<$>> activate query' <<*>> (IxState . forward ffnInputWeight2 $ query')
     in runIxState $
          ireturn query
            >>>= IxState . forward ffnLayerNorm
            >>>= gate
            >>>= IxState . forward ffnActivation
            >>>= IxState . forward ffnActivationDropout
            >>>= IxState . forward ffnOutputWeight
            >>>= IxState . forward ffnDropout
            >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'BART@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (gradient <|> queryGradient)
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'BART queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> ((device <+> queryDevice) <+> generatorDevice)) <+> ((device <+> queryDevice) <+> generatorDevice))
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BART gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayerNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'BERT@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (gradient <|> queryGradient)
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'BERT queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BERT gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayerNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'RoBERTa@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (gradient <|> queryGradient)
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'RoBERTa queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'RoBERTa gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayerNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'Pegasus@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'Pegasus gradient device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnLayerNorm
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)

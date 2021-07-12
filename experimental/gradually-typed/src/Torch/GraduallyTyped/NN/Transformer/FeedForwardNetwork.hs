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
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (sing))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), GeluNew (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (Linear (..), LinearSpec (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Random (Generator, sGeneratorToDevice)
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
  where
  TransformerFeedForwardNetwork ::
    forall style gradient device dataType queryEmbedDim ffnDim.
    GTransformerFeedForwardNetwork
      (FFNInputWeight1F style gradient device dataType queryEmbedDim ffnDim)
      (FFNInputWeight2F style gradient device dataType queryEmbedDim ffnDim)
      (FFNOutputWeightF style gradient device dataType queryEmbedDim ffnDim)
      (FFNActivationF style)
      (FFNActivationDropoutF style)
      (FFNLayerNormF style gradient device dataType queryEmbedDim)
      (FFNDropoutF style) ->
    TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim

data
  TransformerFeedForwardNetworkSpec
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
  where
  TransformerFeedForwardNetworkSpec ::
    forall style gradient device dataType queryEmbedDim ffnDim.
    STransformerStyle style ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim queryEmbedDim ->
    SDim ffnDim ->
    Double ->
    Double ->
    TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim

type instance ModelSpec (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim) = TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim

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
    (style :: TransformerStyle) ::
    Type
  where
  FFNActivationDropoutF 'T5 = Dropout
  FFNActivationDropoutF 'ByT5 = FFNActivationDropoutF 'T5
  FFNActivationDropoutF 'BART = Dropout
  FFNActivationDropoutF 'MBART = FFNActivationDropoutF 'BART
  FFNActivationDropoutF 'Pegasus = FFNActivationDropoutF 'BART
  FFNActivationDropoutF 'BERT = ()
  FFNActivationDropoutF 'RoBERTa = FFNActivationDropoutF 'BERT
  FFNActivationDropoutF 'GPT2 = ()

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
    (style :: TransformerStyle) ::
    Type
  where
  FFNDropoutF _ = Dropout

instance
  ( inputWeight1 ~ FFNInputWeight1F style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight1 device inputWeight1 device,
    inputWeight2 ~ FFNInputWeight2F style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight2 device inputWeight2 device,
    outputWeight ~ FFNOutputWeightF style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize outputWeight device outputWeight device,
    activation ~ FFNActivationF style,
    HasInitialize activation device activation device,
    activationDropout ~ FFNActivationDropoutF style,
    HasInitialize activationDropout device activationDropout device,
    layerNorm ~ FFNLayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    dropout ~ FFNDropoutF style,
    HasInitialize dropout device dropout device
  ) =>
  HasInitialize
    (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim)
    generatorDevice
    (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim)
    device
  where
  initialize (TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        inputWeight1WithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType queryEmbedDim ffnDim
        inputWeight1WithBiasSpec = LinearSpec SWithBias gradient device dataType queryEmbedDim ffnDim
        inputWeight1 = IxStateT . initialize @inputWeight1 $
          case style of
            ST5 -> inputWeight1WithoutBiasSpec
            SByT5 -> inputWeight1WithoutBiasSpec
            SBART -> inputWeight1WithBiasSpec
            SMBART -> inputWeight1WithBiasSpec
            SPegasus -> inputWeight1WithBiasSpec
            SBERT -> inputWeight1WithBiasSpec
            SRoBERTa -> inputWeight1WithBiasSpec
            SGPT2 -> undefined
        inputWeight2 = IxStateT . initialize @inputWeight2 $
          case style of
            ST5 -> ()
            SByT5 -> LinearSpec SWithoutBias gradient device dataType queryEmbedDim ffnDim
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        outputWeightWithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType ffnDim queryEmbedDim
        outputWeightWithBiasSpec = LinearSpec SWithBias gradient device dataType ffnDim queryEmbedDim
        outputWeight = IxStateT . initialize @outputWeight $
          case style of
            ST5 -> outputWeightWithoutBiasSpec
            SByT5 -> outputWeightWithoutBiasSpec
            SBART -> outputWeightWithBiasSpec
            SMBART -> outputWeightWithBiasSpec
            SPegasus -> outputWeightWithBiasSpec
            SBERT -> outputWeightWithBiasSpec
            SRoBERTa -> outputWeightWithBiasSpec
            SGPT2 -> undefined
        activation = IxStateT . initialize @activation $
          case style of
            ST5 -> Relu
            SByT5 -> GeluNew
            SBART -> Gelu
            SMBART -> Gelu
            SPegasus -> Relu
            SBERT -> Gelu
            SRoBERTa -> Gelu
            SGPT2 -> undefined
        activationDropout = IxStateT . initialize @activationDropout $
          case style of
            ST5 -> Dropout dropoutP
            SByT5 -> Dropout dropoutP
            SBART -> Dropout dropoutP
            SMBART -> Dropout dropoutP
            SPegasus -> Dropout dropoutP
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $
          case style of
            ST5 -> layerNormWithoutBiasSpec
            SByT5 -> layerNormWithoutBiasSpec
            SBART -> layerNormWithBiasSpec
            SMBART -> layerNormWithBiasSpec
            SPegasus -> layerNormWithBiasSpec
            SBERT -> layerNormWithBiasSpec
            SRoBERTa -> layerNormWithBiasSpec
            SGPT2 -> undefined
        dropout = IxStateT . initialize @dropout $ Dropout dropoutP
        gffn =
          GTransformerFeedForwardNetwork
            <<$>> inputWeight1
            <<*>> inputWeight2
            <<*>> outputWeight
            <<*>> activation
            <<*>> activationDropout
            <<*>> layerNorm
            <<*>> dropout
     in runIxStateT (gffn >>>= ireturn . TransformerFeedForwardNetwork) generator'

instance
  SingI style =>
  HasStateDict
    (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim)
  where
  fromStateDict (TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps) k =
    let inputWeight1WithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType queryEmbedDim ffnDim
        inputWeight1WithBiasSpec = LinearSpec SWithBias gradient device dataType queryEmbedDim ffnDim
        inputWeight1 ST5 = fromStateDict inputWeight1WithoutBiasSpec (k <> "DenseReluDense.wi.")
        inputWeight1 SByT5 = fromStateDict inputWeight1WithoutBiasSpec (k <> "DenseReluDense.wi_0.")
        inputWeight1 SBART = fromStateDict inputWeight1WithBiasSpec (k <> "fc1.")
        inputWeight1 SMBART = fromStateDict inputWeight1WithBiasSpec (k <> "fc1.")
        inputWeight1 SPegasus = fromStateDict inputWeight1WithBiasSpec (k <> "fc1.")
        inputWeight1 SBERT = fromStateDict inputWeight1WithBiasSpec (k <> "intermediate.dense.")
        inputWeight1 SRoBERTa = fromStateDict inputWeight1WithBiasSpec (k <> "intermediate.dense.")
        inputWeight1 SGPT2 = undefined
        inputWeight2 ST5 = fromStateDict () k
        inputWeight2 SByT5 = fromStateDict (LinearSpec SWithoutBias gradient device dataType queryEmbedDim ffnDim) (k <> "DenseReluDense.wi_1.")
        inputWeight2 SBART = fromStateDict () k
        inputWeight2 SMBART = fromStateDict () k
        inputWeight2 SPegasus = fromStateDict () k
        inputWeight2 SBERT = fromStateDict () k
        inputWeight2 SRoBERTa = fromStateDict () k
        inputWeight2 SGPT2 = fromStateDict () k
        outputWeightWithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType ffnDim queryEmbedDim
        outputWeightWithBiasSpec = LinearSpec SWithBias gradient device dataType ffnDim queryEmbedDim
        outputWeight ST5 = fromStateDict outputWeightWithoutBiasSpec (k <> "DenseReluDense.wo.")
        outputWeight SByT5 = fromStateDict outputWeightWithoutBiasSpec (k <> "DenseReluDense.wo.")
        outputWeight SBART = fromStateDict outputWeightWithBiasSpec (k <> "fc2.")
        outputWeight SMBART = fromStateDict outputWeightWithBiasSpec (k <> "fc2.")
        outputWeight SPegasus = fromStateDict outputWeightWithBiasSpec (k <> "fc2.")
        outputWeight SBERT = fromStateDict outputWeightWithBiasSpec (k <> "output.dense.")
        outputWeight SRoBERTa = fromStateDict outputWeightWithBiasSpec (k <> "output.dense.")
        outputWeight SGPT2 = undefined
        activation ST5 = fromStateDict Relu k
        activation SByT5 = fromStateDict GeluNew k
        activation SBART = fromStateDict Gelu k
        activation SMBART = fromStateDict Gelu k
        activation SPegasus = fromStateDict Relu k
        activation SBERT = fromStateDict Gelu k
        activation SRoBERTa = fromStateDict Gelu k
        activation SGPT2 = undefined
        activationDropout ST5 = fromStateDict (Dropout dropoutP) k
        activationDropout SByT5 = fromStateDict (Dropout dropoutP) k
        activationDropout SBART = fromStateDict (Dropout dropoutP) k
        activationDropout SMBART = fromStateDict (Dropout dropoutP) k
        activationDropout SPegasus = fromStateDict (Dropout dropoutP) k
        activationDropout SBERT = fromStateDict () k
        activationDropout SRoBERTa = fromStateDict () k
        activationDropout SGPT2 = undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNorm ST5 = fromStateDict layerNormWithoutBiasSpec (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict layerNormWithoutBiasSpec (k <> "layer_norm.")
        layerNorm SBART = fromStateDict layerNormWithBiasSpec (k <> "final_layer_norm.")
        layerNorm SMBART = fromStateDict layerNormWithBiasSpec (k <> "final_layer_norm.")
        layerNorm SPegasus = fromStateDict layerNormWithBiasSpec (k <> "final_layer_norm.")
        layerNorm SBERT = fromStateDict layerNormWithBiasSpec (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict layerNormWithBiasSpec (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict (Dropout dropoutP) k
     in TransformerFeedForwardNetwork
          <$> ( GTransformerFeedForwardNetwork
                  <$> inputWeight1 style
                  <*> inputWeight2 style
                  <*> outputWeight style
                  <*> activation style
                  <*> activationDropout style
                  <*> layerNorm style
                  <*> dropout style
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
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape),
    generatorOutputDevice ~ (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'T5 gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnLayerNorm
        >>>= IxStateT . forward ffnInputWeight1
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnActivationDropout
        >>>= IxStateT . forward ffnOutputWeight
        >>>= IxStateT . forward ffnDropout
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
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape),
    generatorOutputDevice ~ (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'ByT5 gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    let activate query' =
          ireturn query'
            >>>= IxStateT . forward ffnInputWeight1
            >>>= IxStateT . forward ffnActivation
        gate query' = (*) <<$>> activate query' <<*>> (IxStateT . forward ffnInputWeight2 $ query')
     in runIxStateT $
          ireturn query
            >>>= IxStateT . forward ffnLayerNorm
            >>>= gate
            >>>= IxStateT . forward ffnActivation
            >>>= IxStateT . forward ffnActivationDropout
            >>>= IxStateT . forward ffnOutputWeight
            >>>= IxStateT . forward ffnDropout
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
    output
      ~ Tensor
          (gradient <|> queryGradient)
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'BART queryEmbedDim ffnDim queryShape),
    generatorOutputDevice
      ~ ((device <+> ((device <+> queryDevice) <+> generatorDevice)) <+> ((device <+> queryDevice) <+> generatorDevice))
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BART gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnInputWeight1
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnActivationDropout
        >>>= IxStateT . forward ffnOutputWeight
        >>>= IxStateT . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward ffnLayerNorm

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
    generatorOutputDevice
      ~ ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BERT gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnInputWeight1
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnOutputWeight
        >>>= IxStateT . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward ffnLayerNorm

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
    output
      ~ Tensor
          (gradient <|> queryGradient)
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'RoBERTa queryEmbedDim ffnDim queryShape),
    generatorOutputDevice
      ~ ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'RoBERTa gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnInputWeight1
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnOutputWeight
        >>>= IxStateT . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward ffnLayerNorm

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
    output
      ~ Tensor
          (queryGradient <|> gradient)
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape),
    generatorOutputDevice ~ (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'Pegasus gradient device dataType queryEmbedDim ffnDim)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnLayerNorm
        >>>= IxStateT . forward ffnInputWeight1
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnActivationDropout
        >>>= IxStateT . forward ffnOutputWeight
        >>>= IxStateT . forward ffnDropout
        >>>= ireturn . (query `add`)

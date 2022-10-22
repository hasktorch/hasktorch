{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GFeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), GeluNew (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Linear (GLinear (..), GLinearF, linearSpec)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), HasDropout (..), SHasBias (..), SHasDropout (..))
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SShape (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic two-layer gate with activation function.
--
-- - @layer0@ is the first layer.
-- - @activation@ is the activation function.
-- - @layer1@ is the second layer.
data
  GGate
    (layer0 :: Type)
    (activation :: Type)
    (layer1 :: Type)
  where
  GGate ::
    forall layer0 activation layer1.
    { -- | first gate layer
      gateLayer0 :: layer0,
      -- | gate activation
      gateActivation :: activation,
      -- | second gate layer
      gateLayer1 :: layer1
    } ->
    GGate layer0 activation layer1
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GGate layer0 activation layer1) =
    GGate (ModelSpec layer0) (ModelSpec activation) (ModelSpec layer1)

instance
  ( HasInitialize layer0 generatorDevice layer0' generatorDevice0,
    HasInitialize activation generatorDevice0 activation' generatorDevice1,
    HasInitialize layer1 generatorDevice1 layer1' generatorOutputDevice
  ) =>
  HasInitialize
    (GGate layer0 activation layer1)
    generatorDevice
    (GGate layer0' activation' layer1')
    generatorOutputDevice

instance
  (HasStateDict layer0, HasStateDict activation, HasStateDict layer1) =>
  HasStateDict (GGate layer0 activation layer1)

instance
  ( HasForward
      layer0
      (Tensor gradient layout device dataType shape)
      generatorDevice
      (Tensor gradient' layout' device' dataType' shape')
      generatorDevice',
    HasForward
      activation
      (Tensor gradient' layout' device' dataType' shape')
      generatorDevice'
      (Tensor gradient' layout' device' dataType' shape')
      generatorDevice',
    HasForward
      layer1
      (Tensor gradient layout device dataType shape)
      generatorDevice'
      (Tensor gradient' layout' device' dataType' shape')
      generatorDevice'',
    output ~ Tensor gradient' layout' device' dataType' shape',
    generatorOutputDevice ~ generatorDevice''
  ) =>
  HasForward
    (GGate layer0 activation layer1)
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GGate {..} input =
    let activate input' =
          ireturn input'
            >>>= IxStateT . forward gateLayer0
            >>>= IxStateT . forward gateActivation
        gate input' = (*) <<$>> activate input' <<*>> (IxStateT . forward gateLayer1 $ input')
     in runIxStateT $ ireturn input >>>= gate

-- | Generic transformer feed-forward network.
--
-- - @inputLayerNorm@ is the layer normalization for the input.
-- - @inputTransformation@ is the input transformation.
-- - @activation@ is the activation function.
-- - @activationDropout@ is the activation dropout layer.
-- - @outputProjection@ is the output projection.
-- - @outputDropout@ is the dropout layer for the output.
-- - @outputLayerNorm@ is the layer normalization for the output.
data
  GTransformerFeedForwardNetwork
    (inputLayerNorm :: Type)
    (inputTransformation :: Type)
    (activation :: Type)
    (activationDropout :: Type)
    (outputProjection :: Type)
    (outputDropout :: Type)
    (outputLayerNorm :: Type)
  where
  GTransformerFeedForwardNetwork ::
    forall inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm.
    { -- | input layer norm
      ffnInputLayerNorm :: inputLayerNorm,
      -- | input transformation
      ffnInputTransformation :: inputTransformation,
      -- | activation
      ffnActivation :: activation,
      -- | activation dropout
      ffnActivationDropout :: activationDropout,
      -- | output projection
      ffnOutputProjection :: outputProjection,
      -- | output dropout
      ffnOutputDropout :: outputDropout,
      -- | output layer norm
      ffnOutputLayerNorm :: outputLayerNorm
    } ->
    GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm) =
    GTransformerFeedForwardNetwork (ModelSpec inputLayerNorm) (ModelSpec inputTransformation) (ModelSpec activation) (ModelSpec activationDropout) (ModelSpec outputProjection) (ModelSpec outputDropout) (ModelSpec outputLayerNorm)

type family
  GTransformerFeedForwardNetworkF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (hasDropout :: HasDropout) ::
    Type
  where
  GTransformerFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim hasDropout =
    GTransformerFeedForwardNetwork
      (FFNInputLayerNormF style gradient device dataType queryEmbedDim)
      (FFNInputTransformationF style gradient device dataType queryEmbedDim ffnDim)
      (FFNActivationF style)
      (FFNActivationDropoutF style hasDropout)
      (FFNOutputProjectionF style gradient device dataType queryEmbedDim ffnDim)
      (FFNOutputDropoutF style hasDropout)
      (FFNOutputLayerNormF style gradient device dataType queryEmbedDim)

-- | Specifies the layer normalization for the input.
type family
  FFNInputLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputLayerNormF 'T5 gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim]))
  FFNInputLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    FFNInputLayerNormF 'T5 gradient device dataType queryEmbedDim
  FFNInputLayerNormF 'BART _ _ _ _ =
    ()
  FFNInputLayerNormF 'MBART gradient device dataType queryEmbedDim =
    FFNInputLayerNormF 'BART gradient device dataType queryEmbedDim
  FFNInputLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  FFNInputLayerNormF 'BERT _ _ _ _ =
    ()
  FFNInputLayerNormF 'RoBERTa gradient device dataType queryEmbedDim =
    FFNInputLayerNormF 'BERT gradient device dataType queryEmbedDim

-- | Specifies the first input projection.
type family
  FFNInputTransformationF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputTransformationF 'T5 gradient device dataType queryEmbedDim ffnDim =
    NamedModel (GLinearF 'WithoutBias gradient device dataType queryEmbedDim ffnDim)
  FFNInputTransformationF 'ByT5 gradient device dataType queryEmbedDim ffnDim =
    GGate
      (NamedModel (GLinearF 'WithoutBias gradient device dataType queryEmbedDim ffnDim))
      GeluNew
      (NamedModel (GLinearF 'WithoutBias gradient device dataType queryEmbedDim ffnDim))
  FFNInputTransformationF _ gradient device dataType queryEmbedDim ffnDim =
    NamedModel (GLinearF 'WithBias gradient device dataType queryEmbedDim ffnDim)

-- | Specifies the activation.
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

-- | Specifies the activation dropout.
type family
  FFNActivationDropoutF
    (style :: TransformerStyle)
    (hasDropout :: HasDropout) ::
    Type
  where
  FFNActivationDropoutF 'T5 'WithDropout = Dropout
  FFNActivationDropoutF 'ByT5 hasDropout = FFNActivationDropoutF 'T5 hasDropout
  FFNActivationDropoutF 'BART 'WithDropout = Dropout
  FFNActivationDropoutF 'MBART hasDropout = FFNActivationDropoutF 'BART hasDropout
  FFNActivationDropoutF 'Pegasus hasDropout = FFNActivationDropoutF 'BART hasDropout
  FFNActivationDropoutF 'BERT _ = ()
  FFNActivationDropoutF 'RoBERTa hasDropout = FFNActivationDropoutF 'BERT hasDropout
  FFNActivationDropoutF _ 'WithoutDropout = ()

-- | Specifies the output projection.
type family
  FFNOutputProjectionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNOutputProjectionF 'T5 gradient device dataType queryEmbedDim ffnDim =
    NamedModel (GLinearF 'WithoutBias gradient device dataType ffnDim queryEmbedDim)
  FFNOutputProjectionF 'ByT5 gradient device dataType queryEmbedDim ffnDim =
    FFNOutputProjectionF 'T5 gradient device dataType queryEmbedDim ffnDim
  FFNOutputProjectionF 'BART gradient device dataType queryEmbedDim ffnDim =
    NamedModel (GLinearF 'WithBias gradient device dataType ffnDim queryEmbedDim)
  FFNOutputProjectionF 'MBART gradient device dataType queryEmbedDim ffnDim =
    FFNOutputProjectionF 'BART gradient device dataType queryEmbedDim ffnDim
  FFNOutputProjectionF 'Pegasus gradient device dataType queryEmbedDim ffnDim =
    FFNOutputProjectionF 'BART gradient device dataType queryEmbedDim ffnDim
  FFNOutputProjectionF 'BERT gradient device dataType queryEmbedDim ffnDim =
    NamedModel (GLinearF 'WithBias gradient device dataType ffnDim queryEmbedDim)
  FFNOutputProjectionF 'RoBERTa gradient device dataType queryEmbedDim ffnDim =
    FFNOutputProjectionF 'BERT gradient device dataType queryEmbedDim ffnDim

-- | Specifies the dropout for the output.
type family
  FFNOutputDropoutF
    (style :: TransformerStyle)
    (hasDropout :: HasDropout) ::
    Type
  where
  FFNOutputDropoutF _ 'WithDropout = Dropout
  FFNOutputDropoutF _ 'WithoutDropout = ()

-- | Specifies the layer normalization for the output.
type family
  FFNOutputLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNOutputLayerNormF 'T5 _ _ _ _ =
    ()
  FFNOutputLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    FFNOutputLayerNormF 'T5 gradient device dataType queryEmbedDim
  FFNOutputLayerNormF 'BART gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  FFNOutputLayerNormF 'MBART gradient device dataType queryEmbedDim =
    FFNOutputLayerNormF 'BART gradient device dataType queryEmbedDim
  FFNOutputLayerNormF 'Pegasus _ _ _ _ =
    ()
  FFNOutputLayerNormF 'BERT gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  FFNOutputLayerNormF 'RoBERTa gradient device dataType queryEmbedDim =
    FFNOutputLayerNormF 'BERT gradient device dataType queryEmbedDim

-- | Specifies the parameters of the transformer feed forward network.
--
-- - @style@: the style of the transformer feed forward network, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the network's parameters.
-- - @device@: the computational device on which the parameters are allocated.
-- - @dataType@: the data type of the parameters.
-- - @queryEmbedDim@: the dimension of the query embedding.
-- - @ffnDim@: the dimension of the feed forward network's hidden state.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
transformerFeedForwardNetworkSpec ::
  forall style gradient device dataType queryEmbedDim ffnDim hasDropout.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim queryEmbedDim ->
  SDim ffnDim ->
  SHasDropout hasDropout ->
  Double ->
  Double ->
  ModelSpec (GTransformerFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim hasDropout)
transformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim hasDropout dropoutP eps =
  let inputLayerNormSpec ST5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      inputLayerNormSpec SByT5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      inputLayerNormSpec SBART = ()
      inputLayerNormSpec SMBART = ()
      inputLayerNormSpec SPegasus = NamedModel "final_layer_norm." layerNormWithBiasSpec
      inputLayerNormSpec SBERT = ()
      inputLayerNormSpec SRoBERTa = ()
      inputLayerNormSpec SGPT2 = undefined
      inputTransformationSpec ST5 = NamedModel "DenseReluDense.wi." $ weightSpecWithoutBias queryEmbedDim ffnDim
      inputTransformationSpec SByT5 =
        GGate
          (NamedModel "DenseReluDense.wi_0." $ weightSpecWithoutBias queryEmbedDim ffnDim)
          GeluNew
          (NamedModel "DenseReluDense.wi_1." $ weightSpecWithoutBias queryEmbedDim ffnDim)
      inputTransformationSpec SBART = NamedModel "fc1." $ weightSpecWithBias queryEmbedDim ffnDim
      inputTransformationSpec SMBART = NamedModel "fc1." $ weightSpecWithBias queryEmbedDim ffnDim
      inputTransformationSpec SPegasus = NamedModel "fc1." $ weightSpecWithBias queryEmbedDim ffnDim
      inputTransformationSpec SBERT = NamedModel "intermediate.dense." $ weightSpecWithBias queryEmbedDim ffnDim
      inputTransformationSpec SRoBERTa = NamedModel "intermediate.dense." $ weightSpecWithBias queryEmbedDim ffnDim
      inputTransformationSpec SGPT2 = undefined
      activationSpec :: STransformerStyle style -> ModelSpec (FFNActivationF style)
      activationSpec ST5 = Relu
      activationSpec SByT5 = GeluNew
      activationSpec SBART = Gelu
      activationSpec SMBART = Gelu
      activationSpec SPegasus = Relu
      activationSpec SBERT = Gelu
      activationSpec SRoBERTa = Gelu
      activationSpec SGPT2 = undefined
      activationDropoutSpec ST5 SWithDropout = Dropout dropoutP
      activationDropoutSpec ST5 SWithoutDropout = ()
      activationDropoutSpec SByT5 SWithDropout = Dropout dropoutP
      activationDropoutSpec SByT5 SWithoutDropout = ()
      activationDropoutSpec SBART SWithDropout = Dropout dropoutP
      activationDropoutSpec SBART SWithoutDropout = ()
      activationDropoutSpec SMBART SWithDropout = Dropout dropoutP
      activationDropoutSpec SMBART SWithoutDropout = ()
      activationDropoutSpec SPegasus SWithDropout = Dropout dropoutP
      activationDropoutSpec SPegasus SWithoutDropout = ()
      activationDropoutSpec SBERT _ = ()
      activationDropoutSpec SRoBERTa _ = ()
      activationDropoutSpec SGPT2 _ = undefined
      outputProjectionSpec ST5 = NamedModel "DenseReluDense.wo." $ weightSpecWithoutBias ffnDim queryEmbedDim
      outputProjectionSpec SByT5 = NamedModel "DenseReluDense.wo." $ weightSpecWithoutBias ffnDim queryEmbedDim
      outputProjectionSpec SBART = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SMBART = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SPegasus = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SBERT = NamedModel "output.dense." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SRoBERTa = NamedModel "output.dense." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SGPT2 = undefined
      outputDropoutSpec _ SWithDropout = Dropout dropoutP
      outputDropoutSpec _ SWithoutDropout = ()
      outputLayerNormSpec ST5 = ()
      outputLayerNormSpec SByT5 = ()
      outputLayerNormSpec SBART = NamedModel "final_layer_norm." layerNormWithBiasSpec
      outputLayerNormSpec SMBART = NamedModel "final_layer_norm." layerNormWithBiasSpec
      outputLayerNormSpec SPegasus = ()
      outputLayerNormSpec SBERT = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      outputLayerNormSpec SRoBERTa = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      outputLayerNormSpec SGPT2 = undefined
   in GTransformerFeedForwardNetwork
        (inputLayerNormSpec style)
        (inputTransformationSpec style)
        (activationSpec style)
        (activationDropoutSpec style hasDropout)
        (outputProjectionSpec style)
        (outputDropoutSpec style hasDropout)
        (outputLayerNormSpec style)
  where
    weightSpecWithoutBias ::
      forall inputDim outputDim.
      SDim inputDim ->
      SDim outputDim ->
      ModelSpec
        ( GLinear
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])))
            (NamedModel ())
        )
    weightSpecWithoutBias = linearSpec SWithoutBias gradient device dataType
    weightSpecWithBias ::
      forall inputDim outputDim.
      SDim inputDim ->
      SDim outputDim ->
      ModelSpec
        ( GLinear
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim, inputDim])))
            (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[outputDim])))
        )
    weightSpecWithBias = linearSpec SWithBias gradient device dataType
    layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
    layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps

instance
  ( HasInitialize inputLayerNorm generatorDevice inputLayerNorm' generatorDevice0,
    HasInitialize inputTransformation generatorDevice0 inputTransformation' generatorDevice1,
    HasInitialize activation generatorDevice1 activation' generatorDevice2,
    HasInitialize activationDropout generatorDevice2 activationDropout' generatorDevice3,
    HasInitialize outputProjection generatorDevice3 outputProjection' generatorDevice4,
    HasInitialize outputDropout generatorDevice4 outputDropout' generatorDevice5,
    HasInitialize outputLayerNorm generatorDevice5 outputLayerNorm' generatorOutputDevice
  ) =>
  HasInitialize
    (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm)
    generatorDevice
    (GTransformerFeedForwardNetwork inputLayerNorm' inputTransformation' activation' activationDropout' outputProjection' outputDropout' outputLayerNorm')
    generatorOutputDevice

instance
  ( HasStateDict inputLayerNorm,
    HasStateDict inputTransformation,
    HasStateDict activation,
    HasStateDict activationDropout,
    HasStateDict outputProjection,
    HasStateDict outputDropout,
    HasStateDict outputLayerNorm
  ) =>
  HasStateDict (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm)

-- | 'HasForward' instance for 'GTransformerFeedForwardNetwork'.
--
-- @
--       ┌───────┐
--       │ query ├────────┐
--       └───┬───┘        │
--           │            │
--           ▼            │
--  (ffnInputLayerNorm)   │
--           ▼            │
-- ffnInputTransformation │
--           ▼            │
--     ffnActivation      │
--           ▼            │
-- (ffnActivationDropout) │
--           ▼            │
--   ffnOutputProjecton   │
--           ▼            │
--    ffnOutputDropout    │
--           │            │
--           ▼            │
--          add◄──────────┘
--           │
--           ▼
--  (ffnOutputLayerNorm)
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( HasForward
      inputLayerNorm
      (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      inputTransformation
      tensor0
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      activation
      tensor1
      generatorDevice1
      tensor2
      generatorDevice2,
    HasForward
      activationDropout
      tensor2
      generatorDevice2
      tensor3
      generatorDevice3,
    HasForward
      outputProjection
      tensor3
      generatorDevice3
      tensor4
      generatorDevice4,
    HasForward
      outputDropout
      tensor4
      generatorDevice4
      (Tensor queryGradient5 queryLayout5 queryDevice5 queryDataType5 queryShape5)
      generatorDevice5,
    HasForward
      outputLayerNorm
      (Tensor (queryGradient <|> queryGradient5) (queryLayout <+> queryLayout5) (queryDevice <+> queryDevice5) (queryDataType <+> queryDataType5) (BroadcastShapesF queryShape queryShape5))
      generatorDevice5
      output
      generatorOutputDevice,
    Catch (BroadcastShapesF queryShape queryShape5)
  ) =>
  HasForward
    (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm)
    (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformerFeedForwardNetwork {..} query =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward ffnInputLayerNorm
        >>>= IxStateT . forward ffnInputTransformation
        >>>= IxStateT . forward ffnActivation
        >>>= IxStateT . forward ffnActivationDropout
        >>>= IxStateT . forward ffnOutputProjection
        >>>= IxStateT . forward ffnOutputDropout
        >>>= ilift . (query `add`)
        >>>= IxStateT . forward ffnOutputLayerNorm

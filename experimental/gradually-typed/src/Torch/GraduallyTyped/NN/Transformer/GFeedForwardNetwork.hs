{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
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
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GFeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map as Map
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), GeluNew (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Linear (GLinear (..), LinearBiasF, LinearWeightF, linearSpec)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
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

type instance
  ModelSpec (GGate layer0 activation layer1) =
    GGate (ModelSpec layer0) (ModelSpec activation) (ModelSpec layer1)

instance
  ( HasInitialize layer0 generatorDevice layer0 generatorDevice,
    HasInitialize activation generatorDevice activation generatorDevice,
    HasInitialize layer1 generatorDevice layer1 generatorDevice
  ) =>
  HasInitialize
    (GGate layer0 activation layer1)
    generatorDevice
    (GGate layer0 activation layer1)
    generatorDevice
  where
  initialize (GGate layer0Spec activationSpec layer1Spec) =
    let layer0 = IxStateT . initialize $ layer0Spec
        activation = IxStateT . initialize $ activationSpec
        layer1 = IxStateT . initialize $ layer1Spec
     in runIxStateT $ GGate <<$>> layer0 <<*>> activation <<*>> layer1

instance
  ( HasStateDict layer0,
    HasStateDict activation,
    HasStateDict layer1
  ) =>
  HasStateDict (GGate layer0 activation layer1)
  where
  fromStateDict (GGate layer0Spec activationSpec layer1Spec) k =
    GGate
      <$> fromStateDict layer0Spec k
      <*> fromStateDict activationSpec k
      <*> fromStateDict layer1Spec k
  toStateDict k GGate {..} = do
    () <- toStateDict k gateLayer0
    () <- toStateDict k gateActivation
    () <- toStateDict k gateLayer1
    pure ()

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

type instance
  ModelSpec (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm) =
    GTransformerFeedForwardNetwork (ModelSpec inputLayerNorm) (ModelSpec inputTransformation) (ModelSpec activation) (ModelSpec activationDropout) (ModelSpec outputProjection) (ModelSpec outputDropout) (ModelSpec outputLayerNorm)

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
  FFNInputLayerNormF _ gradient device dataType queryEmbedDim =
    ()

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
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType queryEmbedDim ffnDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType ffnDim))
      )
  FFNInputTransformationF 'ByT5 gradient device dataType queryEmbedDim ffnDim =
    GGate
      ( NamedModel
          ( GLinear
              (NamedModel (LinearWeightF gradient device dataType queryEmbedDim ffnDim))
              (NamedModel (LinearBiasF 'WithoutBias gradient device dataType ffnDim))
          )
      )
      GeluNew
      ( NamedModel
          ( GLinear
              (NamedModel (LinearWeightF gradient device dataType queryEmbedDim ffnDim))
              (NamedModel (LinearBiasF 'WithoutBias gradient device dataType ffnDim))
          )
      )
  FFNInputTransformationF _ gradient device dataType queryEmbedDim ffnDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType queryEmbedDim ffnDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType ffnDim))
      )

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
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType ffnDim queryEmbedDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType queryEmbedDim))
      )
  FFNOutputProjectionF 'ByT5 gradient device dataType queryEmbedDim ffnDim =
    FFNOutputProjectionF 'T5 gradient device dataType queryEmbedDim ffnDim
  FFNOutputProjectionF _ gradient device dataType queryEmbedDim ffnDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType ffnDim queryEmbedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType queryEmbedDim))
      )

-- | Specifies the dropout for the output.
type family
  FFNOutputDropoutF
    (style :: TransformerStyle) ::
    Type
  where
  FFNOutputDropoutF _ = Dropout

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
  FFNOutputLayerNormF _ gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))

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
  forall style gradient device dataType queryEmbedDim ffnDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim queryEmbedDim ->
  SDim ffnDim ->
  Double ->
  Double ->
  ModelSpec
    ( GTransformerFeedForwardNetwork
        (FFNInputLayerNormF style gradient device dataType queryEmbedDim)
        (FFNInputTransformationF style gradient device dataType queryEmbedDim ffnDim)
        (FFNActivationF style)
        (FFNActivationDropoutF style)
        (FFNOutputProjectionF style gradient device dataType queryEmbedDim ffnDim)
        (FFNOutputDropoutF style)
        (FFNOutputLayerNormF style gradient device dataType queryEmbedDim)
    )
transformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps =
  let inputLayerNormSpec ST5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      inputLayerNormSpec SByT5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      inputLayerNormSpec SBART = ()
      inputLayerNormSpec SMBART = ()
      inputLayerNormSpec SPegasus = ()
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
      activationDropoutSpec ST5 = Dropout dropoutP
      activationDropoutSpec SByT5 = Dropout dropoutP
      activationDropoutSpec SBART = Dropout dropoutP
      activationDropoutSpec SMBART = Dropout dropoutP
      activationDropoutSpec SPegasus = Dropout dropoutP
      activationDropoutSpec SBERT = ()
      activationDropoutSpec SRoBERTa = ()
      activationDropoutSpec SGPT2 = undefined
      outputProjectionSpec ST5 = NamedModel "DenseReluDense.wo." $ weightSpecWithoutBias ffnDim queryEmbedDim
      outputProjectionSpec SByT5 = NamedModel "DenseReluDense.wo." $ weightSpecWithoutBias ffnDim queryEmbedDim
      outputProjectionSpec SBART = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SMBART = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SPegasus = NamedModel "fc2." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SBERT = NamedModel "output.dense." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SRoBERTa = NamedModel "output.dense." $ weightSpecWithBias ffnDim queryEmbedDim
      outputProjectionSpec SGPT2 = undefined
      outputDropoutSpec _ = Dropout dropoutP
      outputLayerNormSpec ST5 = ()
      outputLayerNormSpec SByT5 = ()
      outputLayerNormSpec SBART = NamedModel "final_layer_norm." layerNormWithBiasSpec
      outputLayerNormSpec SMBART = NamedModel "final_layer_norm." layerNormWithBiasSpec
      outputLayerNormSpec SPegasus = NamedModel "final_layer_norm." layerNormWithBiasSpec
      outputLayerNormSpec SBERT = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      outputLayerNormSpec SRoBERTa = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      outputLayerNormSpec SGPT2 = undefined
   in GTransformerFeedForwardNetwork
        (inputLayerNormSpec style)
        (inputTransformationSpec style)
        (activationSpec style)
        (activationDropoutSpec style)
        (outputProjectionSpec style)
        (outputDropoutSpec style)
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
  ( HasInitialize inputLayerNorm generatorDevice inputLayerNorm generatorDevice,
    HasInitialize inputTransformation generatorDevice inputTransformation generatorDevice,
    HasInitialize activation generatorDevice activation generatorDevice,
    HasInitialize activationDropout generatorDevice activationDropout generatorDevice,
    HasInitialize outputProjection generatorDevice outputProjection generatorDevice,
    HasInitialize outputDropout generatorDevice outputDropout generatorDevice,
    HasInitialize outputLayerNorm generatorDevice outputLayerNorm generatorDevice
  ) =>
  HasInitialize
    (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm)
    generatorDevice
    (GTransformerFeedForwardNetwork inputLayerNorm inputTransformation activation activationDropout outputProjection outputDropout outputLayerNorm)
    generatorDevice
  where
  initialize (GTransformerFeedForwardNetwork inputLayerNormSpec inputTransformationSpec activationSpec activationDropoutSpec outputProjectionSpec outputDropoutSpec outputLayerNormSpec) =
    let inputLayerNorm = IxStateT . initialize $ inputLayerNormSpec
        inputTransformation = IxStateT . initialize $ inputTransformationSpec
        activation = IxStateT . initialize $ activationSpec
        activationDropout = IxStateT . initialize $ activationDropoutSpec
        outputProjection = IxStateT . initialize $ outputProjectionSpec
        outputDropout = IxStateT . initialize $ outputDropoutSpec
        outputLayerNorm = IxStateT . initialize $ outputLayerNormSpec
     in runIxStateT $
          GTransformerFeedForwardNetwork
            <<$>> inputLayerNorm
            <<*>> inputTransformation
            <<*>> activation
            <<*>> activationDropout
            <<*>> outputProjection
            <<*>> outputDropout
            <<*>> outputLayerNorm

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
  where
  fromStateDict (GTransformerFeedForwardNetwork inputLayerNormSpec inputTransformationSpec activationSpec activationDropoutSpec outputProjectionSpec outputDropoutSpec outputLayerNormSpec) k =
    GTransformerFeedForwardNetwork
      <$> fromStateDict inputLayerNormSpec k
      <*> fromStateDict inputTransformationSpec k
      <*> fromStateDict activationSpec k
      <*> fromStateDict activationDropoutSpec k
      <*> fromStateDict outputProjectionSpec k
      <*> fromStateDict outputDropoutSpec k
      <*> fromStateDict outputLayerNormSpec k
  toStateDict k GTransformerFeedForwardNetwork {..} = do
    () <- toStateDict k ffnInputLayerNorm
    () <- toStateDict k ffnInputTransformation
    () <- toStateDict k ffnActivation
    () <- toStateDict k ffnActivationDropout
    () <- toStateDict k ffnOutputProjection
    () <- toStateDict k ffnOutputDropout
    () <- toStateDict k ffnOutputLayerNorm
    pure ()

-- | The shape of the output tensor of the transformer feed forward network.
-- type family
--   FeedForwardNetworkOutputShape
--     (style :: TransformerStyle)
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (queryShape :: Shape [Dim (Name Symbol) (Size Nat)]) ::
--     Shape [Dim (Name Symbol) (Size Nat)]
--   where
--   FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape =
--     BroadcastShapesF
--       queryShape
--       ( LinearWithoutBiasF
--           ('Shape '[queryEmbedDim, ffnDim])
--           ( LinearWithoutBiasF
--               ('Shape '[ffnDim, queryEmbedDim])
--               ( LayerNormWithoutBiasF
--                   ('Shape '[queryEmbedDim])
--                   queryShape
--               )
--           )
--       )
--   FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape = FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape
--   FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape =
--     BroadcastShapesF
--       queryShape
--       ( LinearWithBiasF
--           ('Shape '[queryEmbedDim, ffnDim])
--           ('Shape '[queryEmbedDim])
--           ( LinearWithBiasF
--               ('Shape '[ffnDim, queryEmbedDim])
--               ('Shape '[ffnDim])
--               ( LayerNormWithBiasF
--                   ('Shape '[queryEmbedDim])
--                   ('Shape '[queryEmbedDim])
--                   queryShape
--               )
--           )
--       )
--   FeedForwardNetworkOutputShape _ queryEmbedDim ffnDim queryShape =
--     LayerNormWithBiasF
--       ('Shape '[queryEmbedDim])
--       ('Shape '[queryEmbedDim])
--       ( BroadcastShapesF
--           queryShape
--           ( LinearWithBiasF
--               ('Shape '[queryEmbedDim, ffnDim])
--               ('Shape '[queryEmbedDim])
--               ( LinearWithBiasF
--                   ('Shape '[ffnDim, queryEmbedDim])
--                   ('Shape '[ffnDim])
--                   queryShape
--               )
--           )
--       )

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
      generatorOutputDevice
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
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward ffnOutputLayerNorm

testFFN :: IO _
testFFN = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      ffnDim = SName @"*" :&: SSize @2
      queryEmbedDim = SName @"*" :&: SSize @3
      dropoutP = 0
      eps = 1e-6
  let g = sMkGenerator device 0
      spec = NamedModel "ffn." $ transformerFeedForwardNetworkSpec SByT5 gradient device dataType queryEmbedDim ffnDim dropoutP eps
  (ffn, g') <- initialize spec g
  ffn' <- flip evalStateT Map.empty $ do
    toStateDict mempty ffn
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @2
      seqDim = SName @"*" :&: SSize @1
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  (output, _) <- forward ffn' query g'
  pure output

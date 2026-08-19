{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module Torch.Typed.NN.Recurrent.GRU where

import Data.Kind
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import System.Environment
import System.IO.Unsafe
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.NN as A
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (sqrt)
import Torch.Typed.NN.Dropout
import Torch.Typed.NN.Recurrent.Auxiliary
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (tanh)

data
  GRULayerSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = GRULayerSpec
  deriving (Show, Eq)

data
  GRULayer
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GRUUnidirectionalLayer ::
    Parameter device dtype (GRUWIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUWHShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBHShape hiddenSize inputSize) ->
    GRULayer inputSize hiddenSize 'Unidirectional dtype device
  GRUBidirectionalLayer ::
    Parameter device dtype (GRUWIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUWHShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBHShape hiddenSize inputSize) ->
    Parameter device dtype (GRUWIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUWHShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBIShape hiddenSize inputSize) ->
    Parameter device dtype (GRUBHShape hiddenSize inputSize) ->
    GRULayer inputSize hiddenSize 'Bidirectional dtype device

deriving instance Show (GRULayer inputSize hiddenSize directionality dtype device)

instance Parameterized (GRULayer inputSize hiddenSize 'Unidirectional dtype device) where
  type
    Parameters (GRULayer inputSize hiddenSize 'Unidirectional dtype device) =
      '[ Parameter device dtype (GRUWIShape hiddenSize inputSize),
         Parameter device dtype (GRUWHShape hiddenSize inputSize),
         Parameter device dtype (GRUBIShape hiddenSize inputSize),
         Parameter device dtype (GRUBHShape hiddenSize inputSize)
       ]
  flattenParameters (GRUUnidirectionalLayer wi wh bi bh) =
    wi :. wh :. bi :. bh :. HNil
  replaceParameters _ (wi :. wh :. bi :. bh :. HNil) =
    GRUUnidirectionalLayer wi wh bi bh

instance Parameterized (GRULayer inputSize hiddenSize 'Bidirectional dtype device) where
  type
    Parameters (GRULayer inputSize hiddenSize 'Bidirectional dtype device) =
      '[ Parameter device dtype (GRUWIShape hiddenSize inputSize),
         Parameter device dtype (GRUWHShape hiddenSize inputSize),
         Parameter device dtype (GRUBIShape hiddenSize inputSize),
         Parameter device dtype (GRUBHShape hiddenSize inputSize),
         Parameter device dtype (GRUWIShape hiddenSize inputSize),
         Parameter device dtype (GRUWHShape hiddenSize inputSize),
         Parameter device dtype (GRUBIShape hiddenSize inputSize),
         Parameter device dtype (GRUBHShape hiddenSize inputSize)
       ]
  flattenParameters (GRUBidirectionalLayer wi wh bi bh wi' wh' bi' bh') =
    wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil
  replaceParameters _ (wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil) =
    GRUBidirectionalLayer wi wh bi bh wi' wh' bi' bh'

instance
  ( RandDTypeIsValid device dtype,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device
  ) =>
  A.Randomizable
    (GRULayerSpec inputSize hiddenSize 'Unidirectional dtype device)
    (GRULayer inputSize hiddenSize 'Unidirectional dtype device)
  where
  sample _ =
    GRUUnidirectionalLayer
      <$> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

instance
  ( RandDTypeIsValid device dtype,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device
  ) =>
  A.Randomizable
    (GRULayerSpec inputSize hiddenSize 'Bidirectional dtype device)
    (GRULayer inputSize hiddenSize 'Bidirectional dtype device)
  where
  sample _ =
    GRUBidirectionalLayer
      <$> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< xavierUniformGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

data
  GRULayerStackSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = GRULayerStackSpec
  deriving (Show, Eq)

-- Input-to-hidden, hidden-to-hidden, and bias parameters for a mulilayered
-- (and optionally) bidirectional GRU.
--
data
  GRULayerStack
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GRULayer1 ::
    GRULayer inputSize hiddenSize directionality dtype device ->
    GRULayerStack inputSize hiddenSize 1 directionality dtype device
  GRULayerK ::
    GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device ->
    GRULayerStack inputSize hiddenSize numLayers directionality dtype device ->
    GRULayerStack inputSize hiddenSize (numLayers + 1) directionality dtype device

deriving instance Show (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)

class GRULayerStackParameterized (flag :: Bool) inputSize hiddenSize numLayers directionality dtype device where
  type GRULayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device :: [Type]
  gruLayerStackFlattenParameters ::
    Proxy flag ->
    GRULayerStack inputSize hiddenSize numLayers directionality dtype device ->
    HList (GRULayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device)
  gruLayerStackReplaceParameters ::
    Proxy flag ->
    GRULayerStack inputSize hiddenSize numLayers directionality dtype device ->
    HList (GRULayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device) ->
    GRULayerStack inputSize hiddenSize numLayers directionality dtype device

instance
  Parameterized (GRULayer inputSize hiddenSize directionality dtype device) =>
  GRULayerStackParameterized 'False inputSize hiddenSize 1 directionality dtype device
  where
  type
    GRULayerStackParameters 'False inputSize hiddenSize 1 directionality dtype device =
      Parameters (GRULayer inputSize hiddenSize directionality dtype device)
  gruLayerStackFlattenParameters _ (GRULayer1 gruLayer) = flattenParameters gruLayer
  gruLayerStackReplaceParameters _ (GRULayer1 gruLayer) parameters = GRULayer1 $ replaceParameters gruLayer parameters

instance
  ( Parameterized
      ( GRULayer
          (hiddenSize * NumberOfDirections directionality)
          hiddenSize
          directionality
          dtype
          device
      ),
    Parameterized (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device),
    HAppendFD
      (Parameters (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device))
      (Parameters (GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device))
      (Parameters (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device) ++ Parameters (GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)),
    1 <= numLayers,
    numLayersM1 ~ numLayers - 1,
    0 <= numLayersM1
  ) =>
  GRULayerStackParameterized 'True inputSize hiddenSize numLayers directionality dtype device
  where
  type
    GRULayerStackParameters 'True inputSize hiddenSize numLayers directionality dtype device =
      Parameters (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
        ++ Parameters (GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
  gruLayerStackFlattenParameters _ (GRULayerK gruLayer gruLayerStack) =
    let parameters = flattenParameters gruLayer
        parameters' = flattenParameters @(GRULayerStack inputSize hiddenSize numLayersM1 directionality dtype device) gruLayerStack
     in parameters' `happendFD` parameters
  gruLayerStackReplaceParameters _ (GRULayerK gruLayer gruLayerStack) parameters'' =
    let (parameters', parameters) = hunappendFD parameters''
        gruLayer' = replaceParameters gruLayer parameters
        gruLayerStack' = replaceParameters @(GRULayerStack inputSize hiddenSize numLayersM1 directionality dtype device) gruLayerStack parameters'
     in GRULayerK gruLayer' gruLayerStack'

instance
  ( 1 <= numLayers,
    (2 <=? numLayers) ~ flag,
    GRULayerStackParameterized flag inputSize hiddenSize numLayers directionality dtype device
  ) =>
  Parameterized (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)
  where
  type
    Parameters (GRULayerStack inputSize hiddenSize numLayers directionality dtype device) =
      GRULayerStackParameters (2 <=? numLayers) inputSize hiddenSize numLayers directionality dtype device
  flattenParameters = gruLayerStackFlattenParameters (Proxy :: Proxy flag)
  replaceParameters = gruLayerStackReplaceParameters (Proxy :: Proxy flag)

class GRULayerStackRandomizable (flag :: Bool) inputSize hiddenSize numLayers directionality dtype device where
  gruLayerStackSample ::
    Proxy flag ->
    GRULayerStackSpec inputSize hiddenSize numLayers directionality dtype device ->
    IO (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)

instance
  ( A.Randomizable
      (GRULayerSpec inputSize hiddenSize directionality dtype device)
      (GRULayer inputSize hiddenSize directionality dtype device)
  ) =>
  GRULayerStackRandomizable 'False inputSize hiddenSize 1 directionality dtype device
  where
  gruLayerStackSample _ _ = GRULayer1 <$> (sample $ GRULayerSpec @inputSize @hiddenSize @directionality @dtype @device)

instance
  ( 1 <= numLayers,
    A.Randomizable
      (GRULayerSpec (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
      (GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device),
    A.Randomizable
      (GRULayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
      (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
  ) =>
  GRULayerStackRandomizable 'True inputSize hiddenSize numLayers directionality dtype device
  where
  gruLayerStackSample _ _ =
    GRULayerK
      <$> (sample $ GRULayerSpec @(hiddenSize * NumberOfDirections directionality) @hiddenSize @directionality @dtype @device)
      <*> ( sample
              @(GRULayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
              @(GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
              $ GRULayerStackSpec
          )

instance
  ( 1 <= numLayers,
    (2 <=? numLayers) ~ flag,
    RandDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device,
    GRULayerStackRandomizable flag inputSize hiddenSize numLayers directionality dtype device
  ) =>
  Randomizable
    (GRULayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
    (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)
  where
  sample = gruLayerStackSample (Proxy :: Proxy flag)

newtype
  GRUSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = GRUSpec DropoutSpec
  deriving (Show, Generic)

data
  GRU
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GRU ::
    (1 <= numLayers) =>
    { gru_layer_stack :: GRULayerStack inputSize hiddenSize numLayers directionality dtype device,
      gru_dropout :: Dropout
    } ->
    GRU inputSize hiddenSize numLayers directionality dtype device

deriving instance Show (GRU inputSize hiddenSize numLayers directionality dtype device)

instance
  (1 <= numLayers) =>
  Generic (GRU inputSize hiddenSize numLayers directionality dtype device)
  where
  type
    Rep (GRU inputSize hiddenSize numLayers directionality dtype device) =
      Rec0 (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 Dropout
  from (GRU {..}) = K1 gru_layer_stack :*: K1 gru_dropout
  to (K1 layerStack :*: K1 dropout) = GRU layerStack dropout

instance
  ( 1 <= numLayers,
    Parameterized (GRULayerStack inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (GRULayerStack inputSize hiddenSize numLayers directionality dtype device))
      (Parameters Dropout)
      ( Parameters (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)
          ++ Parameters Dropout
      )
  ) =>
  Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device)

-- TODO: when we have cannonical initializers do this correctly:
-- https://github.com/pytorch/pytorch/issues/9221
-- https://discuss.pytorch.org/t/initializing-rnn-gru-and-gru-correctly/23605

-- | Helper to do xavier uniform initializations on weight matrices and
-- orthagonal initializations for the gates. (When implemented.)
xavierUniformGRU ::
  forall device dtype hiddenSize featureSize.
  ( KnownDType dtype,
    KnownNat hiddenSize,
    KnownNat featureSize,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  IO (Tensor device dtype '[3 * hiddenSize, featureSize])
xavierUniformGRU = do
  init <- randn :: IO (Tensor device dtype '[3 * hiddenSize, featureSize])
  UnsafeMkTensor
    <$> xavierUniformFIXME
      (toDynamic init)
      (5.0 / 3)
      (shape @device @dtype @'[3 * hiddenSize, featureSize] init)

instance
  ( KnownDType dtype,
    KnownDevice device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat (NumberOfDirections directionality),
    RandDTypeIsValid device dtype,
    A.Randomizable
      (GRULayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
      (GRULayerStack inputSize hiddenSize numLayers directionality dtype device),
    1 <= numLayers
  ) =>
  A.Randomizable
    (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
    (GRU inputSize hiddenSize numLayers directionality dtype device)
  where
  sample (GRUSpec dropoutSpec) =
    GRU
      <$> A.sample (GRULayerStackSpec @inputSize @hiddenSize @numLayers @directionality @dtype @device)
      <*> A.sample dropoutSpec

-- | A specification for a long, short-term memory layer.
data
  GRUWithInitSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (initialization :: RNNInitialization)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases and cell states.
  GRUWithZerosInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    GRUSpec inputSize hiddenSize numLayers directionality dtype device ->
    GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and user-provided cell states.
  GRUWithConstInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    GRUSpec inputSize hiddenSize numLayers directionality dtype device ->
    -- | The initial values of the hidden state
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and learned cell states.
  GRUWithLearnedInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    GRUSpec inputSize hiddenSize numLayers directionality dtype device ->
    -- | The initial (learnable)
    -- values of the hidden state
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    GRUWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (GRUWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)

-- | A long, short-term memory layer with either fixed initial
-- states for the memory cells and hidden state or learnable
-- inital states for the memory cells and hidden state.
data
  GRUWithInit
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (initialization :: RNNInitialization)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GRUWithConstInit ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    { gruWithConstInit_gru :: GRU inputSize hiddenSize numLayers directionality dtype device,
      gruWithConstInit_h :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
    } ->
    GRUWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      'ConstantInitialization
      dtype
      device
  GRUWithLearnedInit ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    { gruWithLearnedInit_gru :: GRU inputSize hiddenSize numLayers directionality dtype device,
      gruWithLearnedInit_h :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
    } ->
    GRUWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      'LearnedInitialization
      dtype
      device

deriving instance Show (GRUWithInit inputSize hiddenSize numLayers directionality initialization dtype device)

instance Generic (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  type
    Rep (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) =
      Rec0 (GRU inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
  from (GRUWithConstInit {..}) = K1 gruWithConstInit_gru :*: K1 gruWithConstInit_h
  to (K1 gru :*: K1 h) = GRUWithConstInit gru h

instance Generic (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  type
    Rep (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) =
      Rec0 (GRU inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
  from (GRUWithLearnedInit {..}) = K1 gruWithLearnedInit_gru :*: K1 gruWithLearnedInit_h
  to (K1 gru :*: K1 h) = GRUWithLearnedInit gru h

instance
  ( Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (GRU inputSize hiddenSize numLayers directionality dtype device))
      '[]
      ( Parameters (GRU inputSize hiddenSize numLayers directionality dtype device) ++ '[]
      )
  ) =>
  Parameterized (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)

instance
  ( Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (GRU inputSize hiddenSize numLayers directionality dtype device))
      '[Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]]
      ( Parameters (GRU inputSize hiddenSize numLayers directionality dtype device)
          ++ '[Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]]
      )
  ) =>
  Parameterized (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)

instance
  ( KnownNat hiddenSize,
    KnownNat numLayers,
    KnownNat (NumberOfDirections directionality),
    KnownDType dtype,
    KnownDevice device,
    A.Randomizable
      (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
      (GRU inputSize hiddenSize numLayers directionality dtype device)
  ) =>
  A.Randomizable
    (GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
    (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
  where
  sample (GRUWithZerosInitSpec gruSpec) =
    GRUWithConstInit
      <$> A.sample gruSpec
      <*> pure zeros
  sample (GRUWithConstInitSpec gruSpec h) =
    GRUWithConstInit
      <$> A.sample gruSpec
      <*> pure h

instance
  ( KnownNat hiddenSize,
    KnownNat numLayers,
    KnownNat (NumberOfDirections directionality),
    KnownDType dtype,
    KnownDevice device,
    A.Randomizable
      (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
      (GRU inputSize hiddenSize numLayers directionality dtype device)
  ) =>
  A.Randomizable
    (GRUWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
    (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
  where
  sample s@(GRUWithLearnedInitSpec gruSpec h) =
    GRUWithLearnedInit
      <$> A.sample gruSpec
      <*> (makeIndependent =<< pure h)

gruForward ::
  forall
    shapeOrder
    batchSize
    seqLen
    directionality
    initialization
    numLayers
    inputSize
    outputSize
    hiddenSize
    inputShape
    outputShape
    hcShape
    parameters
    tensorParameters
    dtype
    device.
  ( KnownNat (NumberOfDirections directionality),
    KnownNat numLayers,
    KnownNat batchSize,
    KnownNat hiddenSize,
    KnownRNNShapeOrder shapeOrder,
    KnownRNNDirectionality directionality,
    outputSize ~ (hiddenSize * NumberOfDirections directionality),
    inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize,
    outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize,
    hcShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize],
    parameters ~ Parameters (GRU inputSize hiddenSize numLayers directionality dtype device),
    Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device),
    tensorParameters ~ GRUR inputSize hiddenSize numLayers directionality dtype device,
    ATen.Castable (HList tensorParameters) [D.ATenTensor],
    HMap' ToDependent parameters tensorParameters
  ) =>
  Bool ->
  GRUWithInit
    inputSize
    hiddenSize
    numLayers
    directionality
    initialization
    dtype
    device ->
  Tensor device dtype inputShape ->
  ( Tensor device dtype outputShape,
    Tensor device dtype hcShape
  )
gruForward dropoutOn (GRUWithConstInit gruModel@(GRU _ (Dropout dropoutProb)) hc) input =
  gru
    @shapeOrder
    @directionality
    @numLayers
    @seqLen
    @batchSize
    @inputSize
    @outputSize
    @hiddenSize
    @inputShape
    @outputShape
    @hcShape
    @tensorParameters
    @dtype
    @device
    (hmap' ToDependent . flattenParameters $ gruModel)
    dropoutProb
    dropoutOn
    hc'
    input
  where
    hc' =
      reshape @hcShape
        . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
        $ hc
gruForward dropoutOn (GRUWithLearnedInit gruModel@(GRU _ (Dropout dropoutProb)) hc) input =
  gru
    @shapeOrder
    @directionality
    @numLayers
    @seqLen
    @batchSize
    @inputSize
    @outputSize
    @hiddenSize
    @inputShape
    @outputShape
    @hcShape
    @tensorParameters
    @dtype
    @device
    (hmap' ToDependent . flattenParameters $ gruModel)
    dropoutProb
    dropoutOn
    hc'
    input
  where
    hc' =
      reshape @hcShape
        . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
        . toDependent
        $ hc

gruForwardWithDropout,
  gruForwardWithoutDropout ::
    forall
      shapeOrder
      batchSize
      seqLen
      directionality
      initialization
      numLayers
      inputSize
      outputSize
      hiddenSize
      inputShape
      outputShape
      hcShape
      parameters
      tensorParameters
      dtype
      device.
    ( KnownNat (NumberOfDirections directionality),
      KnownNat numLayers,
      KnownNat batchSize,
      KnownNat hiddenSize,
      KnownRNNShapeOrder shapeOrder,
      KnownRNNDirectionality directionality,
      outputSize ~ (hiddenSize * NumberOfDirections directionality),
      inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize,
      outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize,
      hcShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize],
      parameters ~ Parameters (GRU inputSize hiddenSize numLayers directionality dtype device),
      Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device),
      tensorParameters ~ GRUR inputSize hiddenSize numLayers directionality dtype device,
      ATen.Castable (HList tensorParameters) [D.ATenTensor],
      HMap' ToDependent parameters tensorParameters
    ) =>
    GRUWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      initialization
      dtype
      device ->
    Tensor device dtype inputShape ->
    ( Tensor device dtype outputShape,
      Tensor device dtype hcShape
    )
-- ^ Forward propagate the `GRU` module and apply dropout on the outputs of each layer.
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = GRUWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (GRUSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t gruForwardWithDropout @'BatchFirst model input
-- gruForwardWithDropout @'BatchFirst model input
--   :: (Tensor '(D.CPU, 0) 'D.Float [5, 16, 60],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30])
-- >>> (a,b) = gruForwardWithDropout @'BatchFirst model input
-- >>> ((dtype a, shape a), (dtype b, shape b))
-- ((Float,[5,16,60]),(Float,[6,5,30]))
gruForwardWithDropout =
  gruForward
    @shapeOrder
    @batchSize
    @seqLen
    @directionality
    @initialization
    @numLayers
    @inputSize
    @outputSize
    @hiddenSize
    @inputShape
    @outputShape
    @hcShape
    @parameters
    @tensorParameters
    @dtype
    @device
    True
-- ^ Forward propagate the `GRU` module (without applying dropout on the outputs of each layer).
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = GRUWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (GRUSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t gruForwardWithoutDropout @'BatchFirst model input
-- gruForwardWithoutDropout @'BatchFirst model input
--   :: (Tensor '(D.CPU, 0) 'D.Float [5, 16, 60],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30])
-- >>> (a,b) = gruForwardWithoutDropout @'BatchFirst model input
-- >>> ((dtype a, shape a), (dtype b, shape b))
-- ((Float,[5,16,60]),(Float,[6,5,30]))
gruForwardWithoutDropout =
  gruForward
    @shapeOrder
    @batchSize
    @seqLen
    @directionality
    @initialization
    @numLayers
    @inputSize
    @outputSize
    @hiddenSize
    @inputShape
    @outputShape
    @hcShape
    @parameters
    @tensorParameters
    @dtype
    @device
    False

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}

module Torch.Typed.NN.Recurrent.GRU where

import           Prelude                 hiding ( tanh )
import           Data.Kind
import           Torch.HList
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Autograd                as A
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional              as D
import qualified Torch.NN                      as A
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import           Torch.Typed
import           Torch.Typed.Factories
import           Torch.Typed.Functional      hiding ( sqrt )
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.NN

data GRULayerSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 = GRULayerSpec deriving (Show, Eq)

data GRULayer
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  GRUUnidirectionalLayer
    :: Parameter device dtype (GRUWIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUWHShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBHShape hiddenSize inputSize)
    -> GRULayer inputSize hiddenSize 'Unidirectional dtype device
  GRUBidirectionalLayer
    :: Parameter device dtype (GRUWIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUWHShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBHShape hiddenSize inputSize)
    -> Parameter device dtype (GRUWIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUWHShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBIShape hiddenSize inputSize)
    -> Parameter device dtype (GRUBHShape hiddenSize inputSize)
    -> GRULayer inputSize hiddenSize 'Bidirectional dtype device

deriving instance Show (GRULayer inputSize hiddenSize directionality dtype device)
-- deriving instance Generic (GRULayer inputSize hiddenSize directionality dtype device)

instance
  ( wiShape ~ (GRUWIShape hiddenSize inputSize)
  , whShape ~ (GRUWHShape hiddenSize inputSize)
  , biShape ~ (GRUBIShape hiddenSize inputSize)
  , bhShape ~ (GRUBHShape hiddenSize inputSize)
  , parameters ~ '[Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape]
  ) => GParameterized (K1 R (GRULayer inputSize hiddenSize 'Unidirectional dtype device)) parameters where
  gFlattenParameters (K1 (GRUUnidirectionalLayer wi wh bi bh)) =
    wi :. wh :. bi :. bh :. HNil
  gReplaceParameters _ (wi :. wh :. bi :. bh :. HNil) =
    K1 (GRUUnidirectionalLayer wi wh bi bh)

instance
  ( wiShape ~ (GRUWIShape hiddenSize inputSize)
  , whShape ~ (GRUWHShape hiddenSize inputSize)
  , biShape ~ (GRUBIShape hiddenSize inputSize)
  , bhShape ~ (GRUBHShape hiddenSize inputSize)
  , parameters ~ '[Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape, Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape]
  ) => GParameterized (K1 R (GRULayer inputSize hiddenSize 'Bidirectional dtype device)) parameters where
  gFlattenParameters (K1 (GRUBidirectionalLayer wi wh bi bh wi' wh' bi' bh'))
    = wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil
  gReplaceParameters _ (wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil)
    = K1 (GRUBidirectionalLayer wi wh bi bh wi' wh' bi' bh')

instance
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (GRULayerSpec inputSize hiddenSize 'Unidirectional dtype device)
                      (GRULayer     inputSize hiddenSize 'Unidirectional dtype device)
 where
  sample _ =
    GRUUnidirectionalLayer
      <$> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

instance
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (GRULayerSpec inputSize hiddenSize 'Bidirectional dtype device)
                      (GRULayer     inputSize hiddenSize 'Bidirectional dtype device)
 where
  sample _ =
    GRUBidirectionalLayer
      <$> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< xavierUniormGRU)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

data GRULayerStackSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = GRULayerStackSpec deriving (Show, Eq)

-- Input-to-hidden, hidden-to-hidden, and bias parameters for a mulilayered
-- (and optionally) bidirectional GRU.
--
data GRULayerStack
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  GRULayer1
    :: GRULayer inputSize hiddenSize directionality dtype device
    -> GRULayerStack inputSize hiddenSize 1 directionality dtype device
  GRULayerK
    :: (2 <= numLayers)
    => GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device
    -> GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device
    -> GRULayerStack inputSize hiddenSize numLayers directionality dtype device

deriving instance Show (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)
--  TODO: Generics? see https://gist.github.com/RyanGlScott/71d9f933e823b4a03f99de54d4b94d51
-- deriving instance Generic (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)

instance {-# OVERLAPS #-}
  ( GParameterized (K1 R (GRULayer inputSize hiddenSize directionality dtype device)) parameters
  ) => GParameterized (K1 R (GRULayerStack inputSize hiddenSize 1 directionality dtype device)) parameters where
  gFlattenParameters (K1 (GRULayer1 gruLayer))
    = gFlattenParameters (K1 @R gruLayer)
  gReplaceParameters (K1 (GRULayer1 gruLayer)) parameters
    = K1 (GRULayer1 (unK1 (gReplaceParameters (K1 @R gruLayer) parameters)))

instance {-# OVERLAPPABLE #-}
  ( 2 <= numLayers
  , GParameterized (K1 R (GRULayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)) parameters
  , GParameterized (K1 R (GRULayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)) parameters'
  , HAppendFD parameters parameters' parameters''
  , parameters'' ~ (parameters ++ parameters')
  ) => GParameterized (K1 R (GRULayerStack inputSize hiddenSize numLayers directionality dtype device)) parameters'' where
  gFlattenParameters (K1 (GRULayerK gruLayerStack gruLayer))
    = let parameters  = gFlattenParameters (K1 @R gruLayerStack)
          parameters' = gFlattenParameters (K1 @R gruLayer)
      in  parameters `happendFD` parameters'
  gReplaceParameters (K1 (GRULayerK gruLayerStack gruLayer)) parameters''
    = let (parameters, parameters') = hunappendFD parameters''
          gruLayerStack'           = unK1 (gReplaceParameters (K1 @R gruLayerStack) parameters)
          gruLayer'                = unK1 (gReplaceParameters (K1 @R gruLayer)      parameters')
      in  K1 (GRULayerK gruLayerStack' gruLayer')

instance {-# OVERLAPS #-}
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (GRULayerSpec inputSize hiddenSize directionality dtype device)
                   (GRULayer     inputSize hiddenSize directionality dtype device)
  ) => A.Randomizable (GRULayerStackSpec inputSize hiddenSize 1 directionality dtype device)
                      (GRULayerStack     inputSize hiddenSize 1 directionality dtype device)
 where
  sample _ = GRULayer1 <$> (A.sample $ GRULayerSpec @inputSize @hiddenSize @directionality @dtype @device)

instance {-# OVERLAPPABLE #-}
  ( 2 <= numLayers
  , RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (GRULayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
                   (GRULayerStack     inputSize hiddenSize (numLayers - 1) directionality dtype device)
  , A.Randomizable (GRULayerSpec (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
                   (GRULayer     (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
  ) => A.Randomizable (GRULayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
                      (GRULayerStack     inputSize hiddenSize numLayers directionality dtype device)
 where
  sample _ =
    GRULayerK
      <$> (A.sample $ GRULayerStackSpec @inputSize @hiddenSize @(numLayers - 1) @directionality @dtype @device)
      <*> (A.sample $ GRULayerSpec @(hiddenSize * NumberOfDirections directionality) @hiddenSize @directionality @dtype @device)

newtype GRUSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = GRUSpec DropoutSpec
  deriving (Show, Generic)

data GRU
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = GRU
      { gru_layer_stack :: GRULayerStack inputSize hiddenSize numLayers directionality dtype device
      , gru_dropout     :: Dropout
      }
  deriving (Show, Generic)

-- TODO: when we have cannonical initializers do this correctly:
-- https://github.com/pytorch/pytorch/issues/9221
-- https://discuss.pytorch.org/t/initializing-rnn-gru-and-gru-correctly/23605

-- | Helper to do xavier uniform initializations on weight matrices and
-- orthagonal initializations for the gates. (When implemented.)
--
xavierUniormGRU
  :: forall device dtype hiddenSize featureSize
   . ( KnownDType dtype
     , KnownNat hiddenSize
     , KnownNat featureSize
     , KnownDevice device
     , RandDTypeIsValid device dtype
     )
  => IO (Tensor device dtype '[3 * hiddenSize, featureSize])
xavierUniormGRU = do
  init <- randn :: IO (Tensor device dtype '[3 * hiddenSize, featureSize])
  UnsafeMkTensor <$> xavierUniformFIXME
    (toDynamic init)
    (5.0 / 3)
    (shape @device @dtype @'[3 * hiddenSize, featureSize] init)

-- TODO: This is taken from the initializers example code and should be replaced with cannonical,
-- tested versions. However, even a potentially incorrect implementation will likely perform
-- better than an ad-hoc random-normal distribution.
-- | Fan-in / Fan-out scaling calculation
calculateFan :: [Int] -> (Int, Int)
calculateFan shape
  | dimT < 2
  = error
    "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
  | dimT == 2
  = (numInputFmaps, numOutputFmaps)
  | otherwise
  = (numInputFmaps * receptiveFieldSize, numOutputFmaps * receptiveFieldSize)
 where
  dimT               = length shape
  numInputFmaps      = shape !! 1
  numOutputFmaps     = shape !! 0
  receptiveFieldSize = product $ tail shape

-- | Xavier Initialization - Uniform
xavierUniformFIXME :: D.Tensor -> Float -> [Int] -> IO D.Tensor
xavierUniformFIXME init gain shape = pure
  $ D.subScalar (D.mulScalar init (bound * 2.0)) bound
 where
  (fanIn, fanOut) = calculateFan shape
  std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
  bound = sqrt 3.0 * std

instance
  ( KnownDType dtype
  , KnownDevice device
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownNat (NumberOfDirections directionality)
  , RandDTypeIsValid device dtype
  , A.Randomizable (GRULayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
                   (GRULayerStack     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
                      (GRU     inputSize hiddenSize numLayers directionality dtype device) where
  sample (GRUSpec dropoutSpec) =
    GRU
      <$> A.sample (GRULayerStackSpec @inputSize @hiddenSize @numLayers @directionality @dtype @device)
      <*> A.sample dropoutSpec

data RNNInitialization = ConstantInitialization | LearnedInitialization deriving (Show, Generic)

-- | A specification for a long, short-term memory layer.
--
data GRUWithInitSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers:: Nat)
  (directionality :: RNNDirectionality)
  (initialization :: RNNInitialization)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases and cell states.
  GRUWithZerosInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . GRUSpec inputSize hiddenSize numLayers directionality dtype device
    -> GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and user-provided cell states.
  GRUWithConstInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . GRUSpec inputSize hiddenSize numLayers directionality dtype device
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial values of the hidden state
    -> GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and learned cell states.
  GRUWithLearnedInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . GRUSpec inputSize hiddenSize numLayers directionality dtype device
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial (learnable)
                                                                                        -- values of the hidden state
    -> GRUWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (GRUWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)
-- deriving instance Generic (GRUWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)

-- | A long, short-term memory layer with either fixed initial
-- states for the memory cells and hidden state or learnable
-- inital states for the memory cells and hidden state.
--
data GRUWithInit
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (initialization :: RNNInitialization)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  GRUWithConstInit
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . { gruWithConstInit_gru :: GRU inputSize hiddenSize numLayers directionality dtype device
       , gruWithConstInit_h    :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       }
    -> GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  GRUWithLearnedInit
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . { gruWithLearnedInit_gru :: GRU inputSize hiddenSize numLayers directionality dtype device
       , gruWithLearnedInit_h    :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       }
    -> GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (GRUWithInit inputSize hiddenSize numLayers directionality initialization dtype device)
-- TODO: https://ryanglscott.github.io/2018/02/11/how-to-derive-generic-for-some-gadts/
-- deriving instance Generic (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)

instance Generic (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  type Rep (GRUWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) =
    Rec0 (GRU inputSize hiddenSize numLayers directionality dtype device)
      :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])

  from (GRUWithConstInit {..}) = K1 gruWithConstInit_gru :*: K1 gruWithConstInit_h
  to (K1 gru :*: K1 h) = GRUWithConstInit gru h

instance Generic (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  type Rep (GRUWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) =
    Rec0 (GRU inputSize hiddenSize numLayers directionality dtype device)
      :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])

  from (GRUWithLearnedInit {..}) = K1 gruWithLearnedInit_gru :*: K1 gruWithLearnedInit_h
  to (K1 gru :*: K1 h) = GRUWithLearnedInit gru h

instance
  ( KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
                   (GRU     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (GRUWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
                      (GRUWithInit     inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  sample (GRUWithZerosInitSpec gruSpec) =
    GRUWithConstInit
      <$> A.sample gruSpec
      <*> pure zeros
  sample (GRUWithConstInitSpec gruSpec h) =
    GRUWithConstInit
      <$> A.sample gruSpec
      <*> pure h

instance
  ( KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (GRUSpec inputSize hiddenSize numLayers directionality dtype device)
                   (GRU     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (GRUWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
                      (GRUWithInit     inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  sample s@(GRUWithLearnedInitSpec gruSpec h) =
    GRUWithLearnedInit
      <$> A.sample gruSpec
      <*> (makeIndependent =<< pure h)

gru
  :: forall
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
       device
   . ( KnownNat (NumberOfDirections directionality)
     , KnownNat numLayers
     , KnownNat batchSize
     , KnownNat hiddenSize
     , KnownRNNShapeOrder shapeOrder
     , KnownRNNDirectionality directionality
     , outputSize ~ (hiddenSize * NumberOfDirections directionality)
     , inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize
     , outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize
     , hcShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device) parameters
     , tensorParameters ~ GRUR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     , HMap' ToDependent parameters tensorParameters
     )
  => Bool
  -> GRUWithInit
       inputSize
       hiddenSize
       numLayers
       directionality
       initialization
       dtype
       device
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hcShape
     )
gru dropoutOn (GRUWithConstInit gru@(GRU _ (Dropout dropoutProb)) hc) input
  = Torch.Typed.Functional.gru
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
    (hmap' ToDependent . flattenParameters $ gru)
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
gru dropoutOn (GRUWithLearnedInit gru@(GRU _ (Dropout dropoutProb)) hc) input
  = Torch.Typed.Functional.gru
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
    (hmap' ToDependent . flattenParameters $ gru)
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

gruWithDropout, gruWithoutDropout
  :: forall
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
       device
   . ( KnownNat (NumberOfDirections directionality)
     , KnownNat numLayers
     , KnownNat batchSize
     , KnownNat hiddenSize
     , KnownRNNShapeOrder shapeOrder
     , KnownRNNDirectionality directionality
     , outputSize ~ (hiddenSize * NumberOfDirections directionality)
     , inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize
     , outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize
     , hcShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , Parameterized (GRU inputSize hiddenSize numLayers directionality dtype device) parameters
     , tensorParameters ~ GRUR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     , HMap' ToDependent parameters tensorParameters
     )
  => GRUWithInit
       inputSize
       hiddenSize
       numLayers
       directionality
       initialization
       dtype
       device
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hcShape
     )
-- ^ Forward propagate the `GRU` module and apply dropout on the outputs of each layer.
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = GRUWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (GRUSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t gruWithDropout @'BatchFirst model input
-- gruWithDropout @'BatchFirst model input
--   :: (Tensor '( 'D.CPU, 0) 'D.Float '[5, 16, 60],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30])
-- >>> gruWithDropout @'BatchFirst model input
-- (Tensor Float [5,16,60] ,Tensor Float [6,5,30] )
gruWithDropout =
  Torch.Typed.NN.Recurrent.GRU.gru
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
-- >>> :t gruWithoutDropout @'BatchFirst model input
-- gruWithoutDropout @'BatchFirst model input
--   :: (Tensor '( 'D.CPU, 0) 'D.Float '[5, 16, 60],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30])
-- >>> gruWithoutDropout @'BatchFirst model input
-- (Tensor Float [5,16,60] ,Tensor Float [6,5,30] )
gruWithoutDropout =
  Torch.Typed.NN.Recurrent.GRU.gru
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

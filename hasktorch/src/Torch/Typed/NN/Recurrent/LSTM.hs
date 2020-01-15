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

module Torch.Typed.NN.Recurrent.LSTM where

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

data LSTMLayerSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 = LSTMLayerSpec deriving (Show, Eq)

data LSTMLayer
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  LSTMUnidirectionalLayer
    :: Parameter device dtype (LSTMWIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMWHShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBHShape hiddenSize inputSize)
    -> LSTMLayer inputSize hiddenSize 'Unidirectional dtype device
  LSTMBidirectionalLayer
    :: Parameter device dtype (LSTMWIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMWHShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBHShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMWIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMWHShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBIShape hiddenSize inputSize)
    -> Parameter device dtype (LSTMBHShape hiddenSize inputSize)
    -> LSTMLayer inputSize hiddenSize 'Bidirectional dtype device

deriving instance Show (LSTMLayer inputSize hiddenSize directionality dtype device)
-- deriving instance Generic (LSTMLayer inputSize hiddenSize directionality dtype device)

instance
  ( wiShape ~ (LSTMWIShape hiddenSize inputSize)
  , whShape ~ (LSTMWHShape hiddenSize inputSize)
  , biShape ~ (LSTMBIShape hiddenSize inputSize)
  , bhShape ~ (LSTMBHShape hiddenSize inputSize)
  , parameters ~ '[Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape]
  ) => GParameterized (K1 R (LSTMLayer inputSize hiddenSize 'Unidirectional dtype device)) parameters where
  gFlattenParameters (K1 (LSTMUnidirectionalLayer wi wh bi bh)) =
    wi :. wh :. bi :. bh :. HNil
  gReplaceParameters _ (wi :. wh :. bi :. bh :. HNil) =
    K1 (LSTMUnidirectionalLayer wi wh bi bh)

instance
  ( wiShape ~ (LSTMWIShape hiddenSize inputSize)
  , whShape ~ (LSTMWHShape hiddenSize inputSize)
  , biShape ~ (LSTMBIShape hiddenSize inputSize)
  , bhShape ~ (LSTMBHShape hiddenSize inputSize)
  , parameters ~ '[Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape, Parameter device dtype wiShape, Parameter device dtype whShape, Parameter device dtype biShape, Parameter device dtype bhShape]
  ) => GParameterized (K1 R (LSTMLayer inputSize hiddenSize 'Bidirectional dtype device)) parameters where
  gFlattenParameters (K1 (LSTMBidirectionalLayer wi wh bi bh wi' wh' bi' bh'))
    = wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil
  gReplaceParameters _ (wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil)
    = K1 (LSTMBidirectionalLayer wi wh bi bh wi' wh' bi' bh')

instance
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (LSTMLayerSpec inputSize hiddenSize 'Unidirectional dtype device)
                      (LSTMLayer     inputSize hiddenSize 'Unidirectional dtype device)
 where
  sample _ =
    LSTMUnidirectionalLayer
      <$> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

instance
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (LSTMLayerSpec inputSize hiddenSize 'Bidirectional dtype device)
                      (LSTMLayer     inputSize hiddenSize 'Bidirectional dtype device)
 where
  sample _ =
    LSTMBidirectionalLayer
      <$> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< xavierUniormLSTM)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

data LSTMLayerStackSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = LSTMLayerStackSpec deriving (Show, Eq)

-- Input-to-hidden, hidden-to-hidden, and bias parameters for a mulilayered
-- (and optionally) bidirectional LSTM.
--
data LSTMLayerStack
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  LSTMLayer1
    :: LSTMLayer inputSize hiddenSize directionality dtype device
    -> LSTMLayerStack inputSize hiddenSize 1 directionality dtype device
  LSTMLayerK
    :: (2 <= numLayers)
    => LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device
    -> LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device
    -> LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device

deriving instance Show (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)
--  TODO: Generics? see https://gist.github.com/RyanGlScott/71d9f933e823b4a03f99de54d4b94d51
-- deriving instance Generic (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)

instance {-# OVERLAPS #-}
  ( GParameterized (K1 R (LSTMLayer inputSize hiddenSize directionality dtype device)) parameters
  ) => GParameterized (K1 R (LSTMLayerStack inputSize hiddenSize 1 directionality dtype device)) parameters where
  gFlattenParameters (K1 (LSTMLayer1 lstmLayer))
    = gFlattenParameters (K1 @R lstmLayer)
  gReplaceParameters (K1 (LSTMLayer1 lstmLayer)) parameters
    = K1 (LSTMLayer1 (unK1 (gReplaceParameters (K1 @R lstmLayer) parameters)))

instance {-# OVERLAPPABLE #-}
  ( 2 <= numLayers
  , GParameterized (K1 R (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)) parameters
  , GParameterized (K1 R (LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)) parameters'
  , HAppendFD parameters parameters' parameters''
  , parameters'' ~ (parameters ++ parameters')
  ) => GParameterized (K1 R (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)) parameters'' where
  gFlattenParameters (K1 (LSTMLayerK lstmLayerStack lstmLayer))
    = let parameters  = gFlattenParameters (K1 @R lstmLayerStack)
          parameters' = gFlattenParameters (K1 @R lstmLayer)
      in  parameters `happendFD` parameters'
  gReplaceParameters (K1 (LSTMLayerK lstmLayerStack lstmLayer)) parameters''
    = let (parameters, parameters') = hunappendFD parameters''
          lstmLayerStack'           = unK1 (gReplaceParameters (K1 @R lstmLayerStack) parameters)
          lstmLayer'                = unK1 (gReplaceParameters (K1 @R lstmLayer)      parameters')
      in  K1 (LSTMLayerK lstmLayerStack' lstmLayer')

instance {-# OVERLAPS #-}
  ( RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (LSTMLayerSpec inputSize hiddenSize directionality dtype device)
                   (LSTMLayer     inputSize hiddenSize directionality dtype device)
  ) => A.Randomizable (LSTMLayerStackSpec inputSize hiddenSize 1 directionality dtype device)
                      (LSTMLayerStack     inputSize hiddenSize 1 directionality dtype device)
 where
  sample _ = LSTMLayer1 <$> (A.sample $ LSTMLayerSpec @inputSize @hiddenSize @directionality @dtype @device)

instance {-# OVERLAPPABLE #-}
  ( 2 <= numLayers
  , RandDTypeIsValid device dtype
  , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (LSTMLayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
                   (LSTMLayerStack     inputSize hiddenSize (numLayers - 1) directionality dtype device)
  , A.Randomizable (LSTMLayerSpec (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
                   (LSTMLayer     (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
  ) => A.Randomizable (LSTMLayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
                      (LSTMLayerStack     inputSize hiddenSize numLayers directionality dtype device)
 where
  sample _ =
    LSTMLayerK
      <$> (A.sample $ LSTMLayerStackSpec @inputSize @hiddenSize @(numLayers - 1) @directionality @dtype @device)
      <*> (A.sample $ LSTMLayerSpec @(hiddenSize * NumberOfDirections directionality) @hiddenSize @directionality @dtype @device)

newtype LSTMSpec
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = LSTMSpec DropoutSpec
  deriving (Show, Generic)

data LSTM
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
  = LSTM
      { lstm_layer_stack :: LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device
      , lstm_dropout     :: Dropout
      }
  deriving (Show, Generic)

-- TODO: when we have cannonical initializers do this correctly:
-- https://github.com/pytorch/pytorch/issues/9221
-- https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605

-- | Helper to do xavier uniform initializations on weight matrices and
-- orthagonal initializations for the gates. (When implemented.)
--
xavierUniormLSTM
  :: forall device dtype hiddenSize featureSize
   . ( KnownDType dtype
     , KnownNat hiddenSize
     , KnownNat featureSize
     , KnownDevice device
     , RandDTypeIsValid device dtype
     )
  => IO (Tensor device dtype '[4 * hiddenSize, featureSize])
xavierUniormLSTM = do
  init <- randn :: IO (Tensor device dtype '[4 * hiddenSize, featureSize])
  UnsafeMkTensor <$> xavierUniformFIXME
    (toDynamic init)
    (5.0 / 3)
    (shape @device @dtype @'[4 * hiddenSize, featureSize] init)

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
  , A.Randomizable (LSTMLayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
                   (LSTMLayerStack     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
                      (LSTM     inputSize hiddenSize numLayers directionality dtype device) where
  sample (LSTMSpec dropoutSpec) =
    LSTM
      <$> A.sample (LSTMLayerStackSpec @inputSize @hiddenSize @numLayers @directionality @dtype @device)
      <*> A.sample dropoutSpec

data RNNInitialization = ConstantInitialization | LearnedInitialization deriving (Show, Generic)

-- | A specification for a long, short-term memory layer.
--
data LSTMWithInitSpec
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
  LSTMWithZerosInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . LSTMSpec inputSize hiddenSize numLayers directionality dtype device
    -> LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and user-provided cell states.
  LSTMWithConstInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . LSTMSpec inputSize hiddenSize numLayers directionality dtype device
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial values of the memory cell
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial values of the hidden state
    -> LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and learned cell states.
  LSTMWithLearnedInitSpec
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . LSTMSpec inputSize hiddenSize numLayers directionality dtype device
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial (learnable)
                                                                                        -- values of the memory cell
    -> Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] -- ^ The initial (learnable)
                                                                                        -- values of the hidden state
    -> LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (LSTMWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)
-- deriving instance Generic (LSTMWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)

-- | A long, short-term memory layer with either fixed initial
-- states for the memory cells and hidden state or learnable
-- inital states for the memory cells and hidden state.
--
data LSTMWithInit
  (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  (initialization :: RNNInitialization)
  (dtype :: D.DType)
  (device :: (D.DeviceType, Nat))
 where
  LSTMWithConstInit
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . { lstmWithConstInit_lstm :: LSTM inputSize hiddenSize numLayers directionality dtype device
       , lstmWithConstInit_c    :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       , lstmWithConstInit_h    :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       }
    -> LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  LSTMWithLearnedInit
    :: forall inputSize hiddenSize numLayers directionality dtype device
     . { lstmWithLearnedInit_lstm :: LSTM inputSize hiddenSize numLayers directionality dtype device
       , lstmWithLearnedInit_c    :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       , lstmWithLearnedInit_h    :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
       }
    -> LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (LSTMWithInit inputSize hiddenSize numLayers directionality initialization dtype device)
-- TODO: https://ryanglscott.github.io/2018/02/11/how-to-derive-generic-for-some-gadts/
-- deriving instance Generic (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)

instance Generic (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  type Rep (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) =
    Rec0 (LSTM inputSize hiddenSize numLayers directionality dtype device)
      :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
      :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])

  from (LSTMWithConstInit {..}) = K1 lstmWithConstInit_lstm :*: K1 lstmWithConstInit_c :*: K1 lstmWithConstInit_h
  to (K1 lstm :*: K1 c :*: K1 h) = LSTMWithConstInit lstm c h

instance Generic (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  type Rep (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) =
    Rec0 (LSTM inputSize hiddenSize numLayers directionality dtype device)
      :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
      :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])

  from (LSTMWithLearnedInit {..}) = K1 lstmWithLearnedInit_lstm :*: K1 lstmWithLearnedInit_c :*: K1 lstmWithLearnedInit_h
  to (K1 lstm :*: K1 c :*: K1 h) = LSTMWithLearnedInit lstm c h

instance
  ( KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
                   (LSTM     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
                      (LSTMWithInit     inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  sample (LSTMWithZerosInitSpec lstmSpec) =
    LSTMWithConstInit
      <$> A.sample lstmSpec
      <*> pure zeros
      <*> pure zeros
  sample (LSTMWithConstInitSpec lstmSpec c h) =
    LSTMWithConstInit
      <$> A.sample lstmSpec
      <*> pure c
      <*> pure h

instance
  ( KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
                   (LSTM     inputSize hiddenSize numLayers directionality dtype device)
  ) => A.Randomizable (LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
                      (LSTMWithInit     inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  sample s@(LSTMWithLearnedInitSpec lstmSpec c h) =
    LSTMWithLearnedInit
      <$> A.sample lstmSpec
      <*> (makeIndependent =<< pure c)
      <*> (makeIndependent =<< pure h)

lstm
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
       hxShape
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
     , hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device) parameters
     , tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     , HMap' ToDependent parameters tensorParameters
     )
  => Bool
  -> LSTMWithInit
       inputSize
       hiddenSize
       numLayers
       directionality
       initialization
       dtype
       device
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hxShape
     , Tensor device dtype hxShape
     )
lstm dropoutOn (LSTMWithConstInit lstm@(LSTM _ (Dropout dropoutProb)) cc hc) input
  = Torch.Typed.Functional.lstm
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
    @hxShape
    @tensorParameters
    @dtype
    @device
    (hmap' ToDependent . flattenParameters $ lstm)
    dropoutProb
    dropoutOn
    (cc', hc')
    input
 where
  cc' =
    reshape @hxShape
      . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
      $ cc
  hc' =
    reshape @hxShape
      . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
      $ hc
lstm dropoutOn (LSTMWithLearnedInit lstm@(LSTM _ (Dropout dropoutProb)) cc hc) input
  = Torch.Typed.Functional.lstm
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
    @hxShape
    @tensorParameters
    @dtype
    @device
    (hmap' ToDependent . flattenParameters $ lstm)
    dropoutProb
    dropoutOn
    (cc', hc')
    input
 where
  cc' =
    reshape @hxShape
      . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
      . toDependent
      $ cc
  hc' =
    reshape @hxShape
      . expand
          @'[batchSize, numLayers * NumberOfDirections directionality, hiddenSize]
          False -- TODO: What does the bool do?
      . toDependent
      $ hc

lstmWithDropout, lstmWithoutDropout
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
       hxShape
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
     , hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device) parameters
     , tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     , HMap' ToDependent parameters tensorParameters
     )
  => LSTMWithInit
       inputSize
       hiddenSize
       numLayers
       directionality
       initialization
       dtype
       device
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hxShape
     , Tensor device dtype hxShape
     )
-- ^ Forward propagate the `LSTM` module and apply dropout on the outputs of each layer.
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = LSTMWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (LSTMSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t lstmWithDropout @'BatchFirst model input
-- lstmWithDropout @'BatchFirst model input
--   :: (Tensor '( 'D.CPU, 0) 'D.Float '[5, 16, 60],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30])
-- >>> lstmWithDropout @'BatchFirst model input
-- (Tensor Float [5,16,60] ,Tensor Float [6,5,30] ,Tensor Float [6,5,30] )
lstmWithDropout =
  Torch.Typed.NN.Recurrent.LSTM.lstm
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
    @hxShape
    @parameters
    @tensorParameters
    @dtype
    @device
    True
-- ^ Forward propagate the `LSTM` module (without applying dropout on the outputs of each layer).
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = LSTMWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (LSTMSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t lstmWithoutDropout @'BatchFirst model input
-- lstmWithoutDropout @'BatchFirst model input
--   :: (Tensor '( 'D.CPU, 0) 'D.Float '[5, 16, 60],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30],
--       Tensor '( 'D.CPU, 0) 'D.Float '[6, 5, 30])
-- >>> lstmWithoutDropout @'BatchFirst model input
-- (Tensor Float [5,16,60] ,Tensor Float [6,5,30] ,Tensor Float [6,5,30] )
lstmWithoutDropout =
  Torch.Typed.NN.Recurrent.LSTM.lstm
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
    @hxShape
    @parameters
    @tensorParameters
    @dtype
    @device
    False

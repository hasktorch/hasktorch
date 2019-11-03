{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NoStarIsType          #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE StrictData            #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}

module Torch.Typed.NN.Recurrent.LSTM
    ( LSTMSpec(..)
    , Directionality(..)
    , NumberOfDirections
    , LSTM(..)
    , LSTMParams(..)
    , ParamsPerDirection(..)
    , forwardNoDropout
    , forwardWithDropout
    )
where

import qualified ATen.Cast                as ATen
import qualified ATen.Class               as ATen
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Type                as ATen
import           Data.Kind
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           Prelude                  hiding (tanh)
import           System.Environment
import           System.IO.Unsafe
import qualified Torch.Autograd           as A
import qualified Torch.Device             as D
import qualified Torch.DType              as D
import qualified Torch.Functions          as D
import qualified Torch.NN                 as A
import qualified Torch.Tensor             as D
import qualified Torch.TensorFactories    as D
import           Torch.Typed
import           Torch.Typed.Factories
import           Torch.Typed.Native       (Directionality (..),
                                           NumberOfDirections, expand, lstm)
import           Torch.Typed.NN
import           Torch.Typed.Tensor


-- | A specification for a long, short-term memory layer.
--
data LSTMSpec (dir :: Directionality)
        (dtype :: D.DType)
        (numLayers:: Nat)
        (inputDim :: Nat)
        (hiddenDim :: Nat)
        (device :: (D.DeviceType, Nat)) =
    LSTMSpecZerosInit DropoutSpec -- ^ Weights drawn from Xavier-Uniform with zeros-value
                                  --   initialized biases and cell states.
    | LSTMSpec                    -- ^ Weights drawn from Xavier-Uniform
                                  --   with zeros-value initialized biases
                                  --   and user-provided cell states.
        (Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]) -- ^ The initial values of the memory cell
        (Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]) -- ^ The initial values of the hidden state
        DropoutSpec

    | LSTMSpecLearnedInit             -- ^ Weights drawn from Xavier-Uniform
                                      --   with zeros-value initialized biases
                                      --   and learned cell states.
        (Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]) -- ^ The initial (learnable)
                                                                                 -- values of the memory cell
        (Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]) -- ^ The initial (learnable)
                                                                                 -- values of the hidden state
        DropoutSpec
    deriving (Show, Generic)


-- | LSTMParams
-- Input-to-hidden, hidden-to-hidden, and bias parameters for a mulilayered
-- (and optionally) bidirectional LSTM.
--
data LSTMParams (dtype :: D.DType) (numDirections :: Nat) (inputDim :: Nat) (hiddenSize :: Nat) (numLayers :: Nat) (device :: (D.DeviceType, Nat)) where
    LSTMLayer1 ::Parameter device dtype '[4 * hiddenSize, inputDim]
                    -> (Parameter device dtype '[4 * hiddenSize, hiddenSize])
                    -> (Parameter device dtype '[4 * hiddenSize ])
                    -> (Parameter device dtype '[4 * hiddenSize ])
                    -> LSTMParams dtype numDirections inputDim hiddenSize 1 device
    LSTMLayerK ::(1 <= numLayers)
                    => LSTMParams dtype numDirections inputDim hiddenSize numLayers device
                    -> (Parameter device dtype '[4 * hiddenSize, numDirections * hiddenSize])
                    -> (Parameter device dtype '[4 * hiddenSize, hiddenSize])
                    -> (Parameter device dtype '[4 * hiddenSize ])
                    -> (Parameter device dtype '[4 * hiddenSize ])
                    -> LSTMParams dtype numDirections inputDim hiddenSize (numLayers + 1) device

--  TODO: Generics? see https://gist.github.com/RyanGlScott/71d9f933e823b4a03f99de54d4b94d51

--  A specialized singlton helper for initializing parameters
class (KnownNat n) => LayerSong (n :: Nat) where
    singLayerSing :: (RandDTypeIsValid device dtype, KnownDevice device, KnownDType dtype, KnownNat numDirections, KnownNat inputDim, KnownNat hiddenSize)
        =>  IO (LSTMParams dtype numDirections inputDim hiddenSize n device)

instance {-# OVERLAPS #-} LayerSong 1 where
    singLayerSing =
        LSTMLayer1
            <$> (makeIndependent =<< xavierUniormLSTM)
            <*> (makeIndependent =<< xavierUniormLSTM)
            <*> (makeIndependent =<< (pure zeros))
            <*> (makeIndependent =<< (pure zeros))

instance {-# OVERLAPPABLE #-} (KnownNat n, KnownNat m, LayerSong n, m ~ (n + 1), 1 <= n) => LayerSong m where
    singLayerSing =
        LSTMLayerK
            <$> singLayerSing
            <*> (makeIndependent =<< xavierUniormLSTM)
            <*> (makeIndependent =<< xavierUniormLSTM)
            <*> (makeIndependent =<< (pure zeros))
            <*> (makeIndependent =<< (pure zeros))

instance A.Parameterized (LSTMParams dtype numDirections inputDim hiddenSize numLayers device) where
    flattenParameters (LSTMLayer1 a b c d) =
        A.flattenParameters a
            <> A.flattenParameters b
            <> A.flattenParameters c
            <> A.flattenParameters d
    flattenParameters (LSTMLayerK l a b c d) =
        A.flattenParameters a
            <> A.flattenParameters b
            <> A.flattenParameters c
            <> A.flattenParameters d
            <> A.flattenParameters l
    replaceOwnParameters (LSTMLayer1 a b c d) =
        LSTMLayer1
            <$> A.replaceOwnParameters a
            <*> A.replaceOwnParameters b
            <*> A.replaceOwnParameters c
            <*> A.replaceOwnParameters d
    replaceOwnParameters (LSTMLayerK l a b c d) =
        LSTMLayerK
            <$> A.replaceOwnParameters l
            <*> A.replaceOwnParameters a
            <*> A.replaceOwnParameters b
            <*> A.replaceOwnParameters c
            <*> A.replaceOwnParameters d

data ParamsPerDirection dtype inputDim hiddenSize numLayers (dir :: Directionality) device where
    BidirectionalParams ::LSTMParams dtype (NumberOfDirections 'Bidirectional) inputDim hiddenSize numLayers device
        -> LSTMParams dtype (NumberOfDirections 'Bidirectional) inputDim hiddenSize numLayers device
        -> ParamsPerDirection dtype inputDim hiddenSize numLayers 'Bidirectional device
    UniDirectionalParams ::LSTMParams dtype (NumberOfDirections 'Unidirectional)  inputDim hiddenSize numLayers device
        -> ParamsPerDirection dtype inputDim hiddenSize numLayers 'Unidirectional device


sampleBidirectionalParams
    :: forall dtype inputDim hiddenDim numLayers device
     . ( KnownNat hiddenDim
       , KnownNat inputDim
       , KnownNat numLayers
       , KnownDType dtype
       , LayerSong numLayers
       , KnownDevice device
       , RandDTypeIsValid device dtype
       )
    => IO
           ( ParamsPerDirection
                 dtype
                 inputDim
                 hiddenDim
                 numLayers
                 'Bidirectional
                 device
           )
sampleBidirectionalParams =
    BidirectionalParams <$> singLayerSing <*> singLayerSing


sampleUniDirectionalParams
    :: ( KnownNat hiddenDim
       , KnownNat inputDim
       , KnownNat numLayers
       , KnownDType dtype
       , LayerSong numLayers
       , KnownDevice device
       , RandDTypeIsValid device dtype
       )
    => IO
           ( ParamsPerDirection
                 dtype
                 inputDim
                 hiddenDim
                 numLayers
                 'Unidirectional
                 device
           )
sampleUniDirectionalParams = UniDirectionalParams <$> singLayerSing

instance A.Parameterized  (ParamsPerDirection dtype inputDim hiddenDim numLayers dir device) where
    flattenParameters (UniDirectionalParams ps) = A.flattenParameters ps
    flattenParameters (BidirectionalParams as bs) =
        A.flattenParameters as <> A.flattenParameters bs
    replaceOwnParameters (UniDirectionalParams ps) =
        UniDirectionalParams <$> A.replaceOwnParameters ps
    replaceOwnParameters (BidirectionalParams as bs) =
        BidirectionalParams
            <$> A.replaceOwnParameters as
            <*> A.replaceOwnParameters bs

-- | A long, short-term memory layer with either fixed initial
-- states for the memory cells and hidden state or learnable
-- inital states for the memory cells and hidden state.
--
data LSTM (dir :: Directionality)
            (dtype :: D.DType)
            (numLayers :: Nat)
            (inputDim :: Nat)
            (hiddenDim :: Nat)
            device
            =  LSTM {
      lstm_params :: ParamsPerDirection dtype inputDim hiddenDim numLayers dir device
      , lstm_dropout :: Dropout
      , lstm_init_c     :: Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]
      , lstm_init_h     :: Tensor device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]
    }
    | LSTMLearnedInit {
        lstmLearnedInit_params :: ParamsPerDirection  dtype inputDim hiddenDim numLayers dir device
        , lstmLearnedInit_dropout :: Dropout
        , lstmLearnedInit_init_c     :: Parameter device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]
        , lstmLearnedInit_init_h     :: Parameter device dtype '[numLayers * (NumberOfDirections dir), hiddenDim]
    }
    deriving Generic

-- TODO: when we have cannonical initializers do this correctly:
-- https://github.com/pytorch/pytorch/issues/9221
-- https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605

-- | Helper to do xavier uniform initializations on weight matrices and
-- orthagonal initializations for the gates. (When implemented.)
xavierUniormLSTM
    :: forall device dtype hiddenSize d
     . ( KnownDType dtype
       , KnownNat d
       , KnownNat hiddenSize
       , KnownDevice device
       , RandDTypeIsValid device dtype
       )
    => IO (Tensor device dtype '[4 * hiddenSize, d])
xavierUniormLSTM = do
    init <- (randn :: IO (Tensor device dtype '[4 * hiddenSize, d]))
    UnsafeMkTensor <$> xavierUniformFIXME
        (toDynamic init)
        (5.0 / 3)
        (shape @device @dtype @'[4 * hiddenSize, d] init)

-- TODO: This is taken from the initializers examplee code and should be replaced with cannonical,
-- tested versions.  However, even a potentially incorrect implementation will likly perform
-- better than an ad-hoc random-normal distribution.
-- | Fan-in / Fan-out scaling calculation
calculateFan :: [Int] -> (Int, Int)
calculateFan shape =
    if dimT < 2 then
        error "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    else if dimT == 2 then
        (shape !! 1, shape !! 0)
        else 
            (numInputFmaps * receptiveFieldSize,
            numOutputFmaps * receptiveFieldSize)
    where
        dimT = length shape
        numInputFmaps = shape !! 1 -- size t 1
        numOutputFmaps = shape !! 0 -- size t 0
        receptiveFieldSize = product $ tail shape

-- | Xavier Initialization - Uniform
xavierUniformFIXME :: D.Tensor -> Float -> [Int] -> IO D.Tensor
xavierUniformFIXME init gain shape = do
    pure $ D.subScalar (D.mulScalar init (bound * 2.0)) bound
    where
        (fanIn, fanOut) = calculateFan shape
        std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
        bound = sqrt 3.0 * std

instance (KnownDType dtype
    , KnownNat inputDim
    , KnownNat hiddenDim
    , KnownNat numLayers
    , KnownDevice device
    , RandDTypeIsValid device dtype
    , LayerSong numLayers) => A.Randomizable
        (LSTMSpec 'Bidirectional dtype numLayers inputDim hiddenDim device)
        (LSTM 'Bidirectional dtype numLayers inputDim hiddenDim device) where
    sample (LSTMSpecZerosInit d) =
        LSTM
            <$> sampleBidirectionalParams
            <*> A.sample d
            <*> (pure zeros)
            <*> (pure zeros)
    sample s@(LSTMSpec c h d) =
        LSTM
            <$> sampleBidirectionalParams
            <*> A.sample d
            <*> (pure c)
            <*> (pure h)
    sample s@(LSTMSpecLearnedInit c h d) =
        LSTMLearnedInit
            <$> sampleBidirectionalParams
            <*> A.sample d
            <*> (makeIndependent =<< pure c)
            <*> (makeIndependent =<< pure h)

instance (KnownDType dtype
    , KnownNat inputDim
    , KnownNat hiddenDim
    , KnownNat numLayers
    , KnownDevice device
    , RandDTypeIsValid device dtype
    , LayerSong numLayers) => A.Randomizable
        (LSTMSpec 'Unidirectional dtype numLayers inputDim hiddenDim device)
        (LSTM 'Unidirectional dtype numLayers inputDim hiddenDim device) where
    sample (LSTMSpecZerosInit d) =
        LSTM
            <$> sampleUniDirectionalParams
            <*> A.sample d
            <*> (pure zeros)
            <*> (pure zeros)
    sample s@(LSTMSpec c h d) =
        LSTM
            <$> sampleUniDirectionalParams
            <*> A.sample d
            <*> (pure c)
            <*> (pure h)
    sample s@(LSTMSpecLearnedInit c h d) =
        LSTMLearnedInit
            <$> sampleUniDirectionalParams
            <*> A.sample d
            <*> (makeIndependent =<< pure c)
            <*> (makeIndependent =<< pure h)

instance A.Parameterized  (LSTM dir dtype numLayers inputDim hiddenDim device) where
    flattenParameters (LSTM p _ _ _) = A.flattenParameters p
    flattenParameters (LSTMLearnedInit p _ lc lh) =
        A.flattenParameters p
            <> A.flattenParameters lc
            <> A.flattenParameters lh
    replaceOwnParameters (LSTM p d c h) = do
        p' <- A.replaceOwnParameters p
        pure $ LSTM p' d c h
    replaceOwnParameters (LSTMLearnedInit p d lc lh) = do
        p' <- A.replaceOwnParameters p
        LSTMLearnedInit p' d
            <$> A.replaceOwnParameters lc
            <*> A.replaceOwnParameters lh

-- helpers to get params in the right order for Aten.
lstmParamsToTlist'
    :: forall dtype numDirections inputDim hiddenSize numLayers device
     . [D.Tensor]
    -> LSTMParams dtype numDirections inputDim hiddenSize numLayers device
    -> [D.Tensor]
lstmParamsToTlist' acc (LSTMLayerK l wi wh bi bh) =
    lstmParamsToTlist' acc l
        <> ( (toDynamic $ toDependent wi)
           : (toDynamic $ toDependent wh)
           : (toDynamic $ toDependent bi)
           : (toDynamic $ toDependent bh)
           : []
           )
lstmParamsToTlist' acc (LSTMLayer1 wi wh bi bh) =
    (toDynamic $ toDependent wi)
        : (toDynamic $ toDependent wh)
        : (toDynamic $ toDependent bi)
        : (toDynamic $ toDependent bh)
        : acc
lstmParamsToTlist l = lstmParamsToTlist' [] l

ziplstmLayers :: [D.Tensor] -> [D.Tensor] -> [D.Tensor] -> [D.Tensor]
ziplstmLayers acc [] [] = acc
ziplstmLayers acc (a : b : c : d : xs) (a' : b' : c' : d' : xs') =
    a : b : c : d : a' : b' : c' : d' : ziplstmLayers acc xs xs'

forward'
    :: forall dtype dir numLayers inputDim hiddenDim seqLen batchSize device
     . ( KnownNat numLayers
       , KnownNat inputDim
       , KnownNat hiddenDim
       , KnownNat seqLen
       , KnownNat batchSize
       , KnownNat (NumberOfDirections dir)
       )
    => Bool
    -> LSTM dir dtype numLayers inputDim hiddenDim device
    -> Tensor device dtype '[seqLen, batchSize, inputDim]
    -> ( Tensor
             device
             dtype
             '[seqLen, batchSize, hiddenDim * NumberOfDirections dir]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       )
forward' doDropout (LSTM p (Dropout d) cc hc) input = lstm @dir @numLayers
    input
    (cc', hc')
    (params p)
    d
    doDropout
  where
    cc' =
        reshape @'[numLayers * (NumberOfDirections dir), batchSize, hiddenDim]
            $ expand
                  @'[batchSize, numLayers * (NumberOfDirections dir), hiddenDim]
                  False -- TODO: What does the bool do?
                  cc
    hc' =
        reshape @'[numLayers * (NumberOfDirections dir), batchSize, hiddenDim]
            $ expand
                  @'[batchSize, numLayers * (NumberOfDirections dir), hiddenDim]
                  False
                  hc
params
    :: forall device dtype inputDim hiddenDim numLayers dir
     . ParamsPerDirection device dtype inputDim hiddenDim numLayers dir
    -> [D.Tensor]
params (BidirectionalParams fwd rvs) =
    ziplstmLayers [] (lstmParamsToTlist' [] fwd) (lstmParamsToTlist' [] rvs)
params (UniDirectionalParams fwd) = lstmParamsToTlist' [] fwd

-- | Forward propagage the `LSTM` module and apply dropout on the outputs of each layer.
--
-- >>> input :: Tensor '(D.CPU,0) 'D.Float '[5,16,10]            <- randn
-- >>> lstm  :: (LSTM 'Bidirectional D.Float 3 10 30 '(D.CPU,0) ) <- A.sample (LSTMSpecZerosInit (DropoutSpec 10) :: LSTMSpec 'Bidirectional D.Float 3 10 30 '(D.CPU,0) )
-- >>> forwardNoDropout lstm input
-- (Tensor Float [5,16,60] ,Tensor Float [6,16,30] ,Tensor Float [6,16,30] )
forwardWithDropout
    :: forall dtype dir numLayers inputDim hiddenDim seqLen batchSize device
     . ( KnownNat numLayers
       , KnownNat inputDim
       , KnownNat hiddenDim
       , KnownNat seqLen
       , KnownNat batchSize
       , KnownNat (NumberOfDirections dir)
       )
    => LSTM dir dtype numLayers inputDim hiddenDim device
    -> Tensor device dtype '[seqLen, batchSize, inputDim]
    -> ( Tensor
             device
             dtype
             '[seqLen, batchSize, hiddenDim * NumberOfDirections dir]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       )
forwardWithDropout = forward' True

-- | Forward propagage the `LSTM` module (without applying dropout on the outputs of each layer).
--
-- >>> input ::Tensor '(D.CPU,0) 'D.Float '[5,16,10]             <- randn
-- >>> lstm :: (LSTM 'Unidirectional D.Float 1 10 30 '(D.CPU,0)) <- A.sample (LSTMSpecZerosInit (DropoutSpec 10) :: LSTMSpec 'Unidirectional D.Float 1 10 30 '(D.CPU,0))
-- >>> forwardNoDropout lstm input
-- (Tensor Float [5,16,30] ,Tensor Float [1,16,30] ,Tensor Float [1,16,30] )
forwardNoDropout
    :: forall dtype dir numLayers inputDim hiddenDim seqLen batchSize device
     . ( KnownNat numLayers
       , KnownNat inputDim
       , KnownNat hiddenDim
       , KnownNat seqLen
       , KnownNat batchSize
       , KnownNat (NumberOfDirections dir)
       )
    => LSTM dir dtype numLayers inputDim hiddenDim device
    -> Tensor device dtype '[seqLen, batchSize, inputDim]
    -> ( Tensor
             device
             dtype
             '[seqLen, batchSize, hiddenDim * NumberOfDirections dir]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       , Tensor
             device
             dtype
             '[numLayers * NumberOfDirections dir, batchSize, hiddenDim]
       )
forwardNoDropout = forward' False

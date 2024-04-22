{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
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

module Torch.Typed.NN.Recurrent.LSTM where

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
  LSTMLayerSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = LSTMLayerSpec
  deriving (Show, Eq)

data
  LSTMLayer
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  LSTMUnidirectionalLayer ::
    Parameter device dtype (LSTMWIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMWHShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBHShape hiddenSize inputSize) ->
    LSTMLayer inputSize hiddenSize 'Unidirectional dtype device
  LSTMBidirectionalLayer ::
    Parameter device dtype (LSTMWIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMWHShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBHShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMWIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMWHShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBIShape hiddenSize inputSize) ->
    Parameter device dtype (LSTMBHShape hiddenSize inputSize) ->
    LSTMLayer inputSize hiddenSize 'Bidirectional dtype device

deriving instance Show (LSTMLayer inputSize hiddenSize directionality dtype device)

instance Parameterized (LSTMLayer inputSize hiddenSize 'Unidirectional dtype device) where
  type
    Parameters (LSTMLayer inputSize hiddenSize 'Unidirectional dtype device) =
      '[ Parameter device dtype (LSTMWIShape hiddenSize inputSize),
         Parameter device dtype (LSTMWHShape hiddenSize inputSize),
         Parameter device dtype (LSTMBIShape hiddenSize inputSize),
         Parameter device dtype (LSTMBHShape hiddenSize inputSize)
       ]
  flattenParameters (LSTMUnidirectionalLayer wi wh bi bh) =
    wi :. wh :. bi :. bh :. HNil
  replaceParameters _ (wi :. wh :. bi :. bh :. HNil) =
    LSTMUnidirectionalLayer wi wh bi bh

instance Parameterized (LSTMLayer inputSize hiddenSize 'Bidirectional dtype device) where
  type
    Parameters (LSTMLayer inputSize hiddenSize 'Bidirectional dtype device) =
      '[ Parameter device dtype (LSTMWIShape hiddenSize inputSize),
         Parameter device dtype (LSTMWHShape hiddenSize inputSize),
         Parameter device dtype (LSTMBIShape hiddenSize inputSize),
         Parameter device dtype (LSTMBHShape hiddenSize inputSize),
         Parameter device dtype (LSTMWIShape hiddenSize inputSize),
         Parameter device dtype (LSTMWHShape hiddenSize inputSize),
         Parameter device dtype (LSTMBIShape hiddenSize inputSize),
         Parameter device dtype (LSTMBHShape hiddenSize inputSize)
       ]
  flattenParameters (LSTMBidirectionalLayer wi wh bi bh wi' wh' bi' bh') =
    wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil
  replaceParameters _ (wi :. wh :. bi :. bh :. wi' :. wh' :. bi' :. bh' :. HNil) =
    LSTMBidirectionalLayer wi wh bi bh wi' wh' bi' bh'

instance
  ( RandDTypeIsValid device dtype,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownDType dtype,
    KnownDevice device
  ) =>
  A.Randomizable
    (LSTMLayerSpec inputSize hiddenSize 'Unidirectional dtype device)
    (LSTMLayer inputSize hiddenSize 'Unidirectional dtype device)
  where
  sample _ =
    LSTMUnidirectionalLayer
      <$> (makeIndependent =<< xavierUniformLSTM)
      <*> (makeIndependent =<< xavierUniformLSTM)
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
    (LSTMLayerSpec inputSize hiddenSize 'Bidirectional dtype device)
    (LSTMLayer inputSize hiddenSize 'Bidirectional dtype device)
  where
  sample _ =
    LSTMBidirectionalLayer
      <$> (makeIndependent =<< xavierUniformLSTM)
      <*> (makeIndependent =<< xavierUniformLSTM)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< xavierUniformLSTM)
      <*> (makeIndependent =<< xavierUniformLSTM)
      <*> (makeIndependent =<< pure zeros)
      <*> (makeIndependent =<< pure zeros)

instance A.Parameterized (LSTMLayer inputSize hiddenSize directionality dtype device) where
  flattenParameters (LSTMUnidirectionalLayer wi wh bi bh) =
    [ untypeParam wi,
      untypeParam wh,
      untypeParam bi,
      untypeParam bh
    ]
  flattenParameters (LSTMBidirectionalLayer wi wh bi bh wi' wh' bi' bh') =
    [ untypeParam wi,
      untypeParam wh,
      untypeParam bi,
      untypeParam bh,
      untypeParam wi',
      untypeParam wh',
      untypeParam bi',
      untypeParam bh'
    ]
  _replaceParameters (LSTMUnidirectionalLayer _wi _wh _bi _bh) = do
    wi <- A.nextParameter
    wh <- A.nextParameter
    bi <- A.nextParameter
    bh <- A.nextParameter
    return
      ( LSTMUnidirectionalLayer
          (UnsafeMkParameter wi)
          (UnsafeMkParameter wh)
          (UnsafeMkParameter bi)
          (UnsafeMkParameter bh)
      )
  _replaceParameters (LSTMBidirectionalLayer _wi _wh _bi _bh _wi' _wh' _bi' _bh') = do
    wi <- A.nextParameter
    wh <- A.nextParameter
    bi <- A.nextParameter
    bh <- A.nextParameter
    wi' <- A.nextParameter
    wh' <- A.nextParameter
    bi' <- A.nextParameter
    bh' <- A.nextParameter
    return
      ( LSTMBidirectionalLayer
          (UnsafeMkParameter wi)
          (UnsafeMkParameter wh)
          (UnsafeMkParameter bi)
          (UnsafeMkParameter bh)
          (UnsafeMkParameter wi')
          (UnsafeMkParameter wh')
          (UnsafeMkParameter bi')
          (UnsafeMkParameter bh')
      )

data
  LSTMLayerStackSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = LSTMLayerStackSpec
  deriving (Show, Eq)

-- Input-to-hidden, hidden-to-hidden, and bias parameters for a mulilayered
-- (and optionally) bidirectional LSTM.
--
data
  LSTMLayerStack
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  LSTMLayer1 ::
    LSTMLayer inputSize hiddenSize directionality dtype device ->
    LSTMLayerStack inputSize hiddenSize 1 directionality dtype device
  LSTMLayerK ::
    LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device ->
    LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device ->
    LSTMLayerStack inputSize hiddenSize (numLayers + 1) directionality dtype device

deriving instance Show (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)

class LSTMLayerStackParameterized (flag :: Bool) inputSize hiddenSize numLayers directionality dtype device where
  type LSTMLayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device :: [Type]
  lstmLayerStackFlattenParameters ::
    Proxy flag ->
    LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device ->
    HList (LSTMLayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device)
  lstmLayerStackReplaceParameters ::
    Proxy flag ->
    LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device ->
    HList (LSTMLayerStackParameters flag inputSize hiddenSize numLayers directionality dtype device) ->
    LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device

instance
  Parameterized (LSTMLayer inputSize hiddenSize directionality dtype device) =>
  LSTMLayerStackParameterized 'False inputSize hiddenSize 1 directionality dtype device
  where
  type
    LSTMLayerStackParameters 'False inputSize hiddenSize 1 directionality dtype device =
      Parameters (LSTMLayer inputSize hiddenSize directionality dtype device)
  lstmLayerStackFlattenParameters _ (LSTMLayer1 lstmLayer) = flattenParameters lstmLayer
  lstmLayerStackReplaceParameters _ (LSTMLayer1 lstmLayer) parameters = LSTMLayer1 $ replaceParameters lstmLayer parameters

instance
  ( Parameterized
      ( LSTMLayer
          (hiddenSize * NumberOfDirections directionality)
          hiddenSize
          directionality
          dtype
          device
      ),
    Parameterized (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device),
    HAppendFD
      (Parameters (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device))
      (Parameters (LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device))
      ( Parameters (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
          ++ Parameters (LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
      ),
    1 <= numLayers,
    numLayersM1 ~ numLayers - 1,
    0 <= numLayersM1
  ) =>
  LSTMLayerStackParameterized 'True inputSize hiddenSize numLayers directionality dtype device
  where
  type
    LSTMLayerStackParameters 'True inputSize hiddenSize numLayers directionality dtype device =
      Parameters (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
        ++ Parameters (LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
  lstmLayerStackFlattenParameters _ (LSTMLayerK lstmLayer lstmLayerStack) =
    let parameters = flattenParameters lstmLayer
        parameters' = flattenParameters @(LSTMLayerStack inputSize hiddenSize numLayersM1 directionality dtype device) lstmLayerStack
     in parameters' `happendFD` parameters
  lstmLayerStackReplaceParameters _ (LSTMLayerK lstmLayer lstmLayerStack) parameters'' =
    let (parameters', parameters) = hunappendFD parameters''
        lstmLayer' = replaceParameters lstmLayer parameters
        lstmLayerStack' = replaceParameters @(LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device) lstmLayerStack parameters'
     in LSTMLayerK lstmLayer' lstmLayerStack'

instance
  ( 1 <= numLayers,
    (2 <=? numLayers) ~ flag,
    LSTMLayerStackParameterized flag inputSize hiddenSize numLayers directionality dtype device
  ) =>
  Parameterized (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)
  where
  type
    Parameters (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device) =
      LSTMLayerStackParameters (2 <=? numLayers) inputSize hiddenSize numLayers directionality dtype device
  flattenParameters = lstmLayerStackFlattenParameters (Proxy :: Proxy flag)
  replaceParameters = lstmLayerStackReplaceParameters (Proxy :: Proxy flag)

class LSTMLayerStackRandomizable (flag :: Bool) inputSize hiddenSize numLayers directionality dtype device where
  lstmLayerStackSample ::
    Proxy flag ->
    LSTMLayerStackSpec inputSize hiddenSize numLayers directionality dtype device ->
    IO (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)

instance
  ( A.Randomizable
      (LSTMLayerSpec inputSize hiddenSize directionality dtype device)
      (LSTMLayer inputSize hiddenSize directionality dtype device)
  ) =>
  LSTMLayerStackRandomizable 'False inputSize hiddenSize 1 directionality dtype device
  where
  lstmLayerStackSample _ _ = LSTMLayer1 <$> (sample $ LSTMLayerSpec @inputSize @hiddenSize @directionality @dtype @device)

instance
  ( 1 <= numLayers,
    A.Randomizable
      (LSTMLayerSpec (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device)
      (LSTMLayer (hiddenSize * NumberOfDirections directionality) hiddenSize directionality dtype device),
    A.Randomizable
      (LSTMLayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
      (LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
  ) =>
  LSTMLayerStackRandomizable 'True inputSize hiddenSize numLayers directionality dtype device
  where
  lstmLayerStackSample _ _ =
    LSTMLayerK
      <$> (sample $ LSTMLayerSpec @(hiddenSize * NumberOfDirections directionality) @hiddenSize @directionality @dtype @device)
      <*> ( sample
              @(LSTMLayerStackSpec inputSize hiddenSize (numLayers - 1) directionality dtype device)
              @(LSTMLayerStack inputSize hiddenSize (numLayers - 1) directionality dtype device)
              $ LSTMLayerStackSpec
          )

instance
  ( 1 <= numLayers,
    (2 <=? numLayers) ~ flag,
    RandDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device,
    LSTMLayerStackRandomizable flag inputSize hiddenSize numLayers directionality dtype device
  ) =>
  Randomizable
    (LSTMLayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
    (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)
  where
  sample = lstmLayerStackSample (Proxy :: Proxy flag)

instance A.Parameterized (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device) where
  flattenParameters (LSTMLayer1 layer) =
    A.flattenParameters layer
  flattenParameters (LSTMLayerK stack layer) =
    A.flattenParameters stack
      ++ A.flattenParameters layer
  _replaceParameters (LSTMLayer1 layer) = do
    layer' <- A._replaceParameters layer
    return $ LSTMLayer1 layer'
  _replaceParameters (LSTMLayerK stack layer) = do
    stack' <- A._replaceParameters stack
    layer' <- A._replaceParameters layer
    return $ LSTMLayerK stack' layer'

newtype
  LSTMSpec
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = LSTMSpec DropoutSpec
  deriving (Show, Generic)

data
  LSTM
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  LSTM ::
    (1 <= numLayers) =>
    { lstm_layer_stack :: LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device,
      lstm_dropout :: Dropout
    } ->
    LSTM inputSize hiddenSize numLayers directionality dtype device

deriving instance Show (LSTM inputSize hiddenSize numLayers directionality dtype device)

instance
  (1 <= numLayers) =>
  Generic (LSTM inputSize hiddenSize numLayers directionality dtype device)
  where
  type
    Rep (LSTM inputSize hiddenSize numLayers directionality dtype device) =
      Rec0 (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 Dropout
  from (LSTM {..}) = K1 lstm_layer_stack :*: K1 lstm_dropout
  to (K1 layerStack :*: K1 dropout) = LSTM layerStack dropout

instance
  ( 1 <= numLayers,
    Parameterized (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device))
      (Parameters Dropout)
      ( Parameters (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device)
          ++ Parameters Dropout
      )
  ) =>
  Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device)

-- TODO: when we have cannonical initializers do this correctly:
-- https://github.com/pytorch/pytorch/issues/9221
-- https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605

instance A.Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device) where
  flattenParameters LSTM {..} = A.flattenParameters lstm_layer_stack
  _replaceParameters LSTM {..} = do
    lstm_layer_stack' <- A._replaceParameters lstm_layer_stack
    return $
      LSTM
        { lstm_layer_stack = lstm_layer_stack',
          ..
        }

-- | Helper to do xavier uniform initializations on weight matrices and
-- orthagonal initializations for the gates. (When implemented.)
xavierUniformLSTM ::
  forall device dtype hiddenSize featureSize.
  ( KnownDType dtype,
    KnownNat hiddenSize,
    KnownNat featureSize,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  IO (Tensor device dtype '[4 * hiddenSize, featureSize])
xavierUniformLSTM = do
  init <- randn :: IO (Tensor device dtype '[4 * hiddenSize, featureSize])
  UnsafeMkTensor
    <$> xavierUniformFIXME
      (toDynamic init)
      (5.0 / 3)
      (shape @device @dtype @'[4 * hiddenSize, featureSize] init)

instance
  ( KnownDType dtype,
    KnownDevice device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat (NumberOfDirections directionality),
    RandDTypeIsValid device dtype,
    A.Randomizable
      (LSTMLayerStackSpec inputSize hiddenSize numLayers directionality dtype device)
      (LSTMLayerStack inputSize hiddenSize numLayers directionality dtype device),
    1 <= numLayers
  ) =>
  A.Randomizable
    (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
    (LSTM inputSize hiddenSize numLayers directionality dtype device)
  where
  sample (LSTMSpec dropoutSpec) =
    LSTM
      <$> A.sample (LSTMLayerStackSpec @inputSize @hiddenSize @numLayers @directionality @dtype @device)
      <*> A.sample dropoutSpec

-- | A specification for a long, short-term memory layer.
data
  LSTMWithInitSpec
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
  LSTMWithZerosInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    LSTMSpec inputSize hiddenSize numLayers directionality dtype device ->
    LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and user-provided cell states.
  LSTMWithConstInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    LSTMSpec inputSize hiddenSize numLayers directionality dtype device ->
    -- | The initial values of the memory cell
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    -- | The initial values of the hidden state
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device
  -- | Weights drawn from Xavier-Uniform
  --   with zeros-value initialized biases
  --   and learned cell states.
  LSTMWithLearnedInitSpec ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    LSTMSpec inputSize hiddenSize numLayers directionality dtype device ->
    -- | The initial (learnable)
    -- values of the memory cell
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    -- | The initial (learnable)
    -- values of the hidden state
    Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize] ->
    LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device

deriving instance Show (LSTMWithInitSpec inputSize hiddenSize numLayers directionality initialization dtype device)

-- | A long, short-term memory layer with either fixed initial
-- states for the memory cells and hidden state or learnable
-- inital states for the memory cells and hidden state.
data
  LSTMWithInit
    (inputSize :: Nat)
    (hiddenSize :: Nat)
    (numLayers :: Nat)
    (directionality :: RNNDirectionality)
    (initialization :: RNNInitialization)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  LSTMWithConstInit ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    { lstmWithConstInit_lstm :: LSTM inputSize hiddenSize numLayers directionality dtype device,
      lstmWithConstInit_c :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize],
      lstmWithConstInit_h :: Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
    } ->
    LSTMWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      'ConstantInitialization
      dtype
      device
  LSTMWithLearnedInit ::
    forall inputSize hiddenSize numLayers directionality dtype device.
    { lstmWithLearnedInit_lstm :: LSTM inputSize hiddenSize numLayers directionality dtype device,
      lstmWithLearnedInit_c :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize],
      lstmWithLearnedInit_h :: Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize]
    } ->
    LSTMWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      'LearnedInitialization
      dtype
      device

deriving instance Show (LSTMWithInit inputSize hiddenSize numLayers directionality initialization dtype device)

instance Generic (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) where
  type
    Rep (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device) =
      Rec0 (LSTM inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
        :*: Rec0 (Tensor device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
  from (LSTMWithConstInit {..}) = K1 lstmWithConstInit_lstm :*: K1 lstmWithConstInit_c :*: K1 lstmWithConstInit_h
  to (K1 lstm :*: K1 c :*: K1 h) = LSTMWithConstInit lstm c h

instance Generic (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) where
  type
    Rep (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device) =
      Rec0 (LSTM inputSize hiddenSize numLayers directionality dtype device)
        :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
        :*: Rec0 (Parameter device dtype '[numLayers * NumberOfDirections directionality, hiddenSize])
  from (LSTMWithLearnedInit {..}) = K1 lstmWithLearnedInit_lstm :*: K1 lstmWithLearnedInit_c :*: K1 lstmWithLearnedInit_h
  to (K1 lstm :*: K1 c :*: K1 h) = LSTMWithLearnedInit lstm c h

instance
  ( Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device))
      '[]
      (Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device) ++ '[])
  ) =>
  Parameterized (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)

instance
  ( Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device),
    HAppendFD
      (Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device))
      '[ Parameter
           device
           dtype
           '[numLayers * NumberOfDirections directionality, hiddenSize],
         Parameter
           device
           dtype
           '[numLayers * NumberOfDirections directionality, hiddenSize]
       ]
      ( Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device)
          ++ '[ Parameter
                  device
                  dtype
                  '[numLayers * NumberOfDirections directionality, hiddenSize],
                Parameter
                  device
                  dtype
                  '[numLayers * NumberOfDirections directionality, hiddenSize]
              ]
      )
  ) =>
  Parameterized (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)

instance
  ( KnownNat hiddenSize,
    KnownNat numLayers,
    KnownNat (NumberOfDirections directionality),
    KnownDType dtype,
    KnownDevice device,
    A.Randomizable
      (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
      (LSTM inputSize hiddenSize numLayers directionality dtype device)
  ) =>
  A.Randomizable
    (LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
    (LSTMWithInit inputSize hiddenSize numLayers directionality 'ConstantInitialization dtype device)
  where
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
  ( KnownNat hiddenSize,
    KnownNat numLayers,
    KnownNat (NumberOfDirections directionality),
    KnownDType dtype,
    KnownDevice device,
    A.Randomizable
      (LSTMSpec inputSize hiddenSize numLayers directionality dtype device)
      (LSTM inputSize hiddenSize numLayers directionality dtype device)
  ) =>
  A.Randomizable
    (LSTMWithInitSpec inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
    (LSTMWithInit inputSize hiddenSize numLayers directionality 'LearnedInitialization dtype device)
  where
  sample s@(LSTMWithLearnedInitSpec lstmSpec c h) =
    LSTMWithLearnedInit
      <$> A.sample lstmSpec
      <*> (makeIndependent =<< pure c)
      <*> (makeIndependent =<< pure h)

instance A.Parameterized (LSTMWithInit inputSize hiddenSize numLayers directionality initialization dtype device) where
  flattenParameters LSTMWithConstInit {..} =
    A.flattenParameters lstmWithConstInit_lstm
  flattenParameters LSTMWithLearnedInit {..} =
    A.flattenParameters lstmWithLearnedInit_lstm
      ++ fmap untypeParam [lstmWithLearnedInit_c, lstmWithLearnedInit_h]
  _replaceParameters LSTMWithConstInit {..} = do
    lstmWithConstInit_lstm' <- A._replaceParameters lstmWithConstInit_lstm
    return $
      LSTMWithConstInit
        { lstmWithConstInit_lstm = lstmWithConstInit_lstm',
          ..
        }
  _replaceParameters LSTMWithLearnedInit {..} = do
    lstmWithLearnedInit_lstm' <- A._replaceParameters lstmWithLearnedInit_lstm
    lstmWithLearnedInit_c' <- A.nextParameter
    lstmWithLearnedInit_h' <- A.nextParameter
    return $
      LSTMWithLearnedInit
        { lstmWithLearnedInit_lstm = lstmWithLearnedInit_lstm',
          lstmWithLearnedInit_c = UnsafeMkParameter lstmWithLearnedInit_c',
          lstmWithLearnedInit_h = UnsafeMkParameter lstmWithLearnedInit_h'
        }

lstmForward ::
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
    hxShape
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
    hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize],
    parameters ~ Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device),
    Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device),
    tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device,
    ATen.Castable (HList tensorParameters) [D.ATenTensor],
    HMap' ToDependent parameters tensorParameters
  ) =>
  Bool ->
  LSTMWithInit
    inputSize
    hiddenSize
    numLayers
    directionality
    initialization
    dtype
    device ->
  Tensor device dtype inputShape ->
  ( Tensor device dtype outputShape,
    Tensor device dtype hxShape,
    Tensor device dtype hxShape
  )
lstmForward dropoutOn (LSTMWithConstInit lstmModel@(LSTM _ (Dropout dropoutProb)) cc hc) input =
  lstm
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
    (hmap' ToDependent . flattenParameters $ lstmModel)
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
lstmForward dropoutOn (LSTMWithLearnedInit lstmModel@(LSTM _ (Dropout dropoutProb)) cc hc) input =
  lstm
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
    (hmap' ToDependent . flattenParameters $ lstmModel)
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

lstmForwardWithDropout,
  lstmForwardWithoutDropout ::
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
      hxShape
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
      hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize],
      parameters ~ Parameters (LSTM inputSize hiddenSize numLayers directionality dtype device),
      Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device),
      tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device,
      ATen.Castable (HList tensorParameters) [D.ATenTensor],
      HMap' ToDependent parameters tensorParameters
    ) =>
    LSTMWithInit
      inputSize
      hiddenSize
      numLayers
      directionality
      initialization
      dtype
      device ->
    Tensor device dtype inputShape ->
    ( Tensor device dtype outputShape,
      Tensor device dtype hxShape,
      Tensor device dtype hxShape
    )
-- ^ Forward propagate the `LSTM` module and apply dropout on the outputs of each layer.
--
-- >>> input :: CPUTensor 'D.Float '[5,16,10] <- randn
-- >>> spec = LSTMWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'(D.CPU, 0) (LSTMSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t lstmForwardWithDropout @'BatchFirst model input
-- lstmForwardWithDropout @'BatchFirst model input
--   :: (Tensor '(D.CPU, 0) 'D.Float [5, 16, 60],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30])
-- >>> (a,b,c) = lstmForwardWithDropout @'BatchFirst model input
-- >>> ((dtype a, shape a), (dtype b, shape b), (dtype c, shape c))
-- ((Float,[5,16,60]),(Float,[6,5,30]),(Float,[6,5,30]))
lstmForwardWithDropout =
  lstmForward
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
-- >>> spec = LSTMWithZerosInitSpec @10 @30 @3 @'Bidirectional @'D.Float @'(D.CPU, 0) (LSTMSpec (DropoutSpec 0.5))
-- >>> model <- A.sample spec
-- >>> :t lstmForwardWithoutDropout @'BatchFirst model input
-- lstmForwardWithoutDropout @'BatchFirst model input
--   :: (Tensor '(D.CPU, 0) 'D.Float [5, 16, 60],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30],
--       Tensor '(D.CPU, 0) 'D.Float [6, 5, 30])
-- >>> (a,b,c) = lstmForwardWithoutDropout @'BatchFirst model input
-- >>> ((dtype a, shape a), (dtype b, shape b), (dtype c, shape c))
-- ((Float,[5,16,60]),(Float,[6,5,30]),(Float,[6,5,30]))
lstmForwardWithoutDropout =
  lstmForward
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

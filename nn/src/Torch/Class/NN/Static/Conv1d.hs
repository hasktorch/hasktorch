{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Class.NN.Static.Conv1d where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
-- import Numeric.Backprop hiding ((:>))
import Data.Kind (Type)

-- | Temporal (1D) Convolutions
class IsTensor t => TemporalConvolutions (t :: [Nat] -> Type) where
  -- | Applies a 1D convolution over an input sequence composed of nInputFrame frames. The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize) or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).
  _temporalConvolution_updateOutput         :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> Int -> Int -> IO ()
  _temporalConvolution_updateGradInput      :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> IO ()
  _temporalConvolution_accGradParameters    :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> Double -> IO ()

  _temporalRowConvolution_updateOutput      :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_updateGradInput   :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_accGradParameters :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> Double -> IO ()

newtype Conv1d t f o kW dW
  = Conv1d { getTensors :: (t '[o, f*kW], t '[o]) }

weights :: Conv1d t f o kW dW -> t '[o, f*kW]
weights (Conv1d (w, _)) = w

bias :: Conv1d t f o kW dW -> t '[o]
bias (Conv1d (_, b)) = b

featureSize :: forall t f o kW dW . KnownNat f => Conv1d t f o kW dW -> Int
featureSize _ = fromIntegral (natVal (Proxy :: Proxy f))

outputSize :: forall t f o kW dW . KnownNat o => Conv1d t f o kW dW -> Int
outputSize _ = fromIntegral (natVal (Proxy :: Proxy o))

-- | kW: The kernel width of the convolution
kernelWidth :: forall t f o kW dW . KnownNat kW => Conv1d t f o kW dW -> Int
kernelWidth _ = fromIntegral (natVal (Proxy :: Proxy kW))

-- | dW: The step of the convolution. Default is 1 in C.
stepSize :: forall t f o kW dW . KnownNat dW => Conv1d t f o kW dW -> Int
stepSize _ = fromIntegral (natVal (Proxy :: Proxy dW))

type TemporalConvC t s f kW dW o =
  ( TemporalConvolutions t
  , KnownNat5 s f kW dW o
  , (s > kW) ~ 'True
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  -- , o ~ ((Div (s - kW) dW) + 1)
  )

-- If the input sequence is a 2D tensor of dimension (nInputFrame x inputFrameSize), the
-- output sequence will be (nOutputFrame x outputFrameSize) where
--
--    nOutputFrame = (nInputFrame - kW) / dW + 1
conv1d_forward :: TemporalConvC t s f kW dW o => t '[s, f] -> Conv1d t f o kW dW -> IO (t '[s, o])
conv1d_forward = _conv1d_forward

conv1d_forwardBatch :: TemporalConvC t s f kW dW o => t '[b,s,f] -> Conv1d t f o kW dW -> IO (t '[b,s,o])
conv1d_forwardBatch = _conv1d_forward

_conv1d_forward :: (KnownNat4 f o kW dW, TemporalConvolutions t) => t d -> Conv1d t f o kW dW -> IO (t d')
_conv1d_forward inp conv = do
  out <- empty
  _temporalConvolution_updateOutput inp out
    (weights conv) (bias conv)
    (kernelWidth conv) (stepSize conv)
    (featureSize conv) (outputSize conv)
  pure out


conv1d_backward
  :: TemporalConvC t s f kW dW o
  => t '[s, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[s, o]                         -- ^ grad output
  -> t '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[s, f])                    -- ^ output
conv1d_backward = _conv1d_backwardBatch

conv1d_backwardBatch
  :: TemporalConvC t s f kW dW o
  => t '[b, s, f]                      -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[b, s, o]                      -- ^ grad output
  -> t '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[b, s, f])                 -- ^ output
conv1d_backwardBatch = _conv1d_backwardBatch

_conv1d_backwardBatch
  :: forall f o kW dW t d d'
  . (KnownNat4 f o kW dW, TemporalConvolutions t)
  => t d
  -> t d'
  -> t '[o, f*kW]
  -> Proxy '(kW, dW)
  -> IO (t d)
_conv1d_backwardBatch input gradOut weight _ = do
  gradIn <- empty
  _temporalConvolution_updateGradInput input gradOut gradIn weight
    (fromIntegral $ natVal (Proxy :: Proxy kW))
    (fromIntegral $ natVal (Proxy :: Proxy dW))
  pure gradIn


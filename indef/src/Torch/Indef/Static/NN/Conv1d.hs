-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Conv1d
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Temporal (1D) Convolutions
-------------------------------------------------------------------------------
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Indef.Static.NN.Conv1d where

import Data.Kind (Type)
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.NN as Dynamic

newtype Conv1d f o kW dW
  = Conv1d { getTensors :: (Tensor '[o, f*kW], Tensor '[o]) }

weights :: Conv1d f o kW dW -> Tensor '[o, f*kW]
weights (Conv1d (w, _)) = w

bias :: Conv1d f o kW dW -> Tensor '[o]
bias (Conv1d (_, b)) = b

featureSize :: forall f o kW dW . KnownNat f => Conv1d f o kW dW -> Int
featureSize _ = fromIntegral (natVal (Proxy :: Proxy f))

outputSize :: forall f o kW dW . KnownNat o => Conv1d f o kW dW -> Int
outputSize _ = fromIntegral (natVal (Proxy :: Proxy o))

-- | kW: The kernel width of the convolution
kernelWidth :: forall f o kW dW . KnownNat kW => Conv1d f o kW dW -> Int
kernelWidth _ = fromIntegral (natVal (Proxy :: Proxy kW))

-- | dW: The step of the convolution. Default is 1 in C.
stepSize :: forall f o kW dW . KnownNat dW => Conv1d f o kW dW -> Int
stepSize _ = fromIntegral (natVal (Proxy :: Proxy dW))

-- ========================================================================= --

-- | Type constraints required for temporal convolution
type TemporalConvC s f kW dW o =
  ( KnownNat5 s f kW dW o
  , (s > kW) ~ 'True
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  -- , o ~ ((Div (s - kW) dW) + 1)
  )

-------------------------------------------------------------------------------

-- | If the input sequence is a 2D tensor of dimension (nInputFrame x inputFrameSize), the
-- output sequence will be (nOutputFrame x outputFrameSize) where
--
--    nOutputFrame = (nInputFrame - kW) / dW + 1
conv1d_forward :: TemporalConvC s f kW dW o => Tensor '[s, f] -> Conv1d f o kW dW -> IO (Tensor '[s, o])
conv1d_forward = _conv1d_forward


conv1d_forwardBatch :: TemporalConvC s f kW dW o => Tensor '[b,s,f] -> Conv1d f o kW dW -> IO (Tensor '[b,s,o])
conv1d_forwardBatch = _conv1d_forward

-- | Applies a 1D convolution over an input sequence composed of nInputFrame frames. The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize) or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).
_conv1d_forward :: (KnownNat4 f o kW dW) => Tensor d -> Conv1d f o kW dW -> IO (Tensor d')
_conv1d_forward inp conv = do
  out <- empty
  Dynamic._temporalConvolution_updateOutput
    (asDynamic inp) (asDynamic out)
    (asDynamic (weights conv)) (asDynamic (bias conv))
    (kernelWidth conv) (stepSize conv)
    (featureSize conv) (outputSize conv)
  pure out


conv1d_backward
  :: TemporalConvC s f kW dW o
  => Tensor '[s, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[s, o]                         -- ^ grad output
  -> Tensor '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (Tensor '[s, f])                    -- ^ output
conv1d_backward = _conv1d_backwardBatch

conv1d_backwardBatch
  :: TemporalConvC s f kW dW o
  => Tensor '[b, s, f]                      -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[b, s, o]                      -- ^ grad output
  -> Tensor '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                        -- ^ kW: The kernel width of the convolution
                                            --   dW: The step of the convolution. Default is 1 in C.
  -> IO (Tensor '[b, s, f])                 -- ^ output
conv1d_backwardBatch = _conv1d_backwardBatch

_conv1d_backwardBatch
  :: forall f o kW dW d d'
  . (KnownNat4 f o kW dW)
  => Tensor d
  -> Tensor d'
  -> Tensor '[o, f*kW]
  -> Proxy '(kW, dW)
  -> IO (Tensor d)
_conv1d_backwardBatch input gradOut w _ = do
  gradIn <- empty
  Dynamic._temporalConvolution_updateGradInput
    (asDynamic input) (asDynamic gradOut) (asDynamic gradIn) (asDynamic w)
    (fromIntegral $ natVal (Proxy :: Proxy kW))
    (fromIntegral $ natVal (Proxy :: Proxy dW))
  pure gradIn

-- | TODO
_temporalConvolution_accGradParameters :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Int -> Int -> Double -> IO ()
_temporalConvolution_accGradParameters t0 t1 t2 t3 = Dynamic._temporalConvolution_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- ========================================================================= --

-- | TODO
_temporalRowConvolution_updateOutput :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_updateOutput t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

-- | TODO
_temporalRowConvolution_updateGradInput :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_updateGradInput t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

-- | TODO
_temporalRowConvolution_accGradParameters :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> Double -> IO ()
_temporalRowConvolution_accGradParameters t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)



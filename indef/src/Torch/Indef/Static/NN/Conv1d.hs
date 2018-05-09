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
module Torch.Indef.Static.NN.Conv1d
  ( Conv1d(..)
  , weights
  , bias
  , featureSize
  , kernelWidth
  , stepSize

  , conv1d
  , conv1d_forward
  , conv1d_backward

  , conv1dBatch
  , conv1d_forwardBatch
  , conv1d_backwardBatch

  -- still need work:
  , _temporalConvolution_accGradParameters
  , _temporalRowConvolution_updateOutput
  , _temporalRowConvolution_updateGradInput
  , _temporalRowConvolution_accGradParameters
  ) where

import Numeric.Backprop
import Data.Kind (Type)
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import System.IO.Unsafe
import Torch.Indef.Static.NN.Backprop ()

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
  ( KnownNatDim5 s f kW dW o
  , (s > kW) ~ 'True
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  -- , o ~ ((Div (s - kW) dW) + 1)
  )

-- | Backprop convolution function
conv1d
  :: Reifies s W
  => TemporalConvC seq f kW dW o
  => Conv1d f o kW dW
  -> BVar s (Tensor '[seq, f])
  -> BVar s (Tensor '[seq, o])
conv1d c = liftOp1 . op1 $ \inp ->
  (unsafePerformIO $ conv1d_forward inp c, \out -> unsafePerformIO $ conv1d_backward inp out c)

-- | Backprop convolution function with batching
conv1dBatch
  :: Reifies s W
  => TemporalConvC seq f kW dW o
  => KnownDim b
  => Conv1d f o kW dW
  -> BVar s (Tensor '[b, seq, f])
  -> BVar s (Tensor '[b, seq, o])
conv1dBatch c = liftOp1 . op1 $ \inp ->
  (unsafePerformIO $ conv1d_forwardBatch inp c, \out -> unsafePerformIO $ conv1d_backwardBatch inp out c)

-------------------------------------------------------------------------------

-- | If the input sequence is a 2D tensor of dimension (nInputFrame x inputFrameSize), the
-- output sequence will be (nOutputFrame x outputFrameSize) where
--
--    nOutputFrame = (nInputFrame - kW) / dW + 1
conv1d_forward :: TemporalConvC s f kW dW o => Tensor '[s, f] -> Conv1d f o kW dW -> IO (Tensor '[s, o])
conv1d_forward = _conv1d_forward

-- | Applies a 1D convolution over an input sequence composed of nInputFrame frames. The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize) or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).
conv1d_forwardBatch :: TemporalConvC s f kW dW o => Tensor '[b,s,f] -> Conv1d f o kW dW -> IO (Tensor '[b,s,o])
conv1d_forwardBatch = _conv1d_forward

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
  :: TemporalConvC seq f kW dW o
  => Tensor '[seq, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[seq, o]                         -- ^ grad output
  -> Conv1d f o kW dW                         -- ^ conv1d state
  -> IO (Tensor '[seq, f])                    -- ^ grad input
conv1d_backward = _conv1d_backwardBatch

conv1d_backwardBatch
  :: TemporalConvC s f kW dW o
  => Tensor '[b, s, f]                      -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[b, s, o]                      -- ^ grad output
  -> Conv1d f o kW dW                       -- ^ conv1d state
  -> IO (Tensor '[b, s, f])                 -- ^ output
conv1d_backwardBatch = _conv1d_backwardBatch

_conv1d_backwardBatch
  :: forall f o kW dW d d'
  . (KnownNat4 f o kW dW)
  => Tensor d
  -> Tensor d'
  -> Conv1d f o kW dW
  -> IO (Tensor d)
_conv1d_backwardBatch input gradOut conv = do
  gradIn <- empty
  Dynamic._temporalConvolution_updateGradInput
    (asDynamic input) (asDynamic gradOut) (asDynamic gradIn) (asDynamic (weights conv))
    (kernelWidth conv) (stepSize conv)
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



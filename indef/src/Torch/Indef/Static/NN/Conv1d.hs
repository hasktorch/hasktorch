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
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE InstanceSigs #-}
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
  , conv1d_backwardGradInput
  , conv1d_updGradParams

  , conv1dBatch
  , conv1d_forwardBatch
  , conv1d_backwardGradInputBatch
  , conv1d_updGradParamsBatch

  -- still need work:
  , _temporalRowConvolution_updateOutput
  , _temporalRowConvolution_updateGradInput
  , _temporalRowConvolution_updGradParameters
  ) where

import Numeric.Backprop
import Data.Kind (Type)
import Data.List (intercalate)
import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.CompareT
import Torch.Indef.Static.NN.Backprop ()

import qualified Torch.Indef.Dynamic.NN as Dynamic

newtype Conv1d f o kW dW
  = Conv1d { getTensors :: (Tensor '[o, f*kW], Tensor '[o]) }

instance KnownDim4 f o kW dW => Show (Conv1d f o kW dW) where
  show c = intercalate ","
    [ "Conv1d ("
    ++ "features: " ++ show (featureSize c)
    , " output: "   ++ show (outputSize c)
    , " kernel: "   ++ show (kernelWidth c)
    , " step: "     ++ show (stepSize c)
    ++ ")"
    ]

instance (KnownDim (f*kW), KnownDim o) => Backprop (Conv1d f o kW dW) where
  zero = const . Conv1d $ (constant 0, constant 0)
  one  = const . Conv1d $ (constant 1, constant 1)
  add c0 c1 = Conv1d (weights c0 + weights c1, bias c0 + bias c1)

weights :: Conv1d f o kW dW -> Tensor '[o, f*kW]
weights (Conv1d (w, _)) = w

bias :: Conv1d f o kW dW -> Tensor '[o]
bias (Conv1d (_, b)) = b

featureSize :: forall f o kW dW . KnownDim f => Conv1d f o kW dW -> Int
featureSize _ = fromIntegral (dimVal (dim :: Dim f))

outputSize :: forall f o kW dW . KnownDim o => Conv1d f o kW dW -> Int
outputSize _ = fromIntegral (dimVal (dim :: Dim o))

-- | kW: The kernel width of the convolution
kernelWidth :: forall f o kW dW . KnownDim kW => Conv1d f o kW dW -> Int
kernelWidth _ = fromIntegral (dimVal (dim :: Dim kW))

-- | dW: The step of the convolution. Default is 1 in C.
stepSize :: forall f o kW dW . KnownDim dW => Conv1d f o kW dW -> Int
stepSize _ = fromIntegral (dimVal (dim :: Dim dW))

-- ========================================================================= --

-- | Type constraints required for temporal convolution
type TemporalConvC s f kW dW o =
  ( KnownDim5 s f kW dW o
  , (s > kW) ~ 'True
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  -- , o ~ ((Div (s - kW) dW) + 1)
  )

-- | Backprop convolution function
conv1d
  :: forall s seq f kW dW o
  .  Reifies s W
  => KnownDim (f*kW)
  => TemporalConvC seq f kW dW o
  => Double
  -> BVar s (Conv1d f o kW dW)
  -> BVar s (Tensor '[seq, f])
  -> BVar s (Tensor '[seq, o])
conv1d = _conv1d

-- | Backprop convolution function with batching
conv1dBatch
  :: forall s seq f kW dW o b
  .  Reifies s W
  => KnownDim b
  => KnownDim (f*kW)
  => TemporalConvC seq f kW dW o
  => Double
  -> BVar s (Conv1d f o kW dW)
  -> BVar s (Tensor '[b, seq, f])
  -> BVar s (Tensor '[b, seq, o])
conv1dBatch = _conv1d

-- | Backprop convolution function without specifying constraints or dimensions
_conv1d
  -- :: forall s seq f kW dW o d d'
  :: Reifies s W
  => KnownDim (f*kW)
  => KnownDim4 f o kW dW
  => Dimensions2 d d'
  => Double
  -> BVar s (Conv1d f o kW dW)
  -> BVar s (Tensor d)
  -> BVar s (Tensor d')
_conv1d lr = liftOp2 . op2 $ \c inp ->
  (unsafePerformIO $ _conv1d_forward c inp, \gout ->
      ( unsafePerformIO $ _conv1d_updGradParams c inp gout lr
      , unsafePerformIO $ _conv1d_backwardGradInput c inp gout
      ) )

-------------------------------------------------------------------------------
-- * Functions for temporal convolution

-- | If the input sequence is a 2D tensor of dimension
-- (nInputFrame x inputFrameSize), the output sequence will be
-- (nOutputFrame x outputFrameSize) where
--
--    nOutputFrame = (nInputFrame - kW) / dW + 1
--
conv1d_forward
  :: TemporalConvC s f kW dW o
  => Conv1d f o kW dW
  -> Tensor '[s, f]
  -> IO (Tensor '[s, o])
conv1d_forward = _conv1d_forward

-- | backward pass, computing the gradient input
conv1d_backwardGradInput
  :: TemporalConvC seq f kW dW o
  => Conv1d f o kW dW             -- ^ conv1d state
  -> Tensor '[seq, f]             -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[seq, o]             -- ^ grad output
  -> IO (Tensor '[seq, f])        -- ^ grad input
conv1d_backwardGradInput = _conv1d_backwardGradInput

-- | backward pass, computing the weight and bias parameters
--
-- WARNING: this is _pure_ which may be slow for large tensors.
-- Speeding this up will be in active development as the need arises
-- (see issue hasktorch/hasktorch#85)
conv1d_updGradParams
  :: TemporalConvC s f kW dW o
  => Conv1d f o kW dW       -- ^ input state of conv1d (which includes weights and bias)
  -> Tensor '[s, f]         -- ^ input tensor
  -> Tensor '[s, o]         -- ^ output gradient
  -> Double                 -- ^ scale
  -> IO (Conv1d f o kW dW)  -- ^ gradient of (weights, bias)
conv1d_updGradParams = _conv1d_updGradParams

-------------------------------------------------------------------------------
-- * Functions for temporal convolution with a batch dimension

-- | Applies a 1D convolution over an input sequence composed of nInputFrame
-- frames. The input tensor in forward(input) is expected to be a 2D tensor
-- (nInputFrame x inputFrameSize) or a 3D tensor
-- (nBatchFrame x nInputFrame x inputFrameSize).
conv1d_forwardBatch
  :: TemporalConvC s f kW dW o
  => Conv1d f o kW dW
  -> Tensor '[b,s,f]
  -> IO (Tensor '[b,s,o])
conv1d_forwardBatch = _conv1d_forward

-- 'conv1d_backwardGradInput' with a batch dimension
conv1d_backwardGradInputBatch
  :: TemporalConvC s f kW dW o
  => KnownDim b
  => Conv1d f o kW dW             -- ^ conv1d state
  -> Tensor '[b, s, f]            -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[b, s, o]            -- ^ grad output
  -> IO (Tensor '[b, s, f])       -- ^ output
conv1d_backwardGradInputBatch = _conv1d_backwardGradInput

-- 'conv1d_updGradParams' with a batch dimension
conv1d_updGradParamsBatch
  :: TemporalConvC s f kW dW o
  => KnownDim b
  => Conv1d f o kW dW             -- ^ conv1d state
  -> Tensor '[b, s, f]            -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> Tensor '[b, s, o]            -- ^ grad output
  -> Double                       -- ^ scale
  -> IO (Conv1d f o kW dW)        -- ^ output
conv1d_updGradParamsBatch = _conv1d_updGradParams



-------------------------------------------------------------------------------
-- * helper functions

-- | forward pass without locking down the tensor dimensions
_conv1d_forward :: (KnownDim4 f o kW dW) => Conv1d f o kW dW -> Tensor d -> IO (Tensor d')
_conv1d_forward conv inp = do
  out <- empty
  Dynamic._temporalConvolution_updateOutput
    (asDynamic inp) (asDynamic out)
    (asDynamic (weights conv)) (asDynamic (bias conv))
    (kernelWidth conv) (stepSize conv)
    (featureSize conv) (outputSize conv)
  pure out

-- | backward pass, computing the gradient input, without locking down the tensor dimensions
_conv1d_backwardGradInput
  :: forall f o kW dW inputDim goutDim
  . (KnownDim4 f o kW dW)
  => Dimensions2 inputDim goutDim
  => Conv1d f o kW dW
  -> Tensor inputDim
  -> Tensor goutDim
  -> IO (Tensor inputDim)
_conv1d_backwardGradInput conv input gradOut = do
  gradIn <- empty
  Dynamic._temporalConvolution_updateGradInput
    (asDynamic input)  (asDynamic gradOut)
    (asDynamic gradIn) (asDynamic (weights conv))
    (kernelWidth conv) (stepSize conv)
  pure gradIn

-- | backward pass, computing the weight updates, without locking down the tensor dimensions
_conv1d_updGradParams
  :: forall f o kW dW inputDim gOutDim
  . (KnownDim4 f o kW dW, Dimensions2 inputDim gOutDim)
  => Conv1d f o kW dW       -- ^ input state of conv1d (which includes weights and bias)
  -> Tensor inputDim        -- ^ input tensor
  -> Tensor gOutDim         -- ^ output gradient
  -> Double                 -- ^ scale
  -> IO (Conv1d f o kW dW)  -- ^ gradient of (weights, bias)
_conv1d_updGradParams c@(Conv1d (w, b)) input gout scale = do
  -- FIXME: this is _not going to scale well_ and coming up with a mutable version of Conv layers will be nessecary
  w' <- copy w
  b' <- copy b
  Dynamic._temporalConvolution_accGradParameters
    (asDynamic input) (asDynamic gout) (asDynamic w') (asDynamic b')
    (kernelWidth c) (stepSize c) scale
  pure $ Conv1d (w', b')

-- ========================================================================= --

-- | TODO
_temporalRowConvolution_updateOutput :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_updateOutput t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

-- | TODO
_temporalRowConvolution_updateGradInput :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_updateGradInput t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

-- | TODO
_temporalRowConvolution_updGradParameters :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d -> Tensor d -> Int -> Int -> Int -> Bool -> Double -> IO ()
_temporalRowConvolution_updGradParameters t0 t1 t2 t3 t4 t5 = Dynamic._temporalRowConvolution_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)



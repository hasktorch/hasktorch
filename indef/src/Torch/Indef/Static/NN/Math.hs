-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Math
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Static.NN.Math where

import Data.Singletons.Prelude.Ord (type (<))
import Data.Singletons.Prelude.List
import Numeric.Dimensions hiding (Length)
import Numeric.Backprop
import System.IO.Unsafe
import Torch.Indef.Static.Tensor.Math.Reduce (sumall, maxall)
import Torch.Indef.Static.Tensor.Math.Pointwise ((^*^), (^-^))
import Torch.Indef.Static.Tensor.Math.Pairwise ((^-), (^/))

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN as Dynamic
import qualified Torch.Indef.Static.Tensor.Math.Pointwise.Floating as Torch

-- | abs forward pass (updates the output tensor)
abs_updateOutput :: Tensor d -> IO (Tensor d)
abs_updateOutput i = empty >>= \o -> Dynamic._abs_updateOutput (asDynamic i) (asDynamic o) >> pure o

-- | abs backward-update (updates the layer and bias tensors)
abs_updateGradInput
  :: (Product d ~ Product d')
  => Tensor d        -- ^ input
  -> Tensor d'       -- ^ gradOutput
  -> IO (Tensor d)   -- ^ gradInput
abs_updateGradInput i go =
  empty >>= \gi -> Dynamic._abs_updateGradInput (asDynamic i) (asDynamic go) (asDynamic gi) >> pure gi

-- |  sqrt forward pass (updates the output tensor)
_sqrt_updateOutput :: Tensor d -> Tensor d -> Double -> IO ()
_sqrt_updateOutput t0 t1 = Dynamic._sqrt_updateOutput (asDynamic t0) (asDynamic t1)
-- |  sqrt backward-update (updates the layer and bias tensors)
_sqrt_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_sqrt_updateGradInput t0 t1 t2 t3 = Dynamic._sqrt_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  square forward pass (updates the output tensor)
_square_updateOutput :: Tensor d -> Tensor d -> IO ()
_square_updateOutput t0 t1 = Dynamic._square_updateOutput (asDynamic t0) (asDynamic t1)
-- |  square backward-update (updates the layer and bias tensors)
_square_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_square_updateGradInput t0 t1 t2 = Dynamic._square_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  logSigmoid forward pass (updates the output tensor)
_logSigmoid_updateOutput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_logSigmoid_updateOutput t0 t1 t2 = Dynamic._logSigmoid_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  logSigmoid backward-update (updates the layer and bias tensors)
_logSigmoid_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_logSigmoid_updateGradInput t0 t1 t2 t3 = Dynamic._logSigmoid_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  sigmoid forward pass (updates the output tensor)
_sigmoid_updateOutput :: Tensor d -> Tensor d -> IO ()
_sigmoid_updateOutput t0 t1 = Dynamic._sigmoid_updateOutput (asDynamic t0) (asDynamic t1)
-- |  sigmoid backward-update (updates the layer and bias tensors)
_sigmoid_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_sigmoid_updateGradInput t0 t1 t2 = Dynamic._sigmoid_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-------------------------------------------------------------------------------

{-# WARNING softmax, softmaxN "softmax gradients may not propagate correctly. This may require updating hasktorch/ATen. In the meantime consider using logSoftMax which is stable." #-}

-- | one dimensional version of 'softmaxN'
softmax
  :: KnownDim n
  => Reifies s W
  => BVar s (Tensor '[n])    -- ^ input
  -> BVar s (Tensor '[n])    -- ^ output
softmax = softmaxN (dim :: Dim 0)

-- | run a threshold function againts two BVar variables
softmaxN
  :: forall s i d
  .  Reifies s W
  => i < Length d ~ 'True
  => Dimensions d
  => Dim i                -- ^ dimension to softmax over
  -> BVar s (Tensor d)    -- ^ input
  -> BVar s (Tensor d)    -- ^ output
softmaxN d = liftOp1 . op1 $ \inp ->
  let
    idim = fromIntegral (dimVal d)
    out = updateOutput inp idim
  in
    (out, \gout -> updateGradInput inp gout out idim)
 where
  updateOutput :: Dimensions d => Tensor d -> Integer -> Tensor d
  updateOutput inp i = unsafeDupablePerformIO . withEmpty $ \out -> do
    Dynamic._softMax_updateOutput
      (asDynamic inp)
      (asDynamic out)
      i

  -- FIXME: There seems to be a bug in softmax. In the mean time, using a translation
  -- of the raw THNN code:
  -- https://github.com/hasktorch/ATen/blob/hasktorch-expand/src/THNN/generic/SoftMax.c#L111
  updateGradInput
    :: Dimensions d
    => Tensor d  -- input
    -> Tensor d  -- gradOutput
    -> Tensor d  -- output
    -> Integer   -- dimension
    -> Tensor d  -- gradInput
  -- updateGradInput inp gout out d = unsafeDupablePerformIO $ do
    -- let mult = gout ^*^ out
    -- pure $ mult ^*^ (gout ^- acc2real (sumall mult))

  -- NOTE: This would have been the original codebase.
  updateGradInput inp gout out d = unsafeDupablePerformIO . withEmpty $ \gin -> do
    Dynamic._softMax_updateGradInput
      (asDynamic inp)  -- input
      (asDynamic gout) -- gradOutput
      (asDynamic gin)  -- gradInput
      (asDynamic out)  -- output
      d                -- dimension


-- | run a threshold function againts two BVar variables
logSoftMax
  :: KnownDim n
  => Reifies s W
  => BVar s (Tensor '[n])    -- ^ input
  -> BVar s (Tensor '[n])    -- ^ output
logSoftMax = logSoftMaxN (dim :: Dim 0)

-- | run a threshold function againts two BVar variables
logSoftMaxN
  :: forall s i d
  .  Reifies s W
  => i < Length d ~ 'True
  => Dimensions d
  => Dim i                -- ^ dimension to logSoftMax over
  -> BVar s (Tensor d)    -- ^ input
  -> BVar s (Tensor d)    -- ^ output
logSoftMaxN i = liftOp1 . op1 $ \inp ->
  let out = updateOutput inp i
  in (updateOutput inp i, \gout -> updateGradInput inp gout out i)
 where
  updateOutput :: Tensor d -> Dim i -> Tensor d
  updateOutput inp i = unsafeDupablePerformIO . withEmpty $ \out ->
    Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) (fromIntegral $ dimVal i)

  updateGradInput
    :: Tensor d  -- input
    -> Tensor d  -- gradOutput
    -> Tensor d  -- output
    -> Dim i     -- dimension
    -> Tensor d  -- gradInput
  updateGradInput inp gout out i = unsafeDupablePerformIO . withEmpty $ \gin ->
    Dynamic._logSoftMax_updateGradInput
      (asDynamic inp)             -- input
      (asDynamic gout)            -- gradOutput
      (asDynamic gin)             -- gradInput
      (asDynamic out)             -- output
      (fromIntegral $ dimVal i)   -- dimension


-- |  softPlus forward pass (updates the output tensor)
_softPlus_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> IO ()
_softPlus_updateOutput t0 t1 = Dynamic._softPlus_updateOutput (asDynamic t0) (asDynamic t1)
-- |  softPlus backward-update (updates the layer and bias tensors)
_softPlus_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
_softPlus_updateGradInput t0 t1 t2 t3 = Dynamic._softPlus_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  softShrink forward pass (updates the output tensor)
_softShrink_updateOutput :: Tensor d -> Tensor d -> Double -> IO ()
_softShrink_updateOutput t0 t1 = Dynamic._softShrink_updateOutput (asDynamic t0) (asDynamic t1)
-- |  softShrink backward-update (updates the layer and bias tensors)
_softShrink_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_softShrink_updateGradInput t0 t1 t2 = Dynamic._softShrink_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  tanh forward pass (updates the output tensor)
_tanh_updateOutput :: Tensor d -> Tensor d -> IO ()
_tanh_updateOutput t0 t1 = Dynamic._tanh_updateOutput (asDynamic t0) (asDynamic t1)
-- |  tanh backward-update (updates the layer and bias tensors)
_tanh_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_tanh_updateGradInput t0 t1 t2 = Dynamic._tanh_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  hardTanh forward pass (updates the output tensor)
_hardTanh_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_hardTanh_updateOutput t0 t1 = Dynamic._hardTanh_updateOutput (asDynamic t0) (asDynamic t1)

-- |  hardTanh backward-update (updates the layer and bias tensors)
_hardTanh_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_hardTanh_updateGradInput t0 t1 t2 = Dynamic._hardTanh_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)


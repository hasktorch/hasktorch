-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Criterion
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Static.NN.Criterion where

import Control.Arrow ((&&&))
import GHC.TypeLits
import Numeric.Dimensions
import Numeric.Backprop
import System.IO.Unsafe

import Torch.Indef.Static.Tensor
import Torch.Indef.Types
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN.Criterion as Dynamic


-- | absCriterion forward pass (updates the output tensor)
_absCriterion_updateOutput
  :: Tensor d     -- ^ input
  -> Tensor d'    -- ^ target
  -> Tensor d''   -- ^ output
  -> Bool    -- ^ size average
  -> Bool    -- ^ reduce
  -> IO ()
_absCriterion_updateOutput i t o = Dynamic._absCriterion_updateOutput (asDynamic i) (asDynamic t) (asDynamic o)

-- | absCriterion backward-update (updates the layer and bias tensors)
_absCriterion_updateGradInput
  :: Tensor d     -- ^ input
  -> Tensor d'    -- ^ target
  -> Tensor d''   -- ^ gradOutput
  -> Tensor d''   -- ^ gradInput
  -> Bool    -- ^ size average
  -> Bool    -- ^ reduce
  -> IO ()
_absCriterion_updateGradInput i t go gi = Dynamic._absCriterion_updateGradInput (asDynamic i) (asDynamic t) (asDynamic go) (asDynamic gi)

-- | Binary cross-entropy for Sigmoid (two-class version of ClassNLLCriterion)
--
-- Creates a criterion that measures the Binary Cross Entropy between the target and
-- the output:
-- @
--   loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
-- @
-- or in the case of the weights argument being specified:
-- @
--   loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
-- @
-- This is used for measuring the error of a reconstruction in for example an
-- auto-encoder. Note that the outputs o[i] should be numbers between 0 and 1,
-- for instance, the output of an nn.Sigmoid layer and should be interpreted as
-- the probability of predicting t[i] = 1. Note t[i] can be either 0 or 1.
--
-- By default, the losses are averaged for each minibatch over observations as
-- well as over dimensions. However, if the field sizeAverage is set to false,
-- the losses are instead summed.
bCECriterion
  :: forall s n
  . (Reifies s W, KnownNat n, KnownDim n)
  => Bool                          -- ^ sizeAverage (TODO: swap this out with 'Reduction')
  -> Bool                          -- ^ reduce (TODO: swap this out with 'Reduction')
  -> Maybe (Tensor '[n])           -- ^ weights
  -> Tensor '[n]                   -- ^ target
  -> BVar s (Tensor '[n])          -- ^ input
  -> BVar s (Tensor '[1])          -- ^ output
bCECriterion savg r mw tar = liftOp1 . op1 $ (updateOutput &&& updateGradInput)
  where
    updateOutput
      :: Tensor '[n]          -- input
      -> Tensor '[1]          -- output
    updateOutput i = unsafeDupablePerformIO . withNew $ \o ->
      Dynamic._bCECriterion_updateOutput
        (asDynamic i) (asDynamic tar) (asDynamic o) savg (asDynamic <$> mw) r

    updateGradInput
      :: Tensor '[n]          -- input
      -> Tensor '[1]          -- grad output
      -> Tensor '[n]          -- grad input
    updateGradInput i go = unsafeDupablePerformIO . withNew $ \gi ->
      Dynamic._bCECriterion_updateGradInput
        (asDynamic i) (asDynamic tar) (asDynamic go) (asDynamic gi) savg (asDynamic <$> mw) r


-- | marginCriterion forward pass (updates the output tensor)
_marginCriterion_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Bool -> Double -> IO ()
_marginCriterion_updateOutput t0 t1 t2 = Dynamic._marginCriterion_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- | marginCriterion backward-update (updates the layer and bias tensors)
_marginCriterion_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Bool -> Double -> IO ()
_marginCriterion_updateGradInput t0 t1 t2 = Dynamic._marginCriterion_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- | softMarginCriterion forward pass (updates the output tensor)
_softMarginCriterion_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_softMarginCriterion_updateOutput t0 t1 t2 = Dynamic._softMarginCriterion_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- | softMarginCriterion backward-update (updates the layer and bias tensors)
_softMarginCriterion_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_softMarginCriterion_updateGradInput t0 t1 t2 t3 = Dynamic._softMarginCriterion_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- | mSECriterion forward pass (updates the output tensor)
_mSECriterion_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_mSECriterion_updateOutput t0 t1 t2 = Dynamic._mSECriterion_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- | mSECriterion backward-update (updates the layer and bias tensors)
_mSECriterion_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_mSECriterion_updateGradInput t0 t1 t2 t3 = Dynamic._mSECriterion_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- | The <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence Kullback-Leibler divergence> Loss
--
-- KL divergence is a useful distance measure for continuous distributions and
-- is often useful when performing direct regression over the space of
-- (discretely sampled) continuous output distributions.
--
-- As with 'NLLLoss', the input given is expected to contain log-probabilities,
-- however unlike 'ClassNLLLoss', input is not restricted to a 2D Tensor,
-- because the criterion is applied element-wise.
--
-- This criterion expects a target Tensor of the same size as the input Tensor.
--
-- The loss can be described as:
-- @
--   \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
--   l_n = y_n \odot \left( \log y_n - x_n \right),
-- @
--
-- where @N@ is the batch size. If @reduce@ is @True@, then:
-- @
--   \begin{split}\ell(x, y) = \begin{cases}
--       \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
--       \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
--   \end{cases}\end{split}
-- @
--
-- By default, the losses are averaged for each minibatch over observations as
-- well as over dimensions. However, if the field @size_average@ is set to
-- @False@, the losses are instead summed.


-- | distKLDivCriterion forward pass (updates the output tensor)
_distKLDivCriterion_updateOutput
  :: Tensor d  -- ^ output tensor to update
  -> Tensor d  -- ^ input tensor
  -> Tensor d  -- ^ comparative tensor
  -> Bool      -- ^ size_average
  -> Bool      -- ^ reduce
  -> IO ()
_distKLDivCriterion_updateOutput t0 t1 t2 = Dynamic._distKLDivCriterion_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- | distKLDivCriterion backward-update (updates the layer and bias tensors)
_distKLDivCriterion_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_distKLDivCriterion_updateGradInput t0 t1 t2 t3 = Dynamic._distKLDivCriterion_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- | smoothL1Criterion forward pass (updates the output tensor)
_smoothL1Criterion_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_smoothL1Criterion_updateOutput t0 t1 t2 = Dynamic._smoothL1Criterion_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- | smoothL1Criterion backward-update (updates the layer and bias tensors)
_smoothL1Criterion_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Bool -> Bool -> IO ()
_smoothL1Criterion_updateGradInput t0 t1 t2 t3 = Dynamic._smoothL1Criterion_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- | l1Cost forward pass (updates the output tensor)
_l1Cost_updateOutput :: Tensor d -> Tensor d -> IO ()
_l1Cost_updateOutput t0 t1 = Dynamic._l1Cost_updateOutput (asDynamic t0) (asDynamic t1)
-- | l1Cost backward-update (updates the layer and bias tensors)
_l1Cost_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_l1Cost_updateGradInput t0 t1 t2 = Dynamic._l1Cost_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

{-
c_ClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_ClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_SpatialClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_SpatialClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_MultiLabelMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MultiLabelMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MultiMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()
c_MultiMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()
-}

{-
c_LookupTable_renorm :: Ptr CNNState -> Ptr CIndexTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_IndexLinear_updateOutput :: Ptr CNNState -> Ptr CIndexTensor -> CLLong -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_IndexLinear_accGradParameters :: Ptr CNNState -> Ptr CIndexTensor -> CLLong -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_IndexLinear_accUpdateGradParameters :: Ptr CNNState -> Ptr CIndexTensor -> CLLong -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_IndexLinear_updateParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> CLLong -> CDouble -> CDouble -> IO ()

-}

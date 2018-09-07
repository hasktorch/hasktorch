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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Criterion where

import Control.Arrow ((&&&))
import GHC.TypeLits
import Numeric.Dimensions
import Numeric.Backprop
import System.IO.Unsafe
import Data.Singletons.Prelude hiding (All, type (*), type (-), type (+))

import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Types
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN.Criterion as Dynamic
import qualified Torch.Sig.Types.Global as Ix

import Torch.Indef.Static.Tensor.Random.THC


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
bCECriterion'
  :: forall s n
  . (Reifies s W, KnownNat n, KnownDim n)
  => Bool                          -- ^ sizeAverage (TODO: swap this out with 'Reduction')
  -> Bool                          -- ^ reduce (TODO: swap this out with 'Reduction')
  -> Maybe (Tensor '[n])           -- ^ weights
  -> Tensor '[n]                   -- ^ target
  -> BVar s (Tensor '[n])          -- ^ input
  -> BVar s (Tensor '[1])          -- ^ output
bCECriterion' savg r mw tar = liftOp1 . op1 $ (updateOutput &&& updateGradInput)
  where
    {-# NOINLINE updateOutput #-}
    updateOutput
      :: Tensor '[n]          -- input
      -> Tensor '[1]          -- output
    updateOutput i = unsafePerformIO . withNew $ \o ->
      Dynamic._bCECriterion_updateOutput
        (asDynamic i) (asDynamic tar) (asDynamic o) savg (asDynamic <$> mw) r

    {-# NOINLINE updateGradInput #-}
    updateGradInput
      :: Tensor '[n]          -- input
      -> Tensor '[1]          -- grad output
      -> Tensor '[n]          -- grad input
    updateGradInput i go = unsafePerformIO . withNew $ \gi ->
      Dynamic._bCECriterion_updateGradInput
        (asDynamic i) (asDynamic tar) (asDynamic go) (asDynamic gi) savg (asDynamic <$> mw) r

bCECriterion
  :: (Reifies s W, KnownNat n, KnownDim n)
  => Tensor '[n]                   -- ^ target
  -> BVar s (Tensor '[n])          -- ^ input
  -> BVar s (Tensor '[1])          -- ^ output
bCECriterion = bCECriterion' True True Nothing

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

-- | MSECriterion
--
-- Creates a criterion that measures the mean squared error between n elements
-- in the input x and output y:
--
-- @
--   loss(x, y) = 1/n \sum |x_i - y_i|^2 .
-- @
--
-- If x and y are d-dimensional Tensors with a total of n elements, the sum
-- operation still operates over all the elements, and divides by n. The two
-- Tensors must have the same number of elements (but their sizes might be
-- different).
--
-- The division by n can be avoided if one sets the internal variable sizeAverage
-- to false:
--
-- criterion = nn.MSECriterion()
-- criterion.sizeAverage = false
--
-- By default, the losses are averaged over observations for each minibatch.
-- However, if the field sizeAverage is set to false, the losses are instead
-- summed.
mSECriterion' :: forall s bs d d' reduce out
  . Reifies s W
  => All Dimensions '[d', d, out]
  => KnownDim bs
  => d ~ (bs :+ d') -- must have minibatch
  => out ~ (If reduce '[1] d)
  => Bool             -- ^ size_average:
                      --     By default, the losses are averaged over each loss element in the batch.
                      --     Note that for some losses, there multiple elements per sample.
                      --     If the field size_average is set to False, the losses are instead
                      --     summed for each minibatch. Ignored when reduce is False. Default: True
  -> SBool reduce     -- ^ reduce:
                      --     By default, the losses are averaged or summed over observations for each
                      --     minibatch depending on size_average. When reduce is False, returns a loss
                      --     per batch element instead and ignores size_average. Default: True
  -> Tensor d
  -> BVar s (Tensor d)
  -> BVar s (Tensor out)
mSECriterion' sizeAvg reduce target = liftOp1 . op1 $ \i -> (updateOutput i, \gout -> updateGradInput i gout)
  where
    -- mSECriterion forward pass (updates the output tensor)
    {-# NOINLINE updateOutput #-}
    updateOutput :: Tensor d -> Tensor out
    updateOutput i = unsafePerformIO $ do
      o <- new
      Dynamic._mSECriterion_updateOutput (asDynamic i) (asDynamic target) (asDynamic o) sizeAvg (fromSing reduce)
      pure o

    -- mSECriterion backward-update (updates the layer and bias tensors)
    {-# NOINLINE updateGradInput #-}
    updateGradInput :: Tensor d -> Tensor out -> Tensor d
    updateGradInput i gout = unsafePerformIO $ do
      gin <- new
      Dynamic._mSECriterion_updateGradInput (asDynamic i) (asDynamic target) (asDynamic gout) (asDynamic gin) sizeAvg (fromSing reduce)
      pure gin

mSECriterion :: forall s bs d d'
  . Reifies s W
  => All Dimensions '[d', d]
  => KnownDim bs
  => d ~ (bs :+ d')
  => Tensor d
  -> BVar s (Tensor d)
  -> BVar s (Tensor '[1])
mSECriterion = mSECriterion' True (sing :: SBool 'True)


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

-- | ClassNLLCriterion
--
-- The negative log likelihood (NLL) criterion. It is useful to train a classification problem with n classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.
--
-- The input given through a forward() is expected to contain log-probabilities of each class: input has to be a 1D Tensor of size n. Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftMax layer in the last layer of your neural network. You may use CrossEntropyCriterion instead, if you prefer not to add an extra layer to your network. This criterion expects a class index (1 to the number of class) as target when calling forward(input, target) and backward(input, target).
--
-- The loss can be described as:
--
-- loss(x, class) = -x[class]
--
-- or in the case of the weights argument, it is specified as follows:
--
-- loss(x, class) = -weights[class] * x[class]
--
-- or in the case of the ignoreIndex argument:
--
-- loss(x, class) = class != ignoreIndex ? -weights[class] * x[class] : 0
--
-- Indeed, the ignoreIndex (defaults to -100) specifies a value for targets to be ignored. The commensurate gradInput for that target will be zero. When sizeAverage=true (the default), the gradInput and output are averaged over non-ignored targets.
--
-- Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when calculating losses in non-batch mode.
--
-- The following is a code fragment showing how to make a gradient step given an input x, a desired output y (an integer 1 to n, in this case n = 2 classes), a network mlp and a learning rate learningRate:
--
-- function gradUpdate(mlp, x, y, learningRate)
--    local criterion = nn.ClassNLLCriterion()
--    local pred = mlp:forward(x)
--    local err = criterion:forward(pred, y)
--    mlp:zeroGradParameters()
--    local t = criterion:backward(pred, y)
--    mlp:backward(x, t)
--    mlp:updateParameters(learningRate)
-- end
--
-- By default, the losses are averaged over observations for each minibatch. However, if the argument sizeAverage is set to false, the losses are instead summed for each minibatch.
-- FIXME: add batch dimension
classNLLCriterion'
  :: forall s i sz ps
  . (Reifies s W, All KnownDim '[sz, ps])
  => Integer                    -- int64_t ignore_index,
  -> Bool                       -- bool sizeAverage,
  -> Bool                       -- bool reduce
  -> IndexTensor '[sz]          -- THIndexTensor *target. _not_ a one-hot encoded vector.
  -- -> Maybe Dynamic           -- THTensor *weights,
  -> BVar s (Tensor '[sz, ps])  -- THTensor *input,
  -> BVar s (Tensor '[1])       -- THTensor *output,
classNLLCriterion' ix szAvg reduce target = liftOp1 . op1 $ \inp ->
  let
    (out, total_weight) = updateOutput inp target szAvg Nothing ix reduce
  in
    (out, \gout -> updateGradInput inp target gout szAvg Nothing total_weight ix reduce)
  where
    {-# NOINLINE updateOutput #-}
    updateOutput
      :: Tensor '[sz, ps]            -- THTensor *input,
      -> IndexTensor '[sz]           -- THIndexTensor *target,
      -> Bool                        -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])    -- THTensor *weights,
      -> Integer                     -- int64_t ignore_index,
      -> Bool                        -- bool reduce
      -> (Tensor '[1], Tensor '[1])
    updateOutput inp tar szAvg mws ix reduce = unsafePerformIO $ do
      out <- new
      let total_weight = constant 1  -- https://github.com/torch/nn/commit/3585e827eb65d071272a4aa4fab567b0b1eeee54#diff-1aa6a505cf16ad0e59498ada8432afb5
      Dynamic._ClassNLLCriterion_updateOutput (asDynamic inp) (Ix.longAsDynamic tar) (asDynamic out)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce
      pure (out, total_weight)

    {-# NOINLINE updateGradInput #-}
    updateGradInput
      :: Tensor '[sz, ps]          -- THTensor *input,
      -> IndexTensor '[sz]         -- THIndexTensor *target,
      -> Tensor '[1]               -- THTensor *gradOutput,
      -> Bool                      -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])  -- THTensor *weights,
      -> Tensor '[1]               -- THTensor *total_weight,
      -> Integer                   -- int64_t ignore_index,
      -> Bool                      -- bool reduce
      -> Tensor '[sz, ps]
    updateGradInput inp tar gout szAvg mws total_weight ix reduce = unsafePerformIO . withEmpty $ \gin ->
      Dynamic._ClassNLLCriterion_updateGradInput (asDynamic inp) (Ix.longAsDynamic tar) (asDynamic gout) (asDynamic gin)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce


-- | Due to behaviour of backend code, it is nessecary to set sizeAverage to False in Non-Batch mode.
classNLLCriterion
  :: (Reifies s W, All KnownDim '[n, c])
  => IndexTensor '[n]            -- THIndexTensor *target,
  -> BVar s (Tensor '[n, c])     -- THTensor *input,
  -> BVar s (Tensor '[1])        -- THTensor *output,
classNLLCriterion = classNLLCriterion' (-100) True True

{-
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

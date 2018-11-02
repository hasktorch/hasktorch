-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Models.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Helper functions which might end up migrating to the -indef codebase
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE CPP #-}

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

module Torch.Initialization
  ( newLinear
  , newConv2d
  , xavierUniformWith_
  , xavierUniform_
  , xavierNormalWith_
  , xavierNormal_

  , Activation(..)
  , FanMode(..)
  , kaimingUniformWith_
  , kaimingUniform_
  , kaimingNormalWith_
  , kaimingNormal_
  ) where

import Data.Maybe (fromMaybe)
import Data.Function ((&))
import GHC.Generics
import Prelude as P
import Data.Singletons.Prelude hiding (type (*), All)
import Data.Singletons.Prelude.List hiding (All)
import Numeric.Dimensions
import Control.Exception.Safe (throwString)

import Torch.Double
import qualified Torch.Double as Torch
import Torch.Double.NN.Linear (Linear(..))
import qualified Torch.Double.NN.Conv2d as NN


-- Layer initialization: These depend on random functions which are not unified and, thus,
-- it's a little trickier to fold these back into their respective NN modules.

-- | initialize a new linear layer
newLinear :: forall o i . All KnownDim '[i,o] => Generator -> IO (Linear i o)
newLinear g = fmap Linear $ do
  let w = new
  kaimingUniformWith_ (LeakyReluFn (Just $ P.sqrt 5)) FanIn g w

  let
    fanin = calculateCorrectFan w FanIn
    bound = 1 / P.sqrt fanin
    bias = new
    Just pair = ord2Tuple (-bound, bound)
  _uniform bias g pair
  pure (w, bias)


-- | initialize a new conv2d layer
newConv2d :: forall o i kH kW . All KnownDim '[i,o,kH,kW,kH*kW] => Generator -> IO (Conv2d i o '(kH,kW))
newConv2d g = fmap Conv2d $ do
  let w = new
  kaimingUniformWith_ (LeakyReluFn (Just $ P.sqrt 5)) FanIn g w

  let
    fanin = calculateCorrectFan w FanIn
    bound = 1 / P.sqrt fanin
    bias = new
    Just pair = ord2Tuple (-bound, bound)
  _uniform bias g pair
  pure (w, bias)


data Activation
  -- linear functions
  = LinearFn   -- ^ Linear activation
  | Conv1dFn   -- ^ Conv1d activation
  | Conv2dFn   -- ^ Conv2d activation
  | Conv3dFn   -- ^ Conv3d activation
  | Conv1dTFn  -- ^ Conv1d transpose activation
  | Conv2dTFn  -- ^ Conv2d transpose activation
  | Conv3dTFn  -- ^ Conv3d transpose activation

  -- non-linear
  | SigmoidFn
  | TanhFn
  | ReluFn
  | LeakyReluFn (Maybe Double)
  deriving (Eq, Show)

isLinear :: Activation -> Bool
isLinear = \case
  LinearFn  -> True
  Conv1dFn  -> True
  Conv2dFn  -> True
  Conv3dFn  -> True
  Conv1dTFn -> True
  Conv2dTFn -> True
  Conv3dTFn -> True
  otherwise -> False



-- |
-- Return the recommended gain value for the given nonlinearity function.
-- The values are as follows:
-- ================= ====================================================
-- nonlinearity      gain
-- ================= ====================================================
-- Linear / Identity :math:`1`
-- Conv{1,2,3}D      :math:`1`
-- Sigmoid           :math:`1`
-- Tanh              :math:`\frac{5}{3}`
-- ReLU              :math:`\sqrt{2}`
-- Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
-- ================= ====================================================
-- Args:
--     param: optional parameter for the non-linear function
-- Examples:
--     >>> gain = nn.init.calculate_gain('leaky_relu')
calculateGain
  :: Activation  -- ^ the non-linear function (`nn.functional` name)
  -- param=None
  -> Double
calculateGain f
  | isLinear f = 1
  | otherwise =
    case f of
      SigmoidFn -> 1
      TanhFn -> 5 / 3
      ReluFn -> P.sqrt 2
      LeakyReluFn mslope -> P.sqrt $ 2 / (1 + fromMaybe 0.001 mslope ** 2)

fanInAndFanOut
  :: forall outs i o
  .  (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Tensor (i:+o:+outs)
  -> (Double, Double)
fanInAndFanOut = const (fan_in, fan_out)
 where
  fan_in  = fromIntegral (dimVal (dim :: Dim o)) * rest
  fan_out = fromIntegral (dimVal (dim :: Dim i)) * rest
  rest    = fromIntegral (dimVal (dim :: Dim (Product outs)))

-- |
-- Fills the input `Tensor` with values according to the method
-- described in "Understanding the difficulty of training deep feedforward
-- neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
-- distribution. The resulting tensor will have values sampled from
-- :math:`\mathcal{U}(-a, a)` where
-- .. math::
--     a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
-- Also known as Glorot initialization.
-- Examples:
--     -set -XScopedTypeVariables
--     w :: Tensor '[3, 5] <- torch.new
--     xavierUniformWith_ w (calculate_gain Relu)
xavierUniformWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => HsReal              -- ^ gain: an optional scaling factor
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
xavierUniformWith_ = xavierDistributedWith_ $ \g pstd t -> do
  let std = positiveValue pstd
      a = P.sqrt 3 * std   -- Calculate uniform bounds from standard deviation
      Just pair = ord2Tuple (-a, a)
  _uniform t g pair

-- | xavierUniformWith_ with default of gain = 1
xavierUniform_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
xavierUniform_ = xavierUniformWith_ 1

xavierNormalWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => HsReal              -- ^ gain: an optional scaling factor
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
xavierNormalWith_ = xavierDistributedWith_ $ \g std t -> _normal t g 0 std

-- | 'xavierNormalWith_' with default of gain = 1
xavierNormal_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
xavierNormal_ = xavierNormalWith_ 1


xavierDistributedWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => (Generator -> Positive HsReal -> Tensor (i:+o:+outs) -> IO ())
  -> HsReal              -- ^ gain: an optional scaling factor
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
xavierDistributedWith_ distribution gain g tensor = do
  let
    (fan_in, fan_out) = fanInAndFanOut tensor
    mstd = gain * P.sqrt(2 / (fan_in + fan_out))
  case positive mstd of
    Just std -> distribution g std tensor
    Nothing -> throwString $
      "standard deviation is not positive. Found: " ++ show mstd ++ ", most likely the gain is negative, which is incorrect: " ++ show gain



data FanMode = FanIn | FanOut
  deriving (Eq, Ord, Show)


calculateCorrectFan
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Tensor (i:+o:+outs) -> FanMode -> Double
calculateCorrectFan t = \case
  FanIn -> fan_in
  FanOut -> fan_out
 where
  (fan_in, fan_out) = fanInAndFanOut t


-- |
-- Fills the input `Tensor` with values according to the method
-- described in "Delving deep into rectifiers: Surpassing human-level
-- performance on ImageNet classification" - He, K. et al. (2015), using a
-- uniform distribution. The resulting tensor will have values sampled from
-- :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
-- .. math::
--     \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}
-- Also known as He initialization.
-- Args:
--     tensor: an n-dimensional `torch.Tensor`
--     a: the negative slope of the rectifier used after this layer (0 for ReLU
--         by default)
--     mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
--         preserves the magnitude of the variance of the weights in the
--         forward pass. Choosing `fan_out` preserves the magnitudes in the
--         backwards pass.
--     nonlinearity: the non-linear function (`nn.functional` name),
--         recommended to use only with 'relu' or 'leaky_relu' (default).
-- Examples:
--     >>> w = torch.empty(3, 5)
--     >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
kaimingUniformWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Activation
  -> FanMode
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
kaimingUniformWith_ = kaimingDisributedWith_ $ \g pstd t -> do
  let a = P.sqrt 3 * (positiveValue pstd)   -- Calculate uniform bounds from standard deviation
      Just pair = ord2Tuple (-a, a)
  _uniform t g pair

kaimingUniform_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
kaimingUniform_ = kaimingUniformWith_ (LeakyReluFn (Just 0)) FanIn

-- |
-- Fills the input `Tensor` with values according to the method
-- described in "Delving deep into rectifiers: Surpassing human-level
-- performance on ImageNet classification" - He, K. et al. (2015), using a
-- normal distribution. The resulting tensor will have values sampled from
-- :math:`\mathcal{N}(0, \text{std})` where
-- .. math::
--     \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}
-- Also known as He initialization.
-- Args:
--     tensor: an n-dimensional `torch.Tensor`
--     a: the negative slope of the rectifier used after this layer (0 for ReLU
--         by default)
--     mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
--         preserves the magnitude of the variance of the weights in the
--         forward pass. Choosing `fan_out` preserves the magnitudes in the
--         backwards pass.
--     nonlinearity: the non-linear function (`nn.functional` name),
--         recommended to use only with 'relu' or 'leaky_relu' (default).
-- Examples:
--     >>> w = torch.empty(3, 5)
--     >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
kaimingNormalWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Activation
  -> FanMode
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
kaimingNormalWith_ = kaimingDisributedWith_ $ \g std t -> _normal t g 0 std

kaimingNormal_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
kaimingNormal_ = kaimingNormalWith_ (LeakyReluFn (Just 0)) FanIn


kaimingDisributedWith_
  :: (Dimensions outs, All KnownDim '[i, o, Product outs])
  => (Generator -> Positive HsReal -> Tensor (i:+o:+outs) -> IO ()) -- ^ randomizing fill which takes a standard of deviation
  -> Activation
  -> FanMode
  -> Generator
  -> Tensor (i:+o:+outs) -- ^ tensor: an n-dimensional `torch.Tensor` (minimum length 2)
  -> IO ()
kaimingDisributedWith_ distribution activation mode g t =
  case positive std of
    Just std -> distribution g std t
    Nothing -> throwString $
      "standard deviation is not positive. Found: " ++ show std ++ ", most likely the gain is negative, which is incorrect: " ++ show gain
 where
  fan = calculateCorrectFan t mode
  gain = calculateGain activation
  std = gain / P.sqrt fan


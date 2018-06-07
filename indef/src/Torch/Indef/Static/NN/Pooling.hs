-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Pooling
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Indef.Static.NN.Pooling where

import Numeric.Backprop
import Numeric.Dimensions
import System.IO.Unsafe
import Data.Singletons.Prelude hiding (All, type (*), type (-), type (+))
import Data.Singletons.TypeLits

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.NN.Backprop ()
import Torch.Indef.Static.NN.Conv2d (Conv2d(..), Param2d(..), Kernel2d(..), Dilation2d(..), Padding2d(..), Step2d(..))
import Data.Singletons.Prelude.Bool

import qualified Torch.Indef.Index as Ix
import qualified Torch.Indef.Static.NN.Conv2d as Conv2d
import qualified Torch.Indef.Dynamic.NN.Pooling as Dynamic


-- |  featureLPPooling forward pass (updates the output tensor)
_featureLPPooling_updateOutput :: Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateOutput t0 t1 = Dynamic._featureLPPooling_updateOutput (asDynamic t0) (asDynamic t1)

-- |  featureLPPooling backward-update (updates the layer and bias tensors)
_featureLPPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateGradInput t0 t1 t2 t3 = Dynamic._featureLPPooling_updateGradInput (asDynamic t0) (asDynamic t1)
 (asDynamic t2) (asDynamic t3)

-- * 1d pooling functions

-- |  temporalMaxPooling forward pass (updates the output tensor)
_temporalMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateOutput t0 t1 ix0 = Dynamic._temporalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  temporalMaxPooling backward-update (updates the layer and bias tensors)
_temporalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._temporalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- * 2d pooling functions

-- | Constraint to assert that all hyperparameters are valid
-- and to make the requirement that all dimension values are
-- 'KnownDim's.
type SpatialDilationCheckC kH kW dH dW pH pW dilH dilW =
  ( All KnownDim '[kH,kW,pH,pW,dH,dW,dilH,dilW]
  , kW > 0 ~ 'True
  , kH > 0 ~ 'True
  , dW > 0 ~ 'True
  , dH > 0 ~ 'True
  , dilW > 0 ~ 'True
  , dilH > 0 ~ 'True
  , (Div kW 2) >= pW ~ 'True
  , (Div kH 2) >= pH ~ 'True
  )

-- | Type-level if statement to indicate what the output dimension should be if
-- CeilMode is turned on.
type CeilModeOutputDims i k d p o dil ceilMode =
  ( If (ceilMode && (Rem (i - (dil * (k - 1) + 1) + (2 * p)) d > 0))
      ((2 + (Div (i - (dil * (k - 1) + 1) + (2 * p)) d)) ~ o)
      ((1 + (Div (i - (dil * (k - 1) + 1) + (2 * p)) d)) ~ o)
  )

-- | Top-level constraint to assert that checks 'CeilModeOutputDims' on
-- height and width dimensions and asserts that all dimensions checks in
-- 'SpatialDilationCheckC' are true.
type SpatialDilationC iH iW kH kW dH dW pH pW oW oH dilH dilW ceilMode =
   ( SpatialDilationCheckC kH kW dH dW pH pW dilH dilW
   , CeilModeOutputDims iH kH dH pH oH dilH ceilMode
   , CeilModeOutputDims iW kW dW pW oW dilW ceilMode
   , All KnownDim '[oH,oW,iH,iW]
   )


-- | run a backprop-aware @dilatedMaxPooling2d@ function
dilatedMaxPooling2d
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH dilH dilW ceilMode)
  => KnownDim inPlane
  => Reifies s W

  -- Parameters
  => Kernel2d kH kW         -- ^ kernel size
  -> Step2d dH dW           -- ^ step size
  -> Padding2d pH pW        -- ^ padding size
  -> Dilation2d dilH dilW   -- ^ dilation size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor '[inPlane, iW, iH])
  -> BVar s (Tensor '[inPlane, oW, oH])
dilatedMaxPooling2d = _dilatedMaxPooling2d

-- | run a backprop-aware @dilatedMaxPooling2d@ function with a batch dimension.
dilatedMaxPooling2dBatch
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH dilH dilW ceilMode)
  => KnownDim inPlane
  => KnownDim b
  => Reifies s W

  -- Parameters
  => Kernel2d kH kW         -- ^ kernel size
  -> Step2d dH dW           -- ^ step size
  -> Padding2d pH pW        -- ^ padding size
  -> Dilation2d dilH dilW   -- ^ dilation size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor '[b, inPlane, iW, iH])
  -> BVar s (Tensor '[b, inPlane, oW, oH])
dilatedMaxPooling2dBatch = _dilatedMaxPooling2d

-- | internal function of 'dilatedMaxPooling2d' and 'dilatedMaxPooling2dBatch'. Should not be used.
_dilatedMaxPooling2d
  :: forall s d d' kH kW dH dW pH pW dilH dilW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW,dilH,dilW]
  => All Dimensions '[d',d]
  => Reifies s W

  -- Parameters
  => Kernel2d kH kW         -- ^ kernel size
  -> Step2d dH dW           -- ^ step size
  -> Padding2d pH pW        -- ^ padding size
  -> Dilation2d dilH dilW   -- ^ dilation size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor d)      -- ^ input
  -> BVar s (Tensor d')     -- ^ output
_dilatedMaxPooling2d ker step pad dil ceil = liftOp1 . op1 $ \inp -> unsafePerformIO $ do
  (ix, out) <- _spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceil
  pure (out, \gout ->
   unsafePerformIO (_spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceil))
 where
  _spatialDilatedMaxPooling_updateOutput
    :: Tensor d              -- ^ input
    -> Kernel2d kH kW        -- ^ kernel size
    -> Step2d dH dW          -- ^ step size
    -> Padding2d pH pW       -- ^ padding size
    -> Dilation2d dilH dilW  -- ^ dilation size
    -> SBool ceilMode        -- ^ ceil mode
    -> IO (IndexTensor d', Tensor d') -- ^ index of each max from the indicies, output of the max pooling
  _spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceilMode = do
    out <- empty
    let ix = Ix.zeroIxNd :: IndexTensor d'
    Dynamic._spatialDilatedMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (param2d dil) (fromSing ceilMode)
    pure (ix, out)

  _spatialDilatedMaxPooling_updateGradInput
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d kH kW        -- ^ kernel size
    -> Step2d dH dW          -- ^ step size
    -> Padding2d pH pW       -- ^ padding size
    -> Dilation2d dilH dilW  -- ^ dilation size
    -> SBool ceilMode        -- ^ ceil mode
    -> IO (Tensor d)         -- ^ gradInput
  _spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceilMode = do
    gin <- empty
    Dynamic._spatialDilatedMaxPooling_updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (param2d dil) (fromSing ceilMode)
    pure gin

-- * 2d max pooling helpers

-- | internal function of 'maxPooling2d' and 'maxPooling2dBatch'. Should not be used.
_maxPooling2d
  :: forall s d d' kH kW dH dW pH pW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW]
  => All Dimensions '[d',d]
  => Reifies s W

  -- Parameters
  => Kernel2d kH kW         -- ^ kernel size
  -> Step2d dH dW           -- ^ step size. Note: default in C is the kernel size.
  -> Padding2d pH pW        -- ^ padding size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor d)      -- ^ input
  -> BVar s (Tensor d')     -- ^ output
_maxPooling2d ker step pad ceil = liftOp1 . op1 $ \inp ->
  let
    (ix, out) = _spatialMaxPooling_updateOutput inp ker step pad ceil
  in
    (out, \gout -> _spatialMaxPooling_updateGradInput inp gout ix ker step pad ceil)

 where

  _spatialMaxPooling_updateOutput
    :: Tensor d              -- ^ input
    -> Kernel2d kH kW        -- ^ kernel size
    -> Step2d dH dW          -- ^ step size
    -> Padding2d pH pW       -- ^ padding size
    -> SBool ceilMode                         -- ^ ceil mode
    -> (IndexTensor d', Tensor d')           -- ^ output
  _spatialMaxPooling_updateOutput inp ker step pad ceilMode = unsafePerformIO $ do
    out <- empty
    let ix = Ix.zeroIxNd :: IndexTensor d'
    Dynamic._spatialMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing ceilMode)
    pure (ix, out)

  _spatialMaxPooling_updateGradInput
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d kH kW        -- ^ kernel size
    -> Step2d dH dW          -- ^ step size
    -> Padding2d pH pW       -- ^ padding size
    -> SBool ceilMode        -- ^ ceil mode
    -> Tensor d              -- ^ gradInput
  _spatialMaxPooling_updateGradInput inp gout ix ker step pad ceilMode = unsafePerformIO $ do
    gin <- empty
    Dynamic._spatialMaxPooling_updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing ceilMode)
    pure gin

-- | backprop-aware @maxPooling2d@ function.
maxPooling2d
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => Reifies s W
  => KnownDim inPlane

  -- Parameters
  => Kernel2d  kH kW       -- ^ kernel size
  -> Step2d    dH dW       -- ^ step size
  -> Padding2d pH pW       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> BVar s (Tensor '[inPlane, iH, iW])
  -> BVar s (Tensor '[inPlane, oH, oW])
maxPooling2d = _maxPooling2d

-- | backprop-aware @maxPooling2d@ function with a batch dimension.
maxPooling2dBatch
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => Reifies s W
  => KnownDim inPlane
  => KnownDim b

  -- Parameters
  => Kernel2d kH kW        -- ^ kernel size
  -> Step2d dH dW          -- ^ step size
  -> Padding2d pH pW       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> BVar s (Tensor '[b, inPlane, iH, iW])
  -> BVar s (Tensor '[b, inPlane, oH, oW])
maxPooling2dBatch = _maxPooling2d


-- |  spatialAdaptiveMaxPooling forward pass (updates the output tensor)
_spatialAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialAdaptiveMaxPooling_updateOutput t0 t1 ix0 = do
  Dynamic._spatialAdaptiveMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  spatialAdaptiveMaxPooling backward-update (updates the layer and bias tensors)
_spatialAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_spatialAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialAdaptiveMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  spatialFractionalMaxPooling forward pass (updates the output tensor)
_spatialFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_spatialFractionalMaxPooling_updateOutput t0 t1 a0 a1 a2 a3 ix0 t2 = Dynamic._spatialFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) a0 a1 a2 a3 (longAsDynamic ix0) (asDynamic t2)

-- |  spatialFractionalMaxPooling backward-update (updates the layer and bias tensors)
_spatialFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_spatialFractionalMaxPooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 ix0 = Dynamic._spatialFractionalMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) a0 a1 a2 a3 (longAsDynamic ix0)

-- |  spatialMaxUnpooling forward pass (updates the output tensor)
_spatialMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._spatialMaxUnpooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  spatialMaxUnpooling backward-update (updates the layer and bias tensors)
_spatialMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialMaxUnpooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  spatialAdaptiveAveragePooling forward pass (updates the output tensor)
_spatialAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_spatialAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._spatialAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

-- |  spatialAdaptiveAveragePooling backward-update (updates the layer and bias tensors)
_spatialAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_spatialAdaptiveAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  spatialAveragePooling forward pass (updates the output tensor)
_spatialAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateOutput t0 t1 = Dynamic._spatialAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

-- |  spatialAveragePooling backward-update (updates the layer and bias tensors)
_spatialAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)


-- * 3D pooling functions

-- |  volumetricFractionalMaxPooling forward pass (updates the output tensor)
_volumetricFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_volumetricFractionalMaxPooling_updateOutput t0 t1 i0 i1 i2 i3 i4 i5 ix0 t2 = Dynamic._volumetricFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0) (asDynamic t2)

-- |  volumetricFractionalMaxPooling backward-update (updates the layer and bias tensors)
_volumetricFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_volumetricFractionalMaxPooling_updateGradInput t0 t1 t2 i0 i1 i2 i3 i4 i5 ix0 = Dynamic._volumetricFractionalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0)

-- |  volumetricMaxPooling forward pass (updates the output tensor)
_volumetricMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  volumetricMaxPooling backward-update (updates the layer and bias tensors)
_volumetricMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  volumetricDilatedMaxPooling forward pass (updates the output tensor)
_volumetricDilatedMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricDilatedMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  volumetricDilatedMaxPooling backward-update (updates the layer and bias tensors)
_volumetricDilatedMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricDilatedMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  volumetricMaxUnpooling forward pass (updates the output tensor)
_volumetricMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxUnpooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  volumetricMaxUnpooling backward-update (updates the layer and bias tensors)
_volumetricMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxUnpooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  volumetricAdaptiveMaxPooling forward pass (updates the output tensor)
_volumetricAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

-- |  volumetricAdaptiveMaxPooling backward-update (updates the layer and bias tensors)
_volumetricAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_volumetricAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- |  volumetricAveragePooling forward pass (updates the output tensor)
_volumetricAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateOutput t0 t1 =
  Dynamic._volumetricAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

-- |  volumetricAveragePooling backward-update (updates the layer and bias tensors)
_volumetricAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  volumetricAdaptiveAveragePooling forward pass (updates the output tensor)
_volumetricAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._volumetricAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

-- |  volumetricAdaptiveAveragePooling backward-update (updates the layer and bias tensors)
_volumetricAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_volumetricAdaptiveAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)



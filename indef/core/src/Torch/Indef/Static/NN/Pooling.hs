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
{-# LANGUAGE CPP #-}

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Pooling where

import Data.Maybe
import Numeric.Backprop
import Numeric.Dimensions
import System.IO.Unsafe
import Data.Singletons.Prelude hiding (All, type (*), type (-), type (+))
import Data.Singletons.TypeLits

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math (zero_)
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
  , (kW > 0) ~ 'True
  , (kH > 0) ~ 'True
  , (dW > 0) ~ 'True
  , (dH > 0) ~ 'True
  , (dilW > 0) ~ 'True
  , (dilH > 0) ~ 'True
  , ((Div kW 2) >= pW) ~ 'True
  , ((Div kH 2) >= pH) ~ 'True
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
  => Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> Dilation2d '(dilH, dilW)   -- ^ dilation size
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
  => Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> Dilation2d '(dilH, dilW)   -- ^ dilation size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor '[b, inPlane, iW, iH])
  -> BVar s (Tensor '[b, inPlane, oW, oH])
dilatedMaxPooling2dBatch = _dilatedMaxPooling2d

-- | internal function of 'dilatedMaxPooling2d' and 'dilatedMaxPooling2dBatch'. Should not be used.
{-# NOINLINE _dilatedMaxPooling2d #-}
_dilatedMaxPooling2d
  :: forall s d d' kH kW dH dW pH pW dilH dilW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW,dilH,dilW]
  => All Dimensions '[d',d]
  => Reifies s W

  -- Parameters
  => Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> Dilation2d '(dilH,dilW)   -- ^ dilation size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor d)      -- ^ input
  -> BVar s (Tensor d')     -- ^ output
_dilatedMaxPooling2d ker step pad dil ceil = liftOp1 . op1 $ \inp -> unsafePerformIO $ do
  (ix, out) <- _spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceil
  pure (out, \gout ->
   unsafePerformIO (_spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceil))
 where
  {-# NOINLINE _spatialDilatedMaxPooling_updateOutput #-}
  _spatialDilatedMaxPooling_updateOutput
    :: Tensor d              -- ^ input
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> Dilation2d '(dilH, dilW)  -- ^ dilation size
    -> SBool ceilMode        -- ^ ceil mode
    -> IO (IndexTensor d', Tensor d') -- ^ index of each max from the indicies, output of the max pooling
  _spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceilMode = do
    let out = empty
    let ix = Ix.zeroIxNd :: IndexTensor d'
    Dynamic._spatialDilatedMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (param2d dil) (fromSing ceilMode)
    pure (ix, out)

  _spatialDilatedMaxPooling_updateGradInput
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> Dilation2d '(dilH, dilW)  -- ^ dilation size
    -> SBool ceilMode        -- ^ ceil mode
    -> IO (Tensor d)         -- ^ gradInput
  _spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceilMode = do
    let gin = empty
    Dynamic._spatialDilatedMaxPooling_updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (param2d dil) (fromSing ceilMode)
    pure gin

-- * 2d max pooling helpers

{-# NOINLINE _maxPooling2d #-}
-- | internal function of 'maxPooling2d' and 'maxPooling2dBatch'. Should not be used.
_maxPooling2d
  :: forall s d d' kH kW dH dW pH pW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW]
  => All Dimensions '[d',d]
  => Reifies s W

  -- Parameters
  => Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size. Note: default in C is the kernel size.
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> BVar s (Tensor d)      -- ^ input
  -> BVar s (Tensor d')     -- ^ output
_maxPooling2d ker step pad ceil = liftOp1 . op1 $ \inp -> unsafePerformIO $ do
  (ix, out) <- _spatialMaxPooling_updateOutput inp ker step pad ceil
  print ("_maxpooling2d forward - input", shape inp)
  pure (out, \gout -> unsafePerformIO $ do
    gin <- _spatialMaxPooling_updateGradInput inp gout ix ker step pad ceil
    print ("_maxpooling2d backward- gin  ", shape gin)
    pure gin)

 where

  {-# NOINLINE _spatialMaxPooling_updateOutput #-}
  _spatialMaxPooling_updateOutput
    :: Tensor d              -- ^ input
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode                         -- ^ ceil mode
    -> IO (IndexTensor d', Tensor d')           -- ^ output
  _spatialMaxPooling_updateOutput inp ker step pad ceilMode = do
    let out = empty
    let ix = Ix.zeroIxNd :: IndexTensor d'
    Dynamic._spatialMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing ceilMode)
    pure (ix, out)

  {-# NOINLINE _spatialMaxPooling_updateGradInput #-}
  _spatialMaxPooling_updateGradInput
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode        -- ^ ceil mode
    -> IO (Tensor d)              -- ^ gradInput
  _spatialMaxPooling_updateGradInput inp gout ix ker step pad ceilMode = do
    let gin = empty
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
  => Kernel2d '(kH, kW)       -- ^ kernel size
  -> Step2d '(dH, dW)       -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
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
  => Kernel2d '(kH, kW)        -- ^ kernel size
  -> Step2d '(dH, dW)          -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> BVar s (Tensor '[b, inPlane, iH, iW])
  -> BVar s (Tensor '[b, inPlane, oH, oW])
maxPooling2dBatch = _maxPooling2d

-- | internal function of 'maxPooling2d' and 'maxPooling2dBatch'. Should not be used.
maxPooling2dWithIO
  :: forall d d' kH kW dH dW pH pW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW]
  => All Dimensions '[d',d]

  -- optional buffers
  => Maybe (IndexTensor d')
  -> Maybe (Tensor d')
  -> Maybe (Tensor d)

  -- Parameters
  -> Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size. Note: default in C is the kernel size.
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> Tensor d
  -> IO (Tensor d', Tensor d' -> IO (Tensor d))
maxPooling2dWithIO mix mout mgin ker step pad ceil inp = do
  -- let ix = fromMaybe new mix
  -- Ix.zero_ ix
  let ix = Ix.zeroIxNd :: IndexTensor d'
  let out = fromMaybe new mout
  zero_ out

  updateOutput_ inp ker step pad ceil (ix, out)
  pure (out, \gout -> do
    let gin = fromMaybe new mgin
    zero_ gin
    updateGradInput_ inp gout ix ker step pad ceil gin
    pure gin)

 where
  updateOutput_
    :: Tensor d              -- ^ input
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode                         -- ^ ceil mode
    -> (IndexTensor d', Tensor d')           -- ^ output
    -> IO ()
  updateOutput_ inp ker step pad sceil (ix, out) = do
    Dynamic._spatialMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing sceil)

  updateGradInput_
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode        -- ^ ceil mode
    -> Tensor d              -- ^ gradInput
    -> IO ()
  updateGradInput_ inp gout ix ker step pad sceil gin =
    Dynamic._spatialMaxPooling_updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing sceil)

-- | backprop-aware @maxPooling2d@ function.
maxPooling2dIO
  :: forall iH iW kH kW dH dW pH pW oW oH ceilMode inPlane
  .  (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane

  -- Parameters
  => Kernel2d '(kH, kW)       -- ^ kernel size
  -> Step2d '(dH, dW)       -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[inPlane, iH, iW])
  -> IO (Tensor '[inPlane, oH, oW], Tensor '[inPlane, oH, oW] -> IO (Tensor '[inPlane, iH, iW]))
maxPooling2dIO =
  maxPooling2dWithIO
    (Just (Ix.newIx :: IndexTensor '[inPlane, oH, oW]))
    (Just (new :: Tensor '[inPlane, oH, oW]))
    (Just (new :: Tensor '[inPlane, iH, iW]))

-- | backprop-aware @maxPooling2d@ function with a batch dimension.
maxPooling2dBatchIO
  :: forall iH iW kH kW dH dW pH pW oW oH ceilMode b inPlane
  .  (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane
  => KnownDim b

  -- Parameters
  => Kernel2d '(kH, kW)        -- ^ kernel size
  -> Step2d '(dH, dW)          -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[b, inPlane, iH, iW])
  -> IO (Tensor '[b, inPlane, oH, oW], Tensor '[b, inPlane, oH, oW] -> IO (Tensor '[b, inPlane, iH, iW]))
maxPooling2dBatchIO =
  maxPooling2dWithIO
    (Just (Ix.newIx :: IndexTensor '[b, inPlane, oH, oW]))
    (Just (new :: Tensor '[b, inPlane, oH, oW]))
    (Just (new :: Tensor '[b, inPlane, iH, iW]))



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

-- | Type-level if statement to indicate what the output dimension should be if
-- CeilMode is turned on.
type AvgPool2dOutputDim i k p s ceilMode o =
  ( If (ceilMode && (Rem (i + (2 * p) - k) s > 0))
      ((2 + (Div (i + (2 * p) - k) s)) ~ o)
      ((1 + (Div (i + (2 * p) - k) s)) ~ o)
  , (k > 0) ~ 'True
  , (s > 0) ~ 'True
  , (o > 0) ~ 'True
  , ((Div k 2) >= p) ~ 'True
  )

-- | spatial global average pooling on batches in IO
gapPool2dBatchIO
  :: forall iH iW b c varlist
  . varlist ~ '[b, c, iH, iW]
  => All KnownNat varlist
  => All KnownDim varlist
  => AvgPool2dOutputDim iH iH 0 iH 'False 1
  => AvgPool2dOutputDim iW iW 0 iW 'False 1
  => Tensor '[b, c, iH, iW]               -- ^ input tensor
  -> IO (Tensor '[b, c], Tensor '[b, c] -> IO (Tensor '[b, c, iH, iW]))
gapPool2dBatchIO inp = do
  (out, getgrad) <- avgPool2dBatchIO (Kernel2d  @'(iH, iW)) inp
  pure (resizeAs out, getgrad . resizeAs)


-- | spatial average pooling with backprop support in IO
avgPool2dWithIO
  :: All KnownNat '[c, iH, iW, oH, oW, kW, kH, dW, dH, padW, padH]
  => All KnownDim '[c, iH, iW, oH, oW, kW, kH, dW, dH, padW, padH]
  => AvgPool2dOutputDim iH kH padH dH ceil_mode oH
  => AvgPool2dOutputDim iW kW padW dW ceil_mode oW
  => Kernel2d  '(kH, kW)      -- ^ kernel sizes
  -> Step2d    '(dH, dW)      -- ^ step sizes
  -> Padding2d '(padH, padW)  -- ^ pad sizes
  -> SBool ceil_mode          -- ^ ceiling mode: when True, will use `ceil` instead of `floor` to compute the output shape
  -> SBool count_include_pad  -- ^ count_include_pad: when True, will include the zero-padding in the averaging calculation
  -> Tensor '[c, iH, iW]      -- ^ input tensor
  -> IO (Tensor '[c, oH, oW], Tensor '[c, oH, oW] -> IO (Tensor '[c, iH, iW]))
avgPool2dWithIO = _avgPool2dWithIO (Just new) (Just new)


-- | spatial average pooling on batches with backprop support in IO and defaults
avgPool2dBatchIO
  :: forall iH iW kH kW oH oW b c
  .  All KnownNat '[b, c, iH, iW, oH, oW, kW, kH]
  => All KnownDim '[b, c, iH, iW, oH, oW, kW, kH]
  => AvgPool2dOutputDim iH kH 0 kH 'False oH
  => AvgPool2dOutputDim iW kW 0 kW 'False oW
  => Kernel2d  '(kH, kW)                  -- ^ kernel sizes
  -> Tensor '[b, c, iH, iW]               -- ^ input tensor
  -> IO (Tensor '[b, c, oH, oW], Tensor '[b, c, oH, oW] -> IO (Tensor '[b, c, iH, iW]))
avgPool2dBatchIO ker = _avgPool2dWithIO (Just new) (Just new) ker (Step2d @'(kH, kW))
      (Padding2d @'(0, 0))
      (sing :: SBool 'False)
      (sing :: SBool 'True)


-- | spatial average pooling on batches with backprop support in IO
avgPool2dBatchWithIO
  :: All KnownNat '[b, c, iH, iW, oH, oW, kW, kH, dW, dH, padW, padH]
  => All KnownDim '[b, c, iH, iW, oH, oW, kW, kH, dW, dH, padW, padH]
  => AvgPool2dOutputDim iH kH padH dH ceil_mode oH
  => AvgPool2dOutputDim iW kW padW dW ceil_mode oW
  => Kernel2d  '(kH, kW)      -- ^ kernel sizes
  -> Step2d    '(dH, dW)      -- ^ step sizes
  -> Padding2d '(padH, padW)  -- ^ pad sizes
  -> SBool ceil_mode          -- ^ ceiling mode: when True, will use `ceil` instead of `floor` to compute the output shape
  -> SBool count_include_pad  -- ^ count_include_pad: when True, will include the zero-padding in the averaging calculation
  -> Tensor '[b, c, iH, iW]               -- ^ input tensor
  -> IO (Tensor '[b, c, oH, oW], Tensor '[b, c, oH, oW] -> IO (Tensor '[b, c, iH, iW]))
avgPool2dBatchWithIO = _avgPool2dWithIO (Just new) (Just new)

-- | generic spatial average pooling with backprop support in IO. This works without constraints and can be applied on either
-- batch or non-batch tensors, but C errors may occur if you misuse this function.
_avgPool2dWithIO
  :: forall din kW kH dW dH padW padH ceil_mode count_include_pad dout
  .  All KnownNat '[kW, kH, dW, dH, padW, padH]
  => All KnownDim '[kW, kH, dW, dH, padW, padH]
  => All Dimensions '[dout, din]
  => Maybe (Tensor dout)      -- ^ cached output (optional)
  -> Maybe (Tensor din)       -- ^ cached input gradient (optional)
  -> Kernel2d  '(kH, kW)      -- ^ kernel sizes
  -> Step2d    '(dH, dW)      -- ^ step sizes
  -> Padding2d '(padH, padW)  -- ^ pad sizes
  -> SBool ceil_mode          -- ^ ceiling mode: when True, will use `ceil` instead of `floor` to compute the output shape
  -> SBool count_include_pad  -- ^ count_include_pad: when True, will include the zero-padding in the averaging calculation
  -> Tensor din               -- ^ input tensor
  -> IO (Tensor dout, Tensor dout -> IO (Tensor din))
_avgPool2dWithIO mout mgin kers steps pads ceilMode countIncludePad inp = do
  let out = fromMaybe new mout
  _updateOutput
    (asDynamic inp) (asDynamic out)
    (param2d kers) (param2d steps) (param2d pads)
    (fromSing ceilMode) (fromSing countIncludePad)
  pure (out, \gout -> do
    let gin = fromMaybe new mgin
    _updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin)
      (param2d kers) (param2d steps) (param2d pads)
      (fromSing ceilMode) (fromSing countIncludePad)
    pure gin)
  where
    -- spatialAveragePooling forward pass (updates the output tensor)
    _updateOutput
      :: Dynamic         -- input tensor
      -> Dynamic         -- output tensor
      -> (Int, Int)      -- kernel sizes
      -> (Int, Int)      -- step sizes
      -> (Int, Int)      -- pad sizes
      -> Bool            -- ceiling mode: when True, will use `ceil` instead of `floor` to compute the output shape
      -> Bool            -- count_include_pad: when True, will include the zero-padding in the averaging calculation
      -> IO ()
    _updateOutput inp out (kW, kH) (dW, dH) (padW, padH) ceil_mode count_include_pad =
      Dynamic._spatialAveragePooling_updateOutput
        inp out kW kH dW dH padW padH ceil_mode count_include_pad

    -- spatialAveragePooling backward-update (updates the layer and bias tensors)
    _updateGradInput
      :: Dynamic         -- input tensor
      -> Dynamic         -- gradient output tensor
      -> Dynamic         -- gradient input tensor
      -> (Int, Int)      -- kernel sizes
      -> (Int, Int)      -- step sizes
      -> (Int, Int)      -- pad sizes
      -> Bool            -- ceiling mode: when True, will use `ceil` instead of `floor` to compute the output shape
      -> Bool            -- count_include_pad: when True, will include the zero-padding in the averaging calculation
      -> IO ()
    _updateGradInput inp gout gin (kW, kH) (dW, dH) (padW, padH) ceil_mode count_include_pad =
      Dynamic._spatialAveragePooling_updateGradInput
        inp gout gin kW kH dW dH padW padH ceil_mode count_include_pad

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



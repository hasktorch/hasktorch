{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Indef.Static.NN.Pooling where

import Numeric.Backprop
import System.IO.Unsafe


import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.NN.Backprop ()
import Torch.Indef.Static.NN.Conv2d (Conv2d(..), Param2d(..), Kernel2d(..), Dilation2d(..), Padding2d(..), Step2d(..))
import Data.Singletons.Prelude.Bool

import qualified Torch.Indef.Index as Ix
import qualified Torch.Indef.Static.NN.Conv2d as Conv2d
import qualified Torch.Indef.Dynamic.NN.Pooling as Dynamic


_featureLPPooling_updateOutput :: Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateOutput t0 t1 = Dynamic._featureLPPooling_updateOutput (asDynamic t0) (asDynamic t1)

_featureLPPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateGradInput t0 t1 t2 t3 = Dynamic._featureLPPooling_updateGradInput (asDynamic t0) (asDynamic t1)
 (asDynamic t2) (asDynamic t3)

-- * 1d pooling functions

_temporalMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateOutput t0 t1 ix0 = Dynamic._temporalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_temporalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._temporalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

-- * 2d pooling functions

type SpatialDilationCheckC kW kH pW pH dW dH dilW dilH =
  ( KnownNatDim2 kW kH
  , KnownNatDim2 pW pH
  , KnownNatDim2 dW dH
  , KnownNatDim2 dilW dilH
  , kW > 0 ~ 'True
  , kH > 0 ~ 'True
  , dW > 0 ~ 'True
  , dH > 0 ~ 'True
  , dilW > 0 ~ 'True
  , dilH > 0 ~ 'True
  , (Div kW 2) >= pW ~ 'True
  , (Div kH 2) >= pH ~ 'True
  )

-- FIXME: need to figure out a type-level ceil function later
-- type SpatialDilationCeilCheckC iH iW oH oW kW kH pW pH dW dH dilW dilH =
--   ( SpatialDilationCheckC kW kH pW pH dW dH dilW dilH
--   , oH ~ (iH - Div (((dilH * (kH - 1)) + 1) + (2 * pH)) dH) + 1
--   , oW ~ (iW - Div (((dilW * (kW - 1)) + 1) + (2 * pW)) dW) + 1
--   )

-- POSSIBLY THIS:
type CeilModeOutputDims i dil k p d o ceilMode =
  ( If (ceilMode && (Rem (i - (dil * (k - 1) + 1) + (2 * p)) d > 0))
      ((2 + (Div (i - (dil * (k - 1) + 1) + (2 * p)) d)) ~ o)
      ((1 + (Div (i - (dil * (k - 1) + 1) + (2 * p)) d)) ~ o)
  )

 -- (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1    1, ceilMode ~ 'False)
type SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH dilW dilH ceilMode =
   ( SpatialDilationCheckC kW kH pW pH dW dH dilW dilH
   , CeilModeOutputDims iH dilH kH pH dH oH ceilMode
   , CeilModeOutputDims iW dilW kW pW dW oW ceilMode
   )

-- dilatedMaxPooling2d
--   :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH dilW dilH, ceilMode ~ 'False)
--   => Reifies s W => KnownNatDim3 inPlane iW iH => KnownNatDim2 oW oH
-- 
--   -- Parameters
--   => Kernel2d kW kH         -- ^ kernel size
--   -> Step2d dW dH           -- ^ step size
--   -> Padding2d pW pH        -- ^ padding size
--   -> Dilation2d dilW dilH   -- ^ dilation size
--   -> Proxy ceilMode         -- ^ ceil mode
-- 
--   -- function arguments
--   -> BVar s (Tensor      '[inPlane, iW, iH])                                  -- ^ input
--   -> BVar s (IndexTensor '[inPlane, oW, oH], Tensor '[inPlane, oW, oH])       -- ^ output
-- dilatedMaxPooling2d ker step pad dil ceil = liftOp1 . op1 $ \inp ->
--   (unsafePerformIO (_spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceil), \(ix, gout) ->
--    unsafePerformIO (_spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceil))
-- 
-- dilatedMaxPooling2dBatch
--   :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH dilW dilH, ceilMode ~ 'False)
--   => Reifies s W => KnownNatDim3 inPlane iW iH => KnownNatDim3 b oW oH
-- 
--   -- Parameters
--   => Kernel2d kW kH         -- ^ kernel size
--   -> Step2d dW dH           -- ^ step size
--   -> Padding2d pW pH        -- ^ padding size
--   -> Dilation2d dilW dilH   -- ^ dilation size
--   -> Proxy ceilMode         -- ^ ceil mode
-- 
--   -- function arguments
--   -> BVar s (Tensor      '[b, inPlane, iW, iH])                                     -- ^ input
--   -> BVar s (IndexTensor '[b, inPlane, oW, oH], Tensor '[b, inPlane, oW, oH])       -- ^ output
-- dilatedMaxPooling2dBatch ker step pad dil ceil = liftOp1 . op1 $ \inp ->
--   (unsafePerformIO (_spatialDilatedMaxPooling_updateOutput inp ker step pad dil ceil), \(ix, gout) ->
--    unsafePerformIO (_spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil ceil))


_spatialDilatedMaxPooling_updateOutput
  :: forall d d' kW kH pW pH dW dH dilW dilH ceilMode
  . (KnownNat4 kW kH pW pH, KnownNat4 dW dH dilW dilH, ceilMode ~ 'False)
  => Dimensions2 d' d
  => Tensor d              -- ^ input
  -> Kernel2d kW kH        -- ^ kernel size
  -> Step2d dW dH          -- ^ step size
  -> Padding2d pW pH       -- ^ padding size
  -> Dilation2d dilW dilH  -- ^ dilation size
  -> Proxy ceilMode        -- ^ ceil mode
  -> IO (IndexTensor d', Tensor d')        -- ^ index of each max from the indicies, output of the max pooling
_spatialDilatedMaxPooling_updateOutput inp ker step pad dil _ = do
  out <- empty
  let ix = Ix.zeroIxNd :: IndexTensor d'
  Dynamic._spatialDilatedMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
    (param2d ker) (param2d step) (param2d pad) (param2d dil) False
  pure (ix, out)

_spatialDilatedMaxPooling_updateGradInput
  :: (KnownNat4 kW kH pW pH, KnownNat4 dW dH dilW dilH, ceilMode ~ 'False)
  => Tensor d              -- ^ input
  -> Tensor d'             -- ^ gradOutput
  -> IndexTensor d'        -- ^ indices
  -> Kernel2d kW kH        -- ^ kernel size
  -> Step2d dW dH          -- ^ step size
  -> Padding2d pW pH       -- ^ padding size
  -> Dilation2d dilW dilH  -- ^ dilation size
  -> Proxy ceilMode        -- ^ ceil mode
  -> IO (Tensor d)         -- ^ gradInput
_spatialDilatedMaxPooling_updateGradInput inp gout ix ker step pad dil _ = do
  gin <- empty
  Dynamic._spatialDilatedMaxPooling_updateGradInput
    (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
    (param2d ker) (param2d step) (param2d pad) (param2d dil) False
  pure gin

-- * 2d max pooling helpers

spatialMaxPooling_updateOutput
  :: forall iW iH oW oH kW kH pW pH dW dH ceilMode inPlane
  .  (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1 ceilMode)
  => Tensor      '[inPlane, iH, iW]       -- ^ input
  -> IndexTensor '[inPlane, iH, iW]       -- ^ indices
  -> Kernel2d kW kH                       -- ^ kernel size
  -> Step2d dW dH                         -- ^ step size
  -> Padding2d pW pH                      -- ^ padding size
  -> SBool ceilMode                       -- ^ ceil mode
  -> IO (Tensor '[inPlane, oH, oW])      -- ^ output
spatialMaxPooling_updateOutput = _spatialMaxPooling_updateOutput

spatialMaxPooling_updateOutputBatch
  :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1 ceilMode)
  => KnownNat2 batch inPlane
  => Tensor      '[batch, inPlane, iH, iW]       -- ^ input
  -> IndexTensor '[batch, inPlane, iH, iW]       -- ^ indices
  -> Kernel2d kW kH                              -- ^ kernel size
  -> Step2d dW dH                                -- ^ step size
  -> Padding2d pW pH                             -- ^ padding size
  -- -> Proxy ceilMode                         -- ^ ceil mode
  -> SBool ceilMode                         -- ^ ceil mode
  -> IO (Tensor '[batch, inPlane, oH, oW])      -- ^ output
spatialMaxPooling_updateOutputBatch = _spatialMaxPooling_updateOutput

_spatialMaxPooling_updateOutput
  :: (KnownNat4 kW kH pW pH, KnownNat2 dW dH)
  => Tensor d              -- ^ input
  -> IndexTensor d         -- ^ indices
  -> Kernel2d kW kH        -- ^ kernel size
  -> Step2d dW dH          -- ^ step size
  -> Padding2d pW pH       -- ^ padding size
  -- -> Proxy ceilMode        -- ^ ceil mode
  -> SBool ceilMode                         -- ^ ceil mode
  -> IO (Tensor d')        -- ^ output
_spatialMaxPooling_updateOutput inp ix ker step pad ceilMode = do
  out <- empty
  Dynamic._spatialMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
    (param2d ker) (param2d step) (param2d pad) (fromSing ceilMode)
  pure out

spatialMaxPooling_updateGradInput
  :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1 ceilMode)
  => Tensor      '[inPlane, iH, iW]         -- ^ input
  -> Tensor      '[inPlane, oH, oW]         -- ^ gradOutput
  -> IndexTensor '[inPlane, iH, iW]         -- ^ indices
  -> Kernel2d kW kH                         -- ^ kernel size
  -> Step2d dW dH                           -- ^ step size
  -> Padding2d pW pH                        -- ^ padding size
  -- -> Proxy ceilMode                         -- ^ ceil mode
  -> SBool ceilMode                         -- ^ ceil mode
  -> IO (Tensor '[inPlane, iH, iW])         -- ^ gradInput
spatialMaxPooling_updateGradInput = _spatialMaxPooling_updateGradInput

spatialMaxPooling_updateGradInputBatch
  :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1 ceilMode)
  => Tensor      '[b, inPlane, iH, iW]      -- ^ input
  -> Tensor      '[b, inPlane, oH, oW]      -- ^ gradOutput
  -> IndexTensor '[b, inPlane, iH, iW]      -- ^ indices
  -> Kernel2d kW kH                         -- ^ kernel size
  -> Step2d dW dH                           -- ^ step size
  -> Padding2d pW pH                        -- ^ padding size
  -> SBool ceilMode                         -- ^ ceil mode
  -> IO (Tensor '[b, inPlane, iH, iW])      -- ^ gradInput
spatialMaxPooling_updateGradInputBatch = _spatialMaxPooling_updateGradInput

_spatialMaxPooling_updateGradInput
  :: (KnownNat4 kW kH pW pH, KnownNat2 dW dH)
  => Tensor d              -- ^ input
  -> Tensor d'             -- ^ gradOutput
  -> IndexTensor d         -- ^ indices
  -> Kernel2d kW kH        -- ^ kernel size
  -> Step2d dW dH          -- ^ step size
  -> Padding2d pW pH       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode
  -> IO (Tensor d)         -- ^ gradInput
_spatialMaxPooling_updateGradInput inp gout ix ker step pad ceilMode = do
  gin <- empty
  Dynamic._spatialMaxPooling_updateGradInput
    (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
    (param2d ker) (param2d step) (param2d pad) (fromSing ceilMode)
  pure gin

-- -- | run a threshold function againts two BVar variables
-- threshold
--   :: Reifies s W
--   => Dimensions d
--   => Double               -- ^ threshold
--   -> Double               -- ^ replacement value
--   -> BVar s (Tensor d)    -- ^ input
--   -> BVar s (Tensor d)    -- ^ output
-- threshold thr value = liftOp1 . op1 $ \inp ->
--   (unsafePerformIO (_threshold_updateOutput thr value False inp), \gout ->
--     unsafePerformIO (_threshold_updateGradInput thr value False inp gout))
-- 
-- -- | ReLU activation function
-- relu :: Reifies s W => Dimensions d => BVar s (Tensor d) -> BVar s (Tensor d)
-- relu = threshold 0 0


_spatialAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialAdaptiveMaxPooling_updateOutput t0 t1 ix0 = do
  Dynamic._spatialAdaptiveMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_spatialAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialAdaptiveMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_spatialFractionalMaxPooling_updateOutput t0 t1 a0 a1 a2 a3 ix0 t2 = Dynamic._spatialFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) a0 a1 a2 a3 (longAsDynamic ix0) (asDynamic t2)

_spatialFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_spatialFractionalMaxPooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 ix0 = Dynamic._spatialFractionalMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) a0 a1 a2 a3 (longAsDynamic ix0)

_spatialMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._spatialMaxUnpooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialMaxUnpooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_spatialAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._spatialAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

_spatialAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_spatialAdaptiveAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_spatialAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateOutput t0 t1 = Dynamic._spatialAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

_spatialAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)


-- * 3D pooling functions

_volumetricFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_volumetricFractionalMaxPooling_updateOutput t0 t1 i0 i1 i2 i3 i4 i5 ix0 t2 = Dynamic._volumetricFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0) (asDynamic t2)

_volumetricFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_volumetricFractionalMaxPooling_updateGradInput t0 t1 t2 i0 i1 i2 i3 i4 i5 ix0 = Dynamic._volumetricFractionalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0)

_volumetricMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricDilatedMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricDilatedMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricDilatedMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricDilatedMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxUnpooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxUnpooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_volumetricAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateOutput t0 t1 =
  Dynamic._volumetricAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

_volumetricAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_volumetricAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._volumetricAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)

_volumetricAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_volumetricAdaptiveAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)



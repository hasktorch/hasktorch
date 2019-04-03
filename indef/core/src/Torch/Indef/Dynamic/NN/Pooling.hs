-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.NN.Pooling
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- DYNAMIC-NN MODULE WARNING: this module is mostly unfinished and
-- undocumented. It provides, in essence, direct calls to the torch neural
-- network libraries: THNN and THCUNN. Because the dynamic tensor code requires
-- a lot of runtime checks which requires a lot of thought regarding a good
-- API, the recommended route is to use Static tensors, which have a much more
-- natural API and is inherently safer.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.NN.Pooling where

import Torch.Sig.Types.NN
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.NN as Sig
import qualified Torch.Indef.Index as Ix

import Torch.Indef.Types

-- | spatialAdaptiveAveragePooling forward pass (updates the output tensor)
_spatialAdaptiveAveragePooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> IO ()
_spatialAdaptiveAveragePooling_updateOutput t0 t1 a0 a1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_SpatialAdaptiveAveragePooling_updateOutput s' t0' t1' (fromIntegral a0) (fromIntegral a1)

-- | spatialAdaptiveAveragePooling backward pass (updates the gradInput tensor)
_spatialAdaptiveAveragePooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_spatialAdaptiveAveragePooling_updateGradInput t0 t1 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Sig.c_SpatialAdaptiveAveragePooling_updateGradInput s' t0' t1' t2'

-- | spatialAveragePooling forward pass (updates the output tensor)
_spatialAveragePooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateOutput t0 t1 a0 a1 a2 a3 a4 a5 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_SpatialAveragePooling_updateOutput s' t0' t1'
      (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)
      (fromIntegral a4) (fromIntegral a5) (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)


-- | spatialAveragePooling backward pass (updates the gradInput tensor)
_spatialAveragePooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 a4 a5 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Sig.c_SpatialAveragePooling_updateGradInput s' t0' t1' t2'
      (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)
      (fromIntegral a4) (fromIntegral a5) (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)


-- | volumetricAveragePooling forward pass (updates the output tensor)
_volumetricAveragePooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateOutput t0 t1 a0 a1 a2 a3 a4 a5 a6 a7 a8 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_VolumetricAveragePooling_updateOutput s' t0' t1'
      (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)
      (fromIntegral a4) (fromIntegral a5) (fromIntegral a6) (fromIntegral a7)
      (fromIntegral a8) (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)


-- | volumetricAveragePooling backward pass (updates the gradInput tensor)
_volumetricAveragePooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 a4 a5 a6 a7 a8 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Sig.c_VolumetricAveragePooling_updateGradInput s' t0' t1' t2'
      (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)
      (fromIntegral a4) (fromIntegral a5) (fromIntegral a6) (fromIntegral a7)
      (fromIntegral a8) (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | volumetricAdaptiveAveragePooling forward pass (updates the output tensor)
_volumetricAdaptiveAveragePooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveAveragePooling_updateOutput t0 t1 a0 a1 a2 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_VolumetricAdaptiveAveragePooling_updateOutput s' t0' t1'
      (fromIntegral a0) (fromIntegral a1) (fromIntegral a2)

-- | volumetricAdaptiveAveragePooling backward pass (updates the gradInput tensor)
_volumetricAdaptiveAveragePooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_volumetricAdaptiveAveragePooling_updateGradInput t0 t1 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
      Sig.c_VolumetricAdaptiveAveragePooling_updateGradInput s' t0' t1' t2'

-- | featureLPPooling forward pass (updates the output tensor)
_featureLPPooling_updateOutput                      :: Dynamic -> Dynamic -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateOutput r t0 v0 v1 v2 b =
  with2DynamicState r t0 $ \s' r' t0' ->
    Sig.c_FeatureLPPooling_updateOutput s' r' t0' (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

-- | featureLPPooling backward pass (updates the gradInput tensor)
_featureLPPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateGradInput t0 t1 t2 t3 v0 v1 v2 b =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_FeatureLPPooling_updateGradInput s' t0' t1' t2' t3'
        (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

-- | temporalMaxPooling forward pass (updates the output tensor)
_temporalMaxPooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> IO ()
_temporalMaxPooling_updateOutput t0 t1 ix0 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_TemporalMaxPooling_updateOutput s' t0' t1' ix0' (fromIntegral i0) (fromIntegral i1)

-- | temporalMaxPooling backward pass (updates the gradInput tensor)
_temporalMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> IO ()
_temporalMaxPooling_updateGradInput t0 t1 t2 ix0 i0 i1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_TemporalMaxPooling_updateGradInput s' t0' t1' t2' ix0' (fromIntegral i0) (fromIntegral i1)

-- | spatialAdaptiveMaxPooling forward pass (updates the output tensor)
_spatialAdaptiveMaxPooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> IO ()
_spatialAdaptiveMaxPooling_updateOutput t0 t1 ix0 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialAdaptiveMaxPooling_updateOutput s' t0' t1' ix0' (fromIntegral i0) (fromIntegral i1)

-- | spatialAdaptiveMaxPooling backward pass (updates the gradInput tensor)
_spatialAdaptiveMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> IO ()
_spatialAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialAdaptiveMaxPooling_updateGradInput s' t0' t1' t2' ix0'


-- | spatialFractionalMaxPooling forward pass (updates the output tensor)
_spatialFractionalMaxPooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IndexDynamic -> Dynamic -> IO ()
_spatialFractionalMaxPooling_updateOutput t0 t1 a0 a1 a2 a3 ix0 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialFractionalMaxPooling_updateOutput s' t0' t1'
        (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)
        ix0' t2'

-- | spatialFractionalMaxPooling backward pass (updates the gradInput tensor)
_spatialFractionalMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IndexDynamic -> IO ()
_spatialFractionalMaxPooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 ix0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialFractionalMaxPooling_updateGradInput s' t0' t1' t2'
        (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3) ix0'


-- | spatialMaxPooling forward pass (updates the output tensor)
_spatialMaxPooling_updateOutput
  :: Dynamic         -- ^ input
  -> Dynamic         -- ^ output
  -> IndexDynamic    -- ^ indices
  -> (Int, Int)      -- ^ kernel size
  -> (Int, Int)      -- ^ step size
  -> (Int, Int)      -- ^ padding size
  -> Bool            -- ^ ceil mode
  -> IO ()
_spatialMaxPooling_updateOutput t0 t1 ix0 (i0,i1) (i2,i3) (i4,i5) b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialMaxPooling_updateOutput s' t0' t1' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (toEnum $ fromEnum b0)


-- | spatialMaxPooling backward pass (updates the gradInput tensor)
_spatialMaxPooling_updateGradInput
  :: Dynamic         -- ^ input
  -> Dynamic         -- ^ grad output
  -> Dynamic         -- ^ grad input
  -> IndexDynamic    -- ^ indices
  -> (Int, Int)      -- ^ kernel size
  -> (Int, Int)      -- ^ step size
  -> (Int, Int)      -- ^ padding size
  -> Bool            -- ^ ceil mode
  -> IO ()
_spatialMaxPooling_updateGradInput t0 t1 t2 ix0 (i0,i1) (i2,i3) (i4,i5) b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialMaxPooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (toEnum $ fromEnum b0)

-- | spatialDilatedMaxPooling forward pass (updates the output tensor)
_spatialDilatedMaxPooling_updateOutput
  :: Dynamic         -- ^ input
  -> Dynamic         -- ^ output
  -> IndexDynamic    -- ^ indices
  -> (Int, Int)      -- ^ kernel size
  -> (Int, Int)      -- ^ step size
  -> (Int, Int)      -- ^ padding size
  -> (Int, Int)      -- ^ dilation size
  -> Bool            -- ^ ceil mode
  -> IO ()
_spatialDilatedMaxPooling_updateOutput t0 t1 ix0 (i0,i1) (i2,i3) (i4,i5) (i6,i7) b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialDilatedMaxPooling_updateOutput s' t0' t1' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (toEnum $ fromEnum b0)

-- | spatialDilatedMaxPooling backward pass (updates the gradInput tensor)
_spatialDilatedMaxPooling_updateGradInput
  :: Dynamic         -- ^ input
  -> Dynamic         -- ^ grad output
  -> Dynamic         -- ^ grad input
  -> IndexDynamic    -- ^ indices
  -> (Int, Int)      -- ^ kernel size
  -> (Int, Int)      -- ^ step size
  -> (Int, Int)      -- ^ padding size
  -> (Int, Int)      -- ^ dilation size
  -> Bool            -- ^ ceil mode
  -> IO ()
_spatialDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 (i0,i1) (i2,i3) (i4,i5) (i6,i7) b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialDilatedMaxPooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (toEnum $ fromEnum b0)

-- | spatialMaxUnpooling forward pass (updates the output tensor)
_spatialMaxUnpooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateOutput t0 t1 ix0 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialMaxUnpooling_updateOutput s' t0' t1' ix0' (fromIntegral i0) (fromIntegral i1)


-- | spatialMaxUnpooling backward pass (updates the gradInput tensor)
_spatialMaxUnpooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateGradInput t0 t1 t2 ix0 i0 i1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_SpatialMaxUnpooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1)


-- | volumetricFractionalMaxPooling forward pass (updates the output tensor)
_volumetricFractionalMaxPooling_updateOutput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IndexDynamic -> Dynamic -> IO ()
_volumetricFractionalMaxPooling_updateOutput t0 t1 i0 i1 i2 i3 i4 i5 ix0 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricFractionalMaxPooling_updateOutput s' t0' t1'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) ix0' t2'


-- | volumetricFractionalMaxPooling backward pass (updates the gradInput tensor)
_volumetricFractionalMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IndexDynamic -> IO ()
_volumetricFractionalMaxPooling_updateGradInput t0 t1 t2 i0 i1 i2 i3 i4 i5 ix0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricFractionalMaxPooling_updateGradInput s' t0' t1' t2'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) ix0'

-- | volumetricMaxPooling forward pass (updates the output tensor)
_volumetricMaxPooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateOutput t0 t1 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricMaxPooling_updateOutput s' t0' t1' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8) (toEnum $ fromEnum b0)

-- | volumetricMaxPooling backward pass (updates the gradInput tensor)
_volumetricMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateGradInput t0 t1 t2 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricMaxPooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8) (toEnum $ fromEnum b0)



-- | volumetricDilatedMaxPooling forward pass (updates the output tensor)
_volumetricDilatedMaxPooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateOutput t0 t1 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricDilatedMaxPooling_updateOutput s' t0' t1' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8) (fromIntegral i9) (fromIntegral i10) (fromIntegral i11)
        (toEnum $ fromEnum b0)

-- | volumetricDilatedMaxPooling backward pass (updates the gradInput tensor)
_volumetricDilatedMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricDilatedMaxPooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8) (fromIntegral i9) (fromIntegral i10) (fromIntegral i11)
        (toEnum $ fromEnum b0)

-- | volumetricMaxUnpooling forward pass (updates the output tensor)
_volumetricMaxUnpooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateOutput t0 t1 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricMaxUnpooling_updateOutput s' t0' t1' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8)

-- | volumetricMaxUnpooling backward pass (updates the gradInput tensor)
_volumetricMaxUnpooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateGradInput t0 t1 t2 ix0 i0 i1 i2 i3 i4 i5 i6 i7 i8 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricMaxUnpooling_updateGradInput s' t0' t1' t2' ix0'
        (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)
        (fromIntegral i4) (fromIntegral i5) (fromIntegral i6) (fromIntegral i7)
        (fromIntegral i8)

-- | volumetricAdaptiveMaxPooling forward pass (updates the output tensor)
_volumetricAdaptiveMaxPooling_updateOutput :: Dynamic -> Dynamic -> IndexDynamic -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveMaxPooling_updateOutput t0 t1 ix0 i0 i1 i2 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricAdaptiveMaxPooling_updateOutput s' t0' t1' ix0' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)

-- | volumetricAdaptiveMaxPooling backward pass (updates the gradInput tensor)
_volumetricAdaptiveMaxPooling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> IO ()
_volumetricAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    Ix.withDynamicState ix0 $ \_ ix0' ->
      Sig.c_VolumetricAdaptiveMaxPooling_updateGradInput s' t0' t1' t2' ix0'


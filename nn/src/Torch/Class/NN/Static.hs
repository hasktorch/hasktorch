module Torch.Class.NN.Static where

import Foreign.C.Types
import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad.Trans.Class
import Control.Monad.IO.Class
import Torch.Class.NN.Static.Abs
import Debug.Trace

class AbsCriterion (t :: [Nat] -> *) where
  _absCriterion_updateOutput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ output
    -> Bool    -- ^ size average
    -> Bool    -- ^ reduce
    -> IO ()

  _absCriterion_updateGradInput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ gradOutput
    -> t d''   -- ^ gradInput
    -> Bool    -- ^ size average
    -> Bool    -- ^ reduce
    -> IO ()

class BCECriterion (t :: [Nat] -> *) where
  _bCECriterion_updateOutput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ output
    -> Bool    -- ^ sizeAverage
    -> t d'''  -- ^ weights
    -> Bool    -- ^ reduce
    -> IO ()
  _bCECriterion_updateGradInput
    :: t d      -- ^ input
    -> t d'     -- ^ target
    -> t d''    -- ^ grad output
    -> t d'''   -- ^ grad input
    -> Bool     -- ^  sizeAvreage
    -> t d''''  -- ^ weights
    -> Bool     -- ^ reduce
    -> IO ()

class BatchNormalization (t :: [Nat] -> *) where
  _batchNormalization_updateOutput 
    :: t d    -- ^ input
    -> t d    -- ^ output
    -> t d    -- ^ weight
    -> t d    -- ^ bias
    -> t d    -- ^ running mean
    -> t d    -- ^ running var
    -> t d    -- ^ save mean
    -> t d    -- ^ save std
    -> Bool   -- ^ train
    -> Double -- ^ momentum
    -> Double -- ^ eps
    -> IO ()

  _batchNormalization_backward
    :: t d      -- ^ input
    -> t d      -- ^ grad output
    -> t d      -- ^ grad input
    -> t d      -- ^ grad weight
    -> t d      -- ^ grad bias
    -> t d      -- ^ weight
    -> t d      -- ^ running mean
    -> t d      -- ^ running var
    -> t d      -- ^ save mean
    -> t d      -- ^ save std
    -> Bool     -- ^ train
    -> Double   -- ^ momentum
    -> Double   -- ^ eps
    -> IO ()

-- In latest ATen master. Need to bump hasktorch's fork to this
-- class ClassNLLCriterion (t :: [Nat] -> *) where

class Col2Im (t :: [Nat] -> *) where
  _col2Im_updateOutput
    :: t d -- ^ input
    -> t d -- ^ output
    -> Int -- ^ output Height
    -> Int -- ^ output Width
    -> Int -- ^ kH
    -> Int -- ^ kW
    -> Int -- ^ dH
    -> Int -- ^ dW
    -> Int -- ^ padH
    -> Int -- ^ padW
    -> Int -- ^ sH
    -> Int -- ^ sW
    -> IO ()
  _col2Im_updateGradInput
    :: t d -- ^ grad output
    -> t d -- ^ grad input
    -> Int -- ^ kH
    -> Int -- ^ kW
    -> Int -- ^ dH
    -> Int -- ^ dW
    -> Int -- ^ padH
    -> Int -- ^ padW
    -> Int -- ^ sH
    -> Int -- ^ sW
    -> IO ()

class SparseLinear (t :: [Nat] -> *) where
  _sparseLinear_updateOutput                          :: t d -> t d -> t d -> t d -> IO ()
  _sparseLinear_accGradParameters                     :: t d -> t d -> t d -> t d -> t d -> t d -> Double -> Double -> IO ()
  _sparseLinear_zeroGradParameters                    :: t d -> t d -> t d -> IO ()
  _sparseLinear_updateParameters                      :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()
  _sparseLinear_legacyUpdateOutput                    :: t d -> t d -> t d -> t d -> IO ()
  _sparseLinear_legacyAccGradParameters               :: t d -> t d -> t d -> t d -> t d -> t d -> Double -> Double -> IO ()

class Sqrt (t :: [Nat] -> *) where
  _sqrt_updateOutput                                  :: t d -> t d -> Double -> IO ()
  _sqrt_updateGradInput                               :: t d -> t d -> t d -> t d -> IO ()

class Square (t :: [Nat] -> *) where
  _square_updateOutput                                :: t d -> t d -> IO ()
  _square_updateGradInput                             :: t d -> t d -> t d -> IO ()

-- DistKLDivCriterion.c
-- ELU.c
-- FeatureLPPooling.c
-- FusedRNNKernel.c
-- GatedLinearUnit.c
-- HardShrink.c
-- HardTanh.c
-- Im2Col.c
-- IndexLinear.c
-- L1Cost.c
-- LeakyReLU.c
-- Linear.c
-- LogSigmoid.c
-- LogSoftMax.c
-- LookupTable.c
-- MarginCriterion.c
-- MSECriterion.c
-- MultiLabelMarginCriterion.c
-- MultiMarginCriterion.c
-- PReLU.c
-- RReLU.c
-- Sigmoid.c
-- SmoothL1Criterion.c
-- SoftMarginCriterion.c
-- SoftMax.c
-- SoftPlus.c
-- SoftShrink.c
-- SpatialAdaptiveAveragePooling.c
-- SpatialAdaptiveMaxPooling.c
-- SpatialAveragePooling.c
-- SpatialClassNLLCriterion.c
-- SpatialConvolutionLocal.c
-- SpatialConvolutionMap.c
-- SpatialConvolutionMM.c
-- SpatialDilatedConvolution.c
-- SpatialDilatedMaxPooling.c
-- SpatialFractionalMaxPooling.c
-- SpatialFullConvolution.c
-- SpatialFullConvolutionMap.c
-- SpatialFullDilatedConvolution.c
-- SpatialGridSamplerBilinear.c
-- SpatialMaxPooling.c
-- SpatialMaxUnpooling.c
-- SpatialReflectionPadding.c
-- SpatialReplicationPadding.c
-- SpatialSubSampling.c
-- SpatialUpSamplingBilinear.c
-- SpatialUpSamplingNearest.c
-- Tanh.c
-- TemporalConvolution.c
-- TemporalMaxPooling.c
-- TemporalReflectionPadding.c
-- TemporalReplicationPadding.c
-- TemporalRowConvolution.c
-- TemporalSubSampling.c
-- TemporalUpSamplingLinear.c
-- TemporalUpSamplingNearest.c
-- Threshold.c
-- unfold.c
-- VolumetricAdaptiveAveragePooling.c
-- VolumetricAdaptiveMaxPooling.c
-- VolumetricAveragePooling.c
-- VolumetricConvolution.c
-- VolumetricConvolutionMM.c
-- VolumetricDilatedConvolution.c
-- VolumetricDilatedMaxPooling.c
-- VolumetricFractionalMaxPooling.c
-- VolumetricFullConvolution.c
-- VolumetricFullDilatedConvolution.c
-- VolumetricGridSamplerBilinear.c
-- VolumetricMaxPooling.c
-- VolumetricMaxUnpooling.c
-- VolumetricReplicationPadding.c
-- VolumetricUpSamplingNearest.c
-- VolumetricUpSamplingTrilinear.c

class
  ( Abs t
  , AbsCriterion t
  , BCECriterion t
  , BatchNormalization t
  , Col2Im t
  , IsTensor t
  ) => NN (t :: [Nat] -> *) where

  eLU_updateOutput             :: t d -> t d -> Double -> Double -> Bool -> IO ()
  eLU_updateGradInput          :: t d -> t d' -> t d'' -> Double -> Double -> IO ()

  im2Col_updateOutput          :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  im2Col_updateGradInput       :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  distKLDivCriterion_updateOutput    :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  distKLDivCriterion_updateGradInput :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()
  gatedLinear_updateOutput     :: t d -> t d -> Int -> IO ()
  gatedLinear_updateGradInput  :: t d -> t d -> t d -> Int -> IO ()
  hardTanh_updateOutput        :: t d -> t d -> Double -> Double -> Bool -> IO ()
  hardTanh_updateGradInput     :: t d -> t d -> t d -> Double -> Double -> Bool -> IO ()
  l1Cost_updateOutput          :: t d -> t d -> IO ()
  l1Cost_updateGradInput       :: t d -> t d -> t d -> IO ()
  leakyReLU_updateOutput       :: t d -> t d -> Double -> Bool -> IO ()
  leakyReLU_updateGradInput    :: t d -> t d -> t d -> Double -> Bool -> IO ()
  gRUFused_updateOutput        :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()
  gRUFused_updateGradInput     :: t d -> t d -> t d -> t d -> t d -> IO ()
  lSTMFused_updateOutput       :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()
  lSTMFused_updateGradInput    :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()
  logSigmoid_updateOutput      :: t d -> t d -> t d -> IO ()
  logSigmoid_updateGradInput   :: t d -> t d -> t d -> t d -> IO ()
  logSoftMax_updateOutput      :: t d -> t d -> DimReal (t d) -> IO ()
  logSoftMax_updateGradInput   :: t d -> t d -> t d -> t d -> DimReal (t d) -> IO ()
  marginCriterion_updateOutput                       :: t d -> t d -> t d -> Bool -> Double -> IO ()
  marginCriterion_updateGradInput                    :: t d -> t d -> t d -> Bool -> Double -> IO ()
  softMarginCriterion_updateOutput                   :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  softMarginCriterion_updateGradInput                :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()
  mSECriterion_updateOutput                          :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  mSECriterion_updateGradInput                       :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()
  pReLU_updateOutput                                 :: t d -> t d -> t d -> IO ()
  pReLU_updateGradInput                              :: t d -> t d -> t d -> t d -> IO ()
  pReLU_accGradParameters                            :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()
  rReLU_updateOutput                                 :: t d -> t d -> t d -> Double -> Double -> Bool -> Bool -> Generator (t d) -> IO ()
  rReLU_updateGradInput                              :: t d -> t d -> t d -> t d -> Double -> Double -> Bool -> Bool -> IO ()
  sigmoid_updateOutput                               :: t d -> t d -> IO ()
  sigmoid_updateGradInput                            :: t d -> t d -> t d -> IO ()
  smoothL1Criterion_updateOutput                     :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  smoothL1Criterion_updateGradInput                  :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()
  softMax_updateOutput                               :: t d -> t d -> DimReal (t d) -> IO ()
  softMax_updateGradInput                            :: t d -> t d -> t d -> t d -> DimReal (t d) -> IO ()
  softPlus_updateOutput                              :: t d -> t d -> Double -> Double -> IO ()
  softPlus_updateGradInput                           :: t d -> t d -> t d -> t d -> Double -> Double -> IO ()
  softShrink_updateOutput                            :: t d -> t d -> Double -> IO ()
  softShrink_updateGradInput                         :: t d -> t d -> t d -> Double -> IO ()

  tanh_updateOutput                                  :: t d -> t d -> IO ()
  tanh_updateGradInput                               :: t d -> t d -> t d -> IO ()
  threshold_updateOutput                             :: t d -> t d -> Double -> Double -> Bool -> IO ()
  threshold_updateGradInput                          :: t d -> t d -> t d -> Double -> Double -> Bool -> IO ()
  temporalConvolution_updateOutput                   :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  temporalConvolution_updateGradInput                :: t d -> t d -> t d -> t d -> Int -> Int -> IO ()
  temporalConvolution_accGradParameters              :: t d -> t d -> t d -> t d -> Int -> Int -> Double -> IO ()
  temporalRowConvolution_updateOutput                :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  temporalRowConvolution_updateGradInput             :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  temporalRowConvolution_accGradParameters           :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> Double -> IO ()
  temporalUpSamplingNearest_updateOutput             :: t d -> t d -> Int -> IO ()
  temporalUpSamplingNearest_updateGradInput          :: t d -> t d -> t d -> Int -> IO ()
  temporalUpSamplingLinear_updateOutput              :: t d -> t d -> Int -> IO ()
  temporalUpSamplingLinear_updateGradInput           :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialConvolutionMM_updateOutput                  :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialConvolutionMM_updateGradInput               :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialConvolutionMM_accGradParameters             :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialConvolutionLocal_updateOutput               :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  spatialConvolutionLocal_updateGradInput            :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  spatialConvolutionLocal_accGradParameters          :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> Double -> IO ()
  spatialAdaptiveAveragePooling_updateOutput         :: t d -> t d -> Int -> Int -> IO ()
  spatialAdaptiveAveragePooling_updateGradInput      :: t d -> t d -> t d -> IO ()
  spatialAveragePooling_updateOutput                 :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  spatialAveragePooling_updateGradInput              :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  spatialFullConvolution_updateOutput                :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolution_updateGradInput             :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolution_accGradParameters           :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialDilatedConvolution_updateOutput             :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialDilatedConvolution_updateGradInput          :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialDilatedConvolution_accGradParameters        :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialFullDilatedConvolution_updateOutput         :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullDilatedConvolution_updateGradInput      :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullDilatedConvolution_accGradParameters    :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialSubSampling_updateOutput                    :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialSubSampling_updateGradInput                 :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialSubSampling_accGradParameters               :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialUpSamplingNearest_updateOutput              :: t d -> t d -> Int -> IO ()
  spatialUpSamplingNearest_updateGradInput           :: t d -> t d -> t d -> Int -> IO ()
  spatialUpSamplingBilinear_updateOutput             :: t d -> t d -> Int -> Int -> IO ()
  spatialUpSamplingBilinear_updateGradInput          :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialGridSamplerBilinear_updateOutput            :: t d -> t d -> t d -> Int -> IO ()
  spatialGridSamplerBilinear_updateGradInput         :: t d -> t d -> t d -> t d -> t d -> Int -> IO ()
  volumetricGridSamplerBilinear_updateOutput         :: t d -> t d -> t d -> Int -> IO ()
  volumetricGridSamplerBilinear_updateGradInput      :: t d -> t d -> t d -> t d -> t d -> Int -> IO ()
  volumetricAveragePooling_updateOutput              :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  volumetricAveragePooling_updateGradInput           :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  volumetricConvolution_updateOutput                 :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolution_updateGradInput              :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolution_accGradParameters            :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  volumetricFullConvolution_updateOutput             :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricFullConvolution_updateGradInput          :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricFullConvolution_accGradParameters        :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  volumetricDilatedConvolution_updateOutput          :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricDilatedConvolution_updateGradInput       :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricDilatedConvolution_accGradParameters     :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  volumetricFullDilatedConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricFullDilatedConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricFullDilatedConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  volumetricAdaptiveAveragePooling_updateOutput      :: t d -> t d -> Int -> Int -> Int -> IO ()
  volumetricAdaptiveAveragePooling_updateGradInput   :: t d -> t d -> t d -> IO ()
  spatialReflectionPadding_updateOutput              :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReflectionPadding_updateGradInput           :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReplicationPadding_updateOutput             :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReplicationPadding_updateGradInput          :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  featureLPPooling_updateOutput                      :: t d -> t d -> Double -> Int -> Int -> Bool -> IO ()
  featureLPPooling_updateGradInput                   :: t d -> t d -> t d -> t d -> Double -> Int -> Int -> Bool -> IO ()
  volumetricReplicationPadding_updateOutput          :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricReplicationPadding_updateGradInput       :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricUpSamplingNearest_updateOutput           :: t d -> t d -> Int -> IO ()
  volumetricUpSamplingNearest_updateGradInput        :: t d -> t d -> t d -> Int -> IO ()
  volumetricUpSamplingTrilinear_updateOutput         :: t d -> t d -> Int -> Int -> Int -> IO ()
  volumetricUpSamplingTrilinear_updateGradInput      :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  temporalReflectionPadding_updateOutput             :: t d -> t d -> Int -> Int -> IO ()
  temporalReflectionPadding_updateGradInput          :: t d -> t d -> t d -> Int -> Int -> IO ()
  temporalReplicationPadding_updateOutput            :: t d -> t d -> Int -> Int -> IO ()
  temporalReplicationPadding_updateGradInput         :: t d -> t d -> t d -> Int -> Int -> IO ()

class CPUNN t d where
  unfolded_acc  :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  unfolded_copy :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_updateOutput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_updateGradInput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  temporalSubSampling_updateOutput :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> IO ()
  temporalSubSampling_updateGradInput :: t d -> t d -> t d -> t d -> Int -> Int -> IO ()
  temporalSubSampling_accGradParameters :: t d -> t d -> t d -> t d -> Int -> Int -> Double -> IO ()
  spatialFullConvolutionMap_updateOutput :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolutionMap_updateGradInput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolutionMap_accGradParameters :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Double -> IO ()
  hardShrink_updateOutput      :: t d -> t d -> Double -> IO ()
  hardShrink_updateGradInput   :: t d -> t d -> t d -> Double -> IO ()
  linear_updateOutput      :: t d -> t d -> t d -> t d -> t d -> IO ()
  linear_updateGradInput   :: t d -> t d -> t d -> t d -> IO ()
  linear_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> t d -> Double -> IO ()
  sparseLinear_legacyZeroGradParameters :: t d -> t d -> t d -> IO ()
  sparseLinear_legacyUpdateParameters   :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()

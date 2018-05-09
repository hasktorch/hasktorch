module Torch.Indef.Dynamic.NN where

import Torch.Dimensions

import Foreign.C.Types
import Torch.Sig.Types.NN
import qualified Torch.Sig.NN as Sig

import Torch.Indef.Types

_abs_updateOutput :: Dynamic -> Dynamic -> IO ()
_abs_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_abs_updateOutput = ten2 Sig.c_Abs_updateOutput
_abs_updateGradInput = ten3 Sig.c_Abs_updateGradInput

_absCriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_absCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_bCECriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Dynamic -> Bool -> IO ()
_bCECriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Dynamic -> Bool -> IO ()
_eLU_updateOutput             :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_eLU_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
_distKLDivCriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_distKLDivCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_gatedLinear_updateOutput     :: Dynamic -> Dynamic -> Int -> IO ()
_gatedLinear_updateGradInput  :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_hardTanh_updateOutput        :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_hardTanh_updateGradInput     :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_im2Col_updateOutput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_im2Col_updateGradInput       :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()

_col2Im_updateOutput
  :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_col2Im_updateOutput t0 t1 a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_Col2Im_updateOutput s' t0' t1'
      (fromIntegral a0)
      (fromIntegral a1)
      (fromIntegral a2)
      (fromIntegral a3)
      (fromIntegral a4)
      (fromIntegral a5)
      (fromIntegral a6)
      (fromIntegral a7)
      (fromIntegral a8)
      (fromIntegral a9)


_col2Im_updateGradInput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_col2Im_updateGradInput t0 t1 a0 a1 a2 a3 a4 a5 a6 a7 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_Col2Im_updateGradInput s' t0' t1'
      (fromIntegral a0)
      (fromIntegral a1)
      (fromIntegral a2)
      (fromIntegral a3)
      (fromIntegral a4)
      (fromIntegral a5)
      (fromIntegral a6)
      (fromIntegral a7)


_l1Cost_updateOutput          :: Dynamic -> Dynamic -> IO ()
_l1Cost_updateGradInput       :: Dynamic -> Dynamic -> Dynamic -> IO ()
_leakyReLU_updateOutput       :: Dynamic -> Dynamic -> Double -> Bool -> IO ()
_leakyReLU_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Double -> Bool -> IO ()
_gRUFused_updateOutput        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_gRUFused_updateGradInput     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_lSTMFused_updateOutput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_lSTMFused_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_logSigmoid_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> IO ()
_logSigmoid_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_logSoftMax_updateOutput      :: Dynamic -> Dynamic -> Integer -> IO ()
_logSoftMax_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Integer -> IO ()
_marginCriterion_updateOutput        :: Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> IO ()
_marginCriterion_updateGradInput     :: Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> IO ()
_softMarginCriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_softMarginCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_mSECriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_mSECriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_pReLU_updateOutput       :: Dynamic -> Dynamic -> Dynamic -> IO ()
_pReLU_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_pReLU_accGradParameters  :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
_rReLU_updateOutput       :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> Bool -> Generator -> IO ()
_rReLU_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> Bool -> IO ()
_sigmoid_updateOutput     :: Dynamic -> Dynamic -> IO ()
_sigmoid_updateGradInput  :: Dynamic -> Dynamic -> Dynamic -> IO ()
_smoothL1Criterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_smoothL1Criterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_softMax_updateOutput       :: Dynamic -> Dynamic -> Integer -> IO ()
_softMax_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Integer -> IO ()
_softPlus_updateOutput      :: Dynamic -> Dynamic -> Double -> Double -> IO ()
_softPlus_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
_softShrink_updateOutput    :: Dynamic -> Dynamic -> Double -> IO ()
_softShrink_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
_sparseLinear_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_sparseLinear_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
_sparseLinear_zeroGradParameters       :: Dynamic -> Dynamic -> Dynamic -> IO ()
_sparseLinear_updateParameters         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
_sparseLinear_legacyUpdateOutput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_sparseLinear_legacyAccGradParameters  :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
_sqrt_updateOutput         :: Dynamic -> Dynamic -> Double -> IO ()
_sqrt_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
_square_updateOutput       :: Dynamic -> Dynamic -> IO ()
_square_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> IO ()
_tanh_updateOutput         :: Dynamic -> Dynamic -> IO ()
_tanh_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> IO ()
_threshold_updateOutput    :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_threshold_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
_temporalConvolution_updateOutput                   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_temporalConvolution_updateGradInput                :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()
_temporalConvolution_accGradParameters              :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Double -> IO ()
_temporalRowConvolution_updateOutput                :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_updateGradInput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> IO ()
_temporalRowConvolution_accGradParameters           :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> Double -> IO ()
_temporalUpSamplingNearest_updateOutput             :: Dynamic -> Dynamic -> Int -> IO ()
_temporalUpSamplingNearest_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_temporalUpSamplingLinear_updateOutput              :: Dynamic -> Dynamic -> Int -> IO ()
_temporalUpSamplingLinear_updateGradInput           :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_batchNormalization_updateOutput                    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> Double -> IO ()
_batchNormalization_backward                        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> Double -> IO ()

_spatialConvolutionMM_updateOutput
  :: Dynamic    -- ^ input
  -> Dynamic    -- ^ output
  -> Dynamic    -- ^ 3D weight tensor (connTable:size(1) x kH x kW) 
  -> Dynamic    -- ^ 1D bias tensor (nOutputPlane) 
  -> Dynamic    -- ^ finput
  -> Dynamic    -- ^ fgradInput
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.
  -> IO ()
_spatialConvolutionMM_updateOutput t0 t1 t2 t3 t4 t5 (kW, kH) (dW, dH) (pW, pH) =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    Sig.c_SpatialConvolutionMM_updateOutput s' t0' t1' t2' t3' t4' t5'
      (fromIntegral kW) (fromIntegral kH)
      (fromIntegral dW) (fromIntegral dH)
      (fromIntegral pW) (fromIntegral pH)

_spatialConvolutionMM_updateGradInput               :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialConvolutionMM_accGradParameters             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialConvolutionLocal_updateOutput               :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
_spatialConvolutionLocal_updateGradInput            :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
_spatialConvolutionLocal_accGradParameters          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> Double -> IO ()
_spatialAdaptiveAveragePooling_updateOutput         :: Dynamic -> Dynamic -> Int -> Int -> IO ()
_spatialAdaptiveAveragePooling_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> IO ()
_spatialAveragePooling_updateOutput                 :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateGradInput              :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialFullConvolution_updateOutput                :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialFullConvolution_updateGradInput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialFullConvolution_accGradParameters           :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialDilatedConvolution_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialDilatedConvolution_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialDilatedConvolution_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialFullDilatedConvolution_updateOutput         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialFullDilatedConvolution_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialFullDilatedConvolution_accGradParameters    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialSubSampling_updateOutput                    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_spatialSubSampling_updateGradInput                 :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_spatialSubSampling_accGradParameters               :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialUpSamplingNearest_updateOutput              :: Dynamic -> Dynamic -> Int -> IO ()
_spatialUpSamplingNearest_updateGradInput           :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_spatialUpSamplingBilinear_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> IO ()
_spatialUpSamplingBilinear_updateGradInput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialGridSamplerBilinear_updateOutput            :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_spatialGridSamplerBilinear_updateGradInput         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_volumetricGridSamplerBilinear_updateOutput         :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_volumetricGridSamplerBilinear_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_volumetricAveragePooling_updateOutput              :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateGradInput           :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricConvolution_updateOutput                 :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricConvolution_updateGradInput              :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricConvolution_accGradParameters            :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_volumetricFullConvolution_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullConvolution_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullConvolution_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_volumetricDilatedConvolution_updateOutput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricDilatedConvolution_updateGradInput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricDilatedConvolution_accGradParameters     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_volumetricFullDilatedConvolution_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullDilatedConvolution_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullDilatedConvolution_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
_volumetricAdaptiveAveragePooling_updateOutput      :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveAveragePooling_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> IO ()
_spatialReflectionPadding_updateOutput              :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_spatialReflectionPadding_updateGradInput           :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
_featureLPPooling_updateOutput                      :: Dynamic -> Dynamic -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateGradInput                   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Int -> Int -> Bool -> IO ()
_volumetricReplicationPadding_updateOutput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricReplicationPadding_updateGradInput       :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricUpSamplingNearest_updateOutput           :: Dynamic -> Dynamic -> Int -> IO ()
_volumetricUpSamplingNearest_updateGradInput        :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
_volumetricUpSamplingTrilinear_updateOutput         :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
_volumetricUpSamplingTrilinear_updateGradInput      :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_temporalReflectionPadding_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> IO ()
_temporalReflectionPadding_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()
_temporalReplicationPadding_updateOutput            :: Dynamic -> Dynamic -> Int -> Int -> IO ()
_temporalReplicationPadding_updateGradInput         :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()

_absCriterion_updateOutput = ten3bool2 Sig.c_AbsCriterion_updateOutput
_absCriterion_updateGradInput = ten4bool2 Sig.c_AbsCriterion_updateGradInput
_sqrt_updateGradInput = ten4 Sig.c_Sqrt_updateGradInput
_square_updateOutput = ten2 Sig.c_Square_updateOutput
_square_updateGradInput = ten3 Sig.c_Square_updateGradInput
_tanh_updateOutput = ten2 Sig.c_Tanh_updateOutput
_tanh_updateGradInput = ten3 Sig.c_Tanh_updateGradInput
_l1Cost_updateOutput = ten2 Sig.c_L1Cost_updateOutput
_l1Cost_updateGradInput = ten3 Sig.c_L1Cost_updateGradInput
_logSigmoid_updateOutput = ten3 Sig.c_LogSigmoid_updateOutput
_logSigmoid_updateGradInput = ten4 Sig.c_LogSigmoid_updateGradInput
_sigmoid_updateOutput = ten2 Sig.c_Sigmoid_updateOutput
_sigmoid_updateGradInput = ten3 Sig.c_Sigmoid_updateGradInput
_eLU_updateOutput = ten2double2bool1 Sig.c_ELU_updateOutput
_logSoftMax_updateOutput = ten2dim1 Sig.c_LogSoftMax_updateOutput
_im2Col_updateOutput = ten2int8 Sig.c_Im2Col_updateOutput
_im2Col_updateGradInput = ten2int10 Sig.c_Im2Col_updateGradInput
_gRUFused_updateGradInput = ten5 Sig.c_GRUFused_updateGradInput
_pReLU_updateGradInput = ten4 Sig.c_PReLU_updateGradInput
_spatialAdaptiveAveragePooling_updateGradInput = ten3 Sig.c_SpatialAdaptiveAveragePooling_updateGradInput
_softMax_updateOutput = ten2dim1 Sig.c_SoftMax_updateOutput
_pReLU_updateOutput = ten3 Sig.c_PReLU_updateOutput
_distKLDivCriterion_updateOutput = ten3bool2 Sig.c_DistKLDivCriterion_updateOutput
_bCECriterion_updateOutput = ten3bool1ten1bool1 Sig.c_BCECriterion_updateOutput
_marginCriterion_updateGradInput = ten3bool1double1 Sig.c_MarginCriterion_updateGradInput
_distKLDivCriterion_updateGradInput = ten4bool2 Sig.c_DistKLDivCriterion_updateGradInput
_marginCriterion_updateOutput = ten3bool1double1 Sig.c_MarginCriterion_updateOutput
_smoothL1Criterion_updateGradInput = ten4bool2 Sig.c_SmoothL1Criterion_updateGradInput
_softMarginCriterion_updateGradInput = ten4bool2 Sig.c_SoftMarginCriterion_updateGradInput
_mSECriterion_updateGradInput = ten4bool2 Sig.c_MSECriterion_updateGradInput
_bCECriterion_updateGradInput = ten4bool1ten1bool1 Sig.c_BCECriterion_updateGradInput
_smoothL1Criterion_updateOutput = ten3bool2 Sig.c_SmoothL1Criterion_updateOutput
_softMarginCriterion_updateOutput = ten3bool2 Sig.c_SoftMarginCriterion_updateOutput
_mSECriterion_updateOutput = ten3bool2 Sig.c_MSECriterion_updateOutput
_leakyReLU_updateOutput = ten2double1bool1 Sig.c_LeakyReLU_updateOutput
_sqrt_updateOutput = ten2double1 Sig.c_Sqrt_updateOutput
_softShrink_updateOutput = ten2double1 Sig.c_SoftShrink_updateOutput
_softPlus_updateOutput = ten2double2 Sig.c_SoftPlus_updateOutput
_threshold_updateOutput = ten2double2bool1 Sig.c_Threshold_updateOutput
_hardTanh_updateOutput = ten2double2bool1 Sig.c_HardTanh_updateOutput
_eLU_updateGradInput = ten3double2 Sig.c_ELU_updateGradInput
_hardTanh_updateGradInput = ten3double2bool1 Sig.c_HardTanh_updateGradInput
_leakyReLU_updateGradInput = ten3double1bool1 Sig.c_LeakyReLU_updateGradInput
_softShrink_updateGradInput = ten3double1 Sig.c_SoftShrink_updateGradInput
_softPlus_updateGradInput = ten4double2 Sig.c_SoftPlus_updateGradInput
_rReLU_updateOutput = ten3double2bool2gen1 Sig.c_RReLU_updateOutput
_rReLU_updateGradInput = ten4double2bool2 Sig.c_RReLU_updateGradInput
_threshold_updateGradInput = ten3double2bool1 Sig.c_Threshold_updateGradInput
_logSoftMax_updateGradInput = ten4dim1 Sig.c_LogSoftMax_updateGradInput
_softMax_updateGradInput = ten4dim1 Sig.c_SoftMax_updateGradInput
_temporalConvolution_accGradParameters = ten4int2double1 Sig.c_TemporalConvolution_accGradParameters
_spatialSubSampling_updateOutput = ten4int4 Sig.c_SpatialSubSampling_updateOutput
_spatialSubSampling_updateGradInput = ten4int4 Sig.c_SpatialSubSampling_updateGradInput
_spatialSubSampling_accGradParameters = ten4int4double1 Sig.c_SpatialSubSampling_accGradParameters
_spatialGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_SpatialGridSamplerBilinear_updateGradInput
_pReLU_accGradParameters = ten5double1 Sig.c_PReLU_accGradParameters
_sparseLinear_updateParameters = ten5double1 Sig.c_SparseLinear_updateParameters
_volumetricGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_VolumetricGridSamplerBilinear_updateGradInput
_spatialGridSamplerBilinear_updateOutput = ten3int1 Sig.c_SpatialGridSamplerBilinear_updateOutput
_volumetricGridSamplerBilinear_updateOutput = ten3int1 Sig.c_VolumetricGridSamplerBilinear_updateOutput
_spatialUpSamplingNearest_updateGradInput = ten3int1 Sig.c_SpatialUpSamplingNearest_updateGradInput
_temporalUpSamplingNearest_updateGradInput = ten3int1 Sig.c_TemporalUpSamplingNearest_updateGradInput
_gatedLinear_updateGradInput = ten3int1 Sig.c_GatedLinear_updateGradInput
_temporalUpSamplingNearest_updateOutput = ten2int1 Sig.c_TemporalUpSamplingNearest_updateOutput
_temporalUpSamplingLinear_updateOutput = ten2int1 Sig.c_TemporalUpSamplingLinear_updateOutput
_gatedLinear_updateOutput = ten2int1 Sig.c_GatedLinear_updateOutput
_spatialUpSamplingNearest_updateOutput = ten2int1 Sig.c_SpatialUpSamplingNearest_updateOutput
_spatialUpSamplingBilinear_updateOutput = ten2int2 Sig.c_SpatialUpSamplingBilinear_updateOutput
_spatialAdaptiveAveragePooling_updateOutput = ten2int2 Sig.c_SpatialAdaptiveAveragePooling_updateOutput
_temporalConvolution_updateGradInput = ten4int2 Sig.c_TemporalConvolution_updateGradInput
_temporalUpSamplingLinear_updateGradInput = ten2int4 Sig.c_TemporalUpSamplingLinear_updateGradInput
_spatialUpSamplingBilinear_updateGradInput = ten2int6 Sig.c_SpatialUpSamplingBilinear_updateGradInput
_volumetricConvolution_updateGradInput = ten5int6 Sig.c_VolumetricConvolution_updateGradInput
_spatialAveragePooling_updateOutput = ten2int6bool2 Sig.c_SpatialAveragePooling_updateOutput
_volumetricAveragePooling_updateOutput = ten2int9bool2 Sig.c_VolumetricAveragePooling_updateOutput
_spatialAveragePooling_updateGradInput = ten3int6bool2 Sig.c_SpatialAveragePooling_updateGradInput
_volumetricAveragePooling_updateGradInput = ten3int9bool2 Sig.c_VolumetricAveragePooling_updateGradInput
_temporalConvolution_updateOutput = ten4int4 Sig.c_TemporalConvolution_updateOutput
_spatialDilatedConvolution_updateGradInput = ten5int8 Sig.c_SpatialDilatedConvolution_updateGradInput
_spatialFullConvolution_updateGradInput = ten5int8 Sig.c_SpatialFullConvolution_updateGradInput
_spatialFullDilatedConvolution_updateGradInput = ten5int10 Sig.c_SpatialFullDilatedConvolution_updateGradInput
_temporalRowConvolution_updateOutput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateOutput
_temporalRowConvolution_updateGradInput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateGradInput
_temporalRowConvolution_accGradParameters = ten6int3bool1double1 Sig.c_TemporalRowConvolution_accGradParameters
_sparseLinear_legacyAccGradParameters = ten6double2 Sig.c_SparseLinear_legacyAccGradParameters
_sparseLinear_accGradParameters = ten6double2 Sig.c_SparseLinear_accGradParameters
_spatialConvolutionMM_updateGradInput = ten6int6 Sig.c_SpatialConvolutionMM_updateGradInput
_spatialConvolutionMM_accGradParameters = ten6int6double1 Sig.c_SpatialConvolutionMM_accGradParameters
_spatialConvolutionLocal_updateOutput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateOutput
_spatialConvolutionLocal_updateGradInput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateGradInput
_spatialConvolutionLocal_accGradParameters = ten6int6long4double1 Sig.c_SpatialConvolutionLocal_accGradParameters
_volumetricConvolution_updateOutput = ten6int6 Sig.c_VolumetricConvolution_updateOutput
_spatialFullConvolution_updateOutput = ten6int8 Sig.c_SpatialFullConvolution_updateOutput
_spatialFullConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialFullConvolution_accGradParameters
_spatialDilatedConvolution_updateOutput = ten6int8 Sig.c_SpatialDilatedConvolution_updateOutput
_spatialDilatedConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialDilatedConvolution_accGradParameters
_spatialFullDilatedConvolution_updateOutput = ten6int10 Sig.c_SpatialFullDilatedConvolution_updateOutput
_spatialFullDilatedConvolution_accGradParameters = ten6int10double1 Sig.c_SpatialFullDilatedConvolution_accGradParameters
_gRUFused_updateOutput = ten7 Sig.c_GRUFused_updateOutput
_lSTMFused_updateOutput = ten7 Sig.c_LSTMFused_updateOutput
_lSTMFused_updateGradInput = ten7 Sig.c_LSTMFused_updateGradInput
_batchNormalization_updateOutput = ten8bool1double2 Sig.c_BatchNormalization_updateOutput
_batchNormalization_backward = ten10bool1double2 Sig.c_BatchNormalization_backward
_volumetricConvolution_accGradParameters = ten6int6double1 Sig.c_VolumetricConvolution_accGradParameters
_volumetricFullConvolution_updateOutput = ten6int12 Sig.c_VolumetricFullConvolution_updateOutput
_volumetricFullConvolution_updateGradInput = ten6int12 Sig.c_VolumetricFullConvolution_updateGradInput
_volumetricFullConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricFullConvolution_accGradParameters
_volumetricDilatedConvolution_updateOutput = ten6int12 Sig.c_VolumetricDilatedConvolution_updateOutput
_volumetricDilatedConvolution_updateGradInput = ten5int12 Sig.c_VolumetricDilatedConvolution_updateGradInput

_sparseLinear_updateOutput = ten4 Sig.c_SparseLinear_updateOutput
_sparseLinear_zeroGradParameters = ten3 Sig.c_SparseLinear_zeroGradParameters
_sparseLinear_legacyUpdateOutput = ten4 Sig.c_SparseLinear_legacyUpdateOutput
_volumetricDilatedConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricDilatedConvolution_accGradParameters
_volumetricFullDilatedConvolution_updateOutput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateOutput
_volumetricFullDilatedConvolution_updateGradInput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateGradInput
_volumetricFullDilatedConvolution_accGradParameters = ten6int15double1 Sig.c_VolumetricFullDilatedConvolution_accGradParameters

_volumetricAdaptiveAveragePooling_updateOutput = ten2int3 Sig.c_VolumetricAdaptiveAveragePooling_updateOutput
_volumetricAdaptiveAveragePooling_updateGradInput = ten3 Sig.c_VolumetricAdaptiveAveragePooling_updateGradInput

_spatialReflectionPadding_updateOutput     = ten2int4 Sig.c_SpatialReflectionPadding_updateOutput
_spatialReflectionPadding_updateGradInput  = ten3int4 Sig.c_SpatialReflectionPadding_updateGradInput
_spatialReplicationPadding_updateOutput    = ten2int4 Sig.c_SpatialReplicationPadding_updateOutput
_spatialReplicationPadding_updateGradInput = ten3int4 Sig.c_SpatialReplicationPadding_updateGradInput

_featureLPPooling_updateOutput r t0 v0 v1 v2 b =
  with2DynamicState r t0 $ \s' r' t0' ->
    Sig.c_FeatureLPPooling_updateOutput s' r' t0' (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

_featureLPPooling_updateGradInput t0 t1 t2 t3 v0 v1 v2 b =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_FeatureLPPooling_updateGradInput s' t0' t1' t2' t3'
        (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

_volumetricReplicationPadding_updateOutput =
  ten2int6 Sig.c_VolumetricReplicationPadding_updateOutput

_volumetricReplicationPadding_updateGradInput =
  ten3int6 Sig.c_VolumetricReplicationPadding_updateGradInput

_volumetricUpSamplingNearest_updateOutput =
  ten2int1 Sig.c_VolumetricUpSamplingNearest_updateOutput

_volumetricUpSamplingNearest_updateGradInput =
  ten3int1 Sig.c_VolumetricUpSamplingNearest_updateGradInput

_volumetricUpSamplingTrilinear_updateOutput =
  ten2int3 Sig.c_VolumetricUpSamplingTrilinear_updateOutput

_volumetricUpSamplingTrilinear_updateGradInput =
  ten2int8 Sig.c_VolumetricUpSamplingTrilinear_updateGradInput

_temporalReflectionPadding_updateOutput =
  ten2int2 Sig.c_TemporalReflectionPadding_updateOutput

_temporalReflectionPadding_updateGradInput =
  ten3int2 Sig.c_TemporalReflectionPadding_updateGradInput

_temporalReplicationPadding_updateOutput =
  ten2int2 Sig.c_TemporalReplicationPadding_updateOutput

_temporalReplicationPadding_updateGradInput =
  ten3int2 Sig.c_TemporalReplicationPadding_updateGradInput

-------------------------------------------------------------------------------
-- Deal love of god...
--
ten1 fn t0 d0 =
  withDynamicState t0 $ \s' t0' -> fn s' t0' (fromIntegral d0)

ten2dim1 fn t0 t1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral d0)

ten2 fn t0 t1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'

ten3 fn t0 t1 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'

ten4 fn t0 t1 t2 t3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'

ten4dim1 fn t0 t1 t2 t3 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral d0)

ten5 fn t0 t1 t2 t3 t4 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with3DynamicState t2 t3 t4 $ \_  t2' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'

ten3bool2 fn t0 t1 t2 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten4bool2 fn t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_ t2' t3' ->
    fn s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten3bool1ten1bool1 fn t0 t1 t2 b0 t3 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_ t2' t3' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) t3' (toEnum $ fromEnum b1)


ten3int1 fn t0 t1 t2 i0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0)

ten3int2 fn t0 t1 t2 i0 i1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1)

ten3int3 fn t0 t1 t2 i0 i1 i2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)

ten3int4 fn t0 t1 t2 i0 i1 i2 i3 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)

ten4int2 fn t0 t1 t2 t3 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)

ten4int4 fn t0 t1 t2 t3 i0 i1 i2 i3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)

ten3int6bool2 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)

ten2int6bool2 fn t0 t1 i0 i1 i2 i3 i4 i5 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)

ten2int9bool2 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)



ten3int9bool2 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)


ten3int6 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten6int16 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 i15 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) (fromIntegral i15)

ten6int15 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) 

ten6int15double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) 
      (realToFrac d0)

ten4int2double1 fn t0 t1 t2 t3 i0 i1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
   with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (realToFrac d0)

ten4int4double1 fn t0 t1 t2 t3 i0 i1 i2 i3 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
   with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (realToFrac d0)

ten6int4double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (realToFrac d0)

ten6int6double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (realToFrac d0)


ten6int10 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)

ten6int6 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten6int6long4 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)

ten6double2 fn t0 t1 t2 t3 t4 t5 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (realToFrac d0)
      (realToFrac d1)

ten6int6long4double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)
      (realToFrac d0)

ten6int6long6double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 l4 l5 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)
      (fromIntegral l4) (fromIntegral l5)
      (realToFrac d0)

ten6int8 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)

ten6int8double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (realToFrac d0)

ten6int10double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (realToFrac d0)


ten6int12double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (realToFrac d0)

ten5double1 fn t0 t1 t2 t3 t4 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4' (realToFrac d0)


ten5int1 fn t0 t1 t2 t3 t4 i0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4' (fromIntegral i0)

ten5int6 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten5int8 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)


ten5int10 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)

ten2int10 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)



ten5int12 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)


ten3double1 fn t0 t1 t2 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0)

ten3double1bool1 fn t0 t1 t2 d0 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0) (toEnum $ fromEnum b0)

ten3double2 fn t0 t1 t2 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0) (realToFrac d1)

ten4double2 fn t0 t1 t2 t3 d0 d1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3' (realToFrac d0) (realToFrac d1)



ten4double2bool2 fn t0 t1 t2 t3 d0 d1 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (realToFrac d0)        (realToFrac d1)
      (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten4bool1ten1bool1 fn t0 t1 t2 t3 b0 t4 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with3DynamicState t2 t3 t4 $ \ _ t2' t3' t4' ->
    fn s' t0' t1' t2' t3' (toEnum $ fromEnum b0) t4' (toEnum $ fromEnum b1)

-- wtf...
ten3double2bool2gen1 fn t0 t1 t2 d0 d1 b0 b1 g = undefined
--   with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->

--     fn s' t0' t1' t2'
--       (realToFrac d0)
--       (realToFrac d1)
--       (toEnum $ fromEnum b0)
--       (toEnum $ fromEnum b1)
--       g

ten2double1 fn t0 t1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)

ten2double2 fn t0 t1 d0 d1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (realToFrac d1)

ten2double1bool1 fn t0 t1 d0 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (toEnum $ fromEnum b0)

ten2double2bool1 fn t0 t1 d0 d1 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (realToFrac d1)
      (toEnum $ fromEnum b0)

ten3double2bool1 fn t0 t1 t2 d0 d1 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (realToFrac d0)
      (realToFrac d1)
      (toEnum $ fromEnum b0)

ten6int3bool1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0)
      (fromIntegral i1)
      (fromIntegral i2)
      (toEnum $ fromEnum b0)

ten6int3bool1double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 b0 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0)
      (fromIntegral i1)
      (fromIntegral i2)
      (toEnum $ fromEnum b0)
      (realToFrac d0)

ten3bool1double1 fn t0 t1 t2 b0 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) (realToFrac d0)

ten10bool1double2 fn t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 b0 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with3DynamicState t5 t6 t7 $ \_ t5' t6' t7' ->
     with2DynamicState t8 t9 $ \_ t8' t9' ->
      fn s' t0' t1' t2' t3' t4' t5' t6' t7' t8' t9'
        (toEnum $ fromEnum b0)
        (realToFrac d0)
        (realToFrac d1)

ten8bool1double2 fn t0 t1 t2 t3 t4 t5 t6 t7 b0 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with3DynamicState t5 t6 t7 $ \_ t5' t6' t7' ->
      fn s' t0' t1' t2' t3' t4' t5' t6' t7'
        (toEnum $ fromEnum b0)
        (realToFrac d0)
        (realToFrac d1)

ten7 fn t0 t1 t2 t3 t4 t5 t6 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with2DynamicState t5 t6 $ \_ t5' t6' ->
      fn s' t0' t1' t2' t3' t4' t5' t6'


ten6int12 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)




-------------------------------------------------------------------------------

ten2int1 fn t0 t1 i0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0)

ten2int2 fn t0 t1 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1)

ten2int3 fn t0 t1 i0 i1 i2 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)
ten2int4 fn t0 t1 i0 i1 i2 i3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)


ten2int6 fn t0 t1 i0 i1 i2 i3 i4 i5 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten2int8 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)

-- ========================================================================= --

-- CPU TENSORS ONLY
-- unfolded_acc  :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- unfolded_copy :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- temporalSubSampling_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
-- temporalSubSampling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()
-- temporalSubSampling_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Double -> IO ()
-- spatialFullConvolutionMap_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolutionMap_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolutionMap_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Double -> IO ()
-- hardShrink_updateOutput      :: Dynamic -> Dynamic -> Double -> IO ()
-- hardShrink_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- col2Im_updateGradInput       :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- linear_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- linear_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- linear_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- sparseLinear_legacyZeroGradParameters :: Dynamic -> Dynamic -> Dynamic -> IO ()
-- sparseLinear_legacyUpdateParameters   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()

{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Dynamic.NN () where

import Torch.Dimensions

import Foreign.C.Types
import Torch.Sig.Types.NN
import qualified Torch.Sig.NN   as Sig
import qualified Torch.Class.NN as Class
import qualified Torch.Class.Types as Class

import Torch.Indef.Types

type instance Class.DimReal Dynamic = Integer

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


instance Class.NN Dynamic where
  abs_updateOutput = ten2 Sig.c_Abs_updateOutput
  abs_updateGradInput = ten3 Sig.c_Abs_updateGradInput
  absCriterion_updateOutput = ten3bool2 Sig.c_AbsCriterion_updateOutput
  absCriterion_updateGradInput = ten4bool2 Sig.c_AbsCriterion_updateGradInput
  sqrt_updateGradInput = ten4 Sig.c_Sqrt_updateGradInput
  square_updateOutput = ten2 Sig.c_Square_updateOutput
  square_updateGradInput = ten3 Sig.c_Square_updateGradInput
  tanh_updateOutput = ten2 Sig.c_Tanh_updateOutput
  tanh_updateGradInput = ten3 Sig.c_Tanh_updateGradInput
  l1Cost_updateOutput = ten2 Sig.c_L1Cost_updateOutput
  l1Cost_updateGradInput = ten3 Sig.c_L1Cost_updateGradInput
  logSigmoid_updateOutput = ten3 Sig.c_LogSigmoid_updateOutput
  logSigmoid_updateGradInput = ten4 Sig.c_LogSigmoid_updateGradInput
  sigmoid_updateOutput = ten2 Sig.c_Sigmoid_updateOutput
  sigmoid_updateGradInput = ten3 Sig.c_Sigmoid_updateGradInput
  eLU_updateOutput = ten2double2bool1 Sig.c_ELU_updateOutput
  logSoftMax_updateOutput = ten2dim1 Sig.c_LogSoftMax_updateOutput
  im2Col_updateOutput = ten2int8 Sig.c_Im2Col_updateOutput
  im2Col_updateGradInput = ten2int10 Sig.c_Im2Col_updateGradInput
  col2Im_updateOutput = ten2int10 Sig.c_Col2Im_updateOutput
  gRUFused_updateGradInput = ten5 Sig.c_GRUFused_updateGradInput
  pReLU_updateGradInput = ten4 Sig.c_PReLU_updateGradInput
  spatialAdaptiveAveragePooling_updateGradInput = ten3 Sig.c_SpatialAdaptiveAveragePooling_updateGradInput
  softMax_updateOutput = ten2dim1 Sig.c_SoftMax_updateOutput
  pReLU_updateOutput = ten3 Sig.c_PReLU_updateOutput
  distKLDivCriterion_updateOutput = ten3bool2 Sig.c_DistKLDivCriterion_updateOutput
  bCECriterion_updateOutput = ten3bool1ten1bool1 Sig.c_BCECriterion_updateOutput
  marginCriterion_updateGradInput = ten3bool1double1 Sig.c_MarginCriterion_updateGradInput
  distKLDivCriterion_updateGradInput = ten4bool2 Sig.c_DistKLDivCriterion_updateGradInput
  marginCriterion_updateOutput = ten3bool1double1 Sig.c_MarginCriterion_updateOutput
  smoothL1Criterion_updateGradInput = ten4bool2 Sig.c_SmoothL1Criterion_updateGradInput
  softMarginCriterion_updateGradInput = ten4bool2 Sig.c_SoftMarginCriterion_updateGradInput
  mSECriterion_updateGradInput = ten4bool2 Sig.c_MSECriterion_updateGradInput
  bCECriterion_updateGradInput = ten4bool1ten1bool1 Sig.c_BCECriterion_updateGradInput
  smoothL1Criterion_updateOutput = ten3bool2 Sig.c_SmoothL1Criterion_updateOutput
  softMarginCriterion_updateOutput = ten3bool2 Sig.c_SoftMarginCriterion_updateOutput
  mSECriterion_updateOutput = ten3bool2 Sig.c_MSECriterion_updateOutput
  leakyReLU_updateOutput = ten2double1bool1 Sig.c_LeakyReLU_updateOutput
  sqrt_updateOutput = ten2double1 Sig.c_Sqrt_updateOutput
  softShrink_updateOutput = ten2double1 Sig.c_SoftShrink_updateOutput
  softPlus_updateOutput = ten2double2 Sig.c_SoftPlus_updateOutput
  threshold_updateOutput = ten2double2bool1 Sig.c_Threshold_updateOutput
  hardTanh_updateOutput = ten2double2bool1 Sig.c_HardTanh_updateOutput
  eLU_updateGradInput = ten3double2 Sig.c_ELU_updateGradInput
  hardTanh_updateGradInput = ten3double2bool1 Sig.c_HardTanh_updateGradInput
  leakyReLU_updateGradInput = ten3double1bool1 Sig.c_LeakyReLU_updateGradInput
  softShrink_updateGradInput = ten3double1 Sig.c_SoftShrink_updateGradInput
  softPlus_updateGradInput = ten4double2 Sig.c_SoftPlus_updateGradInput
  rReLU_updateOutput = ten3double2bool2gen1 Sig.c_RReLU_updateOutput
  rReLU_updateGradInput = ten4double2bool2 Sig.c_RReLU_updateGradInput
  threshold_updateGradInput = ten3double2bool1 Sig.c_Threshold_updateGradInput
  logSoftMax_updateGradInput = ten4dim1 Sig.c_LogSoftMax_updateGradInput
  softMax_updateGradInput = ten4dim1 Sig.c_SoftMax_updateGradInput
  temporalConvolution_accGradParameters = ten4int2double1 Sig.c_TemporalConvolution_accGradParameters
  spatialSubSampling_updateOutput = ten4int4 Sig.c_SpatialSubSampling_updateOutput
  spatialSubSampling_updateGradInput = ten4int4 Sig.c_SpatialSubSampling_updateGradInput
  spatialSubSampling_accGradParameters = ten4int4double1 Sig.c_SpatialSubSampling_accGradParameters
  spatialGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_SpatialGridSamplerBilinear_updateGradInput
  pReLU_accGradParameters = ten5double1 Sig.c_PReLU_accGradParameters
  sparseLinear_updateParameters = ten5double1 Sig.c_SparseLinear_updateParameters
  volumetricGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_VolumetricGridSamplerBilinear_updateGradInput
  spatialGridSamplerBilinear_updateOutput = ten3int1 Sig.c_SpatialGridSamplerBilinear_updateOutput
  volumetricGridSamplerBilinear_updateOutput = ten3int1 Sig.c_VolumetricGridSamplerBilinear_updateOutput
  spatialUpSamplingNearest_updateGradInput = ten3int1 Sig.c_SpatialUpSamplingNearest_updateGradInput
  temporalUpSamplingNearest_updateGradInput = ten3int1 Sig.c_TemporalUpSamplingNearest_updateGradInput
  gatedLinear_updateGradInput = ten3int1 Sig.c_GatedLinear_updateGradInput
  temporalUpSamplingNearest_updateOutput = ten2int1 Sig.c_TemporalUpSamplingNearest_updateOutput
  temporalUpSamplingLinear_updateOutput = ten2int1 Sig.c_TemporalUpSamplingLinear_updateOutput
  gatedLinear_updateOutput = ten2int1 Sig.c_GatedLinear_updateOutput
  spatialUpSamplingNearest_updateOutput = ten2int1 Sig.c_SpatialUpSamplingNearest_updateOutput
  spatialUpSamplingBilinear_updateOutput = ten2int2 Sig.c_SpatialUpSamplingBilinear_updateOutput
  spatialAdaptiveAveragePooling_updateOutput = ten2int2 Sig.c_SpatialAdaptiveAveragePooling_updateOutput
  temporalConvolution_updateGradInput = ten4int2 Sig.c_TemporalConvolution_updateGradInput
  temporalUpSamplingLinear_updateGradInput = ten2int4 Sig.c_TemporalUpSamplingLinear_updateGradInput
  spatialUpSamplingBilinear_updateGradInput = ten2int6 Sig.c_SpatialUpSamplingBilinear_updateGradInput
  volumetricConvolution_updateGradInput = ten5int6 Sig.c_VolumetricConvolution_updateGradInput
  spatialAveragePooling_updateOutput = ten2int6bool2 Sig.c_SpatialAveragePooling_updateOutput
  volumetricAveragePooling_updateOutput = ten2int9bool2 Sig.c_VolumetricAveragePooling_updateOutput
  spatialAveragePooling_updateGradInput = ten3int6bool2 Sig.c_SpatialAveragePooling_updateGradInput
  volumetricAveragePooling_updateGradInput = ten3int9bool2 Sig.c_VolumetricAveragePooling_updateGradInput
  temporalConvolution_updateOutput = ten4int4 Sig.c_TemporalConvolution_updateOutput
  spatialDilatedConvolution_updateGradInput = ten5int8 Sig.c_SpatialDilatedConvolution_updateGradInput
  spatialFullConvolution_updateGradInput = ten5int8 Sig.c_SpatialFullConvolution_updateGradInput
  spatialFullDilatedConvolution_updateGradInput = ten5int10 Sig.c_SpatialFullDilatedConvolution_updateGradInput
  temporalRowConvolution_updateOutput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateOutput
  temporalRowConvolution_updateGradInput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateGradInput
  temporalRowConvolution_accGradParameters = ten6int3bool1double1 Sig.c_TemporalRowConvolution_accGradParameters
  spatialConvolutionMM_updateOutput = ten6int6 Sig.c_SpatialConvolutionMM_updateOutput
  sparseLinear_legacyAccGradParameters = ten6double2 Sig.c_SparseLinear_legacyAccGradParameters
  sparseLinear_accGradParameters = ten6double2 Sig.c_SparseLinear_accGradParameters
  spatialConvolutionMM_updateGradInput = ten6int6 Sig.c_SpatialConvolutionMM_updateGradInput
  spatialConvolutionMM_accGradParameters = ten6int6double1 Sig.c_SpatialConvolutionMM_accGradParameters
  spatialConvolutionLocal_updateOutput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateOutput
  spatialConvolutionLocal_updateGradInput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateGradInput
  spatialConvolutionLocal_accGradParameters = ten6int6long4double1 Sig.c_SpatialConvolutionLocal_accGradParameters
  volumetricConvolution_updateOutput = ten6int6 Sig.c_VolumetricConvolution_updateOutput
  spatialFullConvolution_updateOutput = ten6int8 Sig.c_SpatialFullConvolution_updateOutput
  spatialFullConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialFullConvolution_accGradParameters
  spatialDilatedConvolution_updateOutput = ten6int8 Sig.c_SpatialDilatedConvolution_updateOutput
  spatialDilatedConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialDilatedConvolution_accGradParameters
  spatialFullDilatedConvolution_updateOutput = ten6int10 Sig.c_SpatialFullDilatedConvolution_updateOutput
  spatialFullDilatedConvolution_accGradParameters = ten6int10double1 Sig.c_SpatialFullDilatedConvolution_accGradParameters
  gRUFused_updateOutput = ten7 Sig.c_GRUFused_updateOutput
  lSTMFused_updateOutput = ten7 Sig.c_LSTMFused_updateOutput
  lSTMFused_updateGradInput = ten7 Sig.c_LSTMFused_updateGradInput
  batchNormalization_updateOutput = ten8bool1double2 Sig.c_BatchNormalization_updateOutput
  batchNormalization_backward = ten10bool1double2 Sig.c_BatchNormalization_backward
  volumetricConvolution_accGradParameters = ten6int6double1 Sig.c_VolumetricConvolution_accGradParameters
  volumetricFullConvolution_updateOutput = ten6int12 Sig.c_VolumetricFullConvolution_updateOutput
  volumetricFullConvolution_updateGradInput = ten6int12 Sig.c_VolumetricFullConvolution_updateGradInput
  volumetricFullConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricFullConvolution_accGradParameters
  volumetricDilatedConvolution_updateOutput = ten6int12 Sig.c_VolumetricDilatedConvolution_updateOutput
  volumetricDilatedConvolution_updateGradInput = ten5int12 Sig.c_VolumetricDilatedConvolution_updateGradInput

  sparseLinear_updateOutput = ten4 Sig.c_SparseLinear_updateOutput
  sparseLinear_zeroGradParameters = ten3 Sig.c_SparseLinear_zeroGradParameters
  sparseLinear_legacyUpdateOutput = ten4 Sig.c_SparseLinear_legacyUpdateOutput
  volumetricDilatedConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricDilatedConvolution_accGradParameters
  volumetricFullDilatedConvolution_updateOutput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateOutput
  volumetricFullDilatedConvolution_updateGradInput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateGradInput
  volumetricFullDilatedConvolution_accGradParameters = ten6int15double1 Sig.c_VolumetricFullDilatedConvolution_accGradParameters

  volumetricAdaptiveAveragePooling_updateOutput = ten2int3 Sig.c_VolumetricAdaptiveAveragePooling_updateOutput
  volumetricAdaptiveAveragePooling_updateGradInput = ten3 Sig.c_VolumetricAdaptiveAveragePooling_updateGradInput

  spatialReflectionPadding_updateOutput     = ten2int4 Sig.c_SpatialReflectionPadding_updateOutput
  spatialReflectionPadding_updateGradInput  = ten3int4 Sig.c_SpatialReflectionPadding_updateGradInput
  spatialReplicationPadding_updateOutput    = ten2int4 Sig.c_SpatialReplicationPadding_updateOutput
  spatialReplicationPadding_updateGradInput = ten3int4 Sig.c_SpatialReplicationPadding_updateGradInput

  featureLPPooling_updateOutput r t0 v0 v1 v2 b =
    with2DynamicState r t0 $ \s' r' t0' ->
      Sig.c_FeatureLPPooling_updateOutput s' r' t0' (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

  featureLPPooling_updateGradInput t0 t1 t2 t3 v0 v1 v2 b =
    with2DynamicState t0 t1 $ \s' t0' t1' ->
      with2DynamicState t2 t3 $ \_ t2' t3' ->
        Sig.c_FeatureLPPooling_updateGradInput s' t0' t1' t2' t3'
          (realToFrac v0) (fromIntegral v1) (fromIntegral v2) (toEnum $ fromEnum b)

  volumetricReplicationPadding_updateOutput =
    ten2int6 Sig.c_VolumetricReplicationPadding_updateOutput

  volumetricReplicationPadding_updateGradInput =
    ten3int6 Sig.c_VolumetricReplicationPadding_updateGradInput

  volumetricUpSamplingNearest_updateOutput =
    ten2int1 Sig.c_VolumetricUpSamplingNearest_updateOutput

  volumetricUpSamplingNearest_updateGradInput =
    ten3int1 Sig.c_VolumetricUpSamplingNearest_updateGradInput

  volumetricUpSamplingTrilinear_updateOutput =
    ten2int3 Sig.c_VolumetricUpSamplingTrilinear_updateOutput

  volumetricUpSamplingTrilinear_updateGradInput =
    ten2int8 Sig.c_VolumetricUpSamplingTrilinear_updateGradInput

  temporalReflectionPadding_updateOutput =
    ten2int2 Sig.c_TemporalReflectionPadding_updateOutput

  temporalReflectionPadding_updateGradInput =
    ten3int2 Sig.c_TemporalReflectionPadding_updateGradInput

  temporalReplicationPadding_updateOutput =
    ten2int2 Sig.c_TemporalReplicationPadding_updateOutput

  temporalReplicationPadding_updateGradInput =
    ten3int2 Sig.c_TemporalReplicationPadding_updateGradInput

-------------------------------------------------------------------------------

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

ten3double2bool2gen1 fn t0 t1 t2 d0 d1 b0 b1 g =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    undefined
    -- fn s' t0' t1' t2'
    --   (realToFrac d0)
    --   (realToFrac d1)
    --   (toEnum $ fromEnum b0)
    --   (toEnum $ fromEnum b1)
    --   g'

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


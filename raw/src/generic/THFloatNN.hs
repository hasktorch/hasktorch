{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatNN (
    c_THFloatNN_Abs_updateOutput,
    c_THFloatNN_Abs_updateGradInput,
    c_THFloatNN_AbsCriterion_updateOutput,
    c_THFloatNN_AbsCriterion_updateGradInput,
    c_THFloatNN_BCECriterion_updateOutput,
    c_THFloatNN_BCECriterion_updateGradInput,
    c_THFloatNN_ClassNLLCriterion_updateOutput,
    c_THFloatNN_ClassNLLCriterion_updateGradInput,
    c_THFloatNN_SpatialClassNLLCriterion_updateOutput,
    c_THFloatNN_SpatialClassNLLCriterion_updateGradInput,
    c_THFloatNN_ELU_updateOutput,
    c_THFloatNN_ELU_updateGradInput,
    c_THFloatNN_DistKLDivCriterion_updateOutput,
    c_THFloatNN_DistKLDivCriterion_updateGradInput,
    c_THFloatNN_GatedLinear_updateOutput,
    c_THFloatNN_GatedLinear_updateGradInput,
    c_THFloatNN_HardShrink_updateOutput,
    c_THFloatNN_HardShrink_updateGradInput,
    c_THFloatNN_HardTanh_updateOutput,
    c_THFloatNN_HardTanh_updateGradInput,
    c_THFloatNN_L1Cost_updateOutput,
    c_THFloatNN_L1Cost_updateGradInput,
    c_THFloatNN_LeakyReLU_updateOutput,
    c_THFloatNN_LeakyReLU_updateGradInput,
    c_THFloatNN_GRUFused_updateOutput,
    c_THFloatNN_GRUFused_updateGradInput,
    c_THFloatNN_LSTMFused_updateOutput,
    c_THFloatNN_LSTMFused_updateGradInput,
    c_THFloatNN_LogSigmoid_updateOutput,
    c_THFloatNN_LogSigmoid_updateGradInput,
    c_THFloatNN_LogSoftMax_updateOutput,
    c_THFloatNN_LogSoftMax_updateGradInput,
    c_THFloatNN_LookupTable_accGradParameters,
    c_THFloatNN_LookupTable_renorm,
    c_THFloatNN_MarginCriterion_updateOutput,
    c_THFloatNN_MarginCriterion_updateGradInput,
    c_THFloatNN_SoftMarginCriterion_updateOutput,
    c_THFloatNN_SoftMarginCriterion_updateGradInput,
    c_THFloatNN_MSECriterion_updateOutput,
    c_THFloatNN_MSECriterion_updateGradInput,
    c_THFloatNN_MultiLabelMarginCriterion_updateOutput,
    c_THFloatNN_MultiLabelMarginCriterion_updateGradInput,
    c_THFloatNN_MultiMarginCriterion_updateOutput,
    c_THFloatNN_MultiMarginCriterion_updateGradInput,
    c_THFloatNN_PReLU_updateOutput,
    c_THFloatNN_PReLU_updateGradInput,
    c_THFloatNN_PReLU_accGradParameters,
    c_THFloatNN_Linear_updateOutput,
    c_THFloatNN_Linear_updateGradInput,
    c_THFloatNN_Linear_accGradParameters,
    c_THFloatNN_RReLU_updateOutput,
    c_THFloatNN_RReLU_updateGradInput,
    c_THFloatNN_Sigmoid_updateOutput,
    c_THFloatNN_Sigmoid_updateGradInput,
    c_THFloatNN_SmoothL1Criterion_updateOutput,
    c_THFloatNN_SmoothL1Criterion_updateGradInput,
    c_THFloatNN_SoftMax_updateOutput,
    c_THFloatNN_SoftMax_updateGradInput,
    c_THFloatNN_SoftPlus_updateOutput,
    c_THFloatNN_SoftPlus_updateGradInput,
    c_THFloatNN_SoftShrink_updateOutput,
    c_THFloatNN_SoftShrink_updateGradInput,
    c_THFloatNN_IndexLinear_updateOutput,
    c_THFloatNN_IndexLinear_accGradParameters,
    c_THFloatNN_IndexLinear_accUpdateGradParameters,
    c_THFloatNN_IndexLinear_updateParameters,
    c_THFloatNN_SparseLinear_updateOutput,
    c_THFloatNN_SparseLinear_accGradParameters,
    c_THFloatNN_SparseLinear_zeroGradParameters,
    c_THFloatNN_SparseLinear_updateParameters,
    c_THFloatNN_SparseLinear_legacyUpdateOutput,
    c_THFloatNN_SparseLinear_legacyAccGradParameters,
    c_THFloatNN_SparseLinear_legacyZeroGradParameters,
    c_THFloatNN_SparseLinear_legacyUpdateParameters,
    c_THFloatNN_Sqrt_updateOutput,
    c_THFloatNN_Sqrt_updateGradInput,
    c_THFloatNN_Square_updateOutput,
    c_THFloatNN_Square_updateGradInput,
    c_THFloatNN_Tanh_updateOutput,
    c_THFloatNN_Tanh_updateGradInput,
    c_THFloatNN_Threshold_updateOutput,
    c_THFloatNN_Threshold_updateGradInput,
    c_THFloatNN_TemporalConvolution_updateOutput,
    c_THFloatNN_TemporalConvolution_updateGradInput,
    c_THFloatNN_TemporalConvolution_accGradParameters,
    c_THFloatNN_TemporalMaxPooling_updateOutput,
    c_THFloatNN_TemporalMaxPooling_updateGradInput,
    c_THFloatNN_TemporalSubSampling_updateOutput,
    c_THFloatNN_TemporalSubSampling_updateGradInput,
    c_THFloatNN_TemporalSubSampling_accGradParameters,
    c_THFloatNN_TemporalRowConvolution_updateOutput,
    c_THFloatNN_TemporalRowConvolution_updateGradInput,
    c_THFloatNN_TemporalRowConvolution_accGradParameters,
    c_THFloatNN_TemporalUpSamplingNearest_updateOutput,
    c_THFloatNN_TemporalUpSamplingNearest_updateGradInput,
    c_THFloatNN_TemporalUpSamplingLinear_updateOutput,
    c_THFloatNN_TemporalUpSamplingLinear_updateGradInput,
    c_THFloatNN_BatchNormalization_updateOutput,
    c_THFloatNN_BatchNormalization_backward,
    c_THFloatNN_SpatialConvolutionMap_updateOutput,
    c_THFloatNN_SpatialConvolutionMap_updateGradInput,
    c_THFloatNN_SpatialConvolutionMap_accGradParameters,
    c_THFloatNN_SpatialConvolutionMM_updateOutput,
    c_THFloatNN_SpatialConvolutionMM_updateGradInput,
    c_THFloatNN_SpatialConvolutionMM_accGradParameters,
    c_THFloatNN_SpatialConvolutionLocal_updateOutput,
    c_THFloatNN_SpatialConvolutionLocal_updateGradInput,
    c_THFloatNN_SpatialConvolutionLocal_accGradParameters,
    c_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput,
    c_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput,
    c_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput,
    c_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput,
    c_THFloatNN_SpatialAveragePooling_updateOutput,
    c_THFloatNN_SpatialAveragePooling_updateGradInput,
    c_THFloatNN_SpatialFractionalMaxPooling_updateOutput,
    c_THFloatNN_SpatialFractionalMaxPooling_updateGradInput,
    c_THFloatNN_SpatialFullConvolution_updateOutput,
    c_THFloatNN_SpatialFullConvolution_updateGradInput,
    c_THFloatNN_SpatialFullConvolution_accGradParameters,
    c_THFloatNN_SpatialFullConvolutionMap_updateOutput,
    c_THFloatNN_SpatialFullConvolutionMap_updateGradInput,
    c_THFloatNN_SpatialFullConvolutionMap_accGradParameters,
    c_THFloatNN_SpatialDilatedConvolution_updateOutput,
    c_THFloatNN_SpatialDilatedConvolution_updateGradInput,
    c_THFloatNN_SpatialDilatedConvolution_accGradParameters,
    c_THFloatNN_SpatialFullDilatedConvolution_updateOutput,
    c_THFloatNN_SpatialFullDilatedConvolution_updateGradInput,
    c_THFloatNN_SpatialFullDilatedConvolution_accGradParameters,
    c_THFloatNN_SpatialMaxPooling_updateOutput,
    c_THFloatNN_SpatialMaxPooling_updateGradInput,
    c_THFloatNN_SpatialDilatedMaxPooling_updateOutput,
    c_THFloatNN_SpatialDilatedMaxPooling_updateGradInput,
    c_THFloatNN_SpatialMaxUnpooling_updateOutput,
    c_THFloatNN_SpatialMaxUnpooling_updateGradInput,
    c_THFloatNN_SpatialSubSampling_updateOutput,
    c_THFloatNN_SpatialSubSampling_updateGradInput,
    c_THFloatNN_SpatialSubSampling_accGradParameters,
    c_THFloatNN_SpatialUpSamplingNearest_updateOutput,
    c_THFloatNN_SpatialUpSamplingNearest_updateGradInput,
    c_THFloatNN_SpatialUpSamplingBilinear_updateOutput,
    c_THFloatNN_SpatialUpSamplingBilinear_updateGradInput,
    c_THFloatNN_SpatialGridSamplerBilinear_updateOutput,
    c_THFloatNN_SpatialGridSamplerBilinear_updateGradInput,
    c_THFloatNN_unfolded_acc,
    c_THFloatNN_unfolded_copy,
    c_THFloatNN_VolumetricAveragePooling_updateOutput,
    c_THFloatNN_VolumetricAveragePooling_updateGradInput,
    c_THFloatNN_VolumetricConvolution_updateOutput,
    c_THFloatNN_VolumetricConvolution_updateGradInput,
    c_THFloatNN_VolumetricConvolution_accGradParameters,
    c_THFloatNN_VolumetricConvolutionMM_updateOutput,
    c_THFloatNN_VolumetricConvolutionMM_updateGradInput,
    c_THFloatNN_VolumetricConvolutionMM_accGradParameters,
    c_THFloatNN_VolumetricFractionalMaxPooling_updateOutput,
    c_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput,
    c_THFloatNN_VolumetricFullConvolution_updateOutput,
    c_THFloatNN_VolumetricFullConvolution_updateGradInput,
    c_THFloatNN_VolumetricFullConvolution_accGradParameters,
    c_THFloatNN_VolumetricDilatedConvolution_updateOutput,
    c_THFloatNN_VolumetricDilatedConvolution_updateGradInput,
    c_THFloatNN_VolumetricDilatedConvolution_accGradParameters,
    c_THFloatNN_VolumetricFullDilatedConvolution_updateOutput,
    c_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput,
    c_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters,
    c_THFloatNN_VolumetricMaxPooling_updateOutput,
    c_THFloatNN_VolumetricMaxPooling_updateGradInput,
    c_THFloatNN_VolumetricDilatedMaxPooling_updateOutput,
    c_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput,
    c_THFloatNN_VolumetricMaxUnpooling_updateOutput,
    c_THFloatNN_VolumetricMaxUnpooling_updateGradInput,
    c_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput,
    c_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput,
    c_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput,
    c_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput,
    c_THFloatNN_SpatialReflectionPadding_updateOutput,
    c_THFloatNN_SpatialReflectionPadding_updateGradInput,
    c_THFloatNN_SpatialReplicationPadding_updateOutput,
    c_THFloatNN_SpatialReplicationPadding_updateGradInput,
    c_THFloatNN_FeatureLPPooling_updateOutput,
    c_THFloatNN_FeatureLPPooling_updateGradInput,
    c_THFloatNN_VolumetricReplicationPadding_updateOutput,
    c_THFloatNN_VolumetricReplicationPadding_updateGradInput,
    c_THFloatNN_VolumetricUpSamplingNearest_updateOutput,
    c_THFloatNN_VolumetricUpSamplingNearest_updateGradInput,
    c_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput,
    c_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput,
    c_THFloatNN_TemporalReflectionPadding_updateOutput,
    c_THFloatNN_TemporalReflectionPadding_updateGradInput,
    c_THFloatNN_TemporalReplicationPadding_updateOutput,
    c_THFloatNN_TemporalReplicationPadding_updateGradInput,
    p_THFloatNN_Abs_updateOutput,
    p_THFloatNN_Abs_updateGradInput,
    p_THFloatNN_AbsCriterion_updateOutput,
    p_THFloatNN_AbsCriterion_updateGradInput,
    p_THFloatNN_BCECriterion_updateOutput,
    p_THFloatNN_BCECriterion_updateGradInput,
    p_THFloatNN_ClassNLLCriterion_updateOutput,
    p_THFloatNN_ClassNLLCriterion_updateGradInput,
    p_THFloatNN_SpatialClassNLLCriterion_updateOutput,
    p_THFloatNN_SpatialClassNLLCriterion_updateGradInput,
    p_THFloatNN_ELU_updateOutput,
    p_THFloatNN_ELU_updateGradInput,
    p_THFloatNN_DistKLDivCriterion_updateOutput,
    p_THFloatNN_DistKLDivCriterion_updateGradInput,
    p_THFloatNN_GatedLinear_updateOutput,
    p_THFloatNN_GatedLinear_updateGradInput,
    p_THFloatNN_HardShrink_updateOutput,
    p_THFloatNN_HardShrink_updateGradInput,
    p_THFloatNN_HardTanh_updateOutput,
    p_THFloatNN_HardTanh_updateGradInput,
    p_THFloatNN_L1Cost_updateOutput,
    p_THFloatNN_L1Cost_updateGradInput,
    p_THFloatNN_LeakyReLU_updateOutput,
    p_THFloatNN_LeakyReLU_updateGradInput,
    p_THFloatNN_GRUFused_updateOutput,
    p_THFloatNN_GRUFused_updateGradInput,
    p_THFloatNN_LSTMFused_updateOutput,
    p_THFloatNN_LSTMFused_updateGradInput,
    p_THFloatNN_LogSigmoid_updateOutput,
    p_THFloatNN_LogSigmoid_updateGradInput,
    p_THFloatNN_LogSoftMax_updateOutput,
    p_THFloatNN_LogSoftMax_updateGradInput,
    p_THFloatNN_LookupTable_accGradParameters,
    p_THFloatNN_LookupTable_renorm,
    p_THFloatNN_MarginCriterion_updateOutput,
    p_THFloatNN_MarginCriterion_updateGradInput,
    p_THFloatNN_SoftMarginCriterion_updateOutput,
    p_THFloatNN_SoftMarginCriterion_updateGradInput,
    p_THFloatNN_MSECriterion_updateOutput,
    p_THFloatNN_MSECriterion_updateGradInput,
    p_THFloatNN_MultiLabelMarginCriterion_updateOutput,
    p_THFloatNN_MultiLabelMarginCriterion_updateGradInput,
    p_THFloatNN_MultiMarginCriterion_updateOutput,
    p_THFloatNN_MultiMarginCriterion_updateGradInput,
    p_THFloatNN_PReLU_updateOutput,
    p_THFloatNN_PReLU_updateGradInput,
    p_THFloatNN_PReLU_accGradParameters,
    p_THFloatNN_Linear_updateOutput,
    p_THFloatNN_Linear_updateGradInput,
    p_THFloatNN_Linear_accGradParameters,
    p_THFloatNN_RReLU_updateOutput,
    p_THFloatNN_RReLU_updateGradInput,
    p_THFloatNN_Sigmoid_updateOutput,
    p_THFloatNN_Sigmoid_updateGradInput,
    p_THFloatNN_SmoothL1Criterion_updateOutput,
    p_THFloatNN_SmoothL1Criterion_updateGradInput,
    p_THFloatNN_SoftMax_updateOutput,
    p_THFloatNN_SoftMax_updateGradInput,
    p_THFloatNN_SoftPlus_updateOutput,
    p_THFloatNN_SoftPlus_updateGradInput,
    p_THFloatNN_SoftShrink_updateOutput,
    p_THFloatNN_SoftShrink_updateGradInput,
    p_THFloatNN_IndexLinear_updateOutput,
    p_THFloatNN_IndexLinear_accGradParameters,
    p_THFloatNN_IndexLinear_accUpdateGradParameters,
    p_THFloatNN_IndexLinear_updateParameters,
    p_THFloatNN_SparseLinear_updateOutput,
    p_THFloatNN_SparseLinear_accGradParameters,
    p_THFloatNN_SparseLinear_zeroGradParameters,
    p_THFloatNN_SparseLinear_updateParameters,
    p_THFloatNN_SparseLinear_legacyUpdateOutput,
    p_THFloatNN_SparseLinear_legacyAccGradParameters,
    p_THFloatNN_SparseLinear_legacyZeroGradParameters,
    p_THFloatNN_SparseLinear_legacyUpdateParameters,
    p_THFloatNN_Sqrt_updateOutput,
    p_THFloatNN_Sqrt_updateGradInput,
    p_THFloatNN_Square_updateOutput,
    p_THFloatNN_Square_updateGradInput,
    p_THFloatNN_Tanh_updateOutput,
    p_THFloatNN_Tanh_updateGradInput,
    p_THFloatNN_Threshold_updateOutput,
    p_THFloatNN_Threshold_updateGradInput,
    p_THFloatNN_TemporalConvolution_updateOutput,
    p_THFloatNN_TemporalConvolution_updateGradInput,
    p_THFloatNN_TemporalConvolution_accGradParameters,
    p_THFloatNN_TemporalMaxPooling_updateOutput,
    p_THFloatNN_TemporalMaxPooling_updateGradInput,
    p_THFloatNN_TemporalSubSampling_updateOutput,
    p_THFloatNN_TemporalSubSampling_updateGradInput,
    p_THFloatNN_TemporalSubSampling_accGradParameters,
    p_THFloatNN_TemporalRowConvolution_updateOutput,
    p_THFloatNN_TemporalRowConvolution_updateGradInput,
    p_THFloatNN_TemporalRowConvolution_accGradParameters,
    p_THFloatNN_TemporalUpSamplingNearest_updateOutput,
    p_THFloatNN_TemporalUpSamplingNearest_updateGradInput,
    p_THFloatNN_TemporalUpSamplingLinear_updateOutput,
    p_THFloatNN_TemporalUpSamplingLinear_updateGradInput,
    p_THFloatNN_BatchNormalization_updateOutput,
    p_THFloatNN_BatchNormalization_backward,
    p_THFloatNN_SpatialConvolutionMap_updateOutput,
    p_THFloatNN_SpatialConvolutionMap_updateGradInput,
    p_THFloatNN_SpatialConvolutionMap_accGradParameters,
    p_THFloatNN_SpatialConvolutionMM_updateOutput,
    p_THFloatNN_SpatialConvolutionMM_updateGradInput,
    p_THFloatNN_SpatialConvolutionMM_accGradParameters,
    p_THFloatNN_SpatialConvolutionLocal_updateOutput,
    p_THFloatNN_SpatialConvolutionLocal_updateGradInput,
    p_THFloatNN_SpatialConvolutionLocal_accGradParameters,
    p_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput,
    p_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput,
    p_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput,
    p_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput,
    p_THFloatNN_SpatialAveragePooling_updateOutput,
    p_THFloatNN_SpatialAveragePooling_updateGradInput,
    p_THFloatNN_SpatialFractionalMaxPooling_updateOutput,
    p_THFloatNN_SpatialFractionalMaxPooling_updateGradInput,
    p_THFloatNN_SpatialFullConvolution_updateOutput,
    p_THFloatNN_SpatialFullConvolution_updateGradInput,
    p_THFloatNN_SpatialFullConvolution_accGradParameters,
    p_THFloatNN_SpatialFullConvolutionMap_updateOutput,
    p_THFloatNN_SpatialFullConvolutionMap_updateGradInput,
    p_THFloatNN_SpatialFullConvolutionMap_accGradParameters,
    p_THFloatNN_SpatialDilatedConvolution_updateOutput,
    p_THFloatNN_SpatialDilatedConvolution_updateGradInput,
    p_THFloatNN_SpatialDilatedConvolution_accGradParameters,
    p_THFloatNN_SpatialFullDilatedConvolution_updateOutput,
    p_THFloatNN_SpatialFullDilatedConvolution_updateGradInput,
    p_THFloatNN_SpatialFullDilatedConvolution_accGradParameters,
    p_THFloatNN_SpatialMaxPooling_updateOutput,
    p_THFloatNN_SpatialMaxPooling_updateGradInput,
    p_THFloatNN_SpatialDilatedMaxPooling_updateOutput,
    p_THFloatNN_SpatialDilatedMaxPooling_updateGradInput,
    p_THFloatNN_SpatialMaxUnpooling_updateOutput,
    p_THFloatNN_SpatialMaxUnpooling_updateGradInput,
    p_THFloatNN_SpatialSubSampling_updateOutput,
    p_THFloatNN_SpatialSubSampling_updateGradInput,
    p_THFloatNN_SpatialSubSampling_accGradParameters,
    p_THFloatNN_SpatialUpSamplingNearest_updateOutput,
    p_THFloatNN_SpatialUpSamplingNearest_updateGradInput,
    p_THFloatNN_SpatialUpSamplingBilinear_updateOutput,
    p_THFloatNN_SpatialUpSamplingBilinear_updateGradInput,
    p_THFloatNN_SpatialGridSamplerBilinear_updateOutput,
    p_THFloatNN_SpatialGridSamplerBilinear_updateGradInput,
    p_THFloatNN_unfolded_acc,
    p_THFloatNN_unfolded_copy,
    p_THFloatNN_VolumetricAveragePooling_updateOutput,
    p_THFloatNN_VolumetricAveragePooling_updateGradInput,
    p_THFloatNN_VolumetricConvolution_updateOutput,
    p_THFloatNN_VolumetricConvolution_updateGradInput,
    p_THFloatNN_VolumetricConvolution_accGradParameters,
    p_THFloatNN_VolumetricConvolutionMM_updateOutput,
    p_THFloatNN_VolumetricConvolutionMM_updateGradInput,
    p_THFloatNN_VolumetricConvolutionMM_accGradParameters,
    p_THFloatNN_VolumetricFractionalMaxPooling_updateOutput,
    p_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput,
    p_THFloatNN_VolumetricFullConvolution_updateOutput,
    p_THFloatNN_VolumetricFullConvolution_updateGradInput,
    p_THFloatNN_VolumetricFullConvolution_accGradParameters,
    p_THFloatNN_VolumetricDilatedConvolution_updateOutput,
    p_THFloatNN_VolumetricDilatedConvolution_updateGradInput,
    p_THFloatNN_VolumetricDilatedConvolution_accGradParameters,
    p_THFloatNN_VolumetricFullDilatedConvolution_updateOutput,
    p_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput,
    p_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters,
    p_THFloatNN_VolumetricMaxPooling_updateOutput,
    p_THFloatNN_VolumetricMaxPooling_updateGradInput,
    p_THFloatNN_VolumetricDilatedMaxPooling_updateOutput,
    p_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput,
    p_THFloatNN_VolumetricMaxUnpooling_updateOutput,
    p_THFloatNN_VolumetricMaxUnpooling_updateGradInput,
    p_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput,
    p_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput,
    p_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput,
    p_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput,
    p_THFloatNN_SpatialReflectionPadding_updateOutput,
    p_THFloatNN_SpatialReflectionPadding_updateGradInput,
    p_THFloatNN_SpatialReplicationPadding_updateOutput,
    p_THFloatNN_SpatialReplicationPadding_updateGradInput,
    p_THFloatNN_FeatureLPPooling_updateOutput,
    p_THFloatNN_FeatureLPPooling_updateGradInput,
    p_THFloatNN_VolumetricReplicationPadding_updateOutput,
    p_THFloatNN_VolumetricReplicationPadding_updateGradInput,
    p_THFloatNN_VolumetricUpSamplingNearest_updateOutput,
    p_THFloatNN_VolumetricUpSamplingNearest_updateGradInput,
    p_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput,
    p_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput,
    p_THFloatNN_TemporalReflectionPadding_updateOutput,
    p_THFloatNN_TemporalReflectionPadding_updateGradInput,
    p_THFloatNN_TemporalReplicationPadding_updateOutput,
    p_THFloatNN_TemporalReplicationPadding_updateGradInput) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THFloatNN_Abs_updateOutput : state input output -> void
foreign import ccall "THNN.h THFloatNN_Abs_updateOutput"
  c_THFloatNN_Abs_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Abs_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THFloatNN_Abs_updateGradInput"
  c_THFloatNN_Abs_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_AbsCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_AbsCriterion_updateOutput"
  c_THFloatNN_AbsCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_AbsCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_AbsCriterion_updateGradInput"
  c_THFloatNN_AbsCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_BCECriterion_updateOutput : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h THFloatNN_BCECriterion_updateOutput"
  c_THFloatNN_BCECriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_BCECriterion_updateGradInput : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h THFloatNN_BCECriterion_updateGradInput"
  c_THFloatNN_BCECriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_ClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THFloatNN_ClassNLLCriterion_updateOutput"
  c_THFloatNN_ClassNLLCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THFloatNN_ClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THFloatNN_ClassNLLCriterion_updateGradInput"
  c_THFloatNN_ClassNLLCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THFloatNN_SpatialClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THFloatNN_SpatialClassNLLCriterion_updateOutput"
  c_THFloatNN_SpatialClassNLLCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THFloatNN_SpatialClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THFloatNN_SpatialClassNLLCriterion_updateGradInput"
  c_THFloatNN_SpatialClassNLLCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THFloatNN_ELU_updateOutput : state input output alpha inplace -> void
foreign import ccall "THNN.h THFloatNN_ELU_updateOutput"
  c_THFloatNN_ELU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_ELU_updateGradInput : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h THFloatNN_ELU_updateGradInput"
  c_THFloatNN_ELU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_DistKLDivCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_DistKLDivCriterion_updateOutput"
  c_THFloatNN_DistKLDivCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_DistKLDivCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_DistKLDivCriterion_updateGradInput"
  c_THFloatNN_DistKLDivCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_GatedLinear_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THFloatNN_GatedLinear_updateOutput"
  c_THFloatNN_GatedLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_GatedLinear_updateGradInput : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THFloatNN_GatedLinear_updateGradInput"
  c_THFloatNN_GatedLinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_HardShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THFloatNN_HardShrink_updateOutput"
  c_THFloatNN_HardShrink_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_HardShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THFloatNN_HardShrink_updateGradInput"
  c_THFloatNN_HardShrink_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_HardTanh_updateOutput : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THFloatNN_HardTanh_updateOutput"
  c_THFloatNN_HardTanh_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_HardTanh_updateGradInput : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THFloatNN_HardTanh_updateGradInput"
  c_THFloatNN_HardTanh_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_L1Cost_updateOutput : state input output -> void
foreign import ccall "THNN.h THFloatNN_L1Cost_updateOutput"
  c_THFloatNN_L1Cost_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_L1Cost_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THFloatNN_L1Cost_updateGradInput"
  c_THFloatNN_L1Cost_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LeakyReLU_updateOutput : state input output negval inplace -> void
foreign import ccall "THNN.h THFloatNN_LeakyReLU_updateOutput"
  c_THFloatNN_LeakyReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_LeakyReLU_updateGradInput : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THFloatNN_LeakyReLU_updateGradInput"
  c_THFloatNN_LeakyReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_GRUFused_updateOutput : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THFloatNN_GRUFused_updateOutput"
  c_THFloatNN_GRUFused_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_GRUFused_updateGradInput : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THFloatNN_GRUFused_updateGradInput"
  c_THFloatNN_GRUFused_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LSTMFused_updateOutput : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THFloatNN_LSTMFused_updateOutput"
  c_THFloatNN_LSTMFused_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LSTMFused_updateGradInput : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THFloatNN_LSTMFused_updateGradInput"
  c_THFloatNN_LSTMFused_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LogSigmoid_updateOutput : state input output buffer -> void
foreign import ccall "THNN.h THFloatNN_LogSigmoid_updateOutput"
  c_THFloatNN_LogSigmoid_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LogSigmoid_updateGradInput : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THFloatNN_LogSigmoid_updateGradInput"
  c_THFloatNN_LogSigmoid_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_LogSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THFloatNN_LogSoftMax_updateOutput"
  c_THFloatNN_LogSoftMax_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_LogSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THFloatNN_LogSoftMax_updateGradInput"
  c_THFloatNN_LogSoftMax_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_LookupTable_accGradParameters : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h THFloatNN_LookupTable_accGradParameters"
  c_THFloatNN_LookupTable_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_LookupTable_renorm : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h THFloatNN_LookupTable_renorm"
  c_THFloatNN_LookupTable_renorm :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_MarginCriterion_updateOutput : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THFloatNN_MarginCriterion_updateOutput"
  c_THFloatNN_MarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ()

-- |c_THFloatNN_MarginCriterion_updateGradInput : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THFloatNN_MarginCriterion_updateGradInput"
  c_THFloatNN_MarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ()

-- |c_THFloatNN_SoftMarginCriterion_updateOutput : state input target output sizeAverage -> void
foreign import ccall "THNN.h THFloatNN_SoftMarginCriterion_updateOutput"
  c_THFloatNN_SoftMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THFloatNN_SoftMarginCriterion_updateGradInput : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h THFloatNN_SoftMarginCriterion_updateGradInput"
  c_THFloatNN_SoftMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THFloatNN_MSECriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_MSECriterion_updateOutput"
  c_THFloatNN_MSECriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_MSECriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_MSECriterion_updateGradInput"
  c_THFloatNN_MSECriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_MultiLabelMarginCriterion_updateOutput : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h THFloatNN_MultiLabelMarginCriterion_updateOutput"
  c_THFloatNN_MultiLabelMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THFloatNN_MultiLabelMarginCriterion_updateGradInput : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h THFloatNN_MultiLabelMarginCriterion_updateGradInput"
  c_THFloatNN_MultiLabelMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THFloatNN_MultiMarginCriterion_updateOutput : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h THFloatNN_MultiMarginCriterion_updateOutput"
  c_THFloatNN_MultiMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_MultiMarginCriterion_updateGradInput : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h THFloatNN_MultiMarginCriterion_updateGradInput"
  c_THFloatNN_MultiMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_PReLU_updateOutput : state input output weight -> void
foreign import ccall "THNN.h THFloatNN_PReLU_updateOutput"
  c_THFloatNN_PReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_PReLU_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THFloatNN_PReLU_updateGradInput"
  c_THFloatNN_PReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_PReLU_accGradParameters : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THFloatNN_PReLU_accGradParameters"
  c_THFloatNN_PReLU_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_Linear_updateOutput : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THFloatNN_Linear_updateOutput"
  c_THFloatNN_Linear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Linear_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THFloatNN_Linear_updateGradInput"
  c_THFloatNN_Linear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Linear_accGradParameters : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THFloatNN_Linear_accGradParameters"
  c_THFloatNN_Linear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_RReLU_updateOutput : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THFloatNN_RReLU_updateOutput"
  c_THFloatNN_RReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ()

-- |c_THFloatNN_RReLU_updateGradInput : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THFloatNN_RReLU_updateGradInput"
  c_THFloatNN_RReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- |c_THFloatNN_Sigmoid_updateOutput : state input output -> void
foreign import ccall "THNN.h THFloatNN_Sigmoid_updateOutput"
  c_THFloatNN_Sigmoid_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Sigmoid_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THFloatNN_Sigmoid_updateGradInput"
  c_THFloatNN_Sigmoid_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SmoothL1Criterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_SmoothL1Criterion_updateOutput"
  c_THFloatNN_SmoothL1Criterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_SmoothL1Criterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THFloatNN_SmoothL1Criterion_updateGradInput"
  c_THFloatNN_SmoothL1Criterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THFloatNN_SoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THFloatNN_SoftMax_updateOutput"
  c_THFloatNN_SoftMax_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_SoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THFloatNN_SoftMax_updateGradInput"
  c_THFloatNN_SoftMax_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_SoftPlus_updateOutput : state input output beta threshold -> void
foreign import ccall "THNN.h THFloatNN_SoftPlus_updateOutput"
  c_THFloatNN_SoftPlus_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SoftPlus_updateGradInput : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THFloatNN_SoftPlus_updateGradInput"
  c_THFloatNN_SoftPlus_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SoftShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THFloatNN_SoftShrink_updateOutput"
  c_THFloatNN_SoftShrink_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_SoftShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THFloatNN_SoftShrink_updateGradInput"
  c_THFloatNN_SoftShrink_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_IndexLinear_updateOutput : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h THFloatNN_IndexLinear_updateOutput"
  c_THFloatNN_IndexLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_IndexLinear_accGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h THFloatNN_IndexLinear_accGradParameters"
  c_THFloatNN_IndexLinear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_IndexLinear_accUpdateGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h THFloatNN_IndexLinear_accUpdateGradParameters"
  c_THFloatNN_IndexLinear_accUpdateGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_IndexLinear_updateParameters : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h THFloatNN_IndexLinear_updateParameters"
  c_THFloatNN_IndexLinear_updateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SparseLinear_updateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_updateOutput"
  c_THFloatNN_SparseLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SparseLinear_accGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_accGradParameters"
  c_THFloatNN_SparseLinear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SparseLinear_zeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_zeroGradParameters"
  c_THFloatNN_SparseLinear_zeroGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SparseLinear_updateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_updateParameters"
  c_THFloatNN_SparseLinear_updateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_SparseLinear_legacyUpdateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_legacyUpdateOutput"
  c_THFloatNN_SparseLinear_legacyUpdateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SparseLinear_legacyAccGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_legacyAccGradParameters"
  c_THFloatNN_SparseLinear_legacyAccGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SparseLinear_legacyZeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_legacyZeroGradParameters"
  c_THFloatNN_SparseLinear_legacyZeroGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SparseLinear_legacyUpdateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THFloatNN_SparseLinear_legacyUpdateParameters"
  c_THFloatNN_SparseLinear_legacyUpdateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_Sqrt_updateOutput : state input output eps -> void
foreign import ccall "THNN.h THFloatNN_Sqrt_updateOutput"
  c_THFloatNN_Sqrt_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatNN_Sqrt_updateGradInput : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THFloatNN_Sqrt_updateGradInput"
  c_THFloatNN_Sqrt_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Square_updateOutput : state input output -> void
foreign import ccall "THNN.h THFloatNN_Square_updateOutput"
  c_THFloatNN_Square_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Square_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THFloatNN_Square_updateGradInput"
  c_THFloatNN_Square_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Tanh_updateOutput : state input output -> void
foreign import ccall "THNN.h THFloatNN_Tanh_updateOutput"
  c_THFloatNN_Tanh_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Tanh_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THFloatNN_Tanh_updateGradInput"
  c_THFloatNN_Tanh_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_Threshold_updateOutput : state input output threshold val inplace -> void
foreign import ccall "THNN.h THFloatNN_Threshold_updateOutput"
  c_THFloatNN_Threshold_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_Threshold_updateGradInput : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THFloatNN_Threshold_updateGradInput"
  c_THFloatNN_Threshold_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THFloatNN_TemporalConvolution_updateOutput : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THFloatNN_TemporalConvolution_updateOutput"
  c_THFloatNN_TemporalConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalConvolution_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THFloatNN_TemporalConvolution_updateGradInput"
  c_THFloatNN_TemporalConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalConvolution_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THFloatNN_TemporalConvolution_accGradParameters"
  c_THFloatNN_TemporalConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_TemporalMaxPooling_updateOutput : state input output indices kW dW -> void
foreign import ccall "THNN.h THFloatNN_TemporalMaxPooling_updateOutput"
  c_THFloatNN_TemporalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalMaxPooling_updateGradInput : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h THFloatNN_TemporalMaxPooling_updateGradInput"
  c_THFloatNN_TemporalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalSubSampling_updateOutput : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THFloatNN_TemporalSubSampling_updateOutput"
  c_THFloatNN_TemporalSubSampling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalSubSampling_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THFloatNN_TemporalSubSampling_updateGradInput"
  c_THFloatNN_TemporalSubSampling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THFloatNN_TemporalSubSampling_accGradParameters"
  c_THFloatNN_TemporalSubSampling_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_TemporalRowConvolution_updateOutput : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THFloatNN_TemporalRowConvolution_updateOutput"
  c_THFloatNN_TemporalRowConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_TemporalRowConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THFloatNN_TemporalRowConvolution_updateGradInput"
  c_THFloatNN_TemporalRowConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_TemporalRowConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THFloatNN_TemporalRowConvolution_accGradParameters"
  c_THFloatNN_TemporalRowConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- |c_THFloatNN_TemporalUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THFloatNN_TemporalUpSamplingNearest_updateOutput"
  c_THFloatNN_TemporalUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_TemporalUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THFloatNN_TemporalUpSamplingNearest_updateGradInput"
  c_THFloatNN_TemporalUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_TemporalUpSamplingLinear_updateOutput : state input output outputWidth -> void
foreign import ccall "THNN.h THFloatNN_TemporalUpSamplingLinear_updateOutput"
  c_THFloatNN_TemporalUpSamplingLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_TemporalUpSamplingLinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h THFloatNN_TemporalUpSamplingLinear_updateGradInput"
  c_THFloatNN_TemporalUpSamplingLinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_BatchNormalization_updateOutput : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THFloatNN_BatchNormalization_updateOutput"
  c_THFloatNN_BatchNormalization_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_BatchNormalization_backward : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THFloatNN_BatchNormalization_backward"
  c_THFloatNN_BatchNormalization_backward :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THFloatNN_SpatialConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMap_updateOutput"
  c_THFloatNN_SpatialConvolutionMap_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMap_updateGradInput"
  c_THFloatNN_SpatialConvolutionMap_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMap_accGradParameters"
  c_THFloatNN_SpatialConvolutionMap_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialConvolutionMM_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMM_updateOutput"
  c_THFloatNN_SpatialConvolutionMM_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMM_updateGradInput"
  c_THFloatNN_SpatialConvolutionMM_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionMM_accGradParameters"
  c_THFloatNN_SpatialConvolutionMM_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialConvolutionLocal_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionLocal_updateOutput"
  c_THFloatNN_SpatialConvolutionLocal_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THFloatNN_SpatialConvolutionLocal_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionLocal_updateGradInput"
  c_THFloatNN_SpatialConvolutionLocal_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THFloatNN_SpatialConvolutionLocal_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialConvolutionLocal_accGradParameters"
  c_THFloatNN_SpatialConvolutionLocal_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- |c_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h THFloatNN_SpatialAdaptiveMaxPooling_updateOutput"
  c_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput"
  c_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput : state input output osizeW osizeH -> void
foreign import ccall "THNN.h THFloatNN_SpatialAdaptiveAveragePooling_updateOutput"
  c_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput"
  c_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SpatialAveragePooling_updateOutput : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THFloatNN_SpatialAveragePooling_updateOutput"
  c_THFloatNN_SpatialAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THFloatNN_SpatialAveragePooling_updateGradInput : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THFloatNN_SpatialAveragePooling_updateGradInput"
  c_THFloatNN_SpatialAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THFloatNN_SpatialFractionalMaxPooling_updateOutput : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THFloatNN_SpatialFractionalMaxPooling_updateOutput"
  c_THFloatNN_SpatialFractionalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_SpatialFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THFloatNN_SpatialFractionalMaxPooling_updateGradInput"
  c_THFloatNN_SpatialFractionalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THFloatNN_SpatialFullConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolution_updateOutput"
  c_THFloatNN_SpatialFullConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolution_updateGradInput"
  c_THFloatNN_SpatialFullConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolution_accGradParameters"
  c_THFloatNN_SpatialFullConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialFullConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolutionMap_updateOutput"
  c_THFloatNN_SpatialFullConvolutionMap_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolutionMap_updateGradInput"
  c_THFloatNN_SpatialFullConvolutionMap_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullConvolutionMap_accGradParameters"
  c_THFloatNN_SpatialFullConvolutionMap_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THFloatNN_SpatialDilatedConvolution_updateOutput"
  c_THFloatNN_SpatialDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THFloatNN_SpatialDilatedConvolution_updateGradInput"
  c_THFloatNN_SpatialDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialDilatedConvolution_accGradParameters"
  c_THFloatNN_SpatialDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialFullDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullDilatedConvolution_updateOutput"
  c_THFloatNN_SpatialFullDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullDilatedConvolution_updateGradInput"
  c_THFloatNN_SpatialFullDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialFullDilatedConvolution_accGradParameters"
  c_THFloatNN_SpatialFullDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialMaxPooling_updateOutput"
  c_THFloatNN_SpatialMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_SpatialMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialMaxPooling_updateGradInput"
  c_THFloatNN_SpatialMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_SpatialDilatedMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialDilatedMaxPooling_updateOutput"
  c_THFloatNN_SpatialDilatedMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_SpatialDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialDilatedMaxPooling_updateGradInput"
  c_THFloatNN_SpatialDilatedMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_SpatialMaxUnpooling_updateOutput : state input output indices owidth oheight -> void
foreign import ccall "THNN.h THFloatNN_SpatialMaxUnpooling_updateOutput"
  c_THFloatNN_SpatialMaxUnpooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialMaxUnpooling_updateGradInput : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h THFloatNN_SpatialMaxUnpooling_updateGradInput"
  c_THFloatNN_SpatialMaxUnpooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialSubSampling_updateOutput : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialSubSampling_updateOutput"
  c_THFloatNN_SpatialSubSampling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialSubSampling_updateGradInput : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THFloatNN_SpatialSubSampling_updateGradInput"
  c_THFloatNN_SpatialSubSampling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THFloatNN_SpatialSubSampling_accGradParameters"
  c_THFloatNN_SpatialSubSampling_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_SpatialUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THFloatNN_SpatialUpSamplingNearest_updateOutput"
  c_THFloatNN_SpatialUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_SpatialUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THFloatNN_SpatialUpSamplingNearest_updateGradInput"
  c_THFloatNN_SpatialUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_SpatialUpSamplingBilinear_updateOutput : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h THFloatNN_SpatialUpSamplingBilinear_updateOutput"
  c_THFloatNN_SpatialUpSamplingBilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialUpSamplingBilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h THFloatNN_SpatialUpSamplingBilinear_updateGradInput"
  c_THFloatNN_SpatialUpSamplingBilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialGridSamplerBilinear_updateOutput : state input grid output padding_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialGridSamplerBilinear_updateOutput"
  c_THFloatNN_SpatialGridSamplerBilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_SpatialGridSamplerBilinear_updateGradInput : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THFloatNN_SpatialGridSamplerBilinear_updateGradInput"
  c_THFloatNN_SpatialGridSamplerBilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_unfolded_acc : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THFloatNN_unfolded_acc"
  c_THFloatNN_unfolded_acc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_unfolded_copy : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THFloatNN_unfolded_copy"
  c_THFloatNN_unfolded_copy :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricAveragePooling_updateOutput : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAveragePooling_updateOutput"
  c_THFloatNN_VolumetricAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THFloatNN_VolumetricAveragePooling_updateGradInput : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAveragePooling_updateGradInput"
  c_THFloatNN_VolumetricAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THFloatNN_VolumetricConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolution_updateOutput"
  c_THFloatNN_VolumetricConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricConvolution_updateGradInput : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolution_updateGradInput"
  c_THFloatNN_VolumetricConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolution_accGradParameters"
  c_THFloatNN_VolumetricConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_VolumetricConvolutionMM_updateOutput : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolutionMM_updateOutput"
  c_THFloatNN_VolumetricConvolutionMM_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolutionMM_updateGradInput"
  c_THFloatNN_VolumetricConvolutionMM_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THFloatNN_VolumetricConvolutionMM_accGradParameters"
  c_THFloatNN_VolumetricConvolutionMM_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_VolumetricFractionalMaxPooling_updateOutput : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFractionalMaxPooling_updateOutput"
  c_THFloatNN_VolumetricFractionalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFractionalMaxPooling_updateGradInput"
  c_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THFloatNN_VolumetricFullConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullConvolution_updateOutput"
  c_THFloatNN_VolumetricFullConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricFullConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullConvolution_updateGradInput"
  c_THFloatNN_VolumetricFullConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullConvolution_accGradParameters"
  c_THFloatNN_VolumetricFullConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_VolumetricDilatedConvolution_updateOutput : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricDilatedConvolution_updateOutput"
  c_THFloatNN_VolumetricDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricDilatedConvolution_updateGradInput"
  c_THFloatNN_VolumetricDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THFloatNN_VolumetricDilatedConvolution_accGradParameters"
  c_THFloatNN_VolumetricDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_VolumetricFullDilatedConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullDilatedConvolution_updateOutput"
  c_THFloatNN_VolumetricFullDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullDilatedConvolution_updateGradInput"
  c_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THFloatNN_VolumetricFullDilatedConvolution_accGradParameters"
  c_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THFloatNN_VolumetricMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THFloatNN_VolumetricMaxPooling_updateOutput"
  c_THFloatNN_VolumetricMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_VolumetricMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THFloatNN_VolumetricMaxPooling_updateGradInput"
  c_THFloatNN_VolumetricMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_VolumetricDilatedMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THFloatNN_VolumetricDilatedMaxPooling_updateOutput"
  c_THFloatNN_VolumetricDilatedMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THFloatNN_VolumetricDilatedMaxPooling_updateGradInput"
  c_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_VolumetricMaxUnpooling_updateOutput : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricMaxUnpooling_updateOutput"
  c_THFloatNN_VolumetricMaxUnpooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricMaxUnpooling_updateGradInput : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricMaxUnpooling_updateGradInput"
  c_THFloatNN_VolumetricMaxUnpooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput"
  c_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput"
  c_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput"
  c_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput"
  c_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THFloatNN_SpatialReflectionPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THFloatNN_SpatialReflectionPadding_updateOutput"
  c_THFloatNN_SpatialReflectionPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THFloatNN_SpatialReflectionPadding_updateGradInput"
  c_THFloatNN_SpatialReflectionPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialReplicationPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THFloatNN_SpatialReplicationPadding_updateOutput"
  c_THFloatNN_SpatialReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_SpatialReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THFloatNN_SpatialReplicationPadding_updateGradInput"
  c_THFloatNN_SpatialReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_FeatureLPPooling_updateOutput : state input output power width stride batchMode -> void
foreign import ccall "THNN.h THFloatNN_FeatureLPPooling_updateOutput"
  c_THFloatNN_FeatureLPPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_FeatureLPPooling_updateGradInput : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THFloatNN_FeatureLPPooling_updateGradInput"
  c_THFloatNN_FeatureLPPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THFloatNN_VolumetricReplicationPadding_updateOutput : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THFloatNN_VolumetricReplicationPadding_updateOutput"
  c_THFloatNN_VolumetricReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricReplicationPadding_updateGradInput : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THFloatNN_VolumetricReplicationPadding_updateGradInput"
  c_THFloatNN_VolumetricReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THFloatNN_VolumetricUpSamplingNearest_updateOutput"
  c_THFloatNN_VolumetricUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_VolumetricUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THFloatNN_VolumetricUpSamplingNearest_updateGradInput"
  c_THFloatNN_VolumetricUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THFloatNN_VolumetricUpSamplingTrilinear_updateOutput"
  c_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput"
  c_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalReflectionPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THFloatNN_TemporalReflectionPadding_updateOutput"
  c_THFloatNN_TemporalReflectionPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THFloatNN_TemporalReflectionPadding_updateGradInput"
  c_THFloatNN_TemporalReflectionPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalReplicationPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THFloatNN_TemporalReplicationPadding_updateOutput"
  c_THFloatNN_TemporalReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatNN_TemporalReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THFloatNN_TemporalReplicationPadding_updateGradInput"
  c_THFloatNN_TemporalReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |p_THFloatNN_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THFloatNN_Abs_updateOutput"
  p_THFloatNN_Abs_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THFloatNN_Abs_updateGradInput"
  p_THFloatNN_Abs_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_AbsCriterion_updateOutput"
  p_THFloatNN_AbsCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_AbsCriterion_updateGradInput"
  p_THFloatNN_AbsCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h &THFloatNN_BCECriterion_updateOutput"
  p_THFloatNN_BCECriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_BCECriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h &THFloatNN_BCECriterion_updateGradInput"
  p_THFloatNN_BCECriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_ClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THFloatNN_ClassNLLCriterion_updateOutput"
  p_THFloatNN_ClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THFloatNN_ClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THFloatNN_ClassNLLCriterion_updateGradInput"
  p_THFloatNN_ClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THFloatNN_SpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THFloatNN_SpatialClassNLLCriterion_updateOutput"
  p_THFloatNN_SpatialClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THFloatNN_SpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THFloatNN_SpatialClassNLLCriterion_updateGradInput"
  p_THFloatNN_SpatialClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THFloatNN_ELU_updateOutput : Pointer to function : state input output alpha inplace -> void
foreign import ccall "THNN.h &THFloatNN_ELU_updateOutput"
  p_THFloatNN_ELU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h &THFloatNN_ELU_updateGradInput"
  p_THFloatNN_ELU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_DistKLDivCriterion_updateOutput"
  p_THFloatNN_DistKLDivCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_DistKLDivCriterion_updateGradInput"
  p_THFloatNN_DistKLDivCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THFloatNN_GatedLinear_updateOutput"
  p_THFloatNN_GatedLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THFloatNN_GatedLinear_updateGradInput"
  p_THFloatNN_GatedLinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_HardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THFloatNN_HardShrink_updateOutput"
  p_THFloatNN_HardShrink_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_HardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THFloatNN_HardShrink_updateGradInput"
  p_THFloatNN_HardShrink_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THFloatNN_HardTanh_updateOutput"
  p_THFloatNN_HardTanh_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THFloatNN_HardTanh_updateGradInput"
  p_THFloatNN_HardTanh_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THFloatNN_L1Cost_updateOutput"
  p_THFloatNN_L1Cost_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THFloatNN_L1Cost_updateGradInput"
  p_THFloatNN_L1Cost_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THFloatNN_LeakyReLU_updateOutput"
  p_THFloatNN_LeakyReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THFloatNN_LeakyReLU_updateGradInput"
  p_THFloatNN_LeakyReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THFloatNN_GRUFused_updateOutput"
  p_THFloatNN_GRUFused_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THFloatNN_GRUFused_updateGradInput"
  p_THFloatNN_GRUFused_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THFloatNN_LSTMFused_updateOutput"
  p_THFloatNN_LSTMFused_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THFloatNN_LSTMFused_updateGradInput"
  p_THFloatNN_LSTMFused_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THFloatNN_LogSigmoid_updateOutput"
  p_THFloatNN_LogSigmoid_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THFloatNN_LogSigmoid_updateGradInput"
  p_THFloatNN_LogSigmoid_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THFloatNN_LogSoftMax_updateOutput"
  p_THFloatNN_LogSoftMax_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THFloatNN_LogSoftMax_updateGradInput"
  p_THFloatNN_LogSoftMax_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_LookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h &THFloatNN_LookupTable_accGradParameters"
  p_THFloatNN_LookupTable_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_LookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h &THFloatNN_LookupTable_renorm"
  p_THFloatNN_LookupTable_renorm :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THFloatNN_MarginCriterion_updateOutput"
  p_THFloatNN_MarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ())

-- |p_THFloatNN_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THFloatNN_MarginCriterion_updateGradInput"
  p_THFloatNN_MarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ())

-- |p_THFloatNN_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage -> void
foreign import ccall "THNN.h &THFloatNN_SoftMarginCriterion_updateOutput"
  p_THFloatNN_SoftMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THFloatNN_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h &THFloatNN_SoftMarginCriterion_updateGradInput"
  p_THFloatNN_SoftMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THFloatNN_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_MSECriterion_updateOutput"
  p_THFloatNN_MSECriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_MSECriterion_updateGradInput"
  p_THFloatNN_MSECriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_MultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h &THFloatNN_MultiLabelMarginCriterion_updateOutput"
  p_THFloatNN_MultiLabelMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THFloatNN_MultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h &THFloatNN_MultiLabelMarginCriterion_updateGradInput"
  p_THFloatNN_MultiLabelMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THFloatNN_MultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THFloatNN_MultiMarginCriterion_updateOutput"
  p_THFloatNN_MultiMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_MultiMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THFloatNN_MultiMarginCriterion_updateGradInput"
  p_THFloatNN_MultiMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THFloatNN_PReLU_updateOutput"
  p_THFloatNN_PReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THFloatNN_PReLU_updateGradInput"
  p_THFloatNN_PReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THFloatNN_PReLU_accGradParameters"
  p_THFloatNN_PReLU_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_Linear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THFloatNN_Linear_updateOutput"
  p_THFloatNN_Linear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Linear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THFloatNN_Linear_updateGradInput"
  p_THFloatNN_Linear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Linear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THFloatNN_Linear_accGradParameters"
  p_THFloatNN_Linear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THFloatNN_RReLU_updateOutput"
  p_THFloatNN_RReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ())

-- |p_THFloatNN_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THFloatNN_RReLU_updateGradInput"
  p_THFloatNN_RReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- |p_THFloatNN_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THFloatNN_Sigmoid_updateOutput"
  p_THFloatNN_Sigmoid_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THFloatNN_Sigmoid_updateGradInput"
  p_THFloatNN_Sigmoid_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_SmoothL1Criterion_updateOutput"
  p_THFloatNN_SmoothL1Criterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THFloatNN_SmoothL1Criterion_updateGradInput"
  p_THFloatNN_SmoothL1Criterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THFloatNN_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THFloatNN_SoftMax_updateOutput"
  p_THFloatNN_SoftMax_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THFloatNN_SoftMax_updateGradInput"
  p_THFloatNN_SoftMax_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THFloatNN_SoftPlus_updateOutput"
  p_THFloatNN_SoftPlus_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THFloatNN_SoftPlus_updateGradInput"
  p_THFloatNN_SoftPlus_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THFloatNN_SoftShrink_updateOutput"
  p_THFloatNN_SoftShrink_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THFloatNN_SoftShrink_updateGradInput"
  p_THFloatNN_SoftShrink_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_IndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h &THFloatNN_IndexLinear_updateOutput"
  p_THFloatNN_IndexLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_IndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h &THFloatNN_IndexLinear_accGradParameters"
  p_THFloatNN_IndexLinear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_IndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THFloatNN_IndexLinear_accUpdateGradParameters"
  p_THFloatNN_IndexLinear_accUpdateGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_IndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h &THFloatNN_IndexLinear_updateParameters"
  p_THFloatNN_IndexLinear_updateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_updateOutput"
  p_THFloatNN_SparseLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_accGradParameters"
  p_THFloatNN_SparseLinear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_zeroGradParameters"
  p_THFloatNN_SparseLinear_zeroGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_updateParameters"
  p_THFloatNN_SparseLinear_updateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_legacyUpdateOutput"
  p_THFloatNN_SparseLinear_legacyUpdateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_legacyAccGradParameters"
  p_THFloatNN_SparseLinear_legacyAccGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_legacyZeroGradParameters"
  p_THFloatNN_SparseLinear_legacyZeroGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THFloatNN_SparseLinear_legacyUpdateParameters"
  p_THFloatNN_SparseLinear_legacyUpdateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THFloatNN_Sqrt_updateOutput"
  p_THFloatNN_Sqrt_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatNN_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THFloatNN_Sqrt_updateGradInput"
  p_THFloatNN_Sqrt_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THFloatNN_Square_updateOutput"
  p_THFloatNN_Square_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THFloatNN_Square_updateGradInput"
  p_THFloatNN_Square_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THFloatNN_Tanh_updateOutput"
  p_THFloatNN_Tanh_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THFloatNN_Tanh_updateGradInput"
  p_THFloatNN_Tanh_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THFloatNN_Threshold_updateOutput"
  p_THFloatNN_Threshold_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THFloatNN_Threshold_updateGradInput"
  p_THFloatNN_Threshold_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THFloatNN_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THFloatNN_TemporalConvolution_updateOutput"
  p_THFloatNN_TemporalConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THFloatNN_TemporalConvolution_updateGradInput"
  p_THFloatNN_TemporalConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THFloatNN_TemporalConvolution_accGradParameters"
  p_THFloatNN_TemporalConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_TemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THNN.h &THFloatNN_TemporalMaxPooling_updateOutput"
  p_THFloatNN_TemporalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h &THFloatNN_TemporalMaxPooling_updateGradInput"
  p_THFloatNN_TemporalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THFloatNN_TemporalSubSampling_updateOutput"
  p_THFloatNN_TemporalSubSampling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THFloatNN_TemporalSubSampling_updateGradInput"
  p_THFloatNN_TemporalSubSampling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THFloatNN_TemporalSubSampling_accGradParameters"
  p_THFloatNN_TemporalSubSampling_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THFloatNN_TemporalRowConvolution_updateOutput"
  p_THFloatNN_TemporalRowConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THFloatNN_TemporalRowConvolution_updateGradInput"
  p_THFloatNN_TemporalRowConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THFloatNN_TemporalRowConvolution_accGradParameters"
  p_THFloatNN_TemporalRowConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- |p_THFloatNN_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_TemporalUpSamplingNearest_updateOutput"
  p_THFloatNN_TemporalUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_TemporalUpSamplingNearest_updateGradInput"
  p_THFloatNN_TemporalUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_TemporalUpSamplingLinear_updateOutput"
  p_THFloatNN_TemporalUpSamplingLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_TemporalUpSamplingLinear_updateGradInput"
  p_THFloatNN_TemporalUpSamplingLinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_BatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THFloatNN_BatchNormalization_updateOutput"
  p_THFloatNN_BatchNormalization_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_BatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THFloatNN_BatchNormalization_backward"
  p_THFloatNN_BatchNormalization_backward :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THFloatNN_SpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMap_updateOutput"
  p_THFloatNN_SpatialConvolutionMap_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMap_updateGradInput"
  p_THFloatNN_SpatialConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMap_accGradParameters"
  p_THFloatNN_SpatialConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMM_updateOutput"
  p_THFloatNN_SpatialConvolutionMM_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMM_updateGradInput"
  p_THFloatNN_SpatialConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionMM_accGradParameters"
  p_THFloatNN_SpatialConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionLocal_updateOutput"
  p_THFloatNN_SpatialConvolutionLocal_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THFloatNN_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionLocal_updateGradInput"
  p_THFloatNN_SpatialConvolutionLocal_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THFloatNN_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialConvolutionLocal_accGradParameters"
  p_THFloatNN_SpatialConvolutionLocal_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- |p_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAdaptiveMaxPooling_updateOutput"
  p_THFloatNN_SpatialAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput"
  p_THFloatNN_SpatialAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAdaptiveAveragePooling_updateOutput"
  p_THFloatNN_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput"
  p_THFloatNN_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAveragePooling_updateOutput"
  p_THFloatNN_SpatialAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THFloatNN_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THFloatNN_SpatialAveragePooling_updateGradInput"
  p_THFloatNN_SpatialAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THFloatNN_SpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFractionalMaxPooling_updateOutput"
  p_THFloatNN_SpatialFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_SpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFractionalMaxPooling_updateGradInput"
  p_THFloatNN_SpatialFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THFloatNN_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolution_updateOutput"
  p_THFloatNN_SpatialFullConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolution_updateGradInput"
  p_THFloatNN_SpatialFullConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolution_accGradParameters"
  p_THFloatNN_SpatialFullConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolutionMap_updateOutput"
  p_THFloatNN_SpatialFullConvolutionMap_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolutionMap_updateGradInput"
  p_THFloatNN_SpatialFullConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullConvolutionMap_accGradParameters"
  p_THFloatNN_SpatialFullConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialDilatedConvolution_updateOutput"
  p_THFloatNN_SpatialDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialDilatedConvolution_updateGradInput"
  p_THFloatNN_SpatialDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialDilatedConvolution_accGradParameters"
  p_THFloatNN_SpatialDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullDilatedConvolution_updateOutput"
  p_THFloatNN_SpatialFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullDilatedConvolution_updateGradInput"
  p_THFloatNN_SpatialFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialFullDilatedConvolution_accGradParameters"
  p_THFloatNN_SpatialFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialMaxPooling_updateOutput"
  p_THFloatNN_SpatialMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_SpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialMaxPooling_updateGradInput"
  p_THFloatNN_SpatialMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_SpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialDilatedMaxPooling_updateOutput"
  p_THFloatNN_SpatialDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_SpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialDilatedMaxPooling_updateGradInput"
  p_THFloatNN_SpatialDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_SpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THNN.h &THFloatNN_SpatialMaxUnpooling_updateOutput"
  p_THFloatNN_SpatialMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h &THFloatNN_SpatialMaxUnpooling_updateGradInput"
  p_THFloatNN_SpatialMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialSubSampling_updateOutput"
  p_THFloatNN_SpatialSubSampling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THFloatNN_SpatialSubSampling_updateGradInput"
  p_THFloatNN_SpatialSubSampling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THFloatNN_SpatialSubSampling_accGradParameters"
  p_THFloatNN_SpatialSubSampling_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_SpatialUpSamplingNearest_updateOutput"
  p_THFloatNN_SpatialUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_SpatialUpSamplingNearest_updateGradInput"
  p_THFloatNN_SpatialUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_SpatialUpSamplingBilinear_updateOutput"
  p_THFloatNN_SpatialUpSamplingBilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_SpatialUpSamplingBilinear_updateGradInput"
  p_THFloatNN_SpatialUpSamplingBilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialGridSamplerBilinear_updateOutput"
  p_THFloatNN_SpatialGridSamplerBilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THFloatNN_SpatialGridSamplerBilinear_updateGradInput"
  p_THFloatNN_SpatialGridSamplerBilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_unfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THFloatNN_unfolded_acc"
  p_THFloatNN_unfolded_acc :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_unfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THFloatNN_unfolded_copy"
  p_THFloatNN_unfolded_copy :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAveragePooling_updateOutput"
  p_THFloatNN_VolumetricAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THFloatNN_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAveragePooling_updateGradInput"
  p_THFloatNN_VolumetricAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THFloatNN_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolution_updateOutput"
  p_THFloatNN_VolumetricConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolution_updateGradInput"
  p_THFloatNN_VolumetricConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolution_accGradParameters"
  p_THFloatNN_VolumetricConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_VolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolutionMM_updateOutput"
  p_THFloatNN_VolumetricConvolutionMM_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolutionMM_updateGradInput"
  p_THFloatNN_VolumetricConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricConvolutionMM_accGradParameters"
  p_THFloatNN_VolumetricConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_VolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFractionalMaxPooling_updateOutput"
  p_THFloatNN_VolumetricFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFractionalMaxPooling_updateGradInput"
  p_THFloatNN_VolumetricFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THFloatNN_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullConvolution_updateOutput"
  p_THFloatNN_VolumetricFullConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullConvolution_updateGradInput"
  p_THFloatNN_VolumetricFullConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullConvolution_accGradParameters"
  p_THFloatNN_VolumetricFullConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricDilatedConvolution_updateOutput"
  p_THFloatNN_VolumetricDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricDilatedConvolution_updateGradInput"
  p_THFloatNN_VolumetricDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricDilatedConvolution_accGradParameters"
  p_THFloatNN_VolumetricDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullDilatedConvolution_updateOutput"
  p_THFloatNN_VolumetricFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullDilatedConvolution_updateGradInput"
  p_THFloatNN_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricFullDilatedConvolution_accGradParameters"
  p_THFloatNN_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THFloatNN_VolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricMaxPooling_updateOutput"
  p_THFloatNN_VolumetricMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_VolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricMaxPooling_updateGradInput"
  p_THFloatNN_VolumetricMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_VolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricDilatedMaxPooling_updateOutput"
  p_THFloatNN_VolumetricDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricDilatedMaxPooling_updateGradInput"
  p_THFloatNN_VolumetricDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_VolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricMaxUnpooling_updateOutput"
  p_THFloatNN_VolumetricMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricMaxUnpooling_updateGradInput"
  p_THFloatNN_VolumetricMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput"
  p_THFloatNN_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput"
  p_THFloatNN_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput"
  p_THFloatNN_VolumetricAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput"
  p_THFloatNN_VolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THFloatNN_SpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THFloatNN_SpatialReflectionPadding_updateOutput"
  p_THFloatNN_SpatialReflectionPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THFloatNN_SpatialReflectionPadding_updateGradInput"
  p_THFloatNN_SpatialReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THFloatNN_SpatialReplicationPadding_updateOutput"
  p_THFloatNN_SpatialReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THFloatNN_SpatialReplicationPadding_updateGradInput"
  p_THFloatNN_SpatialReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_FeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THFloatNN_FeatureLPPooling_updateOutput"
  p_THFloatNN_FeatureLPPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THFloatNN_FeatureLPPooling_updateGradInput"
  p_THFloatNN_FeatureLPPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THFloatNN_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricReplicationPadding_updateOutput"
  p_THFloatNN_VolumetricReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricReplicationPadding_updateGradInput"
  p_THFloatNN_VolumetricReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricUpSamplingNearest_updateOutput"
  p_THFloatNN_VolumetricUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricUpSamplingNearest_updateGradInput"
  p_THFloatNN_VolumetricUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricUpSamplingTrilinear_updateOutput"
  p_THFloatNN_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput"
  p_THFloatNN_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THFloatNN_TemporalReflectionPadding_updateOutput"
  p_THFloatNN_TemporalReflectionPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THFloatNN_TemporalReflectionPadding_updateGradInput"
  p_THFloatNN_TemporalReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THFloatNN_TemporalReplicationPadding_updateOutput"
  p_THFloatNN_TemporalReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatNN_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THFloatNN_TemporalReplicationPadding_updateGradInput"
  p_THFloatNN_TemporalReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

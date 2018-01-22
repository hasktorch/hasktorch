{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleNN (
    c_THDoubleNN_Abs_updateOutput,
    c_THDoubleNN_Abs_updateGradInput,
    c_THDoubleNN_AbsCriterion_updateOutput,
    c_THDoubleNN_AbsCriterion_updateGradInput,
    c_THDoubleNN_BCECriterion_updateOutput,
    c_THDoubleNN_BCECriterion_updateGradInput,
    c_THDoubleNN_ClassNLLCriterion_updateOutput,
    c_THDoubleNN_ClassNLLCriterion_updateGradInput,
    c_THDoubleNN_SpatialClassNLLCriterion_updateOutput,
    c_THDoubleNN_SpatialClassNLLCriterion_updateGradInput,
    c_THDoubleNN_ELU_updateOutput,
    c_THDoubleNN_ELU_updateGradInput,
    c_THDoubleNN_DistKLDivCriterion_updateOutput,
    c_THDoubleNN_DistKLDivCriterion_updateGradInput,
    c_THDoubleNN_GatedLinear_updateOutput,
    c_THDoubleNN_GatedLinear_updateGradInput,
    c_THDoubleNN_HardShrink_updateOutput,
    c_THDoubleNN_HardShrink_updateGradInput,
    c_THDoubleNN_HardTanh_updateOutput,
    c_THDoubleNN_HardTanh_updateGradInput,
    c_THDoubleNN_L1Cost_updateOutput,
    c_THDoubleNN_L1Cost_updateGradInput,
    c_THDoubleNN_LeakyReLU_updateOutput,
    c_THDoubleNN_LeakyReLU_updateGradInput,
    c_THDoubleNN_GRUFused_updateOutput,
    c_THDoubleNN_GRUFused_updateGradInput,
    c_THDoubleNN_LSTMFused_updateOutput,
    c_THDoubleNN_LSTMFused_updateGradInput,
    c_THDoubleNN_LogSigmoid_updateOutput,
    c_THDoubleNN_LogSigmoid_updateGradInput,
    c_THDoubleNN_LogSoftMax_updateOutput,
    c_THDoubleNN_LogSoftMax_updateGradInput,
    c_THDoubleNN_LookupTable_accGradParameters,
    c_THDoubleNN_LookupTable_renorm,
    c_THDoubleNN_MarginCriterion_updateOutput,
    c_THDoubleNN_MarginCriterion_updateGradInput,
    c_THDoubleNN_SoftMarginCriterion_updateOutput,
    c_THDoubleNN_SoftMarginCriterion_updateGradInput,
    c_THDoubleNN_MSECriterion_updateOutput,
    c_THDoubleNN_MSECriterion_updateGradInput,
    c_THDoubleNN_MultiLabelMarginCriterion_updateOutput,
    c_THDoubleNN_MultiLabelMarginCriterion_updateGradInput,
    c_THDoubleNN_MultiMarginCriterion_updateOutput,
    c_THDoubleNN_MultiMarginCriterion_updateGradInput,
    c_THDoubleNN_PReLU_updateOutput,
    c_THDoubleNN_PReLU_updateGradInput,
    c_THDoubleNN_PReLU_accGradParameters,
    c_THDoubleNN_Linear_updateOutput,
    c_THDoubleNN_Linear_updateGradInput,
    c_THDoubleNN_Linear_accGradParameters,
    c_THDoubleNN_RReLU_updateOutput,
    c_THDoubleNN_RReLU_updateGradInput,
    c_THDoubleNN_Sigmoid_updateOutput,
    c_THDoubleNN_Sigmoid_updateGradInput,
    c_THDoubleNN_SmoothL1Criterion_updateOutput,
    c_THDoubleNN_SmoothL1Criterion_updateGradInput,
    c_THDoubleNN_SoftMax_updateOutput,
    c_THDoubleNN_SoftMax_updateGradInput,
    c_THDoubleNN_SoftPlus_updateOutput,
    c_THDoubleNN_SoftPlus_updateGradInput,
    c_THDoubleNN_SoftShrink_updateOutput,
    c_THDoubleNN_SoftShrink_updateGradInput,
    c_THDoubleNN_IndexLinear_updateOutput,
    c_THDoubleNN_IndexLinear_accGradParameters,
    c_THDoubleNN_IndexLinear_accUpdateGradParameters,
    c_THDoubleNN_IndexLinear_updateParameters,
    c_THDoubleNN_SparseLinear_updateOutput,
    c_THDoubleNN_SparseLinear_accGradParameters,
    c_THDoubleNN_SparseLinear_zeroGradParameters,
    c_THDoubleNN_SparseLinear_updateParameters,
    c_THDoubleNN_SparseLinear_legacyUpdateOutput,
    c_THDoubleNN_SparseLinear_legacyAccGradParameters,
    c_THDoubleNN_SparseLinear_legacyZeroGradParameters,
    c_THDoubleNN_SparseLinear_legacyUpdateParameters,
    c_THDoubleNN_Sqrt_updateOutput,
    c_THDoubleNN_Sqrt_updateGradInput,
    c_THDoubleNN_Square_updateOutput,
    c_THDoubleNN_Square_updateGradInput,
    c_THDoubleNN_Tanh_updateOutput,
    c_THDoubleNN_Tanh_updateGradInput,
    c_THDoubleNN_Threshold_updateOutput,
    c_THDoubleNN_Threshold_updateGradInput,
    c_THDoubleNN_TemporalConvolution_updateOutput,
    c_THDoubleNN_TemporalConvolution_updateGradInput,
    c_THDoubleNN_TemporalConvolution_accGradParameters,
    c_THDoubleNN_TemporalMaxPooling_updateOutput,
    c_THDoubleNN_TemporalMaxPooling_updateGradInput,
    c_THDoubleNN_TemporalSubSampling_updateOutput,
    c_THDoubleNN_TemporalSubSampling_updateGradInput,
    c_THDoubleNN_TemporalSubSampling_accGradParameters,
    c_THDoubleNN_TemporalRowConvolution_updateOutput,
    c_THDoubleNN_TemporalRowConvolution_updateGradInput,
    c_THDoubleNN_TemporalRowConvolution_accGradParameters,
    c_THDoubleNN_TemporalUpSamplingNearest_updateOutput,
    c_THDoubleNN_TemporalUpSamplingNearest_updateGradInput,
    c_THDoubleNN_TemporalUpSamplingLinear_updateOutput,
    c_THDoubleNN_TemporalUpSamplingLinear_updateGradInput,
    c_THDoubleNN_BatchNormalization_updateOutput,
    c_THDoubleNN_BatchNormalization_backward,
    c_THDoubleNN_SpatialConvolutionMap_updateOutput,
    c_THDoubleNN_SpatialConvolutionMap_updateGradInput,
    c_THDoubleNN_SpatialConvolutionMap_accGradParameters,
    c_THDoubleNN_SpatialConvolutionMM_updateOutput,
    c_THDoubleNN_SpatialConvolutionMM_updateGradInput,
    c_THDoubleNN_SpatialConvolutionMM_accGradParameters,
    c_THDoubleNN_SpatialConvolutionLocal_updateOutput,
    c_THDoubleNN_SpatialConvolutionLocal_updateGradInput,
    c_THDoubleNN_SpatialConvolutionLocal_accGradParameters,
    c_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput,
    c_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput,
    c_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput,
    c_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput,
    c_THDoubleNN_SpatialAveragePooling_updateOutput,
    c_THDoubleNN_SpatialAveragePooling_updateGradInput,
    c_THDoubleNN_SpatialFractionalMaxPooling_updateOutput,
    c_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput,
    c_THDoubleNN_SpatialFullConvolution_updateOutput,
    c_THDoubleNN_SpatialFullConvolution_updateGradInput,
    c_THDoubleNN_SpatialFullConvolution_accGradParameters,
    c_THDoubleNN_SpatialFullConvolutionMap_updateOutput,
    c_THDoubleNN_SpatialFullConvolutionMap_updateGradInput,
    c_THDoubleNN_SpatialFullConvolutionMap_accGradParameters,
    c_THDoubleNN_SpatialDilatedConvolution_updateOutput,
    c_THDoubleNN_SpatialDilatedConvolution_updateGradInput,
    c_THDoubleNN_SpatialDilatedConvolution_accGradParameters,
    c_THDoubleNN_SpatialFullDilatedConvolution_updateOutput,
    c_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput,
    c_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters,
    c_THDoubleNN_SpatialMaxPooling_updateOutput,
    c_THDoubleNN_SpatialMaxPooling_updateGradInput,
    c_THDoubleNN_SpatialDilatedMaxPooling_updateOutput,
    c_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput,
    c_THDoubleNN_SpatialMaxUnpooling_updateOutput,
    c_THDoubleNN_SpatialMaxUnpooling_updateGradInput,
    c_THDoubleNN_SpatialSubSampling_updateOutput,
    c_THDoubleNN_SpatialSubSampling_updateGradInput,
    c_THDoubleNN_SpatialSubSampling_accGradParameters,
    c_THDoubleNN_SpatialUpSamplingNearest_updateOutput,
    c_THDoubleNN_SpatialUpSamplingNearest_updateGradInput,
    c_THDoubleNN_SpatialUpSamplingBilinear_updateOutput,
    c_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput,
    c_THDoubleNN_SpatialGridSamplerBilinear_updateOutput,
    c_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput,
    c_THDoubleNN_unfolded_acc,
    c_THDoubleNN_unfolded_copy,
    c_THDoubleNN_VolumetricAveragePooling_updateOutput,
    c_THDoubleNN_VolumetricAveragePooling_updateGradInput,
    c_THDoubleNN_VolumetricConvolution_updateOutput,
    c_THDoubleNN_VolumetricConvolution_updateGradInput,
    c_THDoubleNN_VolumetricConvolution_accGradParameters,
    c_THDoubleNN_VolumetricConvolutionMM_updateOutput,
    c_THDoubleNN_VolumetricConvolutionMM_updateGradInput,
    c_THDoubleNN_VolumetricConvolutionMM_accGradParameters,
    c_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput,
    c_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput,
    c_THDoubleNN_VolumetricFullConvolution_updateOutput,
    c_THDoubleNN_VolumetricFullConvolution_updateGradInput,
    c_THDoubleNN_VolumetricFullConvolution_accGradParameters,
    c_THDoubleNN_VolumetricDilatedConvolution_updateOutput,
    c_THDoubleNN_VolumetricDilatedConvolution_updateGradInput,
    c_THDoubleNN_VolumetricDilatedConvolution_accGradParameters,
    c_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput,
    c_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput,
    c_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters,
    c_THDoubleNN_VolumetricMaxPooling_updateOutput,
    c_THDoubleNN_VolumetricMaxPooling_updateGradInput,
    c_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput,
    c_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput,
    c_THDoubleNN_VolumetricMaxUnpooling_updateOutput,
    c_THDoubleNN_VolumetricMaxUnpooling_updateGradInput,
    c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput,
    c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput,
    c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput,
    c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput,
    c_THDoubleNN_SpatialReflectionPadding_updateOutput,
    c_THDoubleNN_SpatialReflectionPadding_updateGradInput,
    c_THDoubleNN_SpatialReplicationPadding_updateOutput,
    c_THDoubleNN_SpatialReplicationPadding_updateGradInput,
    c_THDoubleNN_FeatureLPPooling_updateOutput,
    c_THDoubleNN_FeatureLPPooling_updateGradInput,
    c_THDoubleNN_VolumetricReplicationPadding_updateOutput,
    c_THDoubleNN_VolumetricReplicationPadding_updateGradInput,
    c_THDoubleNN_VolumetricUpSamplingNearest_updateOutput,
    c_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput,
    c_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput,
    c_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput,
    c_THDoubleNN_TemporalReflectionPadding_updateOutput,
    c_THDoubleNN_TemporalReflectionPadding_updateGradInput,
    c_THDoubleNN_TemporalReplicationPadding_updateOutput,
    c_THDoubleNN_TemporalReplicationPadding_updateGradInput,
    p_THDoubleNN_Abs_updateOutput,
    p_THDoubleNN_Abs_updateGradInput,
    p_THDoubleNN_AbsCriterion_updateOutput,
    p_THDoubleNN_AbsCriterion_updateGradInput,
    p_THDoubleNN_BCECriterion_updateOutput,
    p_THDoubleNN_BCECriterion_updateGradInput,
    p_THDoubleNN_ClassNLLCriterion_updateOutput,
    p_THDoubleNN_ClassNLLCriterion_updateGradInput,
    p_THDoubleNN_SpatialClassNLLCriterion_updateOutput,
    p_THDoubleNN_SpatialClassNLLCriterion_updateGradInput,
    p_THDoubleNN_ELU_updateOutput,
    p_THDoubleNN_ELU_updateGradInput,
    p_THDoubleNN_DistKLDivCriterion_updateOutput,
    p_THDoubleNN_DistKLDivCriterion_updateGradInput,
    p_THDoubleNN_GatedLinear_updateOutput,
    p_THDoubleNN_GatedLinear_updateGradInput,
    p_THDoubleNN_HardShrink_updateOutput,
    p_THDoubleNN_HardShrink_updateGradInput,
    p_THDoubleNN_HardTanh_updateOutput,
    p_THDoubleNN_HardTanh_updateGradInput,
    p_THDoubleNN_L1Cost_updateOutput,
    p_THDoubleNN_L1Cost_updateGradInput,
    p_THDoubleNN_LeakyReLU_updateOutput,
    p_THDoubleNN_LeakyReLU_updateGradInput,
    p_THDoubleNN_GRUFused_updateOutput,
    p_THDoubleNN_GRUFused_updateGradInput,
    p_THDoubleNN_LSTMFused_updateOutput,
    p_THDoubleNN_LSTMFused_updateGradInput,
    p_THDoubleNN_LogSigmoid_updateOutput,
    p_THDoubleNN_LogSigmoid_updateGradInput,
    p_THDoubleNN_LogSoftMax_updateOutput,
    p_THDoubleNN_LogSoftMax_updateGradInput,
    p_THDoubleNN_LookupTable_accGradParameters,
    p_THDoubleNN_LookupTable_renorm,
    p_THDoubleNN_MarginCriterion_updateOutput,
    p_THDoubleNN_MarginCriterion_updateGradInput,
    p_THDoubleNN_SoftMarginCriterion_updateOutput,
    p_THDoubleNN_SoftMarginCriterion_updateGradInput,
    p_THDoubleNN_MSECriterion_updateOutput,
    p_THDoubleNN_MSECriterion_updateGradInput,
    p_THDoubleNN_MultiLabelMarginCriterion_updateOutput,
    p_THDoubleNN_MultiLabelMarginCriterion_updateGradInput,
    p_THDoubleNN_MultiMarginCriterion_updateOutput,
    p_THDoubleNN_MultiMarginCriterion_updateGradInput,
    p_THDoubleNN_PReLU_updateOutput,
    p_THDoubleNN_PReLU_updateGradInput,
    p_THDoubleNN_PReLU_accGradParameters,
    p_THDoubleNN_Linear_updateOutput,
    p_THDoubleNN_Linear_updateGradInput,
    p_THDoubleNN_Linear_accGradParameters,
    p_THDoubleNN_RReLU_updateOutput,
    p_THDoubleNN_RReLU_updateGradInput,
    p_THDoubleNN_Sigmoid_updateOutput,
    p_THDoubleNN_Sigmoid_updateGradInput,
    p_THDoubleNN_SmoothL1Criterion_updateOutput,
    p_THDoubleNN_SmoothL1Criterion_updateGradInput,
    p_THDoubleNN_SoftMax_updateOutput,
    p_THDoubleNN_SoftMax_updateGradInput,
    p_THDoubleNN_SoftPlus_updateOutput,
    p_THDoubleNN_SoftPlus_updateGradInput,
    p_THDoubleNN_SoftShrink_updateOutput,
    p_THDoubleNN_SoftShrink_updateGradInput,
    p_THDoubleNN_IndexLinear_updateOutput,
    p_THDoubleNN_IndexLinear_accGradParameters,
    p_THDoubleNN_IndexLinear_accUpdateGradParameters,
    p_THDoubleNN_IndexLinear_updateParameters,
    p_THDoubleNN_SparseLinear_updateOutput,
    p_THDoubleNN_SparseLinear_accGradParameters,
    p_THDoubleNN_SparseLinear_zeroGradParameters,
    p_THDoubleNN_SparseLinear_updateParameters,
    p_THDoubleNN_SparseLinear_legacyUpdateOutput,
    p_THDoubleNN_SparseLinear_legacyAccGradParameters,
    p_THDoubleNN_SparseLinear_legacyZeroGradParameters,
    p_THDoubleNN_SparseLinear_legacyUpdateParameters,
    p_THDoubleNN_Sqrt_updateOutput,
    p_THDoubleNN_Sqrt_updateGradInput,
    p_THDoubleNN_Square_updateOutput,
    p_THDoubleNN_Square_updateGradInput,
    p_THDoubleNN_Tanh_updateOutput,
    p_THDoubleNN_Tanh_updateGradInput,
    p_THDoubleNN_Threshold_updateOutput,
    p_THDoubleNN_Threshold_updateGradInput,
    p_THDoubleNN_TemporalConvolution_updateOutput,
    p_THDoubleNN_TemporalConvolution_updateGradInput,
    p_THDoubleNN_TemporalConvolution_accGradParameters,
    p_THDoubleNN_TemporalMaxPooling_updateOutput,
    p_THDoubleNN_TemporalMaxPooling_updateGradInput,
    p_THDoubleNN_TemporalSubSampling_updateOutput,
    p_THDoubleNN_TemporalSubSampling_updateGradInput,
    p_THDoubleNN_TemporalSubSampling_accGradParameters,
    p_THDoubleNN_TemporalRowConvolution_updateOutput,
    p_THDoubleNN_TemporalRowConvolution_updateGradInput,
    p_THDoubleNN_TemporalRowConvolution_accGradParameters,
    p_THDoubleNN_TemporalUpSamplingNearest_updateOutput,
    p_THDoubleNN_TemporalUpSamplingNearest_updateGradInput,
    p_THDoubleNN_TemporalUpSamplingLinear_updateOutput,
    p_THDoubleNN_TemporalUpSamplingLinear_updateGradInput,
    p_THDoubleNN_BatchNormalization_updateOutput,
    p_THDoubleNN_BatchNormalization_backward,
    p_THDoubleNN_SpatialConvolutionMap_updateOutput,
    p_THDoubleNN_SpatialConvolutionMap_updateGradInput,
    p_THDoubleNN_SpatialConvolutionMap_accGradParameters,
    p_THDoubleNN_SpatialConvolutionMM_updateOutput,
    p_THDoubleNN_SpatialConvolutionMM_updateGradInput,
    p_THDoubleNN_SpatialConvolutionMM_accGradParameters,
    p_THDoubleNN_SpatialConvolutionLocal_updateOutput,
    p_THDoubleNN_SpatialConvolutionLocal_updateGradInput,
    p_THDoubleNN_SpatialConvolutionLocal_accGradParameters,
    p_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput,
    p_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput,
    p_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput,
    p_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput,
    p_THDoubleNN_SpatialAveragePooling_updateOutput,
    p_THDoubleNN_SpatialAveragePooling_updateGradInput,
    p_THDoubleNN_SpatialFractionalMaxPooling_updateOutput,
    p_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput,
    p_THDoubleNN_SpatialFullConvolution_updateOutput,
    p_THDoubleNN_SpatialFullConvolution_updateGradInput,
    p_THDoubleNN_SpatialFullConvolution_accGradParameters,
    p_THDoubleNN_SpatialFullConvolutionMap_updateOutput,
    p_THDoubleNN_SpatialFullConvolutionMap_updateGradInput,
    p_THDoubleNN_SpatialFullConvolutionMap_accGradParameters,
    p_THDoubleNN_SpatialDilatedConvolution_updateOutput,
    p_THDoubleNN_SpatialDilatedConvolution_updateGradInput,
    p_THDoubleNN_SpatialDilatedConvolution_accGradParameters,
    p_THDoubleNN_SpatialFullDilatedConvolution_updateOutput,
    p_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput,
    p_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters,
    p_THDoubleNN_SpatialMaxPooling_updateOutput,
    p_THDoubleNN_SpatialMaxPooling_updateGradInput,
    p_THDoubleNN_SpatialDilatedMaxPooling_updateOutput,
    p_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput,
    p_THDoubleNN_SpatialMaxUnpooling_updateOutput,
    p_THDoubleNN_SpatialMaxUnpooling_updateGradInput,
    p_THDoubleNN_SpatialSubSampling_updateOutput,
    p_THDoubleNN_SpatialSubSampling_updateGradInput,
    p_THDoubleNN_SpatialSubSampling_accGradParameters,
    p_THDoubleNN_SpatialUpSamplingNearest_updateOutput,
    p_THDoubleNN_SpatialUpSamplingNearest_updateGradInput,
    p_THDoubleNN_SpatialUpSamplingBilinear_updateOutput,
    p_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput,
    p_THDoubleNN_SpatialGridSamplerBilinear_updateOutput,
    p_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput,
    p_THDoubleNN_unfolded_acc,
    p_THDoubleNN_unfolded_copy,
    p_THDoubleNN_VolumetricAveragePooling_updateOutput,
    p_THDoubleNN_VolumetricAveragePooling_updateGradInput,
    p_THDoubleNN_VolumetricConvolution_updateOutput,
    p_THDoubleNN_VolumetricConvolution_updateGradInput,
    p_THDoubleNN_VolumetricConvolution_accGradParameters,
    p_THDoubleNN_VolumetricConvolutionMM_updateOutput,
    p_THDoubleNN_VolumetricConvolutionMM_updateGradInput,
    p_THDoubleNN_VolumetricConvolutionMM_accGradParameters,
    p_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput,
    p_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput,
    p_THDoubleNN_VolumetricFullConvolution_updateOutput,
    p_THDoubleNN_VolumetricFullConvolution_updateGradInput,
    p_THDoubleNN_VolumetricFullConvolution_accGradParameters,
    p_THDoubleNN_VolumetricDilatedConvolution_updateOutput,
    p_THDoubleNN_VolumetricDilatedConvolution_updateGradInput,
    p_THDoubleNN_VolumetricDilatedConvolution_accGradParameters,
    p_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput,
    p_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput,
    p_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters,
    p_THDoubleNN_VolumetricMaxPooling_updateOutput,
    p_THDoubleNN_VolumetricMaxPooling_updateGradInput,
    p_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput,
    p_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput,
    p_THDoubleNN_VolumetricMaxUnpooling_updateOutput,
    p_THDoubleNN_VolumetricMaxUnpooling_updateGradInput,
    p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput,
    p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput,
    p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput,
    p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput,
    p_THDoubleNN_SpatialReflectionPadding_updateOutput,
    p_THDoubleNN_SpatialReflectionPadding_updateGradInput,
    p_THDoubleNN_SpatialReplicationPadding_updateOutput,
    p_THDoubleNN_SpatialReplicationPadding_updateGradInput,
    p_THDoubleNN_FeatureLPPooling_updateOutput,
    p_THDoubleNN_FeatureLPPooling_updateGradInput,
    p_THDoubleNN_VolumetricReplicationPadding_updateOutput,
    p_THDoubleNN_VolumetricReplicationPadding_updateGradInput,
    p_THDoubleNN_VolumetricUpSamplingNearest_updateOutput,
    p_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput,
    p_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput,
    p_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput,
    p_THDoubleNN_TemporalReflectionPadding_updateOutput,
    p_THDoubleNN_TemporalReflectionPadding_updateGradInput,
    p_THDoubleNN_TemporalReplicationPadding_updateOutput,
    p_THDoubleNN_TemporalReplicationPadding_updateGradInput) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THDoubleNN_Abs_updateOutput : state input output -> void
foreign import ccall "THNN.h THDoubleNN_Abs_updateOutput"
  c_THDoubleNN_Abs_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Abs_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THDoubleNN_Abs_updateGradInput"
  c_THDoubleNN_Abs_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_AbsCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_AbsCriterion_updateOutput"
  c_THDoubleNN_AbsCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_AbsCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_AbsCriterion_updateGradInput"
  c_THDoubleNN_AbsCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_BCECriterion_updateOutput : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h THDoubleNN_BCECriterion_updateOutput"
  c_THDoubleNN_BCECriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_BCECriterion_updateGradInput : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h THDoubleNN_BCECriterion_updateGradInput"
  c_THDoubleNN_BCECriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_ClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THDoubleNN_ClassNLLCriterion_updateOutput"
  c_THDoubleNN_ClassNLLCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THDoubleNN_ClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THDoubleNN_ClassNLLCriterion_updateGradInput"
  c_THDoubleNN_ClassNLLCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THDoubleNN_SpatialClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THDoubleNN_SpatialClassNLLCriterion_updateOutput"
  c_THDoubleNN_SpatialClassNLLCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THDoubleNN_SpatialClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THDoubleNN_SpatialClassNLLCriterion_updateGradInput"
  c_THDoubleNN_SpatialClassNLLCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THDoubleNN_ELU_updateOutput : state input output alpha inplace -> void
foreign import ccall "THNN.h THDoubleNN_ELU_updateOutput"
  c_THDoubleNN_ELU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_ELU_updateGradInput : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h THDoubleNN_ELU_updateGradInput"
  c_THDoubleNN_ELU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_DistKLDivCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_DistKLDivCriterion_updateOutput"
  c_THDoubleNN_DistKLDivCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_DistKLDivCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_DistKLDivCriterion_updateGradInput"
  c_THDoubleNN_DistKLDivCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_GatedLinear_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THDoubleNN_GatedLinear_updateOutput"
  c_THDoubleNN_GatedLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_GatedLinear_updateGradInput : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THDoubleNN_GatedLinear_updateGradInput"
  c_THDoubleNN_GatedLinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_HardShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THDoubleNN_HardShrink_updateOutput"
  c_THDoubleNN_HardShrink_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_HardShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THDoubleNN_HardShrink_updateGradInput"
  c_THDoubleNN_HardShrink_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_HardTanh_updateOutput : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THDoubleNN_HardTanh_updateOutput"
  c_THDoubleNN_HardTanh_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_HardTanh_updateGradInput : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THDoubleNN_HardTanh_updateGradInput"
  c_THDoubleNN_HardTanh_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_L1Cost_updateOutput : state input output -> void
foreign import ccall "THNN.h THDoubleNN_L1Cost_updateOutput"
  c_THDoubleNN_L1Cost_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_L1Cost_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THDoubleNN_L1Cost_updateGradInput"
  c_THDoubleNN_L1Cost_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LeakyReLU_updateOutput : state input output negval inplace -> void
foreign import ccall "THNN.h THDoubleNN_LeakyReLU_updateOutput"
  c_THDoubleNN_LeakyReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_LeakyReLU_updateGradInput : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THDoubleNN_LeakyReLU_updateGradInput"
  c_THDoubleNN_LeakyReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_GRUFused_updateOutput : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THDoubleNN_GRUFused_updateOutput"
  c_THDoubleNN_GRUFused_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_GRUFused_updateGradInput : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THDoubleNN_GRUFused_updateGradInput"
  c_THDoubleNN_GRUFused_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LSTMFused_updateOutput : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THDoubleNN_LSTMFused_updateOutput"
  c_THDoubleNN_LSTMFused_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LSTMFused_updateGradInput : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THDoubleNN_LSTMFused_updateGradInput"
  c_THDoubleNN_LSTMFused_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LogSigmoid_updateOutput : state input output buffer -> void
foreign import ccall "THNN.h THDoubleNN_LogSigmoid_updateOutput"
  c_THDoubleNN_LogSigmoid_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LogSigmoid_updateGradInput : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THDoubleNN_LogSigmoid_updateGradInput"
  c_THDoubleNN_LogSigmoid_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_LogSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THDoubleNN_LogSoftMax_updateOutput"
  c_THDoubleNN_LogSoftMax_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_LogSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THDoubleNN_LogSoftMax_updateGradInput"
  c_THDoubleNN_LogSoftMax_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_LookupTable_accGradParameters : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h THDoubleNN_LookupTable_accGradParameters"
  c_THDoubleNN_LookupTable_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIntegerTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CBool -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_LookupTable_renorm : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h THDoubleNN_LookupTable_renorm"
  c_THDoubleNN_LookupTable_renorm :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_MarginCriterion_updateOutput : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THDoubleNN_MarginCriterion_updateOutput"
  c_THDoubleNN_MarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ()

-- |c_THDoubleNN_MarginCriterion_updateGradInput : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THDoubleNN_MarginCriterion_updateGradInput"
  c_THDoubleNN_MarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ()

-- |c_THDoubleNN_SoftMarginCriterion_updateOutput : state input target output sizeAverage -> void
foreign import ccall "THNN.h THDoubleNN_SoftMarginCriterion_updateOutput"
  c_THDoubleNN_SoftMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THDoubleNN_SoftMarginCriterion_updateGradInput : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h THDoubleNN_SoftMarginCriterion_updateGradInput"
  c_THDoubleNN_SoftMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THDoubleNN_MSECriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_MSECriterion_updateOutput"
  c_THDoubleNN_MSECriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_MSECriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_MSECriterion_updateGradInput"
  c_THDoubleNN_MSECriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_MultiLabelMarginCriterion_updateOutput : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h THDoubleNN_MultiLabelMarginCriterion_updateOutput"
  c_THDoubleNN_MultiLabelMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THDoubleNN_MultiLabelMarginCriterion_updateGradInput : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h THDoubleNN_MultiLabelMarginCriterion_updateGradInput"
  c_THDoubleNN_MultiLabelMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THDoubleNN_MultiMarginCriterion_updateOutput : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h THDoubleNN_MultiMarginCriterion_updateOutput"
  c_THDoubleNN_MultiMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_MultiMarginCriterion_updateGradInput : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h THDoubleNN_MultiMarginCriterion_updateGradInput"
  c_THDoubleNN_MultiMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_PReLU_updateOutput : state input output weight -> void
foreign import ccall "THNN.h THDoubleNN_PReLU_updateOutput"
  c_THDoubleNN_PReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_PReLU_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THDoubleNN_PReLU_updateGradInput"
  c_THDoubleNN_PReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_PReLU_accGradParameters : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THDoubleNN_PReLU_accGradParameters"
  c_THDoubleNN_PReLU_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_Linear_updateOutput : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THDoubleNN_Linear_updateOutput"
  c_THDoubleNN_Linear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Linear_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THDoubleNN_Linear_updateGradInput"
  c_THDoubleNN_Linear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Linear_accGradParameters : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THDoubleNN_Linear_accGradParameters"
  c_THDoubleNN_Linear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_RReLU_updateOutput : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THDoubleNN_RReLU_updateOutput"
  c_THDoubleNN_RReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ()

-- |c_THDoubleNN_RReLU_updateGradInput : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THDoubleNN_RReLU_updateGradInput"
  c_THDoubleNN_RReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_Sigmoid_updateOutput : state input output -> void
foreign import ccall "THNN.h THDoubleNN_Sigmoid_updateOutput"
  c_THDoubleNN_Sigmoid_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Sigmoid_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THDoubleNN_Sigmoid_updateGradInput"
  c_THDoubleNN_Sigmoid_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SmoothL1Criterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_SmoothL1Criterion_updateOutput"
  c_THDoubleNN_SmoothL1Criterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_SmoothL1Criterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THDoubleNN_SmoothL1Criterion_updateGradInput"
  c_THDoubleNN_SmoothL1Criterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_SoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THDoubleNN_SoftMax_updateOutput"
  c_THDoubleNN_SoftMax_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_SoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THDoubleNN_SoftMax_updateGradInput"
  c_THDoubleNN_SoftMax_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_SoftPlus_updateOutput : state input output beta threshold -> void
foreign import ccall "THNN.h THDoubleNN_SoftPlus_updateOutput"
  c_THDoubleNN_SoftPlus_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SoftPlus_updateGradInput : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THDoubleNN_SoftPlus_updateGradInput"
  c_THDoubleNN_SoftPlus_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SoftShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THDoubleNN_SoftShrink_updateOutput"
  c_THDoubleNN_SoftShrink_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_SoftShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THDoubleNN_SoftShrink_updateGradInput"
  c_THDoubleNN_SoftShrink_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_IndexLinear_updateOutput : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h THDoubleNN_IndexLinear_updateOutput"
  c_THDoubleNN_IndexLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_IndexLinear_accGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h THDoubleNN_IndexLinear_accGradParameters"
  c_THDoubleNN_IndexLinear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_IndexLinear_accUpdateGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h THDoubleNN_IndexLinear_accUpdateGradParameters"
  c_THDoubleNN_IndexLinear_accUpdateGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_IndexLinear_updateParameters : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h THDoubleNN_IndexLinear_updateParameters"
  c_THDoubleNN_IndexLinear_updateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SparseLinear_updateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_updateOutput"
  c_THDoubleNN_SparseLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SparseLinear_accGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_accGradParameters"
  c_THDoubleNN_SparseLinear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SparseLinear_zeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_zeroGradParameters"
  c_THDoubleNN_SparseLinear_zeroGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SparseLinear_updateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_updateParameters"
  c_THDoubleNN_SparseLinear_updateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_SparseLinear_legacyUpdateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_legacyUpdateOutput"
  c_THDoubleNN_SparseLinear_legacyUpdateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SparseLinear_legacyAccGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_legacyAccGradParameters"
  c_THDoubleNN_SparseLinear_legacyAccGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SparseLinear_legacyZeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_legacyZeroGradParameters"
  c_THDoubleNN_SparseLinear_legacyZeroGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SparseLinear_legacyUpdateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THDoubleNN_SparseLinear_legacyUpdateParameters"
  c_THDoubleNN_SparseLinear_legacyUpdateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_Sqrt_updateOutput : state input output eps -> void
foreign import ccall "THNN.h THDoubleNN_Sqrt_updateOutput"
  c_THDoubleNN_Sqrt_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleNN_Sqrt_updateGradInput : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THDoubleNN_Sqrt_updateGradInput"
  c_THDoubleNN_Sqrt_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Square_updateOutput : state input output -> void
foreign import ccall "THNN.h THDoubleNN_Square_updateOutput"
  c_THDoubleNN_Square_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Square_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THDoubleNN_Square_updateGradInput"
  c_THDoubleNN_Square_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Tanh_updateOutput : state input output -> void
foreign import ccall "THNN.h THDoubleNN_Tanh_updateOutput"
  c_THDoubleNN_Tanh_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Tanh_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THDoubleNN_Tanh_updateGradInput"
  c_THDoubleNN_Tanh_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_Threshold_updateOutput : state input output threshold val inplace -> void
foreign import ccall "THNN.h THDoubleNN_Threshold_updateOutput"
  c_THDoubleNN_Threshold_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_Threshold_updateGradInput : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THDoubleNN_Threshold_updateGradInput"
  c_THDoubleNN_Threshold_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THDoubleNN_TemporalConvolution_updateOutput : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THDoubleNN_TemporalConvolution_updateOutput"
  c_THDoubleNN_TemporalConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalConvolution_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THDoubleNN_TemporalConvolution_updateGradInput"
  c_THDoubleNN_TemporalConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalConvolution_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THDoubleNN_TemporalConvolution_accGradParameters"
  c_THDoubleNN_TemporalConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_TemporalMaxPooling_updateOutput : state input output indices kW dW -> void
foreign import ccall "THNN.h THDoubleNN_TemporalMaxPooling_updateOutput"
  c_THDoubleNN_TemporalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalMaxPooling_updateGradInput : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h THDoubleNN_TemporalMaxPooling_updateGradInput"
  c_THDoubleNN_TemporalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalSubSampling_updateOutput : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THDoubleNN_TemporalSubSampling_updateOutput"
  c_THDoubleNN_TemporalSubSampling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalSubSampling_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THDoubleNN_TemporalSubSampling_updateGradInput"
  c_THDoubleNN_TemporalSubSampling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THDoubleNN_TemporalSubSampling_accGradParameters"
  c_THDoubleNN_TemporalSubSampling_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_TemporalRowConvolution_updateOutput : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THDoubleNN_TemporalRowConvolution_updateOutput"
  c_THDoubleNN_TemporalRowConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_TemporalRowConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THDoubleNN_TemporalRowConvolution_updateGradInput"
  c_THDoubleNN_TemporalRowConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_TemporalRowConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THDoubleNN_TemporalRowConvolution_accGradParameters"
  c_THDoubleNN_TemporalRowConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- |c_THDoubleNN_TemporalUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_TemporalUpSamplingNearest_updateOutput"
  c_THDoubleNN_TemporalUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_TemporalUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_TemporalUpSamplingNearest_updateGradInput"
  c_THDoubleNN_TemporalUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_TemporalUpSamplingLinear_updateOutput : state input output outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_TemporalUpSamplingLinear_updateOutput"
  c_THDoubleNN_TemporalUpSamplingLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_TemporalUpSamplingLinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_TemporalUpSamplingLinear_updateGradInput"
  c_THDoubleNN_TemporalUpSamplingLinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_BatchNormalization_updateOutput : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THDoubleNN_BatchNormalization_updateOutput"
  c_THDoubleNN_BatchNormalization_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_BatchNormalization_backward : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THDoubleNN_BatchNormalization_backward"
  c_THDoubleNN_BatchNormalization_backward :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMap_updateOutput"
  c_THDoubleNN_SpatialConvolutionMap_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMap_updateGradInput"
  c_THDoubleNN_SpatialConvolutionMap_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMap_accGradParameters"
  c_THDoubleNN_SpatialConvolutionMap_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMM_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMM_updateOutput"
  c_THDoubleNN_SpatialConvolutionMM_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMM_updateGradInput"
  c_THDoubleNN_SpatialConvolutionMM_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionMM_accGradParameters"
  c_THDoubleNN_SpatialConvolutionMM_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialConvolutionLocal_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionLocal_updateOutput"
  c_THDoubleNN_SpatialConvolutionLocal_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THDoubleNN_SpatialConvolutionLocal_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionLocal_updateGradInput"
  c_THDoubleNN_SpatialConvolutionLocal_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THDoubleNN_SpatialConvolutionLocal_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialConvolutionLocal_accGradParameters"
  c_THDoubleNN_SpatialConvolutionLocal_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput"
  c_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput"
  c_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> IO ()

-- |c_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput : state input output osizeW osizeH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput"
  c_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput"
  c_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SpatialAveragePooling_updateOutput : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAveragePooling_updateOutput"
  c_THDoubleNN_SpatialAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_SpatialAveragePooling_updateGradInput : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THDoubleNN_SpatialAveragePooling_updateGradInput"
  c_THDoubleNN_SpatialAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_SpatialFractionalMaxPooling_updateOutput : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFractionalMaxPooling_updateOutput"
  c_THDoubleNN_SpatialFractionalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFractionalMaxPooling_updateGradInput"
  c_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> IO ()

-- |c_THDoubleNN_SpatialFullConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolution_updateOutput"
  c_THDoubleNN_SpatialFullConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolution_updateGradInput"
  c_THDoubleNN_SpatialFullConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolution_accGradParameters"
  c_THDoubleNN_SpatialFullConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialFullConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolutionMap_updateOutput"
  c_THDoubleNN_SpatialFullConvolutionMap_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolutionMap_updateGradInput"
  c_THDoubleNN_SpatialFullConvolutionMap_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullConvolutionMap_accGradParameters"
  c_THDoubleNN_SpatialFullConvolutionMap_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialDilatedConvolution_updateOutput"
  c_THDoubleNN_SpatialDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialDilatedConvolution_updateGradInput"
  c_THDoubleNN_SpatialDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialDilatedConvolution_accGradParameters"
  c_THDoubleNN_SpatialDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialFullDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullDilatedConvolution_updateOutput"
  c_THDoubleNN_SpatialFullDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullDilatedConvolution_updateGradInput"
  c_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialFullDilatedConvolution_accGradParameters"
  c_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialMaxPooling_updateOutput"
  c_THDoubleNN_SpatialMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_SpatialMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialMaxPooling_updateGradInput"
  c_THDoubleNN_SpatialMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_SpatialDilatedMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialDilatedMaxPooling_updateOutput"
  c_THDoubleNN_SpatialDilatedMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialDilatedMaxPooling_updateGradInput"
  c_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_SpatialMaxUnpooling_updateOutput : state input output indices owidth oheight -> void
foreign import ccall "THNN.h THDoubleNN_SpatialMaxUnpooling_updateOutput"
  c_THDoubleNN_SpatialMaxUnpooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialMaxUnpooling_updateGradInput : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h THDoubleNN_SpatialMaxUnpooling_updateGradInput"
  c_THDoubleNN_SpatialMaxUnpooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialSubSampling_updateOutput : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialSubSampling_updateOutput"
  c_THDoubleNN_SpatialSubSampling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialSubSampling_updateGradInput : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THDoubleNN_SpatialSubSampling_updateGradInput"
  c_THDoubleNN_SpatialSubSampling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THDoubleNN_SpatialSubSampling_accGradParameters"
  c_THDoubleNN_SpatialSubSampling_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_SpatialUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_SpatialUpSamplingNearest_updateOutput"
  c_THDoubleNN_SpatialUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_SpatialUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_SpatialUpSamplingNearest_updateGradInput"
  c_THDoubleNN_SpatialUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_SpatialUpSamplingBilinear_updateOutput : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_SpatialUpSamplingBilinear_updateOutput"
  c_THDoubleNN_SpatialUpSamplingBilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_SpatialUpSamplingBilinear_updateGradInput"
  c_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialGridSamplerBilinear_updateOutput : state input grid output padding_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialGridSamplerBilinear_updateOutput"
  c_THDoubleNN_SpatialGridSamplerBilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THDoubleNN_SpatialGridSamplerBilinear_updateGradInput"
  c_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_unfolded_acc : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THDoubleNN_unfolded_acc"
  c_THDoubleNN_unfolded_acc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_unfolded_copy : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THDoubleNN_unfolded_copy"
  c_THDoubleNN_unfolded_copy :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricAveragePooling_updateOutput : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAveragePooling_updateOutput"
  c_THDoubleNN_VolumetricAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricAveragePooling_updateGradInput : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAveragePooling_updateGradInput"
  c_THDoubleNN_VolumetricAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolution_updateOutput"
  c_THDoubleNN_VolumetricConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricConvolution_updateGradInput : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolution_updateGradInput"
  c_THDoubleNN_VolumetricConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolution_accGradParameters"
  c_THDoubleNN_VolumetricConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_VolumetricConvolutionMM_updateOutput : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolutionMM_updateOutput"
  c_THDoubleNN_VolumetricConvolutionMM_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolutionMM_updateGradInput"
  c_THDoubleNN_VolumetricConvolutionMM_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricConvolutionMM_accGradParameters"
  c_THDoubleNN_VolumetricConvolutionMM_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFractionalMaxPooling_updateOutput"
  c_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput"
  c_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> IO ()

-- |c_THDoubleNN_VolumetricFullConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullConvolution_updateOutput"
  c_THDoubleNN_VolumetricFullConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricFullConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullConvolution_updateGradInput"
  c_THDoubleNN_VolumetricFullConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullConvolution_accGradParameters"
  c_THDoubleNN_VolumetricFullConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_VolumetricDilatedConvolution_updateOutput : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricDilatedConvolution_updateOutput"
  c_THDoubleNN_VolumetricDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricDilatedConvolution_updateGradInput"
  c_THDoubleNN_VolumetricDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricDilatedConvolution_accGradParameters"
  c_THDoubleNN_VolumetricDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullDilatedConvolution_updateOutput"
  c_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput"
  c_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters"
  c_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THDoubleNN_VolumetricMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricMaxPooling_updateOutput"
  c_THDoubleNN_VolumetricMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricMaxPooling_updateGradInput"
  c_THDoubleNN_VolumetricMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricDilatedMaxPooling_updateOutput"
  c_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput"
  c_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricMaxUnpooling_updateOutput : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricMaxUnpooling_updateOutput"
  c_THDoubleNN_VolumetricMaxUnpooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricMaxUnpooling_updateGradInput : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricMaxUnpooling_updateGradInput"
  c_THDoubleNN_VolumetricMaxUnpooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput"
  c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput"
  c_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput"
  c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput"
  c_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> IO ()

-- |c_THDoubleNN_SpatialReflectionPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THDoubleNN_SpatialReflectionPadding_updateOutput"
  c_THDoubleNN_SpatialReflectionPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THDoubleNN_SpatialReflectionPadding_updateGradInput"
  c_THDoubleNN_SpatialReflectionPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialReplicationPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THDoubleNN_SpatialReplicationPadding_updateOutput"
  c_THDoubleNN_SpatialReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_SpatialReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THDoubleNN_SpatialReplicationPadding_updateGradInput"
  c_THDoubleNN_SpatialReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_FeatureLPPooling_updateOutput : state input output power width stride batchMode -> void
foreign import ccall "THNN.h THDoubleNN_FeatureLPPooling_updateOutput"
  c_THDoubleNN_FeatureLPPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_FeatureLPPooling_updateGradInput : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THDoubleNN_FeatureLPPooling_updateGradInput"
  c_THDoubleNN_FeatureLPPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THDoubleNN_VolumetricReplicationPadding_updateOutput : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricReplicationPadding_updateOutput"
  c_THDoubleNN_VolumetricReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricReplicationPadding_updateGradInput : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricReplicationPadding_updateGradInput"
  c_THDoubleNN_VolumetricReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricUpSamplingNearest_updateOutput"
  c_THDoubleNN_VolumetricUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricUpSamplingNearest_updateGradInput"
  c_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput"
  c_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput"
  c_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalReflectionPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THDoubleNN_TemporalReflectionPadding_updateOutput"
  c_THDoubleNN_TemporalReflectionPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THDoubleNN_TemporalReflectionPadding_updateGradInput"
  c_THDoubleNN_TemporalReflectionPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalReplicationPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THDoubleNN_TemporalReplicationPadding_updateOutput"
  c_THDoubleNN_TemporalReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleNN_TemporalReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THDoubleNN_TemporalReplicationPadding_updateGradInput"
  c_THDoubleNN_TemporalReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |p_THDoubleNN_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THDoubleNN_Abs_updateOutput"
  p_THDoubleNN_Abs_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THDoubleNN_Abs_updateGradInput"
  p_THDoubleNN_Abs_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_AbsCriterion_updateOutput"
  p_THDoubleNN_AbsCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_AbsCriterion_updateGradInput"
  p_THDoubleNN_AbsCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h &THDoubleNN_BCECriterion_updateOutput"
  p_THDoubleNN_BCECriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_BCECriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h &THDoubleNN_BCECriterion_updateGradInput"
  p_THDoubleNN_BCECriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_ClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THDoubleNN_ClassNLLCriterion_updateOutput"
  p_THDoubleNN_ClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THDoubleNN_ClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THDoubleNN_ClassNLLCriterion_updateGradInput"
  p_THDoubleNN_ClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THDoubleNN_SpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialClassNLLCriterion_updateOutput"
  p_THDoubleNN_SpatialClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THDoubleNN_SpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialClassNLLCriterion_updateGradInput"
  p_THDoubleNN_SpatialClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THDoubleNN_ELU_updateOutput : Pointer to function : state input output alpha inplace -> void
foreign import ccall "THNN.h &THDoubleNN_ELU_updateOutput"
  p_THDoubleNN_ELU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h &THDoubleNN_ELU_updateGradInput"
  p_THDoubleNN_ELU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_DistKLDivCriterion_updateOutput"
  p_THDoubleNN_DistKLDivCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_DistKLDivCriterion_updateGradInput"
  p_THDoubleNN_DistKLDivCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THDoubleNN_GatedLinear_updateOutput"
  p_THDoubleNN_GatedLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THDoubleNN_GatedLinear_updateGradInput"
  p_THDoubleNN_GatedLinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_HardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THDoubleNN_HardShrink_updateOutput"
  p_THDoubleNN_HardShrink_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_HardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THDoubleNN_HardShrink_updateGradInput"
  p_THDoubleNN_HardShrink_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THDoubleNN_HardTanh_updateOutput"
  p_THDoubleNN_HardTanh_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THDoubleNN_HardTanh_updateGradInput"
  p_THDoubleNN_HardTanh_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THDoubleNN_L1Cost_updateOutput"
  p_THDoubleNN_L1Cost_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THDoubleNN_L1Cost_updateGradInput"
  p_THDoubleNN_L1Cost_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THDoubleNN_LeakyReLU_updateOutput"
  p_THDoubleNN_LeakyReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THDoubleNN_LeakyReLU_updateGradInput"
  p_THDoubleNN_LeakyReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THDoubleNN_GRUFused_updateOutput"
  p_THDoubleNN_GRUFused_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THDoubleNN_GRUFused_updateGradInput"
  p_THDoubleNN_GRUFused_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THDoubleNN_LSTMFused_updateOutput"
  p_THDoubleNN_LSTMFused_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THDoubleNN_LSTMFused_updateGradInput"
  p_THDoubleNN_LSTMFused_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THDoubleNN_LogSigmoid_updateOutput"
  p_THDoubleNN_LogSigmoid_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THDoubleNN_LogSigmoid_updateGradInput"
  p_THDoubleNN_LogSigmoid_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THDoubleNN_LogSoftMax_updateOutput"
  p_THDoubleNN_LogSoftMax_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THDoubleNN_LogSoftMax_updateGradInput"
  p_THDoubleNN_LogSoftMax_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_LookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h &THDoubleNN_LookupTable_accGradParameters"
  p_THDoubleNN_LookupTable_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIntegerTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CBool -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_LookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h &THDoubleNN_LookupTable_renorm"
  p_THDoubleNN_LookupTable_renorm :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THDoubleNN_MarginCriterion_updateOutput"
  p_THDoubleNN_MarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ())

-- |p_THDoubleNN_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THDoubleNN_MarginCriterion_updateGradInput"
  p_THDoubleNN_MarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ())

-- |p_THDoubleNN_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage -> void
foreign import ccall "THNN.h &THDoubleNN_SoftMarginCriterion_updateOutput"
  p_THDoubleNN_SoftMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THDoubleNN_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h &THDoubleNN_SoftMarginCriterion_updateGradInput"
  p_THDoubleNN_SoftMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THDoubleNN_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_MSECriterion_updateOutput"
  p_THDoubleNN_MSECriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_MSECriterion_updateGradInput"
  p_THDoubleNN_MSECriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_MultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h &THDoubleNN_MultiLabelMarginCriterion_updateOutput"
  p_THDoubleNN_MultiLabelMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THDoubleNN_MultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h &THDoubleNN_MultiLabelMarginCriterion_updateGradInput"
  p_THDoubleNN_MultiLabelMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THDoubleNN_MultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THDoubleNN_MultiMarginCriterion_updateOutput"
  p_THDoubleNN_MultiMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_MultiMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THDoubleNN_MultiMarginCriterion_updateGradInput"
  p_THDoubleNN_MultiMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THDoubleNN_PReLU_updateOutput"
  p_THDoubleNN_PReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THDoubleNN_PReLU_updateGradInput"
  p_THDoubleNN_PReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THDoubleNN_PReLU_accGradParameters"
  p_THDoubleNN_PReLU_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_Linear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THDoubleNN_Linear_updateOutput"
  p_THDoubleNN_Linear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Linear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THDoubleNN_Linear_updateGradInput"
  p_THDoubleNN_Linear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Linear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THDoubleNN_Linear_accGradParameters"
  p_THDoubleNN_Linear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THDoubleNN_RReLU_updateOutput"
  p_THDoubleNN_RReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ())

-- |p_THDoubleNN_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THDoubleNN_RReLU_updateGradInput"
  p_THDoubleNN_RReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THDoubleNN_Sigmoid_updateOutput"
  p_THDoubleNN_Sigmoid_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THDoubleNN_Sigmoid_updateGradInput"
  p_THDoubleNN_Sigmoid_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_SmoothL1Criterion_updateOutput"
  p_THDoubleNN_SmoothL1Criterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THDoubleNN_SmoothL1Criterion_updateGradInput"
  p_THDoubleNN_SmoothL1Criterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THDoubleNN_SoftMax_updateOutput"
  p_THDoubleNN_SoftMax_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THDoubleNN_SoftMax_updateGradInput"
  p_THDoubleNN_SoftMax_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THDoubleNN_SoftPlus_updateOutput"
  p_THDoubleNN_SoftPlus_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THDoubleNN_SoftPlus_updateGradInput"
  p_THDoubleNN_SoftPlus_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THDoubleNN_SoftShrink_updateOutput"
  p_THDoubleNN_SoftShrink_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THDoubleNN_SoftShrink_updateGradInput"
  p_THDoubleNN_SoftShrink_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_IndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h &THDoubleNN_IndexLinear_updateOutput"
  p_THDoubleNN_IndexLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_IndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h &THDoubleNN_IndexLinear_accGradParameters"
  p_THDoubleNN_IndexLinear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_IndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THDoubleNN_IndexLinear_accUpdateGradParameters"
  p_THDoubleNN_IndexLinear_accUpdateGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_IndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h &THDoubleNN_IndexLinear_updateParameters"
  p_THDoubleNN_IndexLinear_updateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_updateOutput"
  p_THDoubleNN_SparseLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_accGradParameters"
  p_THDoubleNN_SparseLinear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_zeroGradParameters"
  p_THDoubleNN_SparseLinear_zeroGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_updateParameters"
  p_THDoubleNN_SparseLinear_updateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_legacyUpdateOutput"
  p_THDoubleNN_SparseLinear_legacyUpdateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_legacyAccGradParameters"
  p_THDoubleNN_SparseLinear_legacyAccGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_legacyZeroGradParameters"
  p_THDoubleNN_SparseLinear_legacyZeroGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THDoubleNN_SparseLinear_legacyUpdateParameters"
  p_THDoubleNN_SparseLinear_legacyUpdateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THDoubleNN_Sqrt_updateOutput"
  p_THDoubleNN_Sqrt_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleNN_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THDoubleNN_Sqrt_updateGradInput"
  p_THDoubleNN_Sqrt_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THDoubleNN_Square_updateOutput"
  p_THDoubleNN_Square_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THDoubleNN_Square_updateGradInput"
  p_THDoubleNN_Square_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THDoubleNN_Tanh_updateOutput"
  p_THDoubleNN_Tanh_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THDoubleNN_Tanh_updateGradInput"
  p_THDoubleNN_Tanh_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THDoubleNN_Threshold_updateOutput"
  p_THDoubleNN_Threshold_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THDoubleNN_Threshold_updateGradInput"
  p_THDoubleNN_Threshold_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THDoubleNN_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalConvolution_updateOutput"
  p_THDoubleNN_TemporalConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalConvolution_updateGradInput"
  p_THDoubleNN_TemporalConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalConvolution_accGradParameters"
  p_THDoubleNN_TemporalConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_TemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalMaxPooling_updateOutput"
  p_THDoubleNN_TemporalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalMaxPooling_updateGradInput"
  p_THDoubleNN_TemporalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalSubSampling_updateOutput"
  p_THDoubleNN_TemporalSubSampling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalSubSampling_updateGradInput"
  p_THDoubleNN_TemporalSubSampling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalSubSampling_accGradParameters"
  p_THDoubleNN_TemporalSubSampling_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalRowConvolution_updateOutput"
  p_THDoubleNN_TemporalRowConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalRowConvolution_updateGradInput"
  p_THDoubleNN_TemporalRowConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalRowConvolution_accGradParameters"
  p_THDoubleNN_TemporalRowConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- |p_THDoubleNN_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalUpSamplingNearest_updateOutput"
  p_THDoubleNN_TemporalUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalUpSamplingNearest_updateGradInput"
  p_THDoubleNN_TemporalUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalUpSamplingLinear_updateOutput"
  p_THDoubleNN_TemporalUpSamplingLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalUpSamplingLinear_updateGradInput"
  p_THDoubleNN_TemporalUpSamplingLinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_BatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THDoubleNN_BatchNormalization_updateOutput"
  p_THDoubleNN_BatchNormalization_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_BatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THDoubleNN_BatchNormalization_backward"
  p_THDoubleNN_BatchNormalization_backward :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMap_updateOutput"
  p_THDoubleNN_SpatialConvolutionMap_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMap_updateGradInput"
  p_THDoubleNN_SpatialConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMap_accGradParameters"
  p_THDoubleNN_SpatialConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMM_updateOutput"
  p_THDoubleNN_SpatialConvolutionMM_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMM_updateGradInput"
  p_THDoubleNN_SpatialConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionMM_accGradParameters"
  p_THDoubleNN_SpatialConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionLocal_updateOutput"
  p_THDoubleNN_SpatialConvolutionLocal_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THDoubleNN_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionLocal_updateGradInput"
  p_THDoubleNN_SpatialConvolutionLocal_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THDoubleNN_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialConvolutionLocal_accGradParameters"
  p_THDoubleNN_SpatialConvolutionLocal_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput"
  p_THDoubleNN_SpatialAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput"
  p_THDoubleNN_SpatialAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> IO ())

-- |p_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput"
  p_THDoubleNN_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput"
  p_THDoubleNN_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAveragePooling_updateOutput"
  p_THDoubleNN_SpatialAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialAveragePooling_updateGradInput"
  p_THDoubleNN_SpatialAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_SpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFractionalMaxPooling_updateOutput"
  p_THDoubleNN_SpatialFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFractionalMaxPooling_updateGradInput"
  p_THDoubleNN_SpatialFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> IO ())

-- |p_THDoubleNN_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolution_updateOutput"
  p_THDoubleNN_SpatialFullConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolution_updateGradInput"
  p_THDoubleNN_SpatialFullConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolution_accGradParameters"
  p_THDoubleNN_SpatialFullConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolutionMap_updateOutput"
  p_THDoubleNN_SpatialFullConvolutionMap_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolutionMap_updateGradInput"
  p_THDoubleNN_SpatialFullConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullConvolutionMap_accGradParameters"
  p_THDoubleNN_SpatialFullConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialDilatedConvolution_updateOutput"
  p_THDoubleNN_SpatialDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialDilatedConvolution_updateGradInput"
  p_THDoubleNN_SpatialDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialDilatedConvolution_accGradParameters"
  p_THDoubleNN_SpatialDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullDilatedConvolution_updateOutput"
  p_THDoubleNN_SpatialFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullDilatedConvolution_updateGradInput"
  p_THDoubleNN_SpatialFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialFullDilatedConvolution_accGradParameters"
  p_THDoubleNN_SpatialFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialMaxPooling_updateOutput"
  p_THDoubleNN_SpatialMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_SpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialMaxPooling_updateGradInput"
  p_THDoubleNN_SpatialMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_SpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialDilatedMaxPooling_updateOutput"
  p_THDoubleNN_SpatialDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialDilatedMaxPooling_updateGradInput"
  p_THDoubleNN_SpatialDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_SpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialMaxUnpooling_updateOutput"
  p_THDoubleNN_SpatialMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialMaxUnpooling_updateGradInput"
  p_THDoubleNN_SpatialMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialSubSampling_updateOutput"
  p_THDoubleNN_SpatialSubSampling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialSubSampling_updateGradInput"
  p_THDoubleNN_SpatialSubSampling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialSubSampling_accGradParameters"
  p_THDoubleNN_SpatialSubSampling_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialUpSamplingNearest_updateOutput"
  p_THDoubleNN_SpatialUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialUpSamplingNearest_updateGradInput"
  p_THDoubleNN_SpatialUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialUpSamplingBilinear_updateOutput"
  p_THDoubleNN_SpatialUpSamplingBilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialUpSamplingBilinear_updateGradInput"
  p_THDoubleNN_SpatialUpSamplingBilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialGridSamplerBilinear_updateOutput"
  p_THDoubleNN_SpatialGridSamplerBilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialGridSamplerBilinear_updateGradInput"
  p_THDoubleNN_SpatialGridSamplerBilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_unfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THDoubleNN_unfolded_acc"
  p_THDoubleNN_unfolded_acc :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_unfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THDoubleNN_unfolded_copy"
  p_THDoubleNN_unfolded_copy :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAveragePooling_updateOutput"
  p_THDoubleNN_VolumetricAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAveragePooling_updateGradInput"
  p_THDoubleNN_VolumetricAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolution_updateOutput"
  p_THDoubleNN_VolumetricConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolution_updateGradInput"
  p_THDoubleNN_VolumetricConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolution_accGradParameters"
  p_THDoubleNN_VolumetricConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_VolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolutionMM_updateOutput"
  p_THDoubleNN_VolumetricConvolutionMM_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolutionMM_updateGradInput"
  p_THDoubleNN_VolumetricConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricConvolutionMM_accGradParameters"
  p_THDoubleNN_VolumetricConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFractionalMaxPooling_updateOutput"
  p_THDoubleNN_VolumetricFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput"
  p_THDoubleNN_VolumetricFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHDoubleIndexTensor) -> IO ())

-- |p_THDoubleNN_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullConvolution_updateOutput"
  p_THDoubleNN_VolumetricFullConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullConvolution_updateGradInput"
  p_THDoubleNN_VolumetricFullConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullConvolution_accGradParameters"
  p_THDoubleNN_VolumetricFullConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricDilatedConvolution_updateOutput"
  p_THDoubleNN_VolumetricDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricDilatedConvolution_updateGradInput"
  p_THDoubleNN_VolumetricDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricDilatedConvolution_accGradParameters"
  p_THDoubleNN_VolumetricDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullDilatedConvolution_updateOutput"
  p_THDoubleNN_VolumetricFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput"
  p_THDoubleNN_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters"
  p_THDoubleNN_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THDoubleNN_VolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricMaxPooling_updateOutput"
  p_THDoubleNN_VolumetricMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricMaxPooling_updateGradInput"
  p_THDoubleNN_VolumetricMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricDilatedMaxPooling_updateOutput"
  p_THDoubleNN_VolumetricDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput"
  p_THDoubleNN_VolumetricDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricMaxUnpooling_updateOutput"
  p_THDoubleNN_VolumetricMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricMaxUnpooling_updateGradInput"
  p_THDoubleNN_VolumetricMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput"
  p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput"
  p_THDoubleNN_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput"
  p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput"
  p_THDoubleNN_VolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleIndexTensor) -> IO ())

-- |p_THDoubleNN_SpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialReflectionPadding_updateOutput"
  p_THDoubleNN_SpatialReflectionPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialReflectionPadding_updateGradInput"
  p_THDoubleNN_SpatialReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialReplicationPadding_updateOutput"
  p_THDoubleNN_SpatialReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THDoubleNN_SpatialReplicationPadding_updateGradInput"
  p_THDoubleNN_SpatialReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_FeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THDoubleNN_FeatureLPPooling_updateOutput"
  p_THDoubleNN_FeatureLPPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THDoubleNN_FeatureLPPooling_updateGradInput"
  p_THDoubleNN_FeatureLPPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THDoubleNN_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricReplicationPadding_updateOutput"
  p_THDoubleNN_VolumetricReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricReplicationPadding_updateGradInput"
  p_THDoubleNN_VolumetricReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricUpSamplingNearest_updateOutput"
  p_THDoubleNN_VolumetricUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricUpSamplingNearest_updateGradInput"
  p_THDoubleNN_VolumetricUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput"
  p_THDoubleNN_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput"
  p_THDoubleNN_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalReflectionPadding_updateOutput"
  p_THDoubleNN_TemporalReflectionPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalReflectionPadding_updateGradInput"
  p_THDoubleNN_TemporalReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalReplicationPadding_updateOutput"
  p_THDoubleNN_TemporalReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleNN_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THDoubleNN_TemporalReplicationPadding_updateGradInput"
  p_THDoubleNN_TemporalReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())
{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleNN (
    c_THNN_DoubleAbs_updateOutput,
    c_THNN_DoubleAbs_updateGradInput,
    c_THNN_DoubleAbsCriterion_updateOutput,
    c_THNN_DoubleAbsCriterion_updateGradInput,
    c_THNN_DoubleBCECriterion_updateOutput,
    c_THNN_DoubleBCECriterion_updateGradInput,
    c_THNN_DoubleClassNLLCriterion_updateOutput,
    c_THNN_DoubleClassNLLCriterion_updateGradInput,
    c_THNN_DoubleSpatialClassNLLCriterion_updateOutput,
    c_THNN_DoubleSpatialClassNLLCriterion_updateGradInput,
    c_THNN_DoubleELU_updateOutput,
    c_THNN_DoubleELU_updateGradInput,
    c_THNN_DoubleDistKLDivCriterion_updateOutput,
    c_THNN_DoubleDistKLDivCriterion_updateGradInput,
    c_THNN_DoubleGatedLinear_updateOutput,
    c_THNN_DoubleGatedLinear_updateGradInput,
    c_THNN_DoubleHardShrink_updateOutput,
    c_THNN_DoubleHardShrink_updateGradInput,
    c_THNN_DoubleHardTanh_updateOutput,
    c_THNN_DoubleHardTanh_updateGradInput,
    c_THNN_DoubleL1Cost_updateOutput,
    c_THNN_DoubleL1Cost_updateGradInput,
    c_THNN_DoubleLeakyReLU_updateOutput,
    c_THNN_DoubleLeakyReLU_updateGradInput,
    c_THNN_DoubleGRUFused_updateOutput,
    c_THNN_DoubleGRUFused_updateGradInput,
    c_THNN_DoubleLSTMFused_updateOutput,
    c_THNN_DoubleLSTMFused_updateGradInput,
    c_THNN_DoubleLogSigmoid_updateOutput,
    c_THNN_DoubleLogSigmoid_updateGradInput,
    c_THNN_DoubleLogSoftMax_updateOutput,
    c_THNN_DoubleLogSoftMax_updateGradInput,
    c_THNN_DoubleLookupTable_accGradParameters,
    c_THNN_DoubleLookupTable_renorm,
    c_THNN_DoubleMarginCriterion_updateOutput,
    c_THNN_DoubleMarginCriterion_updateGradInput,
    c_THNN_DoubleSoftMarginCriterion_updateOutput,
    c_THNN_DoubleSoftMarginCriterion_updateGradInput,
    c_THNN_DoubleMSECriterion_updateOutput,
    c_THNN_DoubleMSECriterion_updateGradInput,
    c_THNN_DoubleMultiLabelMarginCriterion_updateOutput,
    c_THNN_DoubleMultiLabelMarginCriterion_updateGradInput,
    c_THNN_DoubleMultiMarginCriterion_updateOutput,
    c_THNN_DoubleMultiMarginCriterion_updateGradInput,
    c_THNN_DoublePReLU_updateOutput,
    c_THNN_DoublePReLU_updateGradInput,
    c_THNN_DoublePReLU_accGradParameters,
    c_THNN_DoubleLinear_updateOutput,
    c_THNN_DoubleLinear_updateGradInput,
    c_THNN_DoubleLinear_accGradParameters,
    c_THNN_DoubleRReLU_updateOutput,
    c_THNN_DoubleRReLU_updateGradInput,
    c_THNN_DoubleSigmoid_updateOutput,
    c_THNN_DoubleSigmoid_updateGradInput,
    c_THNN_DoubleSmoothL1Criterion_updateOutput,
    c_THNN_DoubleSmoothL1Criterion_updateGradInput,
    c_THNN_DoubleSoftMax_updateOutput,
    c_THNN_DoubleSoftMax_updateGradInput,
    c_THNN_DoubleSoftPlus_updateOutput,
    c_THNN_DoubleSoftPlus_updateGradInput,
    c_THNN_DoubleSoftShrink_updateOutput,
    c_THNN_DoubleSoftShrink_updateGradInput,
    c_THNN_DoubleIndexLinear_updateOutput,
    c_THNN_DoubleIndexLinear_accGradParameters,
    c_THNN_DoubleIndexLinear_accUpdateGradParameters,
    c_THNN_DoubleIndexLinear_updateParameters,
    c_THNN_DoubleSparseLinear_updateOutput,
    c_THNN_DoubleSparseLinear_accGradParameters,
    c_THNN_DoubleSparseLinear_zeroGradParameters,
    c_THNN_DoubleSparseLinear_updateParameters,
    c_THNN_DoubleSparseLinear_legacyUpdateOutput,
    c_THNN_DoubleSparseLinear_legacyAccGradParameters,
    c_THNN_DoubleSparseLinear_legacyZeroGradParameters,
    c_THNN_DoubleSparseLinear_legacyUpdateParameters,
    c_THNN_DoubleSqrt_updateOutput,
    c_THNN_DoubleSqrt_updateGradInput,
    c_THNN_DoubleSquare_updateOutput,
    c_THNN_DoubleSquare_updateGradInput,
    c_THNN_DoubleTanh_updateOutput,
    c_THNN_DoubleTanh_updateGradInput,
    c_THNN_DoubleThreshold_updateOutput,
    c_THNN_DoubleThreshold_updateGradInput,
    c_THNN_DoubleTemporalConvolution_updateOutput,
    c_THNN_DoubleTemporalConvolution_updateGradInput,
    c_THNN_DoubleTemporalConvolution_accGradParameters,
    c_THNN_DoubleTemporalMaxPooling_updateOutput,
    c_THNN_DoubleTemporalMaxPooling_updateGradInput,
    c_THNN_DoubleTemporalSubSampling_updateOutput,
    c_THNN_DoubleTemporalSubSampling_updateGradInput,
    c_THNN_DoubleTemporalSubSampling_accGradParameters,
    c_THNN_DoubleTemporalRowConvolution_updateOutput,
    c_THNN_DoubleTemporalRowConvolution_updateGradInput,
    c_THNN_DoubleTemporalRowConvolution_accGradParameters,
    c_THNN_DoubleTemporalUpSamplingNearest_updateOutput,
    c_THNN_DoubleTemporalUpSamplingNearest_updateGradInput,
    c_THNN_DoubleTemporalUpSamplingLinear_updateOutput,
    c_THNN_DoubleTemporalUpSamplingLinear_updateGradInput,
    c_THNN_DoubleBatchNormalization_updateOutput,
    c_THNN_DoubleBatchNormalization_backward,
    c_THNN_DoubleSpatialConvolutionMap_updateOutput,
    c_THNN_DoubleSpatialConvolutionMap_updateGradInput,
    c_THNN_DoubleSpatialConvolutionMap_accGradParameters,
    c_THNN_DoubleSpatialConvolutionMM_updateOutput,
    c_THNN_DoubleSpatialConvolutionMM_updateGradInput,
    c_THNN_DoubleSpatialConvolutionMM_accGradParameters,
    c_THNN_DoubleSpatialConvolutionLocal_updateOutput,
    c_THNN_DoubleSpatialConvolutionLocal_updateGradInput,
    c_THNN_DoubleSpatialConvolutionLocal_accGradParameters,
    c_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput,
    c_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput,
    c_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput,
    c_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput,
    c_THNN_DoubleSpatialAveragePooling_updateOutput,
    c_THNN_DoubleSpatialAveragePooling_updateGradInput,
    c_THNN_DoubleSpatialFractionalMaxPooling_updateOutput,
    c_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput,
    c_THNN_DoubleSpatialFullConvolution_updateOutput,
    c_THNN_DoubleSpatialFullConvolution_updateGradInput,
    c_THNN_DoubleSpatialFullConvolution_accGradParameters,
    c_THNN_DoubleSpatialFullConvolutionMap_updateOutput,
    c_THNN_DoubleSpatialFullConvolutionMap_updateGradInput,
    c_THNN_DoubleSpatialFullConvolutionMap_accGradParameters,
    c_THNN_DoubleSpatialDilatedConvolution_updateOutput,
    c_THNN_DoubleSpatialDilatedConvolution_updateGradInput,
    c_THNN_DoubleSpatialDilatedConvolution_accGradParameters,
    c_THNN_DoubleSpatialFullDilatedConvolution_updateOutput,
    c_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput,
    c_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters,
    c_THNN_DoubleSpatialMaxPooling_updateOutput,
    c_THNN_DoubleSpatialMaxPooling_updateGradInput,
    c_THNN_DoubleSpatialDilatedMaxPooling_updateOutput,
    c_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput,
    c_THNN_DoubleSpatialMaxUnpooling_updateOutput,
    c_THNN_DoubleSpatialMaxUnpooling_updateGradInput,
    c_THNN_DoubleSpatialSubSampling_updateOutput,
    c_THNN_DoubleSpatialSubSampling_updateGradInput,
    c_THNN_DoubleSpatialSubSampling_accGradParameters,
    c_THNN_DoubleSpatialUpSamplingNearest_updateOutput,
    c_THNN_DoubleSpatialUpSamplingNearest_updateGradInput,
    c_THNN_DoubleSpatialUpSamplingBilinear_updateOutput,
    c_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput,
    c_THNN_DoubleSpatialGridSamplerBilinear_updateOutput,
    c_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput,
    c_THNN_Doubleunfolded_acc,
    c_THNN_Doubleunfolded_copy,
    c_THNN_DoubleVolumetricAveragePooling_updateOutput,
    c_THNN_DoubleVolumetricAveragePooling_updateGradInput,
    c_THNN_DoubleVolumetricConvolution_updateOutput,
    c_THNN_DoubleVolumetricConvolution_updateGradInput,
    c_THNN_DoubleVolumetricConvolution_accGradParameters,
    c_THNN_DoubleVolumetricConvolutionMM_updateOutput,
    c_THNN_DoubleVolumetricConvolutionMM_updateGradInput,
    c_THNN_DoubleVolumetricConvolutionMM_accGradParameters,
    c_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput,
    c_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput,
    c_THNN_DoubleVolumetricFullConvolution_updateOutput,
    c_THNN_DoubleVolumetricFullConvolution_updateGradInput,
    c_THNN_DoubleVolumetricFullConvolution_accGradParameters,
    c_THNN_DoubleVolumetricDilatedConvolution_updateOutput,
    c_THNN_DoubleVolumetricDilatedConvolution_updateGradInput,
    c_THNN_DoubleVolumetricDilatedConvolution_accGradParameters,
    c_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput,
    c_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput,
    c_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters,
    c_THNN_DoubleVolumetricMaxPooling_updateOutput,
    c_THNN_DoubleVolumetricMaxPooling_updateGradInput,
    c_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput,
    c_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput,
    c_THNN_DoubleVolumetricMaxUnpooling_updateOutput,
    c_THNN_DoubleVolumetricMaxUnpooling_updateGradInput,
    c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput,
    c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput,
    c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput,
    c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput,
    c_THNN_DoubleSpatialReflectionPadding_updateOutput,
    c_THNN_DoubleSpatialReflectionPadding_updateGradInput,
    c_THNN_DoubleSpatialReplicationPadding_updateOutput,
    c_THNN_DoubleSpatialReplicationPadding_updateGradInput,
    c_THNN_DoubleFeatureLPPooling_updateOutput,
    c_THNN_DoubleFeatureLPPooling_updateGradInput,
    c_THNN_DoubleVolumetricReplicationPadding_updateOutput,
    c_THNN_DoubleVolumetricReplicationPadding_updateGradInput,
    c_THNN_DoubleVolumetricUpSamplingNearest_updateOutput,
    c_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput,
    c_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput,
    c_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput,
    c_THNN_DoubleTemporalReflectionPadding_updateOutput,
    c_THNN_DoubleTemporalReflectionPadding_updateGradInput,
    c_THNN_DoubleTemporalReplicationPadding_updateOutput,
    c_THNN_DoubleTemporalReplicationPadding_updateGradInput,
    p_THNN_DoubleAbs_updateOutput,
    p_THNN_DoubleAbs_updateGradInput,
    p_THNN_DoubleAbsCriterion_updateOutput,
    p_THNN_DoubleAbsCriterion_updateGradInput,
    p_THNN_DoubleBCECriterion_updateOutput,
    p_THNN_DoubleBCECriterion_updateGradInput,
    p_THNN_DoubleClassNLLCriterion_updateOutput,
    p_THNN_DoubleClassNLLCriterion_updateGradInput,
    p_THNN_DoubleSpatialClassNLLCriterion_updateOutput,
    p_THNN_DoubleSpatialClassNLLCriterion_updateGradInput,
    p_THNN_DoubleELU_updateOutput,
    p_THNN_DoubleELU_updateGradInput,
    p_THNN_DoubleDistKLDivCriterion_updateOutput,
    p_THNN_DoubleDistKLDivCriterion_updateGradInput,
    p_THNN_DoubleGatedLinear_updateOutput,
    p_THNN_DoubleGatedLinear_updateGradInput,
    p_THNN_DoubleHardShrink_updateOutput,
    p_THNN_DoubleHardShrink_updateGradInput,
    p_THNN_DoubleHardTanh_updateOutput,
    p_THNN_DoubleHardTanh_updateGradInput,
    p_THNN_DoubleL1Cost_updateOutput,
    p_THNN_DoubleL1Cost_updateGradInput,
    p_THNN_DoubleLeakyReLU_updateOutput,
    p_THNN_DoubleLeakyReLU_updateGradInput,
    p_THNN_DoubleGRUFused_updateOutput,
    p_THNN_DoubleGRUFused_updateGradInput,
    p_THNN_DoubleLSTMFused_updateOutput,
    p_THNN_DoubleLSTMFused_updateGradInput,
    p_THNN_DoubleLogSigmoid_updateOutput,
    p_THNN_DoubleLogSigmoid_updateGradInput,
    p_THNN_DoubleLogSoftMax_updateOutput,
    p_THNN_DoubleLogSoftMax_updateGradInput,
    p_THNN_DoubleLookupTable_accGradParameters,
    p_THNN_DoubleLookupTable_renorm,
    p_THNN_DoubleMarginCriterion_updateOutput,
    p_THNN_DoubleMarginCriterion_updateGradInput,
    p_THNN_DoubleSoftMarginCriterion_updateOutput,
    p_THNN_DoubleSoftMarginCriterion_updateGradInput,
    p_THNN_DoubleMSECriterion_updateOutput,
    p_THNN_DoubleMSECriterion_updateGradInput,
    p_THNN_DoubleMultiLabelMarginCriterion_updateOutput,
    p_THNN_DoubleMultiLabelMarginCriterion_updateGradInput,
    p_THNN_DoubleMultiMarginCriterion_updateOutput,
    p_THNN_DoubleMultiMarginCriterion_updateGradInput,
    p_THNN_DoublePReLU_updateOutput,
    p_THNN_DoublePReLU_updateGradInput,
    p_THNN_DoublePReLU_accGradParameters,
    p_THNN_DoubleLinear_updateOutput,
    p_THNN_DoubleLinear_updateGradInput,
    p_THNN_DoubleLinear_accGradParameters,
    p_THNN_DoubleRReLU_updateOutput,
    p_THNN_DoubleRReLU_updateGradInput,
    p_THNN_DoubleSigmoid_updateOutput,
    p_THNN_DoubleSigmoid_updateGradInput,
    p_THNN_DoubleSmoothL1Criterion_updateOutput,
    p_THNN_DoubleSmoothL1Criterion_updateGradInput,
    p_THNN_DoubleSoftMax_updateOutput,
    p_THNN_DoubleSoftMax_updateGradInput,
    p_THNN_DoubleSoftPlus_updateOutput,
    p_THNN_DoubleSoftPlus_updateGradInput,
    p_THNN_DoubleSoftShrink_updateOutput,
    p_THNN_DoubleSoftShrink_updateGradInput,
    p_THNN_DoubleIndexLinear_updateOutput,
    p_THNN_DoubleIndexLinear_accGradParameters,
    p_THNN_DoubleIndexLinear_accUpdateGradParameters,
    p_THNN_DoubleIndexLinear_updateParameters,
    p_THNN_DoubleSparseLinear_updateOutput,
    p_THNN_DoubleSparseLinear_accGradParameters,
    p_THNN_DoubleSparseLinear_zeroGradParameters,
    p_THNN_DoubleSparseLinear_updateParameters,
    p_THNN_DoubleSparseLinear_legacyUpdateOutput,
    p_THNN_DoubleSparseLinear_legacyAccGradParameters,
    p_THNN_DoubleSparseLinear_legacyZeroGradParameters,
    p_THNN_DoubleSparseLinear_legacyUpdateParameters,
    p_THNN_DoubleSqrt_updateOutput,
    p_THNN_DoubleSqrt_updateGradInput,
    p_THNN_DoubleSquare_updateOutput,
    p_THNN_DoubleSquare_updateGradInput,
    p_THNN_DoubleTanh_updateOutput,
    p_THNN_DoubleTanh_updateGradInput,
    p_THNN_DoubleThreshold_updateOutput,
    p_THNN_DoubleThreshold_updateGradInput,
    p_THNN_DoubleTemporalConvolution_updateOutput,
    p_THNN_DoubleTemporalConvolution_updateGradInput,
    p_THNN_DoubleTemporalConvolution_accGradParameters,
    p_THNN_DoubleTemporalMaxPooling_updateOutput,
    p_THNN_DoubleTemporalMaxPooling_updateGradInput,
    p_THNN_DoubleTemporalSubSampling_updateOutput,
    p_THNN_DoubleTemporalSubSampling_updateGradInput,
    p_THNN_DoubleTemporalSubSampling_accGradParameters,
    p_THNN_DoubleTemporalRowConvolution_updateOutput,
    p_THNN_DoubleTemporalRowConvolution_updateGradInput,
    p_THNN_DoubleTemporalRowConvolution_accGradParameters,
    p_THNN_DoubleTemporalUpSamplingNearest_updateOutput,
    p_THNN_DoubleTemporalUpSamplingNearest_updateGradInput,
    p_THNN_DoubleTemporalUpSamplingLinear_updateOutput,
    p_THNN_DoubleTemporalUpSamplingLinear_updateGradInput,
    p_THNN_DoubleBatchNormalization_updateOutput,
    p_THNN_DoubleBatchNormalization_backward,
    p_THNN_DoubleSpatialConvolutionMap_updateOutput,
    p_THNN_DoubleSpatialConvolutionMap_updateGradInput,
    p_THNN_DoubleSpatialConvolutionMap_accGradParameters,
    p_THNN_DoubleSpatialConvolutionMM_updateOutput,
    p_THNN_DoubleSpatialConvolutionMM_updateGradInput,
    p_THNN_DoubleSpatialConvolutionMM_accGradParameters,
    p_THNN_DoubleSpatialConvolutionLocal_updateOutput,
    p_THNN_DoubleSpatialConvolutionLocal_updateGradInput,
    p_THNN_DoubleSpatialConvolutionLocal_accGradParameters,
    p_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput,
    p_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput,
    p_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput,
    p_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput,
    p_THNN_DoubleSpatialAveragePooling_updateOutput,
    p_THNN_DoubleSpatialAveragePooling_updateGradInput,
    p_THNN_DoubleSpatialFractionalMaxPooling_updateOutput,
    p_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput,
    p_THNN_DoubleSpatialFullConvolution_updateOutput,
    p_THNN_DoubleSpatialFullConvolution_updateGradInput,
    p_THNN_DoubleSpatialFullConvolution_accGradParameters,
    p_THNN_DoubleSpatialFullConvolutionMap_updateOutput,
    p_THNN_DoubleSpatialFullConvolutionMap_updateGradInput,
    p_THNN_DoubleSpatialFullConvolutionMap_accGradParameters,
    p_THNN_DoubleSpatialDilatedConvolution_updateOutput,
    p_THNN_DoubleSpatialDilatedConvolution_updateGradInput,
    p_THNN_DoubleSpatialDilatedConvolution_accGradParameters,
    p_THNN_DoubleSpatialFullDilatedConvolution_updateOutput,
    p_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput,
    p_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters,
    p_THNN_DoubleSpatialMaxPooling_updateOutput,
    p_THNN_DoubleSpatialMaxPooling_updateGradInput,
    p_THNN_DoubleSpatialDilatedMaxPooling_updateOutput,
    p_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput,
    p_THNN_DoubleSpatialMaxUnpooling_updateOutput,
    p_THNN_DoubleSpatialMaxUnpooling_updateGradInput,
    p_THNN_DoubleSpatialSubSampling_updateOutput,
    p_THNN_DoubleSpatialSubSampling_updateGradInput,
    p_THNN_DoubleSpatialSubSampling_accGradParameters,
    p_THNN_DoubleSpatialUpSamplingNearest_updateOutput,
    p_THNN_DoubleSpatialUpSamplingNearest_updateGradInput,
    p_THNN_DoubleSpatialUpSamplingBilinear_updateOutput,
    p_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput,
    p_THNN_DoubleSpatialGridSamplerBilinear_updateOutput,
    p_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput,
    p_THNN_Doubleunfolded_acc,
    p_THNN_Doubleunfolded_copy,
    p_THNN_DoubleVolumetricAveragePooling_updateOutput,
    p_THNN_DoubleVolumetricAveragePooling_updateGradInput,
    p_THNN_DoubleVolumetricConvolution_updateOutput,
    p_THNN_DoubleVolumetricConvolution_updateGradInput,
    p_THNN_DoubleVolumetricConvolution_accGradParameters,
    p_THNN_DoubleVolumetricConvolutionMM_updateOutput,
    p_THNN_DoubleVolumetricConvolutionMM_updateGradInput,
    p_THNN_DoubleVolumetricConvolutionMM_accGradParameters,
    p_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput,
    p_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput,
    p_THNN_DoubleVolumetricFullConvolution_updateOutput,
    p_THNN_DoubleVolumetricFullConvolution_updateGradInput,
    p_THNN_DoubleVolumetricFullConvolution_accGradParameters,
    p_THNN_DoubleVolumetricDilatedConvolution_updateOutput,
    p_THNN_DoubleVolumetricDilatedConvolution_updateGradInput,
    p_THNN_DoubleVolumetricDilatedConvolution_accGradParameters,
    p_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput,
    p_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput,
    p_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters,
    p_THNN_DoubleVolumetricMaxPooling_updateOutput,
    p_THNN_DoubleVolumetricMaxPooling_updateGradInput,
    p_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput,
    p_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput,
    p_THNN_DoubleVolumetricMaxUnpooling_updateOutput,
    p_THNN_DoubleVolumetricMaxUnpooling_updateGradInput,
    p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput,
    p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput,
    p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput,
    p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput,
    p_THNN_DoubleSpatialReflectionPadding_updateOutput,
    p_THNN_DoubleSpatialReflectionPadding_updateGradInput,
    p_THNN_DoubleSpatialReplicationPadding_updateOutput,
    p_THNN_DoubleSpatialReplicationPadding_updateGradInput,
    p_THNN_DoubleFeatureLPPooling_updateOutput,
    p_THNN_DoubleFeatureLPPooling_updateGradInput,
    p_THNN_DoubleVolumetricReplicationPadding_updateOutput,
    p_THNN_DoubleVolumetricReplicationPadding_updateGradInput,
    p_THNN_DoubleVolumetricUpSamplingNearest_updateOutput,
    p_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput,
    p_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput,
    p_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput,
    p_THNN_DoubleTemporalReflectionPadding_updateOutput,
    p_THNN_DoubleTemporalReflectionPadding_updateGradInput,
    p_THNN_DoubleTemporalReplicationPadding_updateOutput,
    p_THNN_DoubleTemporalReplicationPadding_updateGradInput) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THNN_DoubleAbs_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_DoubleAbs_updateOutput"
  c_THNN_DoubleAbs_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleAbs_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleAbs_updateGradInput"
  c_THNN_DoubleAbs_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleAbsCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleAbsCriterion_updateOutput"
  c_THNN_DoubleAbsCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleAbsCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleAbsCriterion_updateGradInput"
  c_THNN_DoubleAbsCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleBCECriterion_updateOutput : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h THNN_DoubleBCECriterion_updateOutput"
  c_THNN_DoubleBCECriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleBCECriterion_updateGradInput : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h THNN_DoubleBCECriterion_updateGradInput"
  c_THNN_DoubleBCECriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_DoubleClassNLLCriterion_updateOutput"
  c_THNN_DoubleClassNLLCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_DoubleClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_DoubleClassNLLCriterion_updateGradInput"
  c_THNN_DoubleClassNLLCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_DoubleSpatialClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_DoubleSpatialClassNLLCriterion_updateOutput"
  c_THNN_DoubleSpatialClassNLLCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_DoubleSpatialClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_DoubleSpatialClassNLLCriterion_updateGradInput"
  c_THNN_DoubleSpatialClassNLLCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_DoubleELU_updateOutput : state input output alpha inplace -> void
foreign import ccall "THNN.h THNN_DoubleELU_updateOutput"
  c_THNN_DoubleELU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleELU_updateGradInput : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h THNN_DoubleELU_updateGradInput"
  c_THNN_DoubleELU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleDistKLDivCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleDistKLDivCriterion_updateOutput"
  c_THNN_DoubleDistKLDivCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleDistKLDivCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleDistKLDivCriterion_updateGradInput"
  c_THNN_DoubleDistKLDivCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleGatedLinear_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleGatedLinear_updateOutput"
  c_THNN_DoubleGatedLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleGatedLinear_updateGradInput : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THNN_DoubleGatedLinear_updateGradInput"
  c_THNN_DoubleGatedLinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleHardShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THNN_DoubleHardShrink_updateOutput"
  c_THNN_DoubleHardShrink_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleHardShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_DoubleHardShrink_updateGradInput"
  c_THNN_DoubleHardShrink_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleHardTanh_updateOutput : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_DoubleHardTanh_updateOutput"
  c_THNN_DoubleHardTanh_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleHardTanh_updateGradInput : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_DoubleHardTanh_updateGradInput"
  c_THNN_DoubleHardTanh_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleL1Cost_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_DoubleL1Cost_updateOutput"
  c_THNN_DoubleL1Cost_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleL1Cost_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleL1Cost_updateGradInput"
  c_THNN_DoubleL1Cost_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLeakyReLU_updateOutput : state input output negval inplace -> void
foreign import ccall "THNN.h THNN_DoubleLeakyReLU_updateOutput"
  c_THNN_DoubleLeakyReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleLeakyReLU_updateGradInput : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THNN_DoubleLeakyReLU_updateGradInput"
  c_THNN_DoubleLeakyReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleGRUFused_updateOutput : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THNN_DoubleGRUFused_updateOutput"
  c_THNN_DoubleGRUFused_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleGRUFused_updateGradInput : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THNN_DoubleGRUFused_updateGradInput"
  c_THNN_DoubleGRUFused_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLSTMFused_updateOutput : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THNN_DoubleLSTMFused_updateOutput"
  c_THNN_DoubleLSTMFused_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLSTMFused_updateGradInput : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THNN_DoubleLSTMFused_updateGradInput"
  c_THNN_DoubleLSTMFused_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLogSigmoid_updateOutput : state input output buffer -> void
foreign import ccall "THNN.h THNN_DoubleLogSigmoid_updateOutput"
  c_THNN_DoubleLogSigmoid_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLogSigmoid_updateGradInput : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THNN_DoubleLogSigmoid_updateGradInput"
  c_THNN_DoubleLogSigmoid_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLogSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleLogSoftMax_updateOutput"
  c_THNN_DoubleLogSoftMax_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleLogSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_DoubleLogSoftMax_updateGradInput"
  c_THNN_DoubleLogSoftMax_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleLookupTable_accGradParameters : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h THNN_DoubleLookupTable_accGradParameters"
  c_THNN_DoubleLookupTable_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleLookupTable_renorm : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h THNN_DoubleLookupTable_renorm"
  c_THNN_DoubleLookupTable_renorm :: (Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleMarginCriterion_updateOutput : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THNN_DoubleMarginCriterion_updateOutput"
  c_THNN_DoubleMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ()

-- |c_THNN_DoubleMarginCriterion_updateGradInput : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THNN_DoubleMarginCriterion_updateGradInput"
  c_THNN_DoubleMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ()

-- |c_THNN_DoubleSoftMarginCriterion_updateOutput : state input target output sizeAverage -> void
foreign import ccall "THNN.h THNN_DoubleSoftMarginCriterion_updateOutput"
  c_THNN_DoubleSoftMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THNN_DoubleSoftMarginCriterion_updateGradInput : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h THNN_DoubleSoftMarginCriterion_updateGradInput"
  c_THNN_DoubleSoftMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THNN_DoubleMSECriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleMSECriterion_updateOutput"
  c_THNN_DoubleMSECriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleMSECriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleMSECriterion_updateGradInput"
  c_THNN_DoubleMSECriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleMultiLabelMarginCriterion_updateOutput : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h THNN_DoubleMultiLabelMarginCriterion_updateOutput"
  c_THNN_DoubleMultiLabelMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THNN_DoubleMultiLabelMarginCriterion_updateGradInput : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h THNN_DoubleMultiLabelMarginCriterion_updateGradInput"
  c_THNN_DoubleMultiLabelMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ()

-- |c_THNN_DoubleMultiMarginCriterion_updateOutput : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h THNN_DoubleMultiMarginCriterion_updateOutput"
  c_THNN_DoubleMultiMarginCriterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleMultiMarginCriterion_updateGradInput : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h THNN_DoubleMultiMarginCriterion_updateGradInput"
  c_THNN_DoubleMultiMarginCriterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoublePReLU_updateOutput : state input output weight -> void
foreign import ccall "THNN.h THNN_DoublePReLU_updateOutput"
  c_THNN_DoublePReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoublePReLU_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_DoublePReLU_updateGradInput"
  c_THNN_DoublePReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoublePReLU_accGradParameters : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THNN_DoublePReLU_accGradParameters"
  c_THNN_DoublePReLU_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleLinear_updateOutput : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THNN_DoubleLinear_updateOutput"
  c_THNN_DoubleLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLinear_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_DoubleLinear_updateGradInput"
  c_THNN_DoubleLinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleLinear_accGradParameters : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THNN_DoubleLinear_accGradParameters"
  c_THNN_DoubleLinear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleRReLU_updateOutput : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THNN_DoubleRReLU_updateOutput"
  c_THNN_DoubleRReLU_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ()

-- |c_THNN_DoubleRReLU_updateGradInput : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THNN_DoubleRReLU_updateGradInput"
  c_THNN_DoubleRReLU_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleSigmoid_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_DoubleSigmoid_updateOutput"
  c_THNN_DoubleSigmoid_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSigmoid_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleSigmoid_updateGradInput"
  c_THNN_DoubleSigmoid_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSmoothL1Criterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSmoothL1Criterion_updateOutput"
  c_THNN_DoubleSmoothL1Criterion_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleSmoothL1Criterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSmoothL1Criterion_updateGradInput"
  c_THNN_DoubleSmoothL1Criterion_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleSoftMax_updateOutput"
  c_THNN_DoubleSoftMax_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_DoubleSoftMax_updateGradInput"
  c_THNN_DoubleSoftMax_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleSoftPlus_updateOutput : state input output beta threshold -> void
foreign import ccall "THNN.h THNN_DoubleSoftPlus_updateOutput"
  c_THNN_DoubleSoftPlus_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSoftPlus_updateGradInput : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THNN_DoubleSoftPlus_updateGradInput"
  c_THNN_DoubleSoftPlus_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSoftShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THNN_DoubleSoftShrink_updateOutput"
  c_THNN_DoubleSoftShrink_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleSoftShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_DoubleSoftShrink_updateGradInput"
  c_THNN_DoubleSoftShrink_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleIndexLinear_updateOutput : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h THNN_DoubleIndexLinear_updateOutput"
  c_THNN_DoubleIndexLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleIndexLinear_accGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleIndexLinear_accGradParameters"
  c_THNN_DoubleIndexLinear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleIndexLinear_accUpdateGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleIndexLinear_accUpdateGradParameters"
  c_THNN_DoubleIndexLinear_accUpdateGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleIndexLinear_updateParameters : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h THNN_DoubleIndexLinear_updateParameters"
  c_THNN_DoubleIndexLinear_updateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSparseLinear_updateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_updateOutput"
  c_THNN_DoubleSparseLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSparseLinear_accGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_accGradParameters"
  c_THNN_DoubleSparseLinear_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSparseLinear_zeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_zeroGradParameters"
  c_THNN_DoubleSparseLinear_zeroGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSparseLinear_updateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_updateParameters"
  c_THNN_DoubleSparseLinear_updateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleSparseLinear_legacyUpdateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyUpdateOutput"
  c_THNN_DoubleSparseLinear_legacyUpdateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSparseLinear_legacyAccGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyAccGradParameters"
  c_THNN_DoubleSparseLinear_legacyAccGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSparseLinear_legacyZeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyZeroGradParameters"
  c_THNN_DoubleSparseLinear_legacyZeroGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSparseLinear_legacyUpdateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyUpdateParameters"
  c_THNN_DoubleSparseLinear_legacyUpdateParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleSqrt_updateOutput : state input output eps -> void
foreign import ccall "THNN.h THNN_DoubleSqrt_updateOutput"
  c_THNN_DoubleSqrt_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THNN_DoubleSqrt_updateGradInput : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleSqrt_updateGradInput"
  c_THNN_DoubleSqrt_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSquare_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_DoubleSquare_updateOutput"
  c_THNN_DoubleSquare_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSquare_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleSquare_updateGradInput"
  c_THNN_DoubleSquare_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleTanh_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_DoubleTanh_updateOutput"
  c_THNN_DoubleTanh_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleTanh_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleTanh_updateGradInput"
  c_THNN_DoubleTanh_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleThreshold_updateOutput : state input output threshold val inplace -> void
foreign import ccall "THNN.h THNN_DoubleThreshold_updateOutput"
  c_THNN_DoubleThreshold_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleThreshold_updateGradInput : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THNN_DoubleThreshold_updateGradInput"
  c_THNN_DoubleThreshold_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_DoubleTemporalConvolution_updateOutput : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_updateOutput"
  c_THNN_DoubleTemporalConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalConvolution_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_updateGradInput"
  c_THNN_DoubleTemporalConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalConvolution_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_accGradParameters"
  c_THNN_DoubleTemporalConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleTemporalMaxPooling_updateOutput : state input output indices kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalMaxPooling_updateOutput"
  c_THNN_DoubleTemporalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalMaxPooling_updateGradInput : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalMaxPooling_updateGradInput"
  c_THNN_DoubleTemporalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalSubSampling_updateOutput : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_updateOutput"
  c_THNN_DoubleTemporalSubSampling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalSubSampling_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_updateGradInput"
  c_THNN_DoubleTemporalSubSampling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_accGradParameters"
  c_THNN_DoubleTemporalSubSampling_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleTemporalRowConvolution_updateOutput : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_updateOutput"
  c_THNN_DoubleTemporalRowConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleTemporalRowConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_updateGradInput"
  c_THNN_DoubleTemporalRowConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleTemporalRowConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_accGradParameters"
  c_THNN_DoubleTemporalRowConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- |c_THNN_DoubleTemporalUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingNearest_updateOutput"
  c_THNN_DoubleTemporalUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleTemporalUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingNearest_updateGradInput"
  c_THNN_DoubleTemporalUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleTemporalUpSamplingLinear_updateOutput : state input output outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingLinear_updateOutput"
  c_THNN_DoubleTemporalUpSamplingLinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleTemporalUpSamplingLinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingLinear_updateGradInput"
  c_THNN_DoubleTemporalUpSamplingLinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleBatchNormalization_updateOutput : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THNN_DoubleBatchNormalization_updateOutput"
  c_THNN_DoubleBatchNormalization_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleBatchNormalization_backward : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THNN_DoubleBatchNormalization_backward"
  c_THNN_DoubleBatchNormalization_backward :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_updateOutput"
  c_THNN_DoubleSpatialConvolutionMap_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_updateGradInput"
  c_THNN_DoubleSpatialConvolutionMap_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_accGradParameters"
  c_THNN_DoubleSpatialConvolutionMap_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMM_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_updateOutput"
  c_THNN_DoubleSpatialConvolutionMM_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_updateGradInput"
  c_THNN_DoubleSpatialConvolutionMM_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_accGradParameters"
  c_THNN_DoubleSpatialConvolutionMM_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialConvolutionLocal_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_updateOutput"
  c_THNN_DoubleSpatialConvolutionLocal_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THNN_DoubleSpatialConvolutionLocal_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_updateGradInput"
  c_THNN_DoubleSpatialConvolutionLocal_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THNN_DoubleSpatialConvolutionLocal_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_accGradParameters"
  c_THNN_DoubleSpatialConvolutionLocal_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput"
  c_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput"
  c_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput : state input output osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput"
  c_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput"
  c_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSpatialAveragePooling_updateOutput : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAveragePooling_updateOutput"
  c_THNN_DoubleSpatialAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleSpatialAveragePooling_updateGradInput : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAveragePooling_updateGradInput"
  c_THNN_DoubleSpatialAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleSpatialFractionalMaxPooling_updateOutput : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFractionalMaxPooling_updateOutput"
  c_THNN_DoubleSpatialFractionalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFractionalMaxPooling_updateGradInput"
  c_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_DoubleSpatialFullConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_updateOutput"
  c_THNN_DoubleSpatialFullConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_updateGradInput"
  c_THNN_DoubleSpatialFullConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_accGradParameters"
  c_THNN_DoubleSpatialFullConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialFullConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_updateOutput"
  c_THNN_DoubleSpatialFullConvolutionMap_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_updateGradInput"
  c_THNN_DoubleSpatialFullConvolutionMap_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_accGradParameters"
  c_THNN_DoubleSpatialFullConvolutionMap_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_updateOutput"
  c_THNN_DoubleSpatialDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_updateGradInput"
  c_THNN_DoubleSpatialDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_accGradParameters"
  c_THNN_DoubleSpatialDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialFullDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_updateOutput"
  c_THNN_DoubleSpatialFullDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_updateGradInput"
  c_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_accGradParameters"
  c_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialMaxPooling_updateOutput"
  c_THNN_DoubleSpatialMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleSpatialMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialMaxPooling_updateGradInput"
  c_THNN_DoubleSpatialMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleSpatialDilatedMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedMaxPooling_updateOutput"
  c_THNN_DoubleSpatialDilatedMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedMaxPooling_updateGradInput"
  c_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleSpatialMaxUnpooling_updateOutput : state input output indices owidth oheight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialMaxUnpooling_updateOutput"
  c_THNN_DoubleSpatialMaxUnpooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialMaxUnpooling_updateGradInput : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialMaxUnpooling_updateGradInput"
  c_THNN_DoubleSpatialMaxUnpooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialSubSampling_updateOutput : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_updateOutput"
  c_THNN_DoubleSpatialSubSampling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialSubSampling_updateGradInput : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_updateGradInput"
  c_THNN_DoubleSpatialSubSampling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_accGradParameters"
  c_THNN_DoubleSpatialSubSampling_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleSpatialUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingNearest_updateOutput"
  c_THNN_DoubleSpatialUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleSpatialUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingNearest_updateGradInput"
  c_THNN_DoubleSpatialUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleSpatialUpSamplingBilinear_updateOutput : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingBilinear_updateOutput"
  c_THNN_DoubleSpatialUpSamplingBilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingBilinear_updateGradInput"
  c_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialGridSamplerBilinear_updateOutput : state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialGridSamplerBilinear_updateOutput"
  c_THNN_DoubleSpatialGridSamplerBilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialGridSamplerBilinear_updateGradInput"
  c_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_Doubleunfolded_acc : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Doubleunfolded_acc"
  c_THNN_Doubleunfolded_acc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_Doubleunfolded_copy : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Doubleunfolded_copy"
  c_THNN_Doubleunfolded_copy :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricAveragePooling_updateOutput : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAveragePooling_updateOutput"
  c_THNN_DoubleVolumetricAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricAveragePooling_updateGradInput : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAveragePooling_updateGradInput"
  c_THNN_DoubleVolumetricAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_updateOutput"
  c_THNN_DoubleVolumetricConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricConvolution_updateGradInput : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_updateGradInput"
  c_THNN_DoubleVolumetricConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_accGradParameters"
  c_THNN_DoubleVolumetricConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleVolumetricConvolutionMM_updateOutput : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_updateOutput"
  c_THNN_DoubleVolumetricConvolutionMM_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_updateGradInput"
  c_THNN_DoubleVolumetricConvolutionMM_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_accGradParameters"
  c_THNN_DoubleVolumetricConvolutionMM_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFractionalMaxPooling_updateOutput"
  c_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput"
  c_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_DoubleVolumetricFullConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_updateOutput"
  c_THNN_DoubleVolumetricFullConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricFullConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_updateGradInput"
  c_THNN_DoubleVolumetricFullConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_accGradParameters"
  c_THNN_DoubleVolumetricFullConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleVolumetricDilatedConvolution_updateOutput : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_updateOutput"
  c_THNN_DoubleVolumetricDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_updateGradInput"
  c_THNN_DoubleVolumetricDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_accGradParameters"
  c_THNN_DoubleVolumetricDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_updateOutput"
  c_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput"
  c_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters"
  c_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_DoubleVolumetricMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricMaxPooling_updateOutput"
  c_THNN_DoubleVolumetricMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricMaxPooling_updateGradInput"
  c_THNN_DoubleVolumetricMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedMaxPooling_updateOutput"
  c_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput"
  c_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricMaxUnpooling_updateOutput : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricMaxUnpooling_updateOutput"
  c_THNN_DoubleVolumetricMaxUnpooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricMaxUnpooling_updateGradInput : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricMaxUnpooling_updateGradInput"
  c_THNN_DoubleVolumetricMaxUnpooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput"
  c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  c_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput"
  c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput"
  c_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_DoubleSpatialReflectionPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReflectionPadding_updateOutput"
  c_THNN_DoubleSpatialReflectionPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReflectionPadding_updateGradInput"
  c_THNN_DoubleSpatialReflectionPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialReplicationPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReplicationPadding_updateOutput"
  c_THNN_DoubleSpatialReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleSpatialReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReplicationPadding_updateGradInput"
  c_THNN_DoubleSpatialReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleFeatureLPPooling_updateOutput : state input output power width stride batchMode -> void
foreign import ccall "THNN.h THNN_DoubleFeatureLPPooling_updateOutput"
  c_THNN_DoubleFeatureLPPooling_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleFeatureLPPooling_updateGradInput : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THNN_DoubleFeatureLPPooling_updateGradInput"
  c_THNN_DoubleFeatureLPPooling_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_DoubleVolumetricReplicationPadding_updateOutput : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricReplicationPadding_updateOutput"
  c_THNN_DoubleVolumetricReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricReplicationPadding_updateGradInput : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricReplicationPadding_updateGradInput"
  c_THNN_DoubleVolumetricReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingNearest_updateOutput"
  c_THNN_DoubleVolumetricUpSamplingNearest_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingNearest_updateGradInput"
  c_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput"
  c_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput"
  c_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalReflectionPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReflectionPadding_updateOutput"
  c_THNN_DoubleTemporalReflectionPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReflectionPadding_updateGradInput"
  c_THNN_DoubleTemporalReflectionPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalReplicationPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReplicationPadding_updateOutput"
  c_THNN_DoubleTemporalReplicationPadding_updateOutput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_DoubleTemporalReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReplicationPadding_updateGradInput"
  c_THNN_DoubleTemporalReplicationPadding_updateGradInput :: (Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |p_THNN_DoubleAbs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleAbs_updateOutput"
  p_THNN_DoubleAbs_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleAbs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleAbs_updateGradInput"
  p_THNN_DoubleAbs_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleAbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleAbsCriterion_updateOutput"
  p_THNN_DoubleAbsCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleAbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleAbsCriterion_updateGradInput"
  p_THNN_DoubleAbsCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleBCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h &THNN_DoubleBCECriterion_updateOutput"
  p_THNN_DoubleBCECriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleBCECriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h &THNN_DoubleBCECriterion_updateGradInput"
  p_THNN_DoubleBCECriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_DoubleClassNLLCriterion_updateOutput"
  p_THNN_DoubleClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_DoubleClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_DoubleClassNLLCriterion_updateGradInput"
  p_THNN_DoubleClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_DoubleSpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialClassNLLCriterion_updateOutput"
  p_THNN_DoubleSpatialClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_DoubleSpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialClassNLLCriterion_updateGradInput"
  p_THNN_DoubleSpatialClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_DoubleELU_updateOutput : Pointer to function : state input output alpha inplace -> void
foreign import ccall "THNN.h &THNN_DoubleELU_updateOutput"
  p_THNN_DoubleELU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h &THNN_DoubleELU_updateGradInput"
  p_THNN_DoubleELU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleDistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleDistKLDivCriterion_updateOutput"
  p_THNN_DoubleDistKLDivCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleDistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleDistKLDivCriterion_updateGradInput"
  p_THNN_DoubleDistKLDivCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleGatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleGatedLinear_updateOutput"
  p_THNN_DoubleGatedLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleGatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THNN_DoubleGatedLinear_updateGradInput"
  p_THNN_DoubleGatedLinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleHardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_DoubleHardShrink_updateOutput"
  p_THNN_DoubleHardShrink_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleHardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_DoubleHardShrink_updateGradInput"
  p_THNN_DoubleHardShrink_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleHardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleHardTanh_updateOutput"
  p_THNN_DoubleHardTanh_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleHardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleHardTanh_updateGradInput"
  p_THNN_DoubleHardTanh_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleL1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleL1Cost_updateOutput"
  p_THNN_DoubleL1Cost_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleL1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleL1Cost_updateGradInput"
  p_THNN_DoubleL1Cost_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THNN_DoubleLeakyReLU_updateOutput"
  p_THNN_DoubleLeakyReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleLeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THNN_DoubleLeakyReLU_updateGradInput"
  p_THNN_DoubleLeakyReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleGRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THNN_DoubleGRUFused_updateOutput"
  p_THNN_DoubleGRUFused_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleGRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THNN_DoubleGRUFused_updateGradInput"
  p_THNN_DoubleGRUFused_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THNN_DoubleLSTMFused_updateOutput"
  p_THNN_DoubleLSTMFused_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THNN_DoubleLSTMFused_updateGradInput"
  p_THNN_DoubleLSTMFused_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THNN_DoubleLogSigmoid_updateOutput"
  p_THNN_DoubleLogSigmoid_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THNN_DoubleLogSigmoid_updateGradInput"
  p_THNN_DoubleLogSigmoid_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleLogSoftMax_updateOutput"
  p_THNN_DoubleLogSoftMax_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleLogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_DoubleLogSoftMax_updateGradInput"
  p_THNN_DoubleLogSoftMax_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleLookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h &THNN_DoubleLookupTable_accGradParameters"
  p_THNN_DoubleLookupTable_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleLookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h &THNN_DoubleLookupTable_renorm"
  p_THNN_DoubleLookupTable_renorm :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_DoubleMarginCriterion_updateOutput"
  p_THNN_DoubleMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ())

-- |p_THNN_DoubleMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_DoubleMarginCriterion_updateGradInput"
  p_THNN_DoubleMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> IO ())

-- |p_THNN_DoubleSoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMarginCriterion_updateOutput"
  p_THNN_DoubleSoftMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THNN_DoubleSoftMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMarginCriterion_updateGradInput"
  p_THNN_DoubleSoftMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THNN_DoubleMSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleMSECriterion_updateOutput"
  p_THNN_DoubleMSECriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleMSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleMSECriterion_updateGradInput"
  p_THNN_DoubleMSECriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleMultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h &THNN_DoubleMultiLabelMarginCriterion_updateOutput"
  p_THNN_DoubleMultiLabelMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THNN_DoubleMultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h &THNN_DoubleMultiLabelMarginCriterion_updateGradInput"
  p_THNN_DoubleMultiLabelMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> IO ())

-- |p_THNN_DoubleMultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THNN_DoubleMultiMarginCriterion_updateOutput"
  p_THNN_DoubleMultiMarginCriterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleMultiMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THNN_DoubleMultiMarginCriterion_updateGradInput"
  p_THNN_DoubleMultiMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CInt -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoublePReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_updateOutput"
  p_THNN_DoublePReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoublePReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_updateGradInput"
  p_THNN_DoublePReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoublePReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_accGradParameters"
  p_THNN_DoublePReLU_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleLinear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_updateOutput"
  p_THNN_DoubleLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLinear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_updateGradInput"
  p_THNN_DoubleLinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleLinear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_accGradParameters"
  p_THNN_DoubleLinear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleRReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THNN_DoubleRReLU_updateOutput"
  p_THNN_DoubleRReLU_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ())

-- |p_THNN_DoubleRReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THNN_DoubleRReLU_updateGradInput"
  p_THNN_DoubleRReLU_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleSigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleSigmoid_updateOutput"
  p_THNN_DoubleSigmoid_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleSigmoid_updateGradInput"
  p_THNN_DoubleSigmoid_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSmoothL1Criterion_updateOutput"
  p_THNN_DoubleSmoothL1Criterion_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleSmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSmoothL1Criterion_updateGradInput"
  p_THNN_DoubleSmoothL1Criterion_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMax_updateOutput"
  p_THNN_DoubleSoftMax_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMax_updateGradInput"
  p_THNN_DoubleSoftMax_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleSoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THNN_DoubleSoftPlus_updateOutput"
  p_THNN_DoubleSoftPlus_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THNN_DoubleSoftPlus_updateGradInput"
  p_THNN_DoubleSoftPlus_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_DoubleSoftShrink_updateOutput"
  p_THNN_DoubleSoftShrink_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleSoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_DoubleSoftShrink_updateGradInput"
  p_THNN_DoubleSoftShrink_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleIndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h &THNN_DoubleIndexLinear_updateOutput"
  p_THNN_DoubleIndexLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleIndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleIndexLinear_accGradParameters"
  p_THNN_DoubleIndexLinear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleIndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleIndexLinear_accUpdateGradParameters"
  p_THNN_DoubleIndexLinear_accUpdateGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleIndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h &THNN_DoubleIndexLinear_updateParameters"
  p_THNN_DoubleIndexLinear_updateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_updateOutput"
  p_THNN_DoubleSparseLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_accGradParameters"
  p_THNN_DoubleSparseLinear_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_zeroGradParameters"
  p_THNN_DoubleSparseLinear_zeroGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_updateParameters"
  p_THNN_DoubleSparseLinear_updateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleSparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyUpdateOutput"
  p_THNN_DoubleSparseLinear_legacyUpdateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyAccGradParameters"
  p_THNN_DoubleSparseLinear_legacyAccGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyZeroGradParameters"
  p_THNN_DoubleSparseLinear_legacyZeroGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyUpdateParameters"
  p_THNN_DoubleSparseLinear_legacyUpdateParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleSqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THNN_DoubleSqrt_updateOutput"
  p_THNN_DoubleSqrt_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THNN_DoubleSqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleSqrt_updateGradInput"
  p_THNN_DoubleSqrt_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSquare_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleSquare_updateOutput"
  p_THNN_DoubleSquare_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSquare_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleSquare_updateGradInput"
  p_THNN_DoubleSquare_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleTanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleTanh_updateOutput"
  p_THNN_DoubleTanh_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleTanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleTanh_updateGradInput"
  p_THNN_DoubleTanh_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleThreshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleThreshold_updateOutput"
  p_THNN_DoubleThreshold_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleThreshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleThreshold_updateGradInput"
  p_THNN_DoubleThreshold_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_DoubleTemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_updateOutput"
  p_THNN_DoubleTemporalConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_updateGradInput"
  p_THNN_DoubleTemporalConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_accGradParameters"
  p_THNN_DoubleTemporalConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleTemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalMaxPooling_updateOutput"
  p_THNN_DoubleTemporalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalMaxPooling_updateGradInput"
  p_THNN_DoubleTemporalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_updateOutput"
  p_THNN_DoubleTemporalSubSampling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_updateGradInput"
  p_THNN_DoubleTemporalSubSampling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_accGradParameters"
  p_THNN_DoubleTemporalSubSampling_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleTemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_updateOutput"
  p_THNN_DoubleTemporalRowConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleTemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_updateGradInput"
  p_THNN_DoubleTemporalRowConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleTemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_accGradParameters"
  p_THNN_DoubleTemporalRowConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- |p_THNN_DoubleTemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingNearest_updateOutput"
  p_THNN_DoubleTemporalUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleTemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingNearest_updateGradInput"
  p_THNN_DoubleTemporalUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleTemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingLinear_updateOutput"
  p_THNN_DoubleTemporalUpSamplingLinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleTemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingLinear_updateGradInput"
  p_THNN_DoubleTemporalUpSamplingLinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleBatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THNN_DoubleBatchNormalization_updateOutput"
  p_THNN_DoubleBatchNormalization_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleBatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THNN_DoubleBatchNormalization_backward"
  p_THNN_DoubleBatchNormalization_backward :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_updateOutput"
  p_THNN_DoubleSpatialConvolutionMap_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_updateGradInput"
  p_THNN_DoubleSpatialConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_accGradParameters"
  p_THNN_DoubleSpatialConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_updateOutput"
  p_THNN_DoubleSpatialConvolutionMM_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_updateGradInput"
  p_THNN_DoubleSpatialConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_accGradParameters"
  p_THNN_DoubleSpatialConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_updateOutput"
  p_THNN_DoubleSpatialConvolutionLocal_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THNN_DoubleSpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_updateGradInput"
  p_THNN_DoubleSpatialConvolutionLocal_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THNN_DoubleSpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_accGradParameters"
  p_THNN_DoubleSpatialConvolutionLocal_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput"
  p_THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput"
  p_THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput"
  p_THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput"
  p_THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAveragePooling_updateOutput"
  p_THNN_DoubleSpatialAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleSpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAveragePooling_updateGradInput"
  p_THNN_DoubleSpatialAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleSpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFractionalMaxPooling_updateOutput"
  p_THNN_DoubleSpatialFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFractionalMaxPooling_updateGradInput"
  p_THNN_DoubleSpatialFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_DoubleSpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_updateOutput"
  p_THNN_DoubleSpatialFullConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_updateGradInput"
  p_THNN_DoubleSpatialFullConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_accGradParameters"
  p_THNN_DoubleSpatialFullConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_updateOutput"
  p_THNN_DoubleSpatialFullConvolutionMap_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_updateGradInput"
  p_THNN_DoubleSpatialFullConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_accGradParameters"
  p_THNN_DoubleSpatialFullConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_updateOutput"
  p_THNN_DoubleSpatialDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_updateGradInput"
  p_THNN_DoubleSpatialDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_accGradParameters"
  p_THNN_DoubleSpatialDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_updateOutput"
  p_THNN_DoubleSpatialFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_updateGradInput"
  p_THNN_DoubleSpatialFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_accGradParameters"
  p_THNN_DoubleSpatialFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialMaxPooling_updateOutput"
  p_THNN_DoubleSpatialMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleSpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialMaxPooling_updateGradInput"
  p_THNN_DoubleSpatialMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleSpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedMaxPooling_updateOutput"
  p_THNN_DoubleSpatialDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedMaxPooling_updateGradInput"
  p_THNN_DoubleSpatialDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleSpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialMaxUnpooling_updateOutput"
  p_THNN_DoubleSpatialMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialMaxUnpooling_updateGradInput"
  p_THNN_DoubleSpatialMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_updateOutput"
  p_THNN_DoubleSpatialSubSampling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_updateGradInput"
  p_THNN_DoubleSpatialSubSampling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_accGradParameters"
  p_THNN_DoubleSpatialSubSampling_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleSpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingNearest_updateOutput"
  p_THNN_DoubleSpatialUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleSpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingNearest_updateGradInput"
  p_THNN_DoubleSpatialUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleSpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingBilinear_updateOutput"
  p_THNN_DoubleSpatialUpSamplingBilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingBilinear_updateGradInput"
  p_THNN_DoubleSpatialUpSamplingBilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialGridSamplerBilinear_updateOutput"
  p_THNN_DoubleSpatialGridSamplerBilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialGridSamplerBilinear_updateGradInput"
  p_THNN_DoubleSpatialGridSamplerBilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_Doubleunfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Doubleunfolded_acc"
  p_THNN_Doubleunfolded_acc :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_Doubleunfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Doubleunfolded_copy"
  p_THNN_Doubleunfolded_copy :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAveragePooling_updateOutput"
  p_THNN_DoubleVolumetricAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAveragePooling_updateGradInput"
  p_THNN_DoubleVolumetricAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_updateOutput"
  p_THNN_DoubleVolumetricConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_updateGradInput"
  p_THNN_DoubleVolumetricConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_accGradParameters"
  p_THNN_DoubleVolumetricConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleVolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_updateOutput"
  p_THNN_DoubleVolumetricConvolutionMM_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_updateGradInput"
  p_THNN_DoubleVolumetricConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_accGradParameters"
  p_THNN_DoubleVolumetricConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFractionalMaxPooling_updateOutput"
  p_THNN_DoubleVolumetricFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput"
  p_THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_DoubleVolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_updateOutput"
  p_THNN_DoubleVolumetricFullConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_updateGradInput"
  p_THNN_DoubleVolumetricFullConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_accGradParameters"
  p_THNN_DoubleVolumetricFullConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleVolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_updateOutput"
  p_THNN_DoubleVolumetricDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_updateGradInput"
  p_THNN_DoubleVolumetricDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_accGradParameters"
  p_THNN_DoubleVolumetricDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_updateOutput"
  p_THNN_DoubleVolumetricFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput"
  p_THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters"
  p_THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_DoubleVolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricMaxPooling_updateOutput"
  p_THNN_DoubleVolumetricMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricMaxPooling_updateGradInput"
  p_THNN_DoubleVolumetricMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedMaxPooling_updateOutput"
  p_THNN_DoubleVolumetricDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput"
  p_THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricMaxUnpooling_updateOutput"
  p_THNN_DoubleVolumetricMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricMaxUnpooling_updateGradInput"
  p_THNN_DoubleVolumetricMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput"
  p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  p_THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput"
  p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput"
  p_THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_DoubleSpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReflectionPadding_updateOutput"
  p_THNN_DoubleSpatialReflectionPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReflectionPadding_updateGradInput"
  p_THNN_DoubleSpatialReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReplicationPadding_updateOutput"
  p_THNN_DoubleSpatialReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleSpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReplicationPadding_updateGradInput"
  p_THNN_DoubleSpatialReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleFeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_DoubleFeatureLPPooling_updateOutput"
  p_THNN_DoubleFeatureLPPooling_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleFeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_DoubleFeatureLPPooling_updateGradInput"
  p_THNN_DoubleFeatureLPPooling_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_DoubleVolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricReplicationPadding_updateOutput"
  p_THNN_DoubleVolumetricReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricReplicationPadding_updateGradInput"
  p_THNN_DoubleVolumetricReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingNearest_updateOutput"
  p_THNN_DoubleVolumetricUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingNearest_updateGradInput"
  p_THNN_DoubleVolumetricUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput"
  p_THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput"
  p_THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReflectionPadding_updateOutput"
  p_THNN_DoubleTemporalReflectionPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReflectionPadding_updateGradInput"
  p_THNN_DoubleTemporalReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReplicationPadding_updateOutput"
  p_THNN_DoubleTemporalReplicationPadding_updateOutput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_DoubleTemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReplicationPadding_updateGradInput"
  p_THNN_DoubleTemporalReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHDoubleNNState) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())
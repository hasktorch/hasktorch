{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatNN (
    c_THNN_FloatAbs_updateOutput,
    c_THNN_FloatAbs_updateGradInput,
    c_THNN_FloatAbsCriterion_updateOutput,
    c_THNN_FloatAbsCriterion_updateGradInput,
    c_THNN_FloatBCECriterion_updateOutput,
    c_THNN_FloatBCECriterion_updateGradInput,
    c_THNN_FloatClassNLLCriterion_updateOutput,
    c_THNN_FloatClassNLLCriterion_updateGradInput,
    c_THNN_FloatSpatialClassNLLCriterion_updateOutput,
    c_THNN_FloatSpatialClassNLLCriterion_updateGradInput,
    c_THNN_FloatELU_updateOutput,
    c_THNN_FloatELU_updateGradInput,
    c_THNN_FloatDistKLDivCriterion_updateOutput,
    c_THNN_FloatDistKLDivCriterion_updateGradInput,
    c_THNN_FloatGatedLinear_updateOutput,
    c_THNN_FloatGatedLinear_updateGradInput,
    c_THNN_FloatHardShrink_updateOutput,
    c_THNN_FloatHardShrink_updateGradInput,
    c_THNN_FloatHardTanh_updateOutput,
    c_THNN_FloatHardTanh_updateGradInput,
    c_THNN_FloatL1Cost_updateOutput,
    c_THNN_FloatL1Cost_updateGradInput,
    c_THNN_FloatLeakyReLU_updateOutput,
    c_THNN_FloatLeakyReLU_updateGradInput,
    c_THNN_FloatGRUFused_updateOutput,
    c_THNN_FloatGRUFused_updateGradInput,
    c_THNN_FloatLSTMFused_updateOutput,
    c_THNN_FloatLSTMFused_updateGradInput,
    c_THNN_FloatLogSigmoid_updateOutput,
    c_THNN_FloatLogSigmoid_updateGradInput,
    c_THNN_FloatLogSoftMax_updateOutput,
    c_THNN_FloatLogSoftMax_updateGradInput,
    c_THNN_FloatLookupTable_accGradParameters,
    c_THNN_FloatLookupTable_renorm,
    c_THNN_FloatMarginCriterion_updateOutput,
    c_THNN_FloatMarginCriterion_updateGradInput,
    c_THNN_FloatSoftMarginCriterion_updateOutput,
    c_THNN_FloatSoftMarginCriterion_updateGradInput,
    c_THNN_FloatMSECriterion_updateOutput,
    c_THNN_FloatMSECriterion_updateGradInput,
    c_THNN_FloatMultiLabelMarginCriterion_updateOutput,
    c_THNN_FloatMultiLabelMarginCriterion_updateGradInput,
    c_THNN_FloatMultiMarginCriterion_updateOutput,
    c_THNN_FloatMultiMarginCriterion_updateGradInput,
    c_THNN_FloatPReLU_updateOutput,
    c_THNN_FloatPReLU_updateGradInput,
    c_THNN_FloatPReLU_accGradParameters,
    c_THNN_FloatLinear_updateOutput,
    c_THNN_FloatLinear_updateGradInput,
    c_THNN_FloatLinear_accGradParameters,
    c_THNN_FloatRReLU_updateOutput,
    c_THNN_FloatRReLU_updateGradInput,
    c_THNN_FloatSigmoid_updateOutput,
    c_THNN_FloatSigmoid_updateGradInput,
    c_THNN_FloatSmoothL1Criterion_updateOutput,
    c_THNN_FloatSmoothL1Criterion_updateGradInput,
    c_THNN_FloatSoftMax_updateOutput,
    c_THNN_FloatSoftMax_updateGradInput,
    c_THNN_FloatSoftPlus_updateOutput,
    c_THNN_FloatSoftPlus_updateGradInput,
    c_THNN_FloatSoftShrink_updateOutput,
    c_THNN_FloatSoftShrink_updateGradInput,
    c_THNN_FloatIndexLinear_updateOutput,
    c_THNN_FloatIndexLinear_accGradParameters,
    c_THNN_FloatIndexLinear_accUpdateGradParameters,
    c_THNN_FloatIndexLinear_updateParameters,
    c_THNN_FloatSparseLinear_updateOutput,
    c_THNN_FloatSparseLinear_accGradParameters,
    c_THNN_FloatSparseLinear_zeroGradParameters,
    c_THNN_FloatSparseLinear_updateParameters,
    c_THNN_FloatSparseLinear_legacyUpdateOutput,
    c_THNN_FloatSparseLinear_legacyAccGradParameters,
    c_THNN_FloatSparseLinear_legacyZeroGradParameters,
    c_THNN_FloatSparseLinear_legacyUpdateParameters,
    c_THNN_FloatSqrt_updateOutput,
    c_THNN_FloatSqrt_updateGradInput,
    c_THNN_FloatSquare_updateOutput,
    c_THNN_FloatSquare_updateGradInput,
    c_THNN_FloatTanh_updateOutput,
    c_THNN_FloatTanh_updateGradInput,
    c_THNN_FloatThreshold_updateOutput,
    c_THNN_FloatThreshold_updateGradInput,
    c_THNN_FloatTemporalConvolution_updateOutput,
    c_THNN_FloatTemporalConvolution_updateGradInput,
    c_THNN_FloatTemporalConvolution_accGradParameters,
    c_THNN_FloatTemporalMaxPooling_updateOutput,
    c_THNN_FloatTemporalMaxPooling_updateGradInput,
    c_THNN_FloatTemporalSubSampling_updateOutput,
    c_THNN_FloatTemporalSubSampling_updateGradInput,
    c_THNN_FloatTemporalSubSampling_accGradParameters,
    c_THNN_FloatTemporalRowConvolution_updateOutput,
    c_THNN_FloatTemporalRowConvolution_updateGradInput,
    c_THNN_FloatTemporalRowConvolution_accGradParameters,
    c_THNN_FloatTemporalUpSamplingNearest_updateOutput,
    c_THNN_FloatTemporalUpSamplingNearest_updateGradInput,
    c_THNN_FloatTemporalUpSamplingLinear_updateOutput,
    c_THNN_FloatTemporalUpSamplingLinear_updateGradInput,
    c_THNN_FloatBatchNormalization_updateOutput,
    c_THNN_FloatBatchNormalization_backward,
    c_THNN_FloatSpatialConvolutionMap_updateOutput,
    c_THNN_FloatSpatialConvolutionMap_updateGradInput,
    c_THNN_FloatSpatialConvolutionMap_accGradParameters,
    c_THNN_FloatSpatialConvolutionMM_updateOutput,
    c_THNN_FloatSpatialConvolutionMM_updateGradInput,
    c_THNN_FloatSpatialConvolutionMM_accGradParameters,
    c_THNN_FloatSpatialConvolutionLocal_updateOutput,
    c_THNN_FloatSpatialConvolutionLocal_updateGradInput,
    c_THNN_FloatSpatialConvolutionLocal_accGradParameters,
    c_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput,
    c_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput,
    c_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput,
    c_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput,
    c_THNN_FloatSpatialAveragePooling_updateOutput,
    c_THNN_FloatSpatialAveragePooling_updateGradInput,
    c_THNN_FloatSpatialFractionalMaxPooling_updateOutput,
    c_THNN_FloatSpatialFractionalMaxPooling_updateGradInput,
    c_THNN_FloatSpatialFullConvolution_updateOutput,
    c_THNN_FloatSpatialFullConvolution_updateGradInput,
    c_THNN_FloatSpatialFullConvolution_accGradParameters,
    c_THNN_FloatSpatialFullConvolutionMap_updateOutput,
    c_THNN_FloatSpatialFullConvolutionMap_updateGradInput,
    c_THNN_FloatSpatialFullConvolutionMap_accGradParameters,
    c_THNN_FloatSpatialDilatedConvolution_updateOutput,
    c_THNN_FloatSpatialDilatedConvolution_updateGradInput,
    c_THNN_FloatSpatialDilatedConvolution_accGradParameters,
    c_THNN_FloatSpatialFullDilatedConvolution_updateOutput,
    c_THNN_FloatSpatialFullDilatedConvolution_updateGradInput,
    c_THNN_FloatSpatialFullDilatedConvolution_accGradParameters,
    c_THNN_FloatSpatialMaxPooling_updateOutput,
    c_THNN_FloatSpatialMaxPooling_updateGradInput,
    c_THNN_FloatSpatialDilatedMaxPooling_updateOutput,
    c_THNN_FloatSpatialDilatedMaxPooling_updateGradInput,
    c_THNN_FloatSpatialMaxUnpooling_updateOutput,
    c_THNN_FloatSpatialMaxUnpooling_updateGradInput,
    c_THNN_FloatSpatialSubSampling_updateOutput,
    c_THNN_FloatSpatialSubSampling_updateGradInput,
    c_THNN_FloatSpatialSubSampling_accGradParameters,
    c_THNN_FloatSpatialUpSamplingNearest_updateOutput,
    c_THNN_FloatSpatialUpSamplingNearest_updateGradInput,
    c_THNN_FloatSpatialUpSamplingBilinear_updateOutput,
    c_THNN_FloatSpatialUpSamplingBilinear_updateGradInput,
    c_THNN_FloatSpatialGridSamplerBilinear_updateOutput,
    c_THNN_FloatSpatialGridSamplerBilinear_updateGradInput,
    c_THNN_Floatunfolded_acc,
    c_THNN_Floatunfolded_copy,
    c_THNN_FloatVolumetricAveragePooling_updateOutput,
    c_THNN_FloatVolumetricAveragePooling_updateGradInput,
    c_THNN_FloatVolumetricConvolution_updateOutput,
    c_THNN_FloatVolumetricConvolution_updateGradInput,
    c_THNN_FloatVolumetricConvolution_accGradParameters,
    c_THNN_FloatVolumetricConvolutionMM_updateOutput,
    c_THNN_FloatVolumetricConvolutionMM_updateGradInput,
    c_THNN_FloatVolumetricConvolutionMM_accGradParameters,
    c_THNN_FloatVolumetricFractionalMaxPooling_updateOutput,
    c_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput,
    c_THNN_FloatVolumetricFullConvolution_updateOutput,
    c_THNN_FloatVolumetricFullConvolution_updateGradInput,
    c_THNN_FloatVolumetricFullConvolution_accGradParameters,
    c_THNN_FloatVolumetricDilatedConvolution_updateOutput,
    c_THNN_FloatVolumetricDilatedConvolution_updateGradInput,
    c_THNN_FloatVolumetricDilatedConvolution_accGradParameters,
    c_THNN_FloatVolumetricFullDilatedConvolution_updateOutput,
    c_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput,
    c_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters,
    c_THNN_FloatVolumetricMaxPooling_updateOutput,
    c_THNN_FloatVolumetricMaxPooling_updateGradInput,
    c_THNN_FloatVolumetricDilatedMaxPooling_updateOutput,
    c_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput,
    c_THNN_FloatVolumetricMaxUnpooling_updateOutput,
    c_THNN_FloatVolumetricMaxUnpooling_updateGradInput,
    c_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput,
    c_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput,
    c_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput,
    c_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput,
    c_THNN_FloatSpatialReflectionPadding_updateOutput,
    c_THNN_FloatSpatialReflectionPadding_updateGradInput,
    c_THNN_FloatSpatialReplicationPadding_updateOutput,
    c_THNN_FloatSpatialReplicationPadding_updateGradInput,
    c_THNN_FloatFeatureLPPooling_updateOutput,
    c_THNN_FloatFeatureLPPooling_updateGradInput,
    c_THNN_FloatVolumetricReplicationPadding_updateOutput,
    c_THNN_FloatVolumetricReplicationPadding_updateGradInput,
    c_THNN_FloatVolumetricUpSamplingNearest_updateOutput,
    c_THNN_FloatVolumetricUpSamplingNearest_updateGradInput,
    c_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput,
    c_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput,
    c_THNN_FloatTemporalReflectionPadding_updateOutput,
    c_THNN_FloatTemporalReflectionPadding_updateGradInput,
    c_THNN_FloatTemporalReplicationPadding_updateOutput,
    c_THNN_FloatTemporalReplicationPadding_updateGradInput,
    p_THNN_FloatAbs_updateOutput,
    p_THNN_FloatAbs_updateGradInput,
    p_THNN_FloatAbsCriterion_updateOutput,
    p_THNN_FloatAbsCriterion_updateGradInput,
    p_THNN_FloatBCECriterion_updateOutput,
    p_THNN_FloatBCECriterion_updateGradInput,
    p_THNN_FloatClassNLLCriterion_updateOutput,
    p_THNN_FloatClassNLLCriterion_updateGradInput,
    p_THNN_FloatSpatialClassNLLCriterion_updateOutput,
    p_THNN_FloatSpatialClassNLLCriterion_updateGradInput,
    p_THNN_FloatELU_updateOutput,
    p_THNN_FloatELU_updateGradInput,
    p_THNN_FloatDistKLDivCriterion_updateOutput,
    p_THNN_FloatDistKLDivCriterion_updateGradInput,
    p_THNN_FloatGatedLinear_updateOutput,
    p_THNN_FloatGatedLinear_updateGradInput,
    p_THNN_FloatHardShrink_updateOutput,
    p_THNN_FloatHardShrink_updateGradInput,
    p_THNN_FloatHardTanh_updateOutput,
    p_THNN_FloatHardTanh_updateGradInput,
    p_THNN_FloatL1Cost_updateOutput,
    p_THNN_FloatL1Cost_updateGradInput,
    p_THNN_FloatLeakyReLU_updateOutput,
    p_THNN_FloatLeakyReLU_updateGradInput,
    p_THNN_FloatGRUFused_updateOutput,
    p_THNN_FloatGRUFused_updateGradInput,
    p_THNN_FloatLSTMFused_updateOutput,
    p_THNN_FloatLSTMFused_updateGradInput,
    p_THNN_FloatLogSigmoid_updateOutput,
    p_THNN_FloatLogSigmoid_updateGradInput,
    p_THNN_FloatLogSoftMax_updateOutput,
    p_THNN_FloatLogSoftMax_updateGradInput,
    p_THNN_FloatLookupTable_accGradParameters,
    p_THNN_FloatLookupTable_renorm,
    p_THNN_FloatMarginCriterion_updateOutput,
    p_THNN_FloatMarginCriterion_updateGradInput,
    p_THNN_FloatSoftMarginCriterion_updateOutput,
    p_THNN_FloatSoftMarginCriterion_updateGradInput,
    p_THNN_FloatMSECriterion_updateOutput,
    p_THNN_FloatMSECriterion_updateGradInput,
    p_THNN_FloatMultiLabelMarginCriterion_updateOutput,
    p_THNN_FloatMultiLabelMarginCriterion_updateGradInput,
    p_THNN_FloatMultiMarginCriterion_updateOutput,
    p_THNN_FloatMultiMarginCriterion_updateGradInput,
    p_THNN_FloatPReLU_updateOutput,
    p_THNN_FloatPReLU_updateGradInput,
    p_THNN_FloatPReLU_accGradParameters,
    p_THNN_FloatLinear_updateOutput,
    p_THNN_FloatLinear_updateGradInput,
    p_THNN_FloatLinear_accGradParameters,
    p_THNN_FloatRReLU_updateOutput,
    p_THNN_FloatRReLU_updateGradInput,
    p_THNN_FloatSigmoid_updateOutput,
    p_THNN_FloatSigmoid_updateGradInput,
    p_THNN_FloatSmoothL1Criterion_updateOutput,
    p_THNN_FloatSmoothL1Criterion_updateGradInput,
    p_THNN_FloatSoftMax_updateOutput,
    p_THNN_FloatSoftMax_updateGradInput,
    p_THNN_FloatSoftPlus_updateOutput,
    p_THNN_FloatSoftPlus_updateGradInput,
    p_THNN_FloatSoftShrink_updateOutput,
    p_THNN_FloatSoftShrink_updateGradInput,
    p_THNN_FloatIndexLinear_updateOutput,
    p_THNN_FloatIndexLinear_accGradParameters,
    p_THNN_FloatIndexLinear_accUpdateGradParameters,
    p_THNN_FloatIndexLinear_updateParameters,
    p_THNN_FloatSparseLinear_updateOutput,
    p_THNN_FloatSparseLinear_accGradParameters,
    p_THNN_FloatSparseLinear_zeroGradParameters,
    p_THNN_FloatSparseLinear_updateParameters,
    p_THNN_FloatSparseLinear_legacyUpdateOutput,
    p_THNN_FloatSparseLinear_legacyAccGradParameters,
    p_THNN_FloatSparseLinear_legacyZeroGradParameters,
    p_THNN_FloatSparseLinear_legacyUpdateParameters,
    p_THNN_FloatSqrt_updateOutput,
    p_THNN_FloatSqrt_updateGradInput,
    p_THNN_FloatSquare_updateOutput,
    p_THNN_FloatSquare_updateGradInput,
    p_THNN_FloatTanh_updateOutput,
    p_THNN_FloatTanh_updateGradInput,
    p_THNN_FloatThreshold_updateOutput,
    p_THNN_FloatThreshold_updateGradInput,
    p_THNN_FloatTemporalConvolution_updateOutput,
    p_THNN_FloatTemporalConvolution_updateGradInput,
    p_THNN_FloatTemporalConvolution_accGradParameters,
    p_THNN_FloatTemporalMaxPooling_updateOutput,
    p_THNN_FloatTemporalMaxPooling_updateGradInput,
    p_THNN_FloatTemporalSubSampling_updateOutput,
    p_THNN_FloatTemporalSubSampling_updateGradInput,
    p_THNN_FloatTemporalSubSampling_accGradParameters,
    p_THNN_FloatTemporalRowConvolution_updateOutput,
    p_THNN_FloatTemporalRowConvolution_updateGradInput,
    p_THNN_FloatTemporalRowConvolution_accGradParameters,
    p_THNN_FloatTemporalUpSamplingNearest_updateOutput,
    p_THNN_FloatTemporalUpSamplingNearest_updateGradInput,
    p_THNN_FloatTemporalUpSamplingLinear_updateOutput,
    p_THNN_FloatTemporalUpSamplingLinear_updateGradInput,
    p_THNN_FloatBatchNormalization_updateOutput,
    p_THNN_FloatBatchNormalization_backward,
    p_THNN_FloatSpatialConvolutionMap_updateOutput,
    p_THNN_FloatSpatialConvolutionMap_updateGradInput,
    p_THNN_FloatSpatialConvolutionMap_accGradParameters,
    p_THNN_FloatSpatialConvolutionMM_updateOutput,
    p_THNN_FloatSpatialConvolutionMM_updateGradInput,
    p_THNN_FloatSpatialConvolutionMM_accGradParameters,
    p_THNN_FloatSpatialConvolutionLocal_updateOutput,
    p_THNN_FloatSpatialConvolutionLocal_updateGradInput,
    p_THNN_FloatSpatialConvolutionLocal_accGradParameters,
    p_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput,
    p_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput,
    p_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput,
    p_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput,
    p_THNN_FloatSpatialAveragePooling_updateOutput,
    p_THNN_FloatSpatialAveragePooling_updateGradInput,
    p_THNN_FloatSpatialFractionalMaxPooling_updateOutput,
    p_THNN_FloatSpatialFractionalMaxPooling_updateGradInput,
    p_THNN_FloatSpatialFullConvolution_updateOutput,
    p_THNN_FloatSpatialFullConvolution_updateGradInput,
    p_THNN_FloatSpatialFullConvolution_accGradParameters,
    p_THNN_FloatSpatialFullConvolutionMap_updateOutput,
    p_THNN_FloatSpatialFullConvolutionMap_updateGradInput,
    p_THNN_FloatSpatialFullConvolutionMap_accGradParameters,
    p_THNN_FloatSpatialDilatedConvolution_updateOutput,
    p_THNN_FloatSpatialDilatedConvolution_updateGradInput,
    p_THNN_FloatSpatialDilatedConvolution_accGradParameters,
    p_THNN_FloatSpatialFullDilatedConvolution_updateOutput,
    p_THNN_FloatSpatialFullDilatedConvolution_updateGradInput,
    p_THNN_FloatSpatialFullDilatedConvolution_accGradParameters,
    p_THNN_FloatSpatialMaxPooling_updateOutput,
    p_THNN_FloatSpatialMaxPooling_updateGradInput,
    p_THNN_FloatSpatialDilatedMaxPooling_updateOutput,
    p_THNN_FloatSpatialDilatedMaxPooling_updateGradInput,
    p_THNN_FloatSpatialMaxUnpooling_updateOutput,
    p_THNN_FloatSpatialMaxUnpooling_updateGradInput,
    p_THNN_FloatSpatialSubSampling_updateOutput,
    p_THNN_FloatSpatialSubSampling_updateGradInput,
    p_THNN_FloatSpatialSubSampling_accGradParameters,
    p_THNN_FloatSpatialUpSamplingNearest_updateOutput,
    p_THNN_FloatSpatialUpSamplingNearest_updateGradInput,
    p_THNN_FloatSpatialUpSamplingBilinear_updateOutput,
    p_THNN_FloatSpatialUpSamplingBilinear_updateGradInput,
    p_THNN_FloatSpatialGridSamplerBilinear_updateOutput,
    p_THNN_FloatSpatialGridSamplerBilinear_updateGradInput,
    p_THNN_Floatunfolded_acc,
    p_THNN_Floatunfolded_copy,
    p_THNN_FloatVolumetricAveragePooling_updateOutput,
    p_THNN_FloatVolumetricAveragePooling_updateGradInput,
    p_THNN_FloatVolumetricConvolution_updateOutput,
    p_THNN_FloatVolumetricConvolution_updateGradInput,
    p_THNN_FloatVolumetricConvolution_accGradParameters,
    p_THNN_FloatVolumetricConvolutionMM_updateOutput,
    p_THNN_FloatVolumetricConvolutionMM_updateGradInput,
    p_THNN_FloatVolumetricConvolutionMM_accGradParameters,
    p_THNN_FloatVolumetricFractionalMaxPooling_updateOutput,
    p_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput,
    p_THNN_FloatVolumetricFullConvolution_updateOutput,
    p_THNN_FloatVolumetricFullConvolution_updateGradInput,
    p_THNN_FloatVolumetricFullConvolution_accGradParameters,
    p_THNN_FloatVolumetricDilatedConvolution_updateOutput,
    p_THNN_FloatVolumetricDilatedConvolution_updateGradInput,
    p_THNN_FloatVolumetricDilatedConvolution_accGradParameters,
    p_THNN_FloatVolumetricFullDilatedConvolution_updateOutput,
    p_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput,
    p_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters,
    p_THNN_FloatVolumetricMaxPooling_updateOutput,
    p_THNN_FloatVolumetricMaxPooling_updateGradInput,
    p_THNN_FloatVolumetricDilatedMaxPooling_updateOutput,
    p_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput,
    p_THNN_FloatVolumetricMaxUnpooling_updateOutput,
    p_THNN_FloatVolumetricMaxUnpooling_updateGradInput,
    p_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput,
    p_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput,
    p_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput,
    p_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput,
    p_THNN_FloatSpatialReflectionPadding_updateOutput,
    p_THNN_FloatSpatialReflectionPadding_updateGradInput,
    p_THNN_FloatSpatialReplicationPadding_updateOutput,
    p_THNN_FloatSpatialReplicationPadding_updateGradInput,
    p_THNN_FloatFeatureLPPooling_updateOutput,
    p_THNN_FloatFeatureLPPooling_updateGradInput,
    p_THNN_FloatVolumetricReplicationPadding_updateOutput,
    p_THNN_FloatVolumetricReplicationPadding_updateGradInput,
    p_THNN_FloatVolumetricUpSamplingNearest_updateOutput,
    p_THNN_FloatVolumetricUpSamplingNearest_updateGradInput,
    p_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput,
    p_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput,
    p_THNN_FloatTemporalReflectionPadding_updateOutput,
    p_THNN_FloatTemporalReflectionPadding_updateGradInput,
    p_THNN_FloatTemporalReplicationPadding_updateOutput,
    p_THNN_FloatTemporalReplicationPadding_updateGradInput) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THNN_FloatAbs_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_FloatAbs_updateOutput"
  c_THNN_FloatAbs_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatAbs_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatAbs_updateGradInput"
  c_THNN_FloatAbs_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatAbsCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatAbsCriterion_updateOutput"
  c_THNN_FloatAbsCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatAbsCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatAbsCriterion_updateGradInput"
  c_THNN_FloatAbsCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatBCECriterion_updateOutput : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h THNN_FloatBCECriterion_updateOutput"
  c_THNN_FloatBCECriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatBCECriterion_updateGradInput : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h THNN_FloatBCECriterion_updateGradInput"
  c_THNN_FloatBCECriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatClassNLLCriterion_updateOutput"
  c_THNN_FloatClassNLLCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_FloatClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatClassNLLCriterion_updateGradInput"
  c_THNN_FloatClassNLLCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_FloatSpatialClassNLLCriterion_updateOutput : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatSpatialClassNLLCriterion_updateOutput"
  c_THNN_FloatSpatialClassNLLCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_FloatSpatialClassNLLCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatSpatialClassNLLCriterion_updateGradInput"
  c_THNN_FloatSpatialClassNLLCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ()

-- |c_THNN_FloatELU_updateOutput : state input output alpha inplace -> void
foreign import ccall "THNN.h THNN_FloatELU_updateOutput"
  c_THNN_FloatELU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatELU_updateGradInput : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h THNN_FloatELU_updateGradInput"
  c_THNN_FloatELU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatDistKLDivCriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatDistKLDivCriterion_updateOutput"
  c_THNN_FloatDistKLDivCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatDistKLDivCriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatDistKLDivCriterion_updateGradInput"
  c_THNN_FloatDistKLDivCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatGatedLinear_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_FloatGatedLinear_updateOutput"
  c_THNN_FloatGatedLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatGatedLinear_updateGradInput : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THNN_FloatGatedLinear_updateGradInput"
  c_THNN_FloatGatedLinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatHardShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THNN_FloatHardShrink_updateOutput"
  c_THNN_FloatHardShrink_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatHardShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_FloatHardShrink_updateGradInput"
  c_THNN_FloatHardShrink_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatHardTanh_updateOutput : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_FloatHardTanh_updateOutput"
  c_THNN_FloatHardTanh_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatHardTanh_updateGradInput : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_FloatHardTanh_updateGradInput"
  c_THNN_FloatHardTanh_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatL1Cost_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_FloatL1Cost_updateOutput"
  c_THNN_FloatL1Cost_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatL1Cost_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatL1Cost_updateGradInput"
  c_THNN_FloatL1Cost_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLeakyReLU_updateOutput : state input output negval inplace -> void
foreign import ccall "THNN.h THNN_FloatLeakyReLU_updateOutput"
  c_THNN_FloatLeakyReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatLeakyReLU_updateGradInput : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THNN_FloatLeakyReLU_updateGradInput"
  c_THNN_FloatLeakyReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatGRUFused_updateOutput : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THNN_FloatGRUFused_updateOutput"
  c_THNN_FloatGRUFused_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatGRUFused_updateGradInput : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THNN_FloatGRUFused_updateGradInput"
  c_THNN_FloatGRUFused_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLSTMFused_updateOutput : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THNN_FloatLSTMFused_updateOutput"
  c_THNN_FloatLSTMFused_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLSTMFused_updateGradInput : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THNN_FloatLSTMFused_updateGradInput"
  c_THNN_FloatLSTMFused_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLogSigmoid_updateOutput : state input output buffer -> void
foreign import ccall "THNN.h THNN_FloatLogSigmoid_updateOutput"
  c_THNN_FloatLogSigmoid_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLogSigmoid_updateGradInput : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THNN_FloatLogSigmoid_updateGradInput"
  c_THNN_FloatLogSigmoid_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLogSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_FloatLogSoftMax_updateOutput"
  c_THNN_FloatLogSoftMax_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatLogSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_FloatLogSoftMax_updateGradInput"
  c_THNN_FloatLogSoftMax_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatLookupTable_accGradParameters : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h THNN_FloatLookupTable_accGradParameters"
  c_THNN_FloatLookupTable_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatLookupTable_renorm : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h THNN_FloatLookupTable_renorm"
  c_THNN_FloatLookupTable_renorm :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatMarginCriterion_updateOutput : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THNN_FloatMarginCriterion_updateOutput"
  c_THNN_FloatMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ()

-- |c_THNN_FloatMarginCriterion_updateGradInput : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THNN_FloatMarginCriterion_updateGradInput"
  c_THNN_FloatMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ()

-- |c_THNN_FloatSoftMarginCriterion_updateOutput : state input target output sizeAverage -> void
foreign import ccall "THNN.h THNN_FloatSoftMarginCriterion_updateOutput"
  c_THNN_FloatSoftMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THNN_FloatSoftMarginCriterion_updateGradInput : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h THNN_FloatSoftMarginCriterion_updateGradInput"
  c_THNN_FloatSoftMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THNN_FloatMSECriterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMSECriterion_updateOutput"
  c_THNN_FloatMSECriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatMSECriterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMSECriterion_updateGradInput"
  c_THNN_FloatMSECriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatMultiLabelMarginCriterion_updateOutput : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h THNN_FloatMultiLabelMarginCriterion_updateOutput"
  c_THNN_FloatMultiLabelMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THNN_FloatMultiLabelMarginCriterion_updateGradInput : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h THNN_FloatMultiLabelMarginCriterion_updateGradInput"
  c_THNN_FloatMultiLabelMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ()

-- |c_THNN_FloatMultiMarginCriterion_updateOutput : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h THNN_FloatMultiMarginCriterion_updateOutput"
  c_THNN_FloatMultiMarginCriterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatMultiMarginCriterion_updateGradInput : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h THNN_FloatMultiMarginCriterion_updateGradInput"
  c_THNN_FloatMultiMarginCriterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatPReLU_updateOutput : state input output weight -> void
foreign import ccall "THNN.h THNN_FloatPReLU_updateOutput"
  c_THNN_FloatPReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatPReLU_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_FloatPReLU_updateGradInput"
  c_THNN_FloatPReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatPReLU_accGradParameters : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THNN_FloatPReLU_accGradParameters"
  c_THNN_FloatPReLU_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatLinear_updateOutput : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THNN_FloatLinear_updateOutput"
  c_THNN_FloatLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLinear_updateGradInput : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_FloatLinear_updateGradInput"
  c_THNN_FloatLinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatLinear_accGradParameters : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THNN_FloatLinear_accGradParameters"
  c_THNN_FloatLinear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatRReLU_updateOutput : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THNN_FloatRReLU_updateOutput"
  c_THNN_FloatRReLU_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ()

-- |c_THNN_FloatRReLU_updateGradInput : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THNN_FloatRReLU_updateGradInput"
  c_THNN_FloatRReLU_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- |c_THNN_FloatSigmoid_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_FloatSigmoid_updateOutput"
  c_THNN_FloatSigmoid_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSigmoid_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatSigmoid_updateGradInput"
  c_THNN_FloatSigmoid_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSmoothL1Criterion_updateOutput : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSmoothL1Criterion_updateOutput"
  c_THNN_FloatSmoothL1Criterion_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatSmoothL1Criterion_updateGradInput : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSmoothL1Criterion_updateGradInput"
  c_THNN_FloatSmoothL1Criterion_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ()

-- |c_THNN_FloatSoftMax_updateOutput : state input output dim -> void
foreign import ccall "THNN.h THNN_FloatSoftMax_updateOutput"
  c_THNN_FloatSoftMax_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatSoftMax_updateGradInput : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_FloatSoftMax_updateGradInput"
  c_THNN_FloatSoftMax_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatSoftPlus_updateOutput : state input output beta threshold -> void
foreign import ccall "THNN.h THNN_FloatSoftPlus_updateOutput"
  c_THNN_FloatSoftPlus_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSoftPlus_updateGradInput : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THNN_FloatSoftPlus_updateGradInput"
  c_THNN_FloatSoftPlus_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSoftShrink_updateOutput : state input output lambda -> void
foreign import ccall "THNN.h THNN_FloatSoftShrink_updateOutput"
  c_THNN_FloatSoftShrink_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatSoftShrink_updateGradInput : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_FloatSoftShrink_updateGradInput"
  c_THNN_FloatSoftShrink_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatIndexLinear_updateOutput : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_updateOutput"
  c_THNN_FloatIndexLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatIndexLinear_accGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_accGradParameters"
  c_THNN_FloatIndexLinear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatIndexLinear_accUpdateGradParameters : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_accUpdateGradParameters"
  c_THNN_FloatIndexLinear_accUpdateGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatIndexLinear_updateParameters : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_updateParameters"
  c_THNN_FloatIndexLinear_updateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSparseLinear_updateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_updateOutput"
  c_THNN_FloatSparseLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSparseLinear_accGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_accGradParameters"
  c_THNN_FloatSparseLinear_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSparseLinear_zeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_zeroGradParameters"
  c_THNN_FloatSparseLinear_zeroGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSparseLinear_updateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_updateParameters"
  c_THNN_FloatSparseLinear_updateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatSparseLinear_legacyUpdateOutput : state input output weight bias -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyUpdateOutput"
  c_THNN_FloatSparseLinear_legacyUpdateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSparseLinear_legacyAccGradParameters : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyAccGradParameters"
  c_THNN_FloatSparseLinear_legacyAccGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSparseLinear_legacyZeroGradParameters : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyZeroGradParameters"
  c_THNN_FloatSparseLinear_legacyZeroGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSparseLinear_legacyUpdateParameters : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyUpdateParameters"
  c_THNN_FloatSparseLinear_legacyUpdateParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatSqrt_updateOutput : state input output eps -> void
foreign import ccall "THNN.h THNN_FloatSqrt_updateOutput"
  c_THNN_FloatSqrt_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THNN_FloatSqrt_updateGradInput : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatSqrt_updateGradInput"
  c_THNN_FloatSqrt_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSquare_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_FloatSquare_updateOutput"
  c_THNN_FloatSquare_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSquare_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatSquare_updateGradInput"
  c_THNN_FloatSquare_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatTanh_updateOutput : state input output -> void
foreign import ccall "THNN.h THNN_FloatTanh_updateOutput"
  c_THNN_FloatTanh_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatTanh_updateGradInput : state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatTanh_updateGradInput"
  c_THNN_FloatTanh_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatThreshold_updateOutput : state input output threshold val inplace -> void
foreign import ccall "THNN.h THNN_FloatThreshold_updateOutput"
  c_THNN_FloatThreshold_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatThreshold_updateGradInput : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THNN_FloatThreshold_updateGradInput"
  c_THNN_FloatThreshold_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ()

-- |c_THNN_FloatTemporalConvolution_updateOutput : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_updateOutput"
  c_THNN_FloatTemporalConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalConvolution_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_updateGradInput"
  c_THNN_FloatTemporalConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalConvolution_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_accGradParameters"
  c_THNN_FloatTemporalConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatTemporalMaxPooling_updateOutput : state input output indices kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalMaxPooling_updateOutput"
  c_THNN_FloatTemporalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalMaxPooling_updateGradInput : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalMaxPooling_updateGradInput"
  c_THNN_FloatTemporalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalSubSampling_updateOutput : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_updateOutput"
  c_THNN_FloatTemporalSubSampling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalSubSampling_updateGradInput : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_updateGradInput"
  c_THNN_FloatTemporalSubSampling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_accGradParameters"
  c_THNN_FloatTemporalSubSampling_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatTemporalRowConvolution_updateOutput : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_updateOutput"
  c_THNN_FloatTemporalRowConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatTemporalRowConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_updateGradInput"
  c_THNN_FloatTemporalRowConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatTemporalRowConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_accGradParameters"
  c_THNN_FloatTemporalRowConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- |c_THNN_FloatTemporalUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingNearest_updateOutput"
  c_THNN_FloatTemporalUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatTemporalUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingNearest_updateGradInput"
  c_THNN_FloatTemporalUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatTemporalUpSamplingLinear_updateOutput : state input output outputWidth -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingLinear_updateOutput"
  c_THNN_FloatTemporalUpSamplingLinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatTemporalUpSamplingLinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingLinear_updateGradInput"
  c_THNN_FloatTemporalUpSamplingLinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatBatchNormalization_updateOutput : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THNN_FloatBatchNormalization_updateOutput"
  c_THNN_FloatBatchNormalization_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatBatchNormalization_backward : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THNN_FloatBatchNormalization_backward"
  c_THNN_FloatBatchNormalization_backward :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ()

-- |c_THNN_FloatSpatialConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_updateOutput"
  c_THNN_FloatSpatialConvolutionMap_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_updateGradInput"
  c_THNN_FloatSpatialConvolutionMap_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_accGradParameters"
  c_THNN_FloatSpatialConvolutionMap_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialConvolutionMM_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_updateOutput"
  c_THNN_FloatSpatialConvolutionMM_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_updateGradInput"
  c_THNN_FloatSpatialConvolutionMM_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_accGradParameters"
  c_THNN_FloatSpatialConvolutionMM_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialConvolutionLocal_updateOutput : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_updateOutput"
  c_THNN_FloatSpatialConvolutionLocal_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THNN_FloatSpatialConvolutionLocal_updateGradInput : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_updateGradInput"
  c_THNN_FloatSpatialConvolutionLocal_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THNN_FloatSpatialConvolutionLocal_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_accGradParameters"
  c_THNN_FloatSpatialConvolutionLocal_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- |c_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveMaxPooling_updateOutput"
  c_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput"
  c_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput : state input output osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveAveragePooling_updateOutput"
  c_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput"
  c_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSpatialAveragePooling_updateOutput : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatSpatialAveragePooling_updateOutput"
  c_THNN_FloatSpatialAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_FloatSpatialAveragePooling_updateGradInput : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatSpatialAveragePooling_updateGradInput"
  c_THNN_FloatSpatialAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_FloatSpatialFractionalMaxPooling_updateOutput : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THNN_FloatSpatialFractionalMaxPooling_updateOutput"
  c_THNN_FloatSpatialFractionalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatSpatialFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THNN_FloatSpatialFractionalMaxPooling_updateGradInput"
  c_THNN_FloatSpatialFractionalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_FloatSpatialFullConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_updateOutput"
  c_THNN_FloatSpatialFullConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_updateGradInput"
  c_THNN_FloatSpatialFullConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_accGradParameters"
  c_THNN_FloatSpatialFullConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialFullConvolutionMap_updateOutput : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_updateOutput"
  c_THNN_FloatSpatialFullConvolutionMap_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullConvolutionMap_updateGradInput : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_updateGradInput"
  c_THNN_FloatSpatialFullConvolutionMap_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullConvolutionMap_accGradParameters : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_accGradParameters"
  c_THNN_FloatSpatialFullConvolutionMap_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_updateOutput"
  c_THNN_FloatSpatialDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_updateGradInput"
  c_THNN_FloatSpatialDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_accGradParameters"
  c_THNN_FloatSpatialDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialFullDilatedConvolution_updateOutput : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_updateOutput"
  c_THNN_FloatSpatialFullDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_updateGradInput"
  c_THNN_FloatSpatialFullDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_accGradParameters"
  c_THNN_FloatSpatialFullDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxPooling_updateOutput"
  c_THNN_FloatSpatialMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatSpatialMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxPooling_updateGradInput"
  c_THNN_FloatSpatialMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatSpatialDilatedMaxPooling_updateOutput : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedMaxPooling_updateOutput"
  c_THNN_FloatSpatialDilatedMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatSpatialDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedMaxPooling_updateGradInput"
  c_THNN_FloatSpatialDilatedMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatSpatialMaxUnpooling_updateOutput : state input output indices owidth oheight -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxUnpooling_updateOutput"
  c_THNN_FloatSpatialMaxUnpooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialMaxUnpooling_updateGradInput : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxUnpooling_updateGradInput"
  c_THNN_FloatSpatialMaxUnpooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialSubSampling_updateOutput : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_updateOutput"
  c_THNN_FloatSpatialSubSampling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialSubSampling_updateGradInput : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_updateGradInput"
  c_THNN_FloatSpatialSubSampling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialSubSampling_accGradParameters : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_accGradParameters"
  c_THNN_FloatSpatialSubSampling_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatSpatialUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingNearest_updateOutput"
  c_THNN_FloatSpatialUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatSpatialUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingNearest_updateGradInput"
  c_THNN_FloatSpatialUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatSpatialUpSamplingBilinear_updateOutput : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingBilinear_updateOutput"
  c_THNN_FloatSpatialUpSamplingBilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialUpSamplingBilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingBilinear_updateGradInput"
  c_THNN_FloatSpatialUpSamplingBilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialGridSamplerBilinear_updateOutput : state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialGridSamplerBilinear_updateOutput"
  c_THNN_FloatSpatialGridSamplerBilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatSpatialGridSamplerBilinear_updateGradInput : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialGridSamplerBilinear_updateGradInput"
  c_THNN_FloatSpatialGridSamplerBilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_Floatunfolded_acc : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Floatunfolded_acc"
  c_THNN_Floatunfolded_acc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_Floatunfolded_copy : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Floatunfolded_copy"
  c_THNN_Floatunfolded_copy :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricAveragePooling_updateOutput : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAveragePooling_updateOutput"
  c_THNN_FloatVolumetricAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_FloatVolumetricAveragePooling_updateGradInput : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAveragePooling_updateGradInput"
  c_THNN_FloatVolumetricAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- |c_THNN_FloatVolumetricConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_updateOutput"
  c_THNN_FloatVolumetricConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricConvolution_updateGradInput : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_updateGradInput"
  c_THNN_FloatVolumetricConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_accGradParameters"
  c_THNN_FloatVolumetricConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatVolumetricConvolutionMM_updateOutput : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_updateOutput"
  c_THNN_FloatVolumetricConvolutionMM_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricConvolutionMM_updateGradInput : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_updateGradInput"
  c_THNN_FloatVolumetricConvolutionMM_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricConvolutionMM_accGradParameters : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_accGradParameters"
  c_THNN_FloatVolumetricConvolutionMM_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatVolumetricFractionalMaxPooling_updateOutput : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFractionalMaxPooling_updateOutput"
  c_THNN_FloatVolumetricFractionalMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFractionalMaxPooling_updateGradInput"
  c_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_FloatVolumetricFullConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_updateOutput"
  c_THNN_FloatVolumetricFullConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricFullConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_updateGradInput"
  c_THNN_FloatVolumetricFullConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricFullConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_accGradParameters"
  c_THNN_FloatVolumetricFullConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatVolumetricDilatedConvolution_updateOutput : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_updateOutput"
  c_THNN_FloatVolumetricDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricDilatedConvolution_updateGradInput : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_updateGradInput"
  c_THNN_FloatVolumetricDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_accGradParameters"
  c_THNN_FloatVolumetricDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatVolumetricFullDilatedConvolution_updateOutput : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_updateOutput"
  c_THNN_FloatVolumetricFullDilatedConvolution_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_updateGradInput"
  c_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_accGradParameters"
  c_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- |c_THNN_FloatVolumetricMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxPooling_updateOutput"
  c_THNN_FloatVolumetricMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatVolumetricMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxPooling_updateGradInput"
  c_THNN_FloatVolumetricMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatVolumetricDilatedMaxPooling_updateOutput : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedMaxPooling_updateOutput"
  c_THNN_FloatVolumetricDilatedMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedMaxPooling_updateGradInput"
  c_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatVolumetricMaxUnpooling_updateOutput : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxUnpooling_updateOutput"
  c_THNN_FloatVolumetricMaxUnpooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricMaxUnpooling_updateGradInput : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxUnpooling_updateGradInput"
  c_THNN_FloatVolumetricMaxUnpooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput"
  c_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput : state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput"
  c_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput"
  c_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput"
  c_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ()

-- |c_THNN_FloatSpatialReflectionPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_FloatSpatialReflectionPadding_updateOutput"
  c_THNN_FloatSpatialReflectionPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_FloatSpatialReflectionPadding_updateGradInput"
  c_THNN_FloatSpatialReflectionPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialReplicationPadding_updateOutput : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_FloatSpatialReplicationPadding_updateOutput"
  c_THNN_FloatSpatialReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatSpatialReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h THNN_FloatSpatialReplicationPadding_updateGradInput"
  c_THNN_FloatSpatialReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatFeatureLPPooling_updateOutput : state input output power width stride batchMode -> void
foreign import ccall "THNN.h THNN_FloatFeatureLPPooling_updateOutput"
  c_THNN_FloatFeatureLPPooling_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatFeatureLPPooling_updateGradInput : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THNN_FloatFeatureLPPooling_updateGradInput"
  c_THNN_FloatFeatureLPPooling_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- |c_THNN_FloatVolumetricReplicationPadding_updateOutput : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THNN_FloatVolumetricReplicationPadding_updateOutput"
  c_THNN_FloatVolumetricReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricReplicationPadding_updateGradInput : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h THNN_FloatVolumetricReplicationPadding_updateGradInput"
  c_THNN_FloatVolumetricReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricUpSamplingNearest_updateOutput : state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingNearest_updateOutput"
  c_THNN_FloatVolumetricUpSamplingNearest_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatVolumetricUpSamplingNearest_updateGradInput : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingNearest_updateGradInput"
  c_THNN_FloatVolumetricUpSamplingNearest_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingTrilinear_updateOutput"
  c_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput"
  c_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalReflectionPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THNN_FloatTemporalReflectionPadding_updateOutput"
  c_THNN_FloatTemporalReflectionPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalReflectionPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THNN_FloatTemporalReflectionPadding_updateGradInput"
  c_THNN_FloatTemporalReflectionPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalReplicationPadding_updateOutput : state input output pad_l pad_r -> void
foreign import ccall "THNN.h THNN_FloatTemporalReplicationPadding_updateOutput"
  c_THNN_FloatTemporalReplicationPadding_updateOutput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THNN_FloatTemporalReplicationPadding_updateGradInput : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h THNN_FloatTemporalReplicationPadding_updateGradInput"
  c_THNN_FloatTemporalReplicationPadding_updateGradInput :: (Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |p_THNN_FloatAbs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatAbs_updateOutput"
  p_THNN_FloatAbs_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatAbs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatAbs_updateGradInput"
  p_THNN_FloatAbs_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatAbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatAbsCriterion_updateOutput"
  p_THNN_FloatAbsCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatAbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatAbsCriterion_updateGradInput"
  p_THNN_FloatAbsCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatBCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights -> void
foreign import ccall "THNN.h &THNN_FloatBCECriterion_updateOutput"
  p_THNN_FloatBCECriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatBCECriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage weights -> void
foreign import ccall "THNN.h &THNN_FloatBCECriterion_updateGradInput"
  p_THNN_FloatBCECriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatClassNLLCriterion_updateOutput"
  p_THNN_FloatClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_FloatClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatClassNLLCriterion_updateGradInput"
  p_THNN_FloatClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_FloatSpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatSpatialClassNLLCriterion_updateOutput"
  p_THNN_FloatSpatialClassNLLCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_FloatSpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatSpatialClassNLLCriterion_updateGradInput"
  p_THNN_FloatSpatialClassNLLCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CBool -> IO ())

-- |p_THNN_FloatELU_updateOutput : Pointer to function : state input output alpha inplace -> void
foreign import ccall "THNN.h &THNN_FloatELU_updateOutput"
  p_THNN_FloatELU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha inplace -> void
foreign import ccall "THNN.h &THNN_FloatELU_updateGradInput"
  p_THNN_FloatELU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatDistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatDistKLDivCriterion_updateOutput"
  p_THNN_FloatDistKLDivCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatDistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatDistKLDivCriterion_updateGradInput"
  p_THNN_FloatDistKLDivCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatGatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatGatedLinear_updateOutput"
  p_THNN_FloatGatedLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatGatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THNN_FloatGatedLinear_updateGradInput"
  p_THNN_FloatGatedLinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatHardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_FloatHardShrink_updateOutput"
  p_THNN_FloatHardShrink_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatHardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_FloatHardShrink_updateGradInput"
  p_THNN_FloatHardShrink_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatHardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_FloatHardTanh_updateOutput"
  p_THNN_FloatHardTanh_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatHardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_FloatHardTanh_updateGradInput"
  p_THNN_FloatHardTanh_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatL1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatL1Cost_updateOutput"
  p_THNN_FloatL1Cost_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatL1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatL1Cost_updateGradInput"
  p_THNN_FloatL1Cost_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THNN_FloatLeakyReLU_updateOutput"
  p_THNN_FloatLeakyReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatLeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THNN_FloatLeakyReLU_updateGradInput"
  p_THNN_FloatLeakyReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatGRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THNN_FloatGRUFused_updateOutput"
  p_THNN_FloatGRUFused_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatGRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THNN_FloatGRUFused_updateGradInput"
  p_THNN_FloatGRUFused_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THNN_FloatLSTMFused_updateOutput"
  p_THNN_FloatLSTMFused_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THNN_FloatLSTMFused_updateGradInput"
  p_THNN_FloatLSTMFused_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THNN_FloatLogSigmoid_updateOutput"
  p_THNN_FloatLogSigmoid_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THNN_FloatLogSigmoid_updateGradInput"
  p_THNN_FloatLogSigmoid_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatLogSoftMax_updateOutput"
  p_THNN_FloatLogSoftMax_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatLogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_FloatLogSoftMax_updateGradInput"
  p_THNN_FloatLogSoftMax_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatLookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h &THNN_FloatLookupTable_accGradParameters"
  p_THNN_FloatLookupTable_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIntegerTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CBool -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatLookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h &THNN_FloatLookupTable_renorm"
  p_THNN_FloatLookupTable_renorm :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_FloatMarginCriterion_updateOutput"
  p_THNN_FloatMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ())

-- |p_THNN_FloatMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_FloatMarginCriterion_updateGradInput"
  p_THNN_FloatMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> IO ())

-- |p_THNN_FloatSoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage -> void
foreign import ccall "THNN.h &THNN_FloatSoftMarginCriterion_updateOutput"
  p_THNN_FloatSoftMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THNN_FloatSoftMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage -> void
foreign import ccall "THNN.h &THNN_FloatSoftMarginCriterion_updateGradInput"
  p_THNN_FloatSoftMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THNN_FloatMSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMSECriterion_updateOutput"
  p_THNN_FloatMSECriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatMSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMSECriterion_updateGradInput"
  p_THNN_FloatMSECriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatMultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output isTarget sizeAverage -> void
foreign import ccall "THNN.h &THNN_FloatMultiLabelMarginCriterion_updateOutput"
  p_THNN_FloatMultiLabelMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THNN_FloatMultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradInput isTarget sizeAverage -> void
foreign import ccall "THNN.h &THNN_FloatMultiLabelMarginCriterion_updateGradInput"
  p_THNN_FloatMultiLabelMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> IO ())

-- |p_THNN_FloatMultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THNN_FloatMultiMarginCriterion_updateOutput"
  p_THNN_FloatMultiMarginCriterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatMultiMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage p weights margin -> void
foreign import ccall "THNN.h &THNN_FloatMultiMarginCriterion_updateGradInput"
  p_THNN_FloatMultiMarginCriterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> CBool -> CInt -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatPReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_updateOutput"
  p_THNN_FloatPReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatPReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_updateGradInput"
  p_THNN_FloatPReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatPReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_accGradParameters"
  p_THNN_FloatPReLU_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatLinear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THNN_FloatLinear_updateOutput"
  p_THNN_FloatLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLinear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_FloatLinear_updateGradInput"
  p_THNN_FloatLinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatLinear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THNN_FloatLinear_accGradParameters"
  p_THNN_FloatLinear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatRReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THNN_FloatRReLU_updateOutput"
  p_THNN_FloatRReLU_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> Ptr CTHGenerator -> IO ())

-- |p_THNN_FloatRReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THNN_FloatRReLU_updateGradInput"
  p_THNN_FloatRReLU_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- |p_THNN_FloatSigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatSigmoid_updateOutput"
  p_THNN_FloatSigmoid_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatSigmoid_updateGradInput"
  p_THNN_FloatSigmoid_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSmoothL1Criterion_updateOutput"
  p_THNN_FloatSmoothL1Criterion_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatSmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSmoothL1Criterion_updateGradInput"
  p_THNN_FloatSmoothL1Criterion_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CBool -> IO ())

-- |p_THNN_FloatSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatSoftMax_updateOutput"
  p_THNN_FloatSoftMax_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_FloatSoftMax_updateGradInput"
  p_THNN_FloatSoftMax_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatSoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THNN_FloatSoftPlus_updateOutput"
  p_THNN_FloatSoftPlus_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THNN_FloatSoftPlus_updateGradInput"
  p_THNN_FloatSoftPlus_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_FloatSoftShrink_updateOutput"
  p_THNN_FloatSoftShrink_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatSoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_FloatSoftShrink_updateGradInput"
  p_THNN_FloatSoftShrink_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatIndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_updateOutput"
  p_THNN_FloatIndexLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatIndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_accGradParameters"
  p_THNN_FloatIndexLinear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatIndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_accUpdateGradParameters"
  p_THNN_FloatIndexLinear_accUpdateGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHIndexTensor) -> CLLong -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatIndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_updateParameters"
  p_THNN_FloatIndexLinear_updateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> (Ptr CTHIndexTensor) -> CLLong -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_updateOutput"
  p_THNN_FloatSparseLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_accGradParameters"
  p_THNN_FloatSparseLinear_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_zeroGradParameters"
  p_THNN_FloatSparseLinear_zeroGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_updateParameters"
  p_THNN_FloatSparseLinear_updateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatSparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyUpdateOutput"
  p_THNN_FloatSparseLinear_legacyUpdateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyAccGradParameters"
  p_THNN_FloatSparseLinear_legacyAccGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyZeroGradParameters"
  p_THNN_FloatSparseLinear_legacyZeroGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyUpdateParameters"
  p_THNN_FloatSparseLinear_legacyUpdateParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatSqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THNN_FloatSqrt_updateOutput"
  p_THNN_FloatSqrt_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THNN_FloatSqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatSqrt_updateGradInput"
  p_THNN_FloatSqrt_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSquare_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatSquare_updateOutput"
  p_THNN_FloatSquare_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSquare_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatSquare_updateGradInput"
  p_THNN_FloatSquare_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatTanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatTanh_updateOutput"
  p_THNN_FloatTanh_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatTanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatTanh_updateGradInput"
  p_THNN_FloatTanh_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatThreshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THNN_FloatThreshold_updateOutput"
  p_THNN_FloatThreshold_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatThreshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THNN_FloatThreshold_updateGradInput"
  p_THNN_FloatThreshold_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CBool -> IO ())

-- |p_THNN_FloatTemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_updateOutput"
  p_THNN_FloatTemporalConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_updateGradInput"
  p_THNN_FloatTemporalConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_accGradParameters"
  p_THNN_FloatTemporalConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatTemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalMaxPooling_updateOutput"
  p_THNN_FloatTemporalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalMaxPooling_updateGradInput"
  p_THNN_FloatTemporalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_updateOutput"
  p_THNN_FloatTemporalSubSampling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_updateGradInput"
  p_THNN_FloatTemporalSubSampling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_accGradParameters"
  p_THNN_FloatTemporalSubSampling_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatTemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_updateOutput"
  p_THNN_FloatTemporalRowConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatTemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_updateGradInput"
  p_THNN_FloatTemporalRowConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatTemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_accGradParameters"
  p_THNN_FloatTemporalRowConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- |p_THNN_FloatTemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingNearest_updateOutput"
  p_THNN_FloatTemporalUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatTemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingNearest_updateGradInput"
  p_THNN_FloatTemporalUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatTemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingLinear_updateOutput"
  p_THNN_FloatTemporalUpSamplingLinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatTemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingLinear_updateGradInput"
  p_THNN_FloatTemporalUpSamplingLinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatBatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THNN_FloatBatchNormalization_updateOutput"
  p_THNN_FloatBatchNormalization_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatBatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THNN_FloatBatchNormalization_backward"
  p_THNN_FloatBatchNormalization_backward :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CBool -> CDouble -> CDouble -> IO ())

-- |p_THNN_FloatSpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_updateOutput"
  p_THNN_FloatSpatialConvolutionMap_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_updateGradInput"
  p_THNN_FloatSpatialConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_accGradParameters"
  p_THNN_FloatSpatialConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_updateOutput"
  p_THNN_FloatSpatialConvolutionMM_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_updateGradInput"
  p_THNN_FloatSpatialConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_accGradParameters"
  p_THNN_FloatSpatialConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_updateOutput"
  p_THNN_FloatSpatialConvolutionLocal_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THNN_FloatSpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_updateGradInput"
  p_THNN_FloatSpatialConvolutionLocal_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THNN_FloatSpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_accGradParameters"
  p_THNN_FloatSpatialConvolutionLocal_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- |p_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveMaxPooling_updateOutput"
  p_THNN_FloatSpatialAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput"
  p_THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveAveragePooling_updateOutput"
  p_THNN_FloatSpatialAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput"
  p_THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAveragePooling_updateOutput"
  p_THNN_FloatSpatialAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_FloatSpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAveragePooling_updateGradInput"
  p_THNN_FloatSpatialAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_FloatSpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFractionalMaxPooling_updateOutput"
  p_THNN_FloatSpatialFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatSpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFractionalMaxPooling_updateGradInput"
  p_THNN_FloatSpatialFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_FloatSpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_updateOutput"
  p_THNN_FloatSpatialFullConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_updateGradInput"
  p_THNN_FloatSpatialFullConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_accGradParameters"
  p_THNN_FloatSpatialFullConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_updateOutput"
  p_THNN_FloatSpatialFullConvolutionMap_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_updateGradInput"
  p_THNN_FloatSpatialFullConvolutionMap_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_accGradParameters"
  p_THNN_FloatSpatialFullConvolutionMap_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_updateOutput"
  p_THNN_FloatSpatialDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_updateGradInput"
  p_THNN_FloatSpatialDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_accGradParameters"
  p_THNN_FloatSpatialDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_updateOutput"
  p_THNN_FloatSpatialFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_updateGradInput"
  p_THNN_FloatSpatialFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_accGradParameters"
  p_THNN_FloatSpatialFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxPooling_updateOutput"
  p_THNN_FloatSpatialMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatSpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxPooling_updateGradInput"
  p_THNN_FloatSpatialMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatSpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedMaxPooling_updateOutput"
  p_THNN_FloatSpatialDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatSpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedMaxPooling_updateGradInput"
  p_THNN_FloatSpatialDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatSpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxUnpooling_updateOutput"
  p_THNN_FloatSpatialMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxUnpooling_updateGradInput"
  p_THNN_FloatSpatialMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_updateOutput"
  p_THNN_FloatSpatialSubSampling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_updateGradInput"
  p_THNN_FloatSpatialSubSampling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_accGradParameters"
  p_THNN_FloatSpatialSubSampling_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatSpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingNearest_updateOutput"
  p_THNN_FloatSpatialUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatSpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingNearest_updateGradInput"
  p_THNN_FloatSpatialUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatSpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingBilinear_updateOutput"
  p_THNN_FloatSpatialUpSamplingBilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingBilinear_updateGradInput"
  p_THNN_FloatSpatialUpSamplingBilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialGridSamplerBilinear_updateOutput"
  p_THNN_FloatSpatialGridSamplerBilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatSpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialGridSamplerBilinear_updateGradInput"
  p_THNN_FloatSpatialGridSamplerBilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_Floatunfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Floatunfolded_acc"
  p_THNN_Floatunfolded_acc :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_Floatunfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Floatunfolded_copy"
  p_THNN_Floatunfolded_copy :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAveragePooling_updateOutput"
  p_THNN_FloatVolumetricAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_FloatVolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAveragePooling_updateGradInput"
  p_THNN_FloatVolumetricAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- |p_THNN_FloatVolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_updateOutput"
  p_THNN_FloatVolumetricConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_updateGradInput"
  p_THNN_FloatVolumetricConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_accGradParameters"
  p_THNN_FloatVolumetricConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatVolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_updateOutput"
  p_THNN_FloatVolumetricConvolutionMM_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_updateGradInput"
  p_THNN_FloatVolumetricConvolutionMM_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_accGradParameters"
  p_THNN_FloatVolumetricConvolutionMM_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatVolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFractionalMaxPooling_updateOutput"
  p_THNN_FloatVolumetricFractionalMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFractionalMaxPooling_updateGradInput"
  p_THNN_FloatVolumetricFractionalMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_FloatVolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_updateOutput"
  p_THNN_FloatVolumetricFullConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_updateGradInput"
  p_THNN_FloatVolumetricFullConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_accGradParameters"
  p_THNN_FloatVolumetricFullConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatVolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_updateOutput"
  p_THNN_FloatVolumetricDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_updateGradInput"
  p_THNN_FloatVolumetricDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_accGradParameters"
  p_THNN_FloatVolumetricDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatVolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_updateOutput"
  p_THNN_FloatVolumetricFullDilatedConvolution_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_updateGradInput"
  p_THNN_FloatVolumetricFullDilatedConvolution_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_accGradParameters"
  p_THNN_FloatVolumetricFullDilatedConvolution_accGradParameters :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- |p_THNN_FloatVolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxPooling_updateOutput"
  p_THNN_FloatVolumetricMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatVolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxPooling_updateGradInput"
  p_THNN_FloatVolumetricMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatVolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedMaxPooling_updateOutput"
  p_THNN_FloatVolumetricDilatedMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedMaxPooling_updateGradInput"
  p_THNN_FloatVolumetricDilatedMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatVolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxUnpooling_updateOutput"
  p_THNN_FloatVolumetricMaxUnpooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxUnpooling_updateGradInput"
  p_THNN_FloatVolumetricMaxUnpooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput"
  p_THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput"
  p_THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput"
  p_THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput"
  p_THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHIndexTensor) -> IO ())

-- |p_THNN_FloatSpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReflectionPadding_updateOutput"
  p_THNN_FloatSpatialReflectionPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReflectionPadding_updateGradInput"
  p_THNN_FloatSpatialReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReplicationPadding_updateOutput"
  p_THNN_FloatSpatialReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatSpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r pad_t pad_b -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReplicationPadding_updateGradInput"
  p_THNN_FloatSpatialReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatFeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_FloatFeatureLPPooling_updateOutput"
  p_THNN_FloatFeatureLPPooling_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatFeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_FloatFeatureLPPooling_updateGradInput"
  p_THNN_FloatFeatureLPPooling_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- |p_THNN_FloatVolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricReplicationPadding_updateOutput"
  p_THNN_FloatVolumetricReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricReplicationPadding_updateGradInput"
  p_THNN_FloatVolumetricReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingNearest_updateOutput"
  p_THNN_FloatVolumetricUpSamplingNearest_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatVolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingNearest_updateGradInput"
  p_THNN_FloatVolumetricUpSamplingNearest_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingTrilinear_updateOutput"
  p_THNN_FloatVolumetricUpSamplingTrilinear_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput"
  p_THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReflectionPadding_updateOutput"
  p_THNN_FloatTemporalReflectionPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReflectionPadding_updateGradInput"
  p_THNN_FloatTemporalReflectionPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReplicationPadding_updateOutput"
  p_THNN_FloatTemporalReplicationPadding_updateOutput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THNN_FloatTemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_l pad_r -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReplicationPadding_updateGradInput"
  p_THNN_FloatTemporalReplicationPadding_updateGradInput :: FunPtr ((Ptr CTHFloatNNState) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())
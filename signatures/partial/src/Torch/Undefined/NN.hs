module Torch.Undefined.NN where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Undefined.Types.NN

c_Abs_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Abs_updateOutput = undefined
c_Abs_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Abs_updateGradInput = undefined
c_AbsCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_AbsCriterion_updateOutput = undefined
c_AbsCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_AbsCriterion_updateGradInput = undefined
c_BCECriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> CBool -> IO ()
c_BCECriterion_updateOutput = undefined
c_BCECriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> CBool -> IO ()
c_BCECriterion_updateGradInput = undefined
c_ELU_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> IO ()
c_ELU_updateOutput = undefined
c_ELU_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_ELU_updateGradInput = undefined
c_DistKLDivCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_DistKLDivCriterion_updateOutput = undefined
c_DistKLDivCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_DistKLDivCriterion_updateGradInput = undefined
c_GatedLinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_GatedLinear_updateOutput = undefined
c_GatedLinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_GatedLinear_updateGradInput = undefined
c_HardTanh_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> IO ()
c_HardTanh_updateOutput = undefined
c_HardTanh_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> IO ()
c_HardTanh_updateGradInput = undefined
c_Im2Col_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_Im2Col_updateOutput = undefined
c_Im2Col_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_Im2Col_updateGradInput = undefined
c_Col2Im_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_Col2Im_updateOutput = undefined
c_L1Cost_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_L1Cost_updateOutput = undefined
c_L1Cost_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_L1Cost_updateGradInput = undefined
c_LeakyReLU_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CBool -> IO ()
c_LeakyReLU_updateOutput = undefined
c_LeakyReLU_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CBool -> IO ()
c_LeakyReLU_updateGradInput = undefined
c_GRUFused_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_GRUFused_updateOutput = undefined
c_GRUFused_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_GRUFused_updateGradInput = undefined
c_LSTMFused_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_LSTMFused_updateOutput = undefined
c_LSTMFused_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_LSTMFused_updateGradInput = undefined
c_LogSigmoid_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_LogSigmoid_updateOutput = undefined
c_LogSigmoid_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_LogSigmoid_updateGradInput = undefined
c_LogSoftMax_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDim -> IO ()
c_LogSoftMax_updateOutput = undefined
c_LogSoftMax_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDim -> IO ()
c_LogSoftMax_updateGradInput = undefined
c_MarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CDouble -> IO ()
c_MarginCriterion_updateOutput = undefined
c_MarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CDouble -> IO ()
c_MarginCriterion_updateGradInput = undefined
c_SoftMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_SoftMarginCriterion_updateOutput = undefined
c_SoftMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_SoftMarginCriterion_updateGradInput = undefined
c_MSECriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MSECriterion_updateOutput = undefined
c_MSECriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MSECriterion_updateGradInput = undefined
c_PReLU_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_PReLU_updateOutput = undefined
c_PReLU_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_PReLU_updateGradInput = undefined
c_PReLU_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> IO ()
c_PReLU_accGradParameters = undefined
c_RReLU_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr CNNGenerator -> IO ()
c_RReLU_updateOutput = undefined
c_RReLU_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()
c_RReLU_updateGradInput = undefined
c_Sigmoid_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Sigmoid_updateOutput = undefined
c_Sigmoid_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Sigmoid_updateGradInput = undefined
c_SmoothL1Criterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_SmoothL1Criterion_updateOutput = undefined
c_SmoothL1Criterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_SmoothL1Criterion_updateGradInput = undefined
c_SoftMax_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDim -> IO ()
c_SoftMax_updateOutput = undefined
c_SoftMax_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDim -> IO ()
c_SoftMax_updateGradInput = undefined
c_SoftPlus_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_SoftPlus_updateOutput = undefined
c_SoftPlus_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_SoftPlus_updateGradInput = undefined
c_SoftShrink_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> IO ()
c_SoftShrink_updateOutput = undefined
c_SoftShrink_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> IO ()
c_SoftShrink_updateGradInput = undefined
c_SparseLinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_SparseLinear_updateOutput = undefined
c_SparseLinear_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_SparseLinear_accGradParameters = undefined
c_SparseLinear_zeroGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_SparseLinear_zeroGradParameters = undefined
c_SparseLinear_updateParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> IO ()
c_SparseLinear_updateParameters = undefined
c_SparseLinear_legacyUpdateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_SparseLinear_legacyUpdateOutput = undefined
c_SparseLinear_legacyAccGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> IO ()
c_SparseLinear_legacyAccGradParameters = undefined
c_Sqrt_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> IO ()
c_Sqrt_updateOutput = undefined
c_Sqrt_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Sqrt_updateGradInput = undefined
c_Square_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Square_updateOutput = undefined
c_Square_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Square_updateGradInput = undefined
c_Tanh_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Tanh_updateOutput = undefined
c_Tanh_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_Tanh_updateGradInput = undefined
c_Threshold_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> IO ()
c_Threshold_updateOutput = undefined
c_Threshold_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CDouble -> CBool -> IO ()
c_Threshold_updateGradInput = undefined
c_TemporalConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_TemporalConvolution_updateOutput = undefined
c_TemporalConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_TemporalConvolution_updateGradInput = undefined
c_TemporalConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CDouble -> IO ()
c_TemporalConvolution_accGradParameters = undefined
c_TemporalRowConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CBool -> IO ()
c_TemporalRowConvolution_updateOutput = undefined
c_TemporalRowConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CBool -> IO ()
c_TemporalRowConvolution_updateGradInput = undefined
c_TemporalRowConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()
c_TemporalRowConvolution_accGradParameters = undefined
c_TemporalUpSamplingNearest_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_TemporalUpSamplingNearest_updateOutput = undefined
c_TemporalUpSamplingNearest_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_TemporalUpSamplingNearest_updateGradInput = undefined
c_TemporalUpSamplingLinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_TemporalUpSamplingLinear_updateOutput = undefined
c_TemporalUpSamplingLinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_TemporalUpSamplingLinear_updateGradInput = undefined
c_BatchNormalization_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CDouble -> CDouble -> IO ()
c_BatchNormalization_updateOutput = undefined
c_BatchNormalization_backward :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CDouble -> CDouble -> IO ()
c_BatchNormalization_backward = undefined
c_SpatialConvolutionMM_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialConvolutionMM_updateOutput = undefined
c_SpatialConvolutionMM_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialConvolutionMM_updateGradInput = undefined
c_SpatialConvolutionMM_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_SpatialConvolutionMM_accGradParameters = undefined
c_SpatialConvolutionLocal_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_SpatialConvolutionLocal_updateOutput = undefined
c_SpatialConvolutionLocal_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_SpatialConvolutionLocal_updateGradInput = undefined
c_SpatialConvolutionLocal_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()
c_SpatialConvolutionLocal_accGradParameters = undefined
c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_SpatialAdaptiveAveragePooling_updateOutput = undefined
c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_SpatialAdaptiveAveragePooling_updateGradInput = undefined
c_SpatialAveragePooling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()
c_SpatialAveragePooling_updateOutput = undefined
c_SpatialAveragePooling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()
c_SpatialAveragePooling_updateGradInput = undefined
c_SpatialFullConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialFullConvolution_updateOutput = undefined
c_SpatialFullConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialFullConvolution_updateGradInput = undefined
c_SpatialFullConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_SpatialFullConvolution_accGradParameters = undefined
c_SpatialDilatedConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialDilatedConvolution_updateOutput = undefined
c_SpatialDilatedConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialDilatedConvolution_updateGradInput = undefined
c_SpatialDilatedConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_SpatialDilatedConvolution_accGradParameters = undefined
c_SpatialFullDilatedConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialFullDilatedConvolution_updateOutput = undefined
c_SpatialFullDilatedConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialFullDilatedConvolution_updateGradInput = undefined
c_SpatialFullDilatedConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_SpatialFullDilatedConvolution_accGradParameters = undefined
c_SpatialSubSampling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialSubSampling_updateOutput = undefined
c_SpatialSubSampling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialSubSampling_updateGradInput = undefined
c_SpatialSubSampling_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_SpatialSubSampling_accGradParameters = undefined
c_SpatialUpSamplingNearest_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_SpatialUpSamplingNearest_updateOutput = undefined
c_SpatialUpSamplingNearest_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_SpatialUpSamplingNearest_updateGradInput = undefined
c_SpatialUpSamplingBilinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_SpatialUpSamplingBilinear_updateOutput = undefined
c_SpatialUpSamplingBilinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialUpSamplingBilinear_updateGradInput = undefined
c_SpatialGridSamplerBilinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_SpatialGridSamplerBilinear_updateOutput = undefined
c_SpatialGridSamplerBilinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_SpatialGridSamplerBilinear_updateGradInput = undefined
c_VolumetricGridSamplerBilinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_VolumetricGridSamplerBilinear_updateOutput = undefined
c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_VolumetricGridSamplerBilinear_updateGradInput = undefined
c_VolumetricAveragePooling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()
c_VolumetricAveragePooling_updateOutput = undefined
c_VolumetricAveragePooling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()
c_VolumetricAveragePooling_updateGradInput = undefined
c_VolumetricConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricConvolution_updateOutput = undefined
c_VolumetricConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricConvolution_updateGradInput = undefined
c_VolumetricConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_VolumetricConvolution_accGradParameters = undefined
c_VolumetricFullConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricFullConvolution_updateOutput = undefined
c_VolumetricFullConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricFullConvolution_updateGradInput = undefined
c_VolumetricFullConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_VolumetricFullConvolution_accGradParameters = undefined
c_VolumetricDilatedConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricDilatedConvolution_updateOutput = undefined
c_VolumetricDilatedConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricDilatedConvolution_updateGradInput = undefined
c_VolumetricDilatedConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_VolumetricDilatedConvolution_accGradParameters = undefined
c_VolumetricFullDilatedConvolution_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricFullDilatedConvolution_updateOutput = undefined
c_VolumetricFullDilatedConvolution_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricFullDilatedConvolution_updateGradInput = undefined
c_VolumetricFullDilatedConvolution_accGradParameters :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()
c_VolumetricFullDilatedConvolution_accGradParameters = undefined
c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
c_VolumetricAdaptiveAveragePooling_updateOutput = undefined
c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_VolumetricAdaptiveAveragePooling_updateGradInput = undefined
c_SpatialReflectionPadding_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialReflectionPadding_updateOutput = undefined
c_SpatialReflectionPadding_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialReflectionPadding_updateGradInput = undefined
c_SpatialReplicationPadding_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialReplicationPadding_updateOutput = undefined
c_SpatialReplicationPadding_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> IO ()
c_SpatialReplicationPadding_updateGradInput = undefined
c_FeatureLPPooling_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()
c_FeatureLPPooling_updateOutput = undefined
c_FeatureLPPooling_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()
c_FeatureLPPooling_updateGradInput = undefined
c_VolumetricReplicationPadding_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricReplicationPadding_updateOutput = undefined
c_VolumetricReplicationPadding_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricReplicationPadding_updateGradInput = undefined
c_VolumetricUpSamplingNearest_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_VolumetricUpSamplingNearest_updateOutput = undefined
c_VolumetricUpSamplingNearest_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> IO ()
c_VolumetricUpSamplingNearest_updateGradInput = undefined
c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
c_VolumetricUpSamplingTrilinear_updateOutput = undefined
c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
c_VolumetricUpSamplingTrilinear_updateGradInput = undefined
c_TemporalReflectionPadding_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_TemporalReflectionPadding_updateOutput = undefined
c_TemporalReflectionPadding_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_TemporalReflectionPadding_updateGradInput = undefined
c_TemporalReplicationPadding_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_TemporalReplicationPadding_updateOutput = undefined
c_TemporalReplicationPadding_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_TemporalReplicationPadding_updateGradInput = undefined

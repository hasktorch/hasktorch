{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THNN.Short. where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.THNN

-- | c_Im2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNNShort_Im2Col_updateOutput"
  c_Im2Col_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Im2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNNShort_Im2Col_updateGradInput"
  c_Im2Col_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNNShort_Col2Im_updateOutput"
  c_Col2Im_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateGradInput :  state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNNShort_Col2Im_updateGradInput"
  c_Col2Im_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_GRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THNNShort_GRUFused_updateGradInput"
  c_GRUFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_LSTMFused_updateGradInput :  state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THNNShort_LSTMFused_updateGradInput"
  c_LSTMFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_SoftMarginCriterion_updateOutput"
  c_SoftMarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_SoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_SoftMarginCriterion_updateGradInput"
  c_SoftMarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_MSECriterion_updateOutput"
  c_MSECriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_MSECriterion_updateGradInput"
  c_MSECriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_PReLU_updateOutput :  state input output weight -> void
foreign import ccall "THNN.h THNNShort_PReLU_updateOutput"
  c_PReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_PReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNNShort_PReLU_updateGradInput"
  c_PReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_PReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THNNShort_PReLU_accGradParameters"
  c_PReLU_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_Linear_updateOutput :  state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THNNShort_Linear_updateOutput"
  c_Linear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Linear_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNNShort_Linear_updateGradInput"
  c_Linear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Linear_accGradParameters :  state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THNNShort_Linear_accGradParameters"
  c_Linear_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_RReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THNNShort_RReLU_updateOutput"
  c_RReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> CBool -> Ptr C'THGenerator -> IO ()

-- | c_RReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THNNShort_RReLU_updateGradInput"
  c_RReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> CBool -> IO ()

-- | c_Sigmoid_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNNShort_Sigmoid_updateOutput"
  c_Sigmoid_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Sigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNNShort_Sigmoid_updateGradInput"
  c_Sigmoid_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_SmoothL1Criterion_updateOutput"
  c_SmoothL1Criterion_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_SmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNNShort_SmoothL1Criterion_updateGradInput"
  c_SmoothL1Criterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ()

-- | c_SoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNNShort_SoftMax_updateOutput"
  c_SoftMax_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLLong -> IO ()

-- | c_SoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNNShort_SoftMax_updateGradInput"
  c_SoftMax_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLLong -> IO ()

-- | c_SoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THNN.h THNNShort_SoftPlus_updateOutput"
  c_SoftPlus_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ()

-- | c_SoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THNNShort_SoftPlus_updateGradInput"
  c_SoftPlus_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ()

-- | c_SoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THNN.h THNNShort_SoftShrink_updateOutput"
  c_SoftShrink_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_SoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNNShort_SoftShrink_updateGradInput"
  c_SoftShrink_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_SparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_updateOutput"
  c_SparseLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_accGradParameters"
  c_SparseLinear_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ()

-- | c_SparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_zeroGradParameters"
  c_SparseLinear_zeroGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_updateParameters"
  c_SparseLinear_updateParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_SparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_legacyUpdateOutput"
  c_SparseLinear_legacyUpdateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_legacyAccGradParameters"
  c_SparseLinear_legacyAccGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ()

-- | c_SparseLinear_legacyZeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_legacyZeroGradParameters"
  c_SparseLinear_legacyZeroGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SparseLinear_legacyUpdateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNNShort_SparseLinear_legacyUpdateParameters"
  c_SparseLinear_legacyUpdateParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_Sqrt_updateOutput :  state input output eps -> void
foreign import ccall "THNN.h THNNShort_Sqrt_updateOutput"
  c_Sqrt_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ()

-- | c_Sqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THNNShort_Sqrt_updateGradInput"
  c_Sqrt_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Square_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNNShort_Square_updateOutput"
  c_Square_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Square_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNNShort_Square_updateGradInput"
  c_Square_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Tanh_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNNShort_Tanh_updateOutput"
  c_Tanh_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Tanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNNShort_Tanh_updateGradInput"
  c_Tanh_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_Threshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THNN.h THNNShort_Threshold_updateOutput"
  c_Threshold_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_Threshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THNNShort_Threshold_updateGradInput"
  c_Threshold_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_TemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THNNShort_TemporalConvolution_updateOutput"
  c_TemporalConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNNShort_TemporalConvolution_updateGradInput"
  c_TemporalConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNNShort_TemporalConvolution_accGradParameters"
  c_TemporalConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CLong -> IO ()

-- | c_TemporalSubSampling_updateOutput :  state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THNNShort_TemporalSubSampling_updateOutput"
  c_TemporalSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNNShort_TemporalSubSampling_updateGradInput"
  c_TemporalSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNNShort_TemporalSubSampling_accGradParameters"
  c_TemporalSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CLong -> IO ()

-- | c_TemporalRowConvolution_updateOutput :  state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNNShort_TemporalRowConvolution_updateOutput"
  c_TemporalRowConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNNShort_TemporalRowConvolution_updateGradInput"
  c_TemporalRowConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THNNShort_TemporalRowConvolution_accGradParameters"
  c_TemporalRowConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> CLong -> IO ()

-- | c_TemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNNShort_TemporalUpSamplingNearest_updateOutput"
  c_TemporalUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNNShort_TemporalUpSamplingNearest_updateGradInput"
  c_TemporalUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateOutput :  state input output osizeW -> void
foreign import ccall "THNN.h THNNShort_TemporalUpSamplingLinear_updateOutput"
  c_TemporalUpSamplingLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h THNNShort_TemporalUpSamplingLinear_updateGradInput"
  c_TemporalUpSamplingLinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNNShort_SpatialConvolutionMM_updateGradInput"
  c_SpatialConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNNShort_SpatialConvolutionLocal_updateOutput"
  c_SpatialConvolutionLocal_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNNShort_SpatialConvolutionLocal_updateGradInput"
  c_SpatialConvolutionLocal_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THNNShort_SpatialConvolutionLocal_accGradParameters"
  c_SpatialConvolutionLocal_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CLong -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THNN.h THNNShort_SpatialAdaptiveAveragePooling_updateOutput"
  c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNNShort_SpatialAdaptiveAveragePooling_updateGradInput"
  c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNNShort_SpatialAveragePooling_updateOutput"
  c_SpatialAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNNShort_SpatialAveragePooling_updateGradInput"
  c_SpatialAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNNShort_SpatialFullConvolution_updateGradInput"
  c_SpatialFullConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNNShort_SpatialDilatedConvolution_updateGradInput"
  c_SpatialDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNNShort_SpatialFullDilatedConvolution_updateGradInput"
  c_SpatialFullDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THNNShort_SpatialSubSampling_updateOutput"
  c_SpatialSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THNNShort_SpatialSubSampling_updateGradInput"
  c_SpatialSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THNNShort_SpatialSubSampling_accGradParameters"
  c_SpatialSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CLong -> IO ()

-- | c_SpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNNShort_SpatialUpSamplingNearest_updateOutput"
  c_SpatialUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNNShort_SpatialUpSamplingNearest_updateGradInput"
  c_SpatialUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateOutput :  state input output osizeH osizeW -> void
foreign import ccall "THNN.h THNNShort_SpatialUpSamplingBilinear_updateOutput"
  c_SpatialUpSamplingBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h THNNShort_SpatialUpSamplingBilinear_updateGradInput"
  c_SpatialUpSamplingBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNNShort_SpatialGridSamplerBilinear_updateOutput"
  c_SpatialGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNNShort_SpatialGridSamplerBilinear_updateGradInput"
  c_SpatialGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNNShort_VolumetricGridSamplerBilinear_updateOutput"
  c_VolumetricGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNNShort_VolumetricGridSamplerBilinear_updateGradInput"
  c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_unfolded_acc :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h THNNShort_unfolded_acc"
  c_unfolded_acc :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_unfolded_copy :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNNShort_unfolded_copy"
  c_unfolded_copy :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNNShort_VolumetricAveragePooling_updateOutput"
  c_VolumetricAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNNShort_VolumetricAveragePooling_updateGradInput"
  c_VolumetricAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNNShort_VolumetricConvolution_updateGradInput"
  c_VolumetricConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNNShort_VolumetricConvolutionMM_updateGradInput"
  c_VolumetricConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNNShort_VolumetricDilatedConvolution_updateGradInput"
  c_VolumetricDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNNShort_VolumetricAdaptiveAveragePooling_updateOutput"
  c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNNShort_VolumetricAdaptiveAveragePooling_updateGradInput"
  c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_SpatialReflectionPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNNShort_SpatialReflectionPadding_updateOutput"
  c_SpatialReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNNShort_SpatialReflectionPadding_updateGradInput"
  c_SpatialReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNNShort_SpatialReplicationPadding_updateOutput"
  c_SpatialReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNNShort_SpatialReplicationPadding_updateGradInput"
  c_SpatialReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_FeatureLPPooling_updateOutput :  state input output power width stride batchMode -> void
foreign import ccall "THNN.h THNNShort_FeatureLPPooling_updateOutput"
  c_FeatureLPPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CInt -> CInt -> CBool -> IO ()

-- | c_FeatureLPPooling_updateGradInput :  state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THNNShort_FeatureLPPooling_updateGradInput"
  c_FeatureLPPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNNShort_VolumetricReplicationPadding_updateOutput"
  c_VolumetricReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNNShort_VolumetricReplicationPadding_updateGradInput"
  c_VolumetricReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNNShort_VolumetricUpSamplingNearest_updateOutput"
  c_VolumetricUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNNShort_VolumetricUpSamplingNearest_updateGradInput"
  c_VolumetricUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateOutput :  state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNNShort_VolumetricUpSamplingTrilinear_updateOutput"
  c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNNShort_VolumetricUpSamplingTrilinear_updateGradInput"
  c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNNShort_TemporalReflectionPadding_updateOutput"
  c_TemporalReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNNShort_TemporalReflectionPadding_updateGradInput"
  c_TemporalReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNNShort_TemporalReplicationPadding_updateOutput"
  c_TemporalReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNNShort_TemporalReplicationPadding_updateGradInput"
  c_TemporalReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | p_Im2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNNShort_Im2Col_updateOutput"
  p_Im2Col_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Im2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNNShort_Im2Col_updateGradInput"
  p_Im2Col_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNNShort_Col2Im_updateOutput"
  p_Col2Im_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateGradInput : Pointer to function : state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNNShort_Col2Im_updateGradInput"
  p_Col2Im_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THNNShort_GRUFused_updateGradInput"
  p_GRUFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THNNShort_LSTMFused_updateGradInput"
  p_LSTMFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_SoftMarginCriterion_updateOutput"
  p_SoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_SoftMarginCriterion_updateGradInput"
  p_SoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_MSECriterion_updateOutput"
  p_MSECriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_MSECriterion_updateGradInput"
  p_MSECriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THNNShort_PReLU_updateOutput"
  p_PReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNNShort_PReLU_updateGradInput"
  p_PReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THNNShort_PReLU_accGradParameters"
  p_PReLU_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_Linear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THNNShort_Linear_updateOutput"
  p_Linear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Linear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNNShort_Linear_updateGradInput"
  p_Linear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Linear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THNNShort_Linear_accGradParameters"
  p_Linear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THNNShort_RReLU_updateOutput"
  p_RReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> CBool -> Ptr C'THGenerator -> IO ())

-- | p_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THNNShort_RReLU_updateGradInput"
  p_RReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> CBool -> IO ())

-- | p_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNNShort_Sigmoid_updateOutput"
  p_Sigmoid_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNNShort_Sigmoid_updateGradInput"
  p_Sigmoid_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_SmoothL1Criterion_updateOutput"
  p_SmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNNShort_SmoothL1Criterion_updateGradInput"
  p_SmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CBool -> CBool -> IO ())

-- | p_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNNShort_SoftMax_updateOutput"
  p_SoftMax_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLLong -> IO ())

-- | p_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNNShort_SoftMax_updateGradInput"
  p_SoftMax_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLLong -> IO ())

-- | p_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THNNShort_SoftPlus_updateOutput"
  p_SoftPlus_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ())

-- | p_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THNNShort_SoftPlus_updateGradInput"
  p_SoftPlus_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ())

-- | p_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNNShort_SoftShrink_updateOutput"
  p_SoftShrink_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNNShort_SoftShrink_updateGradInput"
  p_SoftShrink_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_updateOutput"
  p_SparseLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_accGradParameters"
  p_SparseLinear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ())

-- | p_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_zeroGradParameters"
  p_SparseLinear_zeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_updateParameters"
  p_SparseLinear_updateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_legacyUpdateOutput"
  p_SparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_legacyAccGradParameters"
  p_SparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> IO ())

-- | p_SparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_legacyZeroGradParameters"
  p_SparseLinear_legacyZeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNNShort_SparseLinear_legacyUpdateParameters"
  p_SparseLinear_legacyUpdateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THNNShort_Sqrt_updateOutput"
  p_Sqrt_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> IO ())

-- | p_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNNShort_Sqrt_updateGradInput"
  p_Sqrt_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNNShort_Square_updateOutput"
  p_Square_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNNShort_Square_updateGradInput"
  p_Square_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNNShort_Tanh_updateOutput"
  p_Tanh_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNNShort_Tanh_updateGradInput"
  p_Tanh_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THNNShort_Threshold_updateOutput"
  p_Threshold_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THNNShort_Threshold_updateGradInput"
  p_Threshold_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THNNShort_TemporalConvolution_updateOutput"
  p_TemporalConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNNShort_TemporalConvolution_updateGradInput"
  p_TemporalConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNNShort_TemporalConvolution_accGradParameters"
  p_TemporalConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CLong -> IO ())

-- | p_TemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THNNShort_TemporalSubSampling_updateOutput"
  p_TemporalSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNNShort_TemporalSubSampling_updateGradInput"
  p_TemporalSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNNShort_TemporalSubSampling_accGradParameters"
  p_TemporalSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CLong -> IO ())

-- | p_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNNShort_TemporalRowConvolution_updateOutput"
  p_TemporalRowConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNNShort_TemporalRowConvolution_updateGradInput"
  p_TemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THNNShort_TemporalRowConvolution_accGradParameters"
  p_TemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CBool -> CLong -> IO ())

-- | p_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNNShort_TemporalUpSamplingNearest_updateOutput"
  p_TemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNNShort_TemporalUpSamplingNearest_updateGradInput"
  p_TemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output osizeW -> void
foreign import ccall "THNN.h &THNNShort_TemporalUpSamplingLinear_updateOutput"
  p_TemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h &THNNShort_TemporalUpSamplingLinear_updateGradInput"
  p_TemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNNShort_SpatialConvolutionMM_updateGradInput"
  p_SpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNNShort_SpatialConvolutionLocal_updateOutput"
  p_SpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNNShort_SpatialConvolutionLocal_updateGradInput"
  p_SpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THNNShort_SpatialConvolutionLocal_accGradParameters"
  p_SpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CLong -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THNNShort_SpatialAdaptiveAveragePooling_updateOutput"
  p_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNNShort_SpatialAdaptiveAveragePooling_updateGradInput"
  p_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNNShort_SpatialAveragePooling_updateOutput"
  p_SpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNNShort_SpatialAveragePooling_updateGradInput"
  p_SpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNNShort_SpatialFullConvolution_updateGradInput"
  p_SpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNNShort_SpatialDilatedConvolution_updateGradInput"
  p_SpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNNShort_SpatialFullDilatedConvolution_updateGradInput"
  p_SpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THNNShort_SpatialSubSampling_updateOutput"
  p_SpatialSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THNNShort_SpatialSubSampling_updateGradInput"
  p_SpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THNNShort_SpatialSubSampling_accGradParameters"
  p_SpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CLong -> IO ())

-- | p_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNNShort_SpatialUpSamplingNearest_updateOutput"
  p_SpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNNShort_SpatialUpSamplingNearest_updateGradInput"
  p_SpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output osizeH osizeW -> void
foreign import ccall "THNN.h &THNNShort_SpatialUpSamplingBilinear_updateOutput"
  p_SpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h &THNNShort_SpatialUpSamplingBilinear_updateGradInput"
  p_SpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNNShort_SpatialGridSamplerBilinear_updateOutput"
  p_SpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNNShort_SpatialGridSamplerBilinear_updateGradInput"
  p_SpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNNShort_VolumetricGridSamplerBilinear_updateOutput"
  p_VolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNNShort_VolumetricGridSamplerBilinear_updateGradInput"
  p_VolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_unfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h &THNNShort_unfolded_acc"
  p_unfolded_acc :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_unfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNNShort_unfolded_copy"
  p_unfolded_copy :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNNShort_VolumetricAveragePooling_updateOutput"
  p_VolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNNShort_VolumetricAveragePooling_updateGradInput"
  p_VolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNNShort_VolumetricConvolution_updateGradInput"
  p_VolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNNShort_VolumetricConvolutionMM_updateGradInput"
  p_VolumetricConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNNShort_VolumetricDilatedConvolution_updateGradInput"
  p_VolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNNShort_VolumetricAdaptiveAveragePooling_updateOutput"
  p_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNNShort_VolumetricAdaptiveAveragePooling_updateGradInput"
  p_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_SpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNNShort_SpatialReflectionPadding_updateOutput"
  p_SpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNNShort_SpatialReflectionPadding_updateGradInput"
  p_SpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNNShort_SpatialReplicationPadding_updateOutput"
  p_SpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNNShort_SpatialReplicationPadding_updateGradInput"
  p_SpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_FeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THNNShort_FeatureLPPooling_updateOutput"
  p_FeatureLPPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CInt -> CInt -> CBool -> IO ())

-- | p_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THNNShort_FeatureLPPooling_updateGradInput"
  p_FeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CLong -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNNShort_VolumetricReplicationPadding_updateOutput"
  p_VolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNNShort_VolumetricReplicationPadding_updateGradInput"
  p_VolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNNShort_VolumetricUpSamplingNearest_updateOutput"
  p_VolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNNShort_VolumetricUpSamplingNearest_updateGradInput"
  p_VolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNNShort_VolumetricUpSamplingTrilinear_updateOutput"
  p_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNNShort_VolumetricUpSamplingTrilinear_updateGradInput"
  p_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNNShort_TemporalReflectionPadding_updateOutput"
  p_TemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNNShort_TemporalReflectionPadding_updateGradInput"
  p_TemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNNShort_TemporalReplicationPadding_updateOutput"
  p_TemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNNShort_TemporalReplicationPadding_updateGradInput"
  p_TemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())
{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THCUNN.Char. where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.THCUNN

-- | c_thnnAbs_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNNChar_Abs_updateOutput"
  c_thnnAbs_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnAbs_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNNChar_Abs_updateGradInput"
  c_thnnAbs_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnAbsCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_AbsCriterion_updateOutput"
  c_thnnAbsCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnAbsCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_AbsCriterion_updateGradInput"
  c_thnnAbsCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnDistKLDivCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_DistKLDivCriterion_updateOutput"
  c_thnnDistKLDivCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnDistKLDivCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_DistKLDivCriterion_updateGradInput"
  c_thnnDistKLDivCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnELU_updateOutput :  state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h THNNChar_ELU_updateOutput"
  c_thnnELU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_thnnELU_updateGradInput :  state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h THNNChar_ELU_updateGradInput"
  c_thnnELU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ()

-- | c_thnnFeatureLPPooling_updateOutput :  state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNNChar_FeatureLPPooling_updateOutput"
  c_thnnFeatureLPPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CInt -> CInt -> CBool -> IO ()

-- | c_thnnFeatureLPPooling_updateGradInput :  state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNNChar_FeatureLPPooling_updateGradInput"
  c_thnnFeatureLPPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CInt -> CInt -> CBool -> IO ()

-- | c_thnnHardTanh_updateOutput :  state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNNChar_HardTanh_updateOutput"
  c_thnnHardTanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_thnnHardTanh_updateGradInput :  state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNNChar_HardTanh_updateGradInput"
  c_thnnHardTanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_thnnGatedLinear_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNNChar_GatedLinear_updateOutput"
  c_thnnGatedLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnGatedLinear_updateGradInput :  state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h THNNChar_GatedLinear_updateGradInput"
  c_thnnGatedLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnIm2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNNChar_Im2Col_updateOutput"
  c_thnnIm2Col_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnIm2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNNChar_Im2Col_updateGradInput"
  c_thnnIm2Col_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnCol2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNNChar_Col2Im_updateOutput"
  c_thnnCol2Im_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnLeakyReLU_updateOutput :  state input output negval inplace -> void
foreign import ccall "THCUNN.h THNNChar_LeakyReLU_updateOutput"
  c_thnnLeakyReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CBool -> IO ()

-- | c_thnnLeakyReLU_updateGradInput :  state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h THNNChar_LeakyReLU_updateGradInput"
  c_thnnLeakyReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CBool -> IO ()

-- | c_thnnGRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h THNNChar_GRUFused_updateGradInput"
  c_thnnGRUFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnLSTMFused_updateGradInput :  state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h THNNChar_LSTMFused_updateGradInput"
  c_thnnLSTMFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnLogSigmoid_updateOutput :  state input output buffer -> void
foreign import ccall "THCUNN.h THNNChar_LogSigmoid_updateOutput"
  c_thnnLogSigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnLogSigmoid_updateGradInput :  state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h THNNChar_LogSigmoid_updateGradInput"
  c_thnnLogSigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnLogSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNNChar_LogSoftMax_updateOutput"
  c_thnnLogSoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnLogSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNNChar_LogSoftMax_updateGradInput"
  c_thnnLogSoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnL1Cost_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNNChar_L1Cost_updateOutput"
  c_thnnL1Cost_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnMarginCriterion_updateOutput :  state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h THNNChar_MarginCriterion_updateOutput"
  c_thnnMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CLong -> IO ()

-- | c_thnnMarginCriterion_updateGradInput :  state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h THNNChar_MarginCriterion_updateGradInput"
  c_thnnMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CLong -> IO ()

-- | c_thnnMSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_MSECriterion_updateOutput"
  c_thnnMSECriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnMSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_MSECriterion_updateGradInput"
  c_thnnMSECriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnPReLU_updateOutput :  state input output weight -> void
foreign import ccall "THCUNN.h THNNChar_PReLU_updateOutput"
  c_thnnPReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnPReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h THNNChar_PReLU_updateGradInput"
  c_thnnPReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnPReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h THNNChar_PReLU_accGradParameters"
  c_thnnPReLU_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ()

-- | c_thnnSmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_SmoothL1Criterion_updateOutput"
  c_thnnSmoothL1Criterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnSmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_SmoothL1Criterion_updateGradInput"
  c_thnnSmoothL1Criterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnSparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_updateOutput"
  c_thnnSparseLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_accGradParameters"
  c_thnnSparseLinear_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ()

-- | c_thnnSparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_legacyUpdateOutput"
  c_thnnSparseLinear_legacyUpdateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_legacyAccGradParameters"
  c_thnnSparseLinear_legacyAccGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ()

-- | c_thnnSparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_zeroGradParameters"
  c_thnnSparseLinear_zeroGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h THNNChar_SparseLinear_updateParameters"
  c_thnnSparseLinear_updateParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ()

-- | c_thnnSpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialAdaptiveAveragePooling_updateOutput"
  c_thnnSpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnSpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNNChar_SpatialAdaptiveAveragePooling_updateGradInput"
  c_thnnSpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNNChar_SpatialAveragePooling_updateOutput"
  c_thnnSpatialAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_thnnSpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNNChar_SpatialAveragePooling_updateGradInput"
  c_thnnSpatialAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_thnnSpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNNChar_SpatialConvolutionLocal_updateOutput"
  c_thnnSpatialConvolutionLocal_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_thnnSpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNNChar_SpatialConvolutionLocal_updateGradInput"
  c_thnnSpatialConvolutionLocal_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_thnnSpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h THNNChar_SpatialConvolutionLocal_accGradParameters"
  c_thnnSpatialConvolutionLocal_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CLong -> IO ()

-- | c_thnnSpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialConvolutionMM_updateGradInput"
  c_thnnSpatialConvolutionMM_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialDepthwiseConvolution_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialDepthwiseConvolution_updateGradInput"
  c_thnnSpatialDepthwiseConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialDepthwiseConvolution_accGradParameters :  state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialDepthwiseConvolution_accGradParameters"
  c_thnnSpatialDepthwiseConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialCrossMapLRN_updateOutput :  state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h THNNChar_SpatialCrossMapLRN_updateOutput"
  c_thnnSpatialCrossMapLRN_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CLong -> CLong -> CLong -> IO ()

-- | c_thnnSpatialCrossMapLRN_updateGradInput :  state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h THNNChar_SpatialCrossMapLRN_updateGradInput"
  c_thnnSpatialCrossMapLRN_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CLong -> CLong -> CLong -> IO ()

-- | c_thnnSpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialDilatedConvolution_updateGradInput"
  c_thnnSpatialDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialFullDilatedConvolution_updateGradInput"
  c_thnnSpatialFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialFullConvolution_updateGradInput"
  c_thnnSpatialFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialReflectionPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNNChar_SpatialReflectionPadding_updateOutput"
  c_thnnSpatialReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNNChar_SpatialReflectionPadding_updateGradInput"
  c_thnnSpatialReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialReplicationPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNNChar_SpatialReplicationPadding_updateOutput"
  c_thnnSpatialReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNNChar_SpatialReplicationPadding_updateGradInput"
  c_thnnSpatialReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialSubSampling_updateOutput"
  c_thnnSpatialSubSampling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h THNNChar_SpatialSubSampling_updateGradInput"
  c_thnnSpatialSubSampling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h THNNChar_SpatialSubSampling_accGradParameters"
  c_thnnSpatialSubSampling_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CLong -> IO ()

-- | c_thnnSpatialUpSamplingBilinear_updateOutput :  state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_SpatialUpSamplingBilinear_updateOutput"
  c_thnnSpatialUpSamplingBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnSpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_SpatialUpSamplingBilinear_updateGradInput"
  c_thnnSpatialUpSamplingBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnSpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_SpatialUpSamplingNearest_updateGradInput"
  c_thnnSpatialUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnSpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_SpatialUpSamplingNearest_updateOutput"
  c_thnnSpatialUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnSpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNNChar_SpatialGridSamplerBilinear_updateOutput"
  c_thnnSpatialGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnSpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNNChar_SpatialGridSamplerBilinear_updateGradInput"
  c_thnnSpatialGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnVolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricGridSamplerBilinear_updateOutput"
  c_thnnVolumetricGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnVolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricGridSamplerBilinear_updateGradInput"
  c_thnnVolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnRReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h THNNChar_RReLU_updateOutput"
  c_thnnRReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ()

-- | c_thnnRReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h THNNChar_RReLU_updateGradInput"
  c_thnnRReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- | c_thnnSigmoid_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNNChar_Sigmoid_updateOutput"
  c_thnnSigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNNChar_Sigmoid_updateGradInput"
  c_thnnSigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_SoftMarginCriterion_updateOutput"
  c_thnnSoftMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnSoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNNChar_SoftMarginCriterion_updateGradInput"
  c_thnnSoftMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ()

-- | c_thnnSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNNChar_SoftMax_updateOutput"
  c_thnnSoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNNChar_SoftMax_updateGradInput"
  c_thnnSoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnSoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THCUNN.h THNNChar_SoftPlus_updateOutput"
  c_thnnSoftPlus_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ()

-- | c_thnnSoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h THNNChar_SoftPlus_updateGradInput"
  c_thnnSoftPlus_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ()

-- | c_thnnSoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THCUNN.h THNNChar_SoftShrink_updateOutput"
  c_thnnSoftShrink_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ()

-- | c_thnnSoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h THNNChar_SoftShrink_updateGradInput"
  c_thnnSoftShrink_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ()

-- | c_thnnSquare_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNNChar_Square_updateOutput"
  c_thnnSquare_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSquare_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNNChar_Square_updateGradInput"
  c_thnnSquare_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnSqrt_updateOutput :  state input output eps -> void
foreign import ccall "THCUNN.h THNNChar_Sqrt_updateOutput"
  c_thnnSqrt_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ()

-- | c_thnnSqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNNChar_Sqrt_updateGradInput"
  c_thnnSqrt_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnTanh_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNNChar_Tanh_updateOutput"
  c_thnnTanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnTanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNNChar_Tanh_updateGradInput"
  c_thnnTanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnTemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h THNNChar_TemporalConvolution_updateOutput"
  c_thnnTemporalConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnTemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h THNNChar_TemporalConvolution_updateGradInput"
  c_thnnTemporalConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnTemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h THNNChar_TemporalConvolution_accGradParameters"
  c_thnnTemporalConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CLong -> IO ()

-- | c_thnnTemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h THNNChar_TemporalRowConvolution_updateGradInput"
  c_thnnTemporalRowConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_thnnTemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h THNNChar_TemporalRowConvolution_accGradParameters"
  c_thnnTemporalRowConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CBool -> CLong -> IO ()

-- | c_thnnTemporalReflectionPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNNChar_TemporalReflectionPadding_updateOutput"
  c_thnnTemporalReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnTemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNNChar_TemporalReflectionPadding_updateGradInput"
  c_thnnTemporalReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnTemporalReplicationPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNNChar_TemporalReplicationPadding_updateOutput"
  c_thnnTemporalReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnTemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNNChar_TemporalReplicationPadding_updateGradInput"
  c_thnnTemporalReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_thnnTemporalUpSamplingLinear_updateOutput :  state input output outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_TemporalUpSamplingLinear_updateOutput"
  c_thnnTemporalUpSamplingLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnTemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_TemporalUpSamplingLinear_updateGradInput"
  c_thnnTemporalUpSamplingLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnTemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_TemporalUpSamplingNearest_updateGradInput"
  c_thnnTemporalUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnTemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_TemporalUpSamplingNearest_updateOutput"
  c_thnnTemporalUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnThreshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THCUNN.h THNNChar_Threshold_updateOutput"
  c_thnnThreshold_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_thnnThreshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h THNNChar_Threshold_updateGradInput"
  c_thnnThreshold_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ()

-- | c_thnnVolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricAveragePooling_updateOutput"
  c_thnnVolumetricAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_thnnVolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricAveragePooling_updateGradInput"
  c_thnnVolumetricAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_thnnVolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricConvolution_updateGradInput"
  c_thnnVolumetricConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricDilatedConvolution_updateGradInput"
  c_thnnVolumetricDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricFullDilatedConvolution_updateGradInput"
  c_thnnVolumetricFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricFullConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricFullConvolution_updateGradInput"
  c_thnnVolumetricFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricAdaptiveAveragePooling_updateOutput"
  c_thnnVolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricAdaptiveAveragePooling_updateGradInput"
  c_thnnVolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_thnnVolumetricReplicationPadding_updateOutput :  state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricReplicationPadding_updateOutput"
  c_thnnVolumetricReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricReplicationPadding_updateGradInput"
  c_thnnVolumetricReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricUpSamplingNearest_updateGradInput"
  c_thnnVolumetricUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnVolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricUpSamplingNearest_updateOutput"
  c_thnnVolumetricUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ()

-- | c_thnnVolumetricUpSamplingTrilinear_updateOutput :  state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricUpSamplingTrilinear_updateOutput"
  c_thnnVolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_thnnVolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNNChar_VolumetricUpSamplingTrilinear_updateGradInput"
  c_thnnVolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | p_thnnAbs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNNChar_Abs_updateOutput"
  p_thnnAbs_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnAbs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNNChar_Abs_updateGradInput"
  p_thnnAbs_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnAbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_AbsCriterion_updateOutput"
  p_thnnAbsCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnAbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_AbsCriterion_updateGradInput"
  p_thnnAbsCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnDistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_DistKLDivCriterion_updateOutput"
  p_thnnDistKLDivCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnDistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_DistKLDivCriterion_updateGradInput"
  p_thnnDistKLDivCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnELU_updateOutput : Pointer to function : state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h &THNNChar_ELU_updateOutput"
  p_thnnELU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_thnnELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h &THNNChar_ELU_updateGradInput"
  p_thnnELU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ())

-- | p_thnnFeatureLPPooling_updateOutput : Pointer to function : state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNNChar_FeatureLPPooling_updateOutput"
  p_thnnFeatureLPPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CInt -> CInt -> CBool -> IO ())

-- | p_thnnFeatureLPPooling_updateGradInput : Pointer to function : state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNNChar_FeatureLPPooling_updateGradInput"
  p_thnnFeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CInt -> CInt -> CBool -> IO ())

-- | p_thnnHardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNNChar_HardTanh_updateOutput"
  p_thnnHardTanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_thnnHardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNNChar_HardTanh_updateGradInput"
  p_thnnHardTanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_thnnGatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNNChar_GatedLinear_updateOutput"
  p_thnnGatedLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnGatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h &THNNChar_GatedLinear_updateGradInput"
  p_thnnGatedLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnIm2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNNChar_Im2Col_updateOutput"
  p_thnnIm2Col_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnIm2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNNChar_Im2Col_updateGradInput"
  p_thnnIm2Col_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnCol2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNNChar_Col2Im_updateOutput"
  p_thnnCol2Im_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnLeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THCUNN.h &THNNChar_LeakyReLU_updateOutput"
  p_thnnLeakyReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CBool -> IO ())

-- | p_thnnLeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h &THNNChar_LeakyReLU_updateGradInput"
  p_thnnLeakyReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CBool -> IO ())

-- | p_thnnGRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h &THNNChar_GRUFused_updateGradInput"
  p_thnnGRUFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnLSTMFused_updateGradInput : Pointer to function : state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h &THNNChar_LSTMFused_updateGradInput"
  p_thnnLSTMFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnLogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THCUNN.h &THNNChar_LogSigmoid_updateOutput"
  p_thnnLogSigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnLogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h &THNNChar_LogSigmoid_updateGradInput"
  p_thnnLogSigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnLogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNNChar_LogSoftMax_updateOutput"
  p_thnnLogSoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnLogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNNChar_LogSoftMax_updateGradInput"
  p_thnnLogSoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnL1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNNChar_L1Cost_updateOutput"
  p_thnnL1Cost_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNNChar_MarginCriterion_updateOutput"
  p_thnnMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CLong -> IO ())

-- | p_thnnMarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNNChar_MarginCriterion_updateGradInput"
  p_thnnMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CLong -> IO ())

-- | p_thnnMSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_MSECriterion_updateOutput"
  p_thnnMSECriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnMSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_MSECriterion_updateGradInput"
  p_thnnMSECriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnPReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THCUNN.h &THNNChar_PReLU_updateOutput"
  p_thnnPReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnPReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h &THNNChar_PReLU_updateGradInput"
  p_thnnPReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnPReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h &THNNChar_PReLU_accGradParameters"
  p_thnnPReLU_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ())

-- | p_thnnSmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_SmoothL1Criterion_updateOutput"
  p_thnnSmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnSmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_SmoothL1Criterion_updateGradInput"
  p_thnnSmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnSparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_updateOutput"
  p_thnnSparseLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_accGradParameters"
  p_thnnSparseLinear_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ())

-- | p_thnnSparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_legacyUpdateOutput"
  p_thnnSparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_legacyAccGradParameters"
  p_thnnSparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ())

-- | p_thnnSparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_zeroGradParameters"
  p_thnnSparseLinear_zeroGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h &THNNChar_SparseLinear_updateParameters"
  p_thnnSparseLinear_updateParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ())

-- | p_thnnSpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialAdaptiveAveragePooling_updateOutput"
  p_thnnSpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnSpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialAdaptiveAveragePooling_updateGradInput"
  p_thnnSpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialAveragePooling_updateOutput"
  p_thnnSpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_thnnSpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialAveragePooling_updateGradInput"
  p_thnnSpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_thnnSpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialConvolutionLocal_updateOutput"
  p_thnnSpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_thnnSpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialConvolutionLocal_updateGradInput"
  p_thnnSpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_thnnSpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialConvolutionLocal_accGradParameters"
  p_thnnSpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CLong -> IO ())

-- | p_thnnSpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialConvolutionMM_updateGradInput"
  p_thnnSpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialDepthwiseConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialDepthwiseConvolution_updateGradInput"
  p_thnnSpatialDepthwiseConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialDepthwiseConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialDepthwiseConvolution_accGradParameters"
  p_thnnSpatialDepthwiseConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialCrossMapLRN_updateOutput : Pointer to function : state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialCrossMapLRN_updateOutput"
  p_thnnSpatialCrossMapLRN_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CLong -> CLong -> CLong -> IO ())

-- | p_thnnSpatialCrossMapLRN_updateGradInput : Pointer to function : state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialCrossMapLRN_updateGradInput"
  p_thnnSpatialCrossMapLRN_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CLong -> CLong -> CLong -> IO ())

-- | p_thnnSpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialDilatedConvolution_updateGradInput"
  p_thnnSpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialFullDilatedConvolution_updateGradInput"
  p_thnnSpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialFullConvolution_updateGradInput"
  p_thnnSpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialReflectionPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialReflectionPadding_updateOutput"
  p_thnnSpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialReflectionPadding_updateGradInput"
  p_thnnSpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialReplicationPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialReplicationPadding_updateOutput"
  p_thnnSpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialReplicationPadding_updateGradInput"
  p_thnnSpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialSubSampling_updateOutput"
  p_thnnSpatialSubSampling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialSubSampling_updateGradInput"
  p_thnnSpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialSubSampling_accGradParameters"
  p_thnnSpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CLong -> IO ())

-- | p_thnnSpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialUpSamplingBilinear_updateOutput"
  p_thnnSpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnSpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialUpSamplingBilinear_updateGradInput"
  p_thnnSpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnSpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialUpSamplingNearest_updateGradInput"
  p_thnnSpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnSpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialUpSamplingNearest_updateOutput"
  p_thnnSpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnSpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialGridSamplerBilinear_updateOutput"
  p_thnnSpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnSpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNNChar_SpatialGridSamplerBilinear_updateGradInput"
  p_thnnSpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnVolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricGridSamplerBilinear_updateOutput"
  p_thnnVolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnVolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricGridSamplerBilinear_updateGradInput"
  p_thnnVolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnRReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h &THNNChar_RReLU_updateOutput"
  p_thnnRReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ())

-- | p_thnnRReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h &THNNChar_RReLU_updateGradInput"
  p_thnnRReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- | p_thnnSigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNNChar_Sigmoid_updateOutput"
  p_thnnSigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNNChar_Sigmoid_updateGradInput"
  p_thnnSigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_SoftMarginCriterion_updateOutput"
  p_thnnSoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnSoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNNChar_SoftMarginCriterion_updateGradInput"
  p_thnnSoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CBool -> CBool -> IO ())

-- | p_thnnSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNNChar_SoftMax_updateOutput"
  p_thnnSoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNNChar_SoftMax_updateGradInput"
  p_thnnSoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnSoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THCUNN.h &THNNChar_SoftPlus_updateOutput"
  p_thnnSoftPlus_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ())

-- | p_thnnSoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h &THNNChar_SoftPlus_updateGradInput"
  p_thnnSoftPlus_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> IO ())

-- | p_thnnSoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THCUNN.h &THNNChar_SoftShrink_updateOutput"
  p_thnnSoftShrink_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ())

-- | p_thnnSoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h &THNNChar_SoftShrink_updateGradInput"
  p_thnnSoftShrink_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ())

-- | p_thnnSquare_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNNChar_Square_updateOutput"
  p_thnnSquare_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSquare_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNNChar_Square_updateGradInput"
  p_thnnSquare_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnSqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THCUNN.h &THNNChar_Sqrt_updateOutput"
  p_thnnSqrt_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> IO ())

-- | p_thnnSqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNNChar_Sqrt_updateGradInput"
  p_thnnSqrt_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnTanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNNChar_Tanh_updateOutput"
  p_thnnTanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnTanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNNChar_Tanh_updateGradInput"
  p_thnnTanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnTemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalConvolution_updateOutput"
  p_thnnTemporalConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnTemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalConvolution_updateGradInput"
  p_thnnTemporalConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnTemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalConvolution_accGradParameters"
  p_thnnTemporalConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CLong -> IO ())

-- | p_thnnTemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalRowConvolution_updateGradInput"
  p_thnnTemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_thnnTemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalRowConvolution_accGradParameters"
  p_thnnTemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CBool -> CLong -> IO ())

-- | p_thnnTemporalReflectionPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalReflectionPadding_updateOutput"
  p_thnnTemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnTemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalReflectionPadding_updateGradInput"
  p_thnnTemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnTemporalReplicationPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalReplicationPadding_updateOutput"
  p_thnnTemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnTemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalReplicationPadding_updateGradInput"
  p_thnnTemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_thnnTemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalUpSamplingLinear_updateOutput"
  p_thnnTemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnTemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalUpSamplingLinear_updateGradInput"
  p_thnnTemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnTemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalUpSamplingNearest_updateGradInput"
  p_thnnTemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnTemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_TemporalUpSamplingNearest_updateOutput"
  p_thnnTemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnThreshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THCUNN.h &THNNChar_Threshold_updateOutput"
  p_thnnThreshold_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_thnnThreshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h &THNNChar_Threshold_updateGradInput"
  p_thnnThreshold_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CLong -> CLong -> CBool -> IO ())

-- | p_thnnVolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricAveragePooling_updateOutput"
  p_thnnVolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_thnnVolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricAveragePooling_updateGradInput"
  p_thnnVolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_thnnVolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricConvolution_updateGradInput"
  p_thnnVolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricDilatedConvolution_updateGradInput"
  p_thnnVolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricFullDilatedConvolution_updateGradInput"
  p_thnnVolumetricFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricFullConvolution_updateGradInput"
  p_thnnVolumetricFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricAdaptiveAveragePooling_updateOutput"
  p_thnnVolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricAdaptiveAveragePooling_updateGradInput"
  p_thnnVolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_thnnVolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricReplicationPadding_updateOutput"
  p_thnnVolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricReplicationPadding_updateGradInput"
  p_thnnVolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricUpSamplingNearest_updateGradInput"
  p_thnnVolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnVolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricUpSamplingNearest_updateOutput"
  p_thnnVolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> IO ())

-- | p_thnnVolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricUpSamplingTrilinear_updateOutput"
  p_thnnVolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_thnnVolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNNChar_VolumetricUpSamplingTrilinear_updateGradInput"
  p_thnnVolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())
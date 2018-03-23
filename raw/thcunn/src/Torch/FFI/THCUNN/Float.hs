{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THCUNN.Float where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.THCUNN

-- | c_Abs_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatAbs_updateOutput"
  c_Abs_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Abs_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatAbs_updateGradInput"
  c_Abs_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_AbsCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatAbsCriterion_updateOutput"
  c_AbsCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_AbsCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatAbsCriterion_updateGradInput"
  c_AbsCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_BatchNormalization_updateOutput :  state input_ output_ weight_ bias_ runningMean_ runningVar_ saveMean_ saveStd_ train momentum eps -> void
foreign import ccall "THCUNN.h THNN_CudaFloatBatchNormalization_updateOutput"
  c_BatchNormalization_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BatchNormalization_backward :  state input_ gradOutput_ gradInput_ gradWeight_ gradBias_ weight_ runningMean_ runningVar_ saveMean_ saveStd_ train scale eps -> void
foreign import ccall "THCUNN.h THNN_CudaFloatBatchNormalization_backward"
  c_BatchNormalization_backward :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BCECriterion_updateOutput :  state input target output sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatBCECriterion_updateOutput"
  c_BCECriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> Ptr C'THCudaFloatTensor -> CBool -> IO ()

-- | c_BCECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatBCECriterion_updateGradInput"
  c_BCECriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> Ptr C'THCudaFloatTensor -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatDistKLDivCriterion_updateOutput"
  c_DistKLDivCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatDistKLDivCriterion_updateGradInput"
  c_DistKLDivCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_ELU_updateOutput :  state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatELU_updateOutput"
  c_ELU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_ELU_updateGradInput :  state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatELU_updateGradInput"
  c_ELU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_FeatureLPPooling_updateOutput :  state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatFeatureLPPooling_updateOutput"
  c_FeatureLPPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_FeatureLPPooling_updateGradInput :  state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatFeatureLPPooling_updateGradInput"
  c_FeatureLPPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_HardTanh_updateOutput :  state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatHardTanh_updateOutput"
  c_HardTanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_HardTanh_updateGradInput :  state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatHardTanh_updateGradInput"
  c_HardTanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_GatedLinear_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatGatedLinear_updateOutput"
  c_GatedLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_GatedLinear_updateGradInput :  state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatGatedLinear_updateGradInput"
  c_GatedLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_Im2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaFloatIm2Col_updateOutput"
  c_Im2Col_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Im2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaFloatIm2Col_updateGradInput"
  c_Im2Col_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaFloatCol2Im_updateOutput"
  c_Col2Im_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_LeakyReLU_updateOutput :  state input output negval inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLeakyReLU_updateOutput"
  c_LeakyReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CBool -> IO ()

-- | c_LeakyReLU_updateGradInput :  state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLeakyReLU_updateGradInput"
  c_LeakyReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CBool -> IO ()

-- | c_GRUFused_updateOutput :  state input hidden bias1 bias2 hx hy storage -> void
foreign import ccall "THCUNN.h THNN_CudaFloatGRUFused_updateOutput"
  c_GRUFused_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_GRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h THNN_CudaFloatGRUFused_updateGradInput"
  c_GRUFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_LSTMFused_updateOutput :  state input hidden bias1 bias2 cx hy cy -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLSTMFused_updateOutput"
  c_LSTMFused_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_LSTMFused_updateGradInput :  state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLSTMFused_updateGradInput"
  c_LSTMFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_LogSigmoid_updateOutput :  state input output buffer -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLogSigmoid_updateOutput"
  c_LogSigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_LogSigmoid_updateGradInput :  state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLogSigmoid_updateGradInput"
  c_LogSigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_LogSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLogSoftMax_updateOutput"
  c_LogSoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_LogSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatLogSoftMax_updateGradInput"
  c_LogSoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_L1Cost_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatL1Cost_updateOutput"
  c_L1Cost_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_L1Cost_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatL1Cost_updateGradInput"
  c_L1Cost_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_MarginCriterion_updateOutput :  state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h THNN_CudaFloatMarginCriterion_updateOutput"
  c_MarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> IO ()

-- | c_MarginCriterion_updateGradInput :  state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h THNN_CudaFloatMarginCriterion_updateGradInput"
  c_MarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> IO ()

-- | c_MSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatMSECriterion_updateOutput"
  c_MSECriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatMSECriterion_updateGradInput"
  c_MSECriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_PReLU_updateOutput :  state input output weight -> void
foreign import ccall "THCUNN.h THNN_CudaFloatPReLU_updateOutput"
  c_PReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_PReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h THNN_CudaFloatPReLU_updateGradInput"
  c_PReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_PReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatPReLU_accGradParameters"
  c_PReLU_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ()

-- | c_SmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSmoothL1Criterion_updateOutput"
  c_SmoothL1Criterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_SmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSmoothL1Criterion_updateGradInput"
  c_SmoothL1Criterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_SparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_updateOutput"
  c_SparseLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_SparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_accGradParameters"
  c_SparseLinear_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_legacyUpdateOutput"
  c_SparseLinear_legacyUpdateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_SparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_legacyAccGradParameters"
  c_SparseLinear_legacyAccGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_zeroGradParameters"
  c_SparseLinear_zeroGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_SparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSparseLinear_updateParameters"
  c_SparseLinear_updateParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialAdaptiveAveragePooling_updateOutput"
  c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialAdaptiveAveragePooling_updateGradInput"
  c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_SpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialAveragePooling_updateOutput"
  c_SpatialAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialAveragePooling_updateGradInput"
  c_SpatialAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionLocal_updateOutput"
  c_SpatialConvolutionLocal_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionLocal_updateGradInput"
  c_SpatialConvolutionLocal_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionLocal_accGradParameters"
  c_SpatialConvolutionLocal_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_SpatialConvolutionMM_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionMM_updateOutput"
  c_SpatialConvolutionMM_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionMM_updateGradInput"
  c_SpatialConvolutionMM_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialConvolutionMM_accGradParameters"
  c_SpatialConvolutionMM_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialDepthwiseConvolution_updateOutput :  state input output weight bias kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDepthwiseConvolution_updateOutput"
  c_SpatialDepthwiseConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDepthwiseConvolution_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDepthwiseConvolution_updateGradInput"
  c_SpatialDepthwiseConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDepthwiseConvolution_accGradParameters :  state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDepthwiseConvolution_accGradParameters"
  c_SpatialDepthwiseConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialCrossMapLRN_updateOutput :  state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialCrossMapLRN_updateOutput"
  c_SpatialCrossMapLRN_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_SpatialCrossMapLRN_updateGradInput :  state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialCrossMapLRN_updateGradInput"
  c_SpatialCrossMapLRN_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_SpatialDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDilatedConvolution_updateOutput"
  c_SpatialDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDilatedConvolution_updateGradInput"
  c_SpatialDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialDilatedConvolution_accGradParameters"
  c_SpatialDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullDilatedConvolution_updateOutput"
  c_SpatialFullDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullDilatedConvolution_updateGradInput"
  c_SpatialFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullDilatedConvolution_accGradParameters"
  c_SpatialFullDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullConvolution_updateOutput"
  c_SpatialFullConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullConvolution_updateGradInput"
  c_SpatialFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialFullConvolution_accGradParameters"
  c_SpatialFullConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialReflectionPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialReflectionPadding_updateOutput"
  c_SpatialReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialReflectionPadding_updateGradInput"
  c_SpatialReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialReplicationPadding_updateOutput"
  c_SpatialReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialReplicationPadding_updateGradInput"
  c_SpatialReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialSubSampling_updateOutput"
  c_SpatialSubSampling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialSubSampling_updateGradInput"
  c_SpatialSubSampling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialSubSampling_accGradParameters"
  c_SpatialSubSampling_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialUpSamplingBilinear_updateOutput :  state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialUpSamplingBilinear_updateOutput"
  c_SpatialUpSamplingBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialUpSamplingBilinear_updateGradInput"
  c_SpatialUpSamplingBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialUpSamplingNearest_updateGradInput"
  c_SpatialUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialUpSamplingNearest_updateOutput"
  c_SpatialUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialGridSamplerBilinear_updateOutput"
  c_SpatialGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSpatialGridSamplerBilinear_updateGradInput"
  c_SpatialGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricGridSamplerBilinear_updateOutput"
  c_VolumetricGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricGridSamplerBilinear_updateGradInput"
  c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_RReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h THNN_CudaFloatRReLU_updateOutput"
  c_RReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ()

-- | c_RReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatRReLU_updateGradInput"
  c_RReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- | c_Sigmoid_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSigmoid_updateOutput"
  c_Sigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Sigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSigmoid_updateGradInput"
  c_Sigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_SoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftMarginCriterion_updateOutput"
  c_SoftMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_SoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftMarginCriterion_updateGradInput"
  c_SoftMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ()

-- | c_SoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftMax_updateOutput"
  c_SoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_SoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftMax_updateGradInput"
  c_SoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_SoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftPlus_updateOutput"
  c_SoftPlus_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftPlus_updateGradInput"
  c_SoftPlus_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftShrink_updateOutput"
  c_SoftShrink_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ()

-- | c_SoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSoftShrink_updateGradInput"
  c_SoftShrink_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ()

-- | c_Square_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSquare_updateOutput"
  c_Square_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Square_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSquare_updateGradInput"
  c_Square_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Sqrt_updateOutput :  state input output eps -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSqrt_updateOutput"
  c_Sqrt_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ()

-- | c_Sqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatSqrt_updateGradInput"
  c_Sqrt_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Tanh_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTanh_updateOutput"
  c_Tanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_Tanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTanh_updateGradInput"
  c_Tanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_TemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalConvolution_updateOutput"
  c_TemporalConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalConvolution_updateGradInput"
  c_TemporalConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalConvolution_accGradParameters"
  c_TemporalConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalRowConvolution_updateOutput :  state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalRowConvolution_updateOutput"
  c_TemporalRowConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalRowConvolution_updateGradInput"
  c_TemporalRowConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalRowConvolution_accGradParameters"
  c_TemporalRowConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- | c_TemporalReflectionPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalReflectionPadding_updateOutput"
  c_TemporalReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalReflectionPadding_updateGradInput"
  c_TemporalReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalReplicationPadding_updateOutput"
  c_TemporalReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalReplicationPadding_updateGradInput"
  c_TemporalReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateOutput :  state input output outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalUpSamplingLinear_updateOutput"
  c_TemporalUpSamplingLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalUpSamplingLinear_updateGradInput"
  c_TemporalUpSamplingLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalUpSamplingNearest_updateGradInput"
  c_TemporalUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatTemporalUpSamplingNearest_updateOutput"
  c_TemporalUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_Threshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatThreshold_updateOutput"
  c_Threshold_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Threshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaFloatThreshold_updateGradInput"
  c_Threshold_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricAveragePooling_updateOutput"
  c_VolumetricAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricAveragePooling_updateGradInput"
  c_VolumetricAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricConvolution_updateOutput :  state input output weight bias finput fgradInput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricConvolution_updateOutput"
  c_VolumetricConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricConvolution_updateGradInput"
  c_VolumetricConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput dT dW dH padT padW padH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricConvolution_accGradParameters"
  c_VolumetricConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricDilatedConvolution_updateOutput :  state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricDilatedConvolution_updateOutput"
  c_VolumetricDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricDilatedConvolution_updateGradInput"
  c_VolumetricDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricDilatedConvolution_accGradParameters"
  c_VolumetricDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullDilatedConvolution_updateOutput"
  c_VolumetricFullDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullDilatedConvolution_updateGradInput"
  c_VolumetricFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullDilatedConvolution_accGradParameters"
  c_VolumetricFullDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullConvolution_updateOutput"
  c_VolumetricFullConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullConvolution_updateGradInput"
  c_VolumetricFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricFullConvolution_accGradParameters"
  c_VolumetricFullConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricAdaptiveAveragePooling_updateOutput"
  c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricAdaptiveAveragePooling_updateGradInput"
  c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_VolumetricReplicationPadding_updateOutput :  state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricReplicationPadding_updateOutput"
  c_VolumetricReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricReplicationPadding_updateGradInput"
  c_VolumetricReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricUpSamplingNearest_updateGradInput"
  c_VolumetricUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricUpSamplingNearest_updateOutput"
  c_VolumetricUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateOutput :  state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricUpSamplingTrilinear_updateOutput"
  c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaFloatVolumetricUpSamplingTrilinear_updateGradInput"
  c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | p_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatAbs_updateOutput"
  p_Abs_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatAbs_updateGradInput"
  p_Abs_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatAbsCriterion_updateOutput"
  p_AbsCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatAbsCriterion_updateGradInput"
  p_AbsCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_BatchNormalization_updateOutput : Pointer to function : state input_ output_ weight_ bias_ runningMean_ runningVar_ saveMean_ saveStd_ train momentum eps -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatBatchNormalization_updateOutput"
  p_BatchNormalization_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BatchNormalization_backward : Pointer to function : state input_ gradOutput_ gradInput_ gradWeight_ gradBias_ weight_ runningMean_ runningVar_ saveMean_ saveStd_ train scale eps -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatBatchNormalization_backward"
  p_BatchNormalization_backward :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatBCECriterion_updateOutput"
  p_BCECriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> Ptr C'THCudaFloatTensor -> CBool -> IO ())

-- | p_BCECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatBCECriterion_updateGradInput"
  p_BCECriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> Ptr C'THCudaFloatTensor -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatDistKLDivCriterion_updateOutput"
  p_DistKLDivCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatDistKLDivCriterion_updateGradInput"
  p_DistKLDivCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_ELU_updateOutput : Pointer to function : state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatELU_updateOutput"
  p_ELU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatELU_updateGradInput"
  p_ELU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_FeatureLPPooling_updateOutput : Pointer to function : state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatFeatureLPPooling_updateOutput"
  p_FeatureLPPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatFeatureLPPooling_updateGradInput"
  p_FeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatHardTanh_updateOutput"
  p_HardTanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatHardTanh_updateGradInput"
  p_HardTanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatGatedLinear_updateOutput"
  p_GatedLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatGatedLinear_updateGradInput"
  p_GatedLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_Im2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatIm2Col_updateOutput"
  p_Im2Col_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Im2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatIm2Col_updateGradInput"
  p_Im2Col_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatCol2Im_updateOutput"
  p_Col2Im_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLeakyReLU_updateOutput"
  p_LeakyReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CBool -> IO ())

-- | p_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLeakyReLU_updateGradInput"
  p_LeakyReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CBool -> IO ())

-- | p_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx hy storage -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatGRUFused_updateOutput"
  p_GRUFused_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatGRUFused_updateGradInput"
  p_GRUFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cx hy cy -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLSTMFused_updateOutput"
  p_LSTMFused_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLSTMFused_updateGradInput"
  p_LSTMFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLogSigmoid_updateOutput"
  p_LogSigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLogSigmoid_updateGradInput"
  p_LogSigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLogSoftMax_updateOutput"
  p_LogSoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatLogSoftMax_updateGradInput"
  p_LogSoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatL1Cost_updateOutput"
  p_L1Cost_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatL1Cost_updateGradInput"
  p_L1Cost_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatMarginCriterion_updateOutput"
  p_MarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> IO ())

-- | p_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatMarginCriterion_updateGradInput"
  p_MarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CDouble -> IO ())

-- | p_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatMSECriterion_updateOutput"
  p_MSECriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatMSECriterion_updateGradInput"
  p_MSECriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatPReLU_updateOutput"
  p_PReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatPReLU_updateGradInput"
  p_PReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatPReLU_accGradParameters"
  p_PReLU_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ())

-- | p_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSmoothL1Criterion_updateOutput"
  p_SmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSmoothL1Criterion_updateGradInput"
  p_SmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_updateOutput"
  p_SparseLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_accGradParameters"
  p_SparseLinear_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_legacyUpdateOutput"
  p_SparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_legacyAccGradParameters"
  p_SparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_zeroGradParameters"
  p_SparseLinear_zeroGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSparseLinear_updateParameters"
  p_SparseLinear_updateParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialAdaptiveAveragePooling_updateOutput"
  p_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialAdaptiveAveragePooling_updateGradInput"
  p_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialAveragePooling_updateOutput"
  p_SpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialAveragePooling_updateGradInput"
  p_SpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionLocal_updateOutput"
  p_SpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionLocal_updateGradInput"
  p_SpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionLocal_accGradParameters"
  p_SpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionMM_updateOutput"
  p_SpatialConvolutionMM_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionMM_updateGradInput"
  p_SpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialConvolutionMM_accGradParameters"
  p_SpatialConvolutionMM_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialDepthwiseConvolution_updateOutput : Pointer to function : state input output weight bias kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDepthwiseConvolution_updateOutput"
  p_SpatialDepthwiseConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDepthwiseConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDepthwiseConvolution_updateGradInput"
  p_SpatialDepthwiseConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDepthwiseConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDepthwiseConvolution_accGradParameters"
  p_SpatialDepthwiseConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialCrossMapLRN_updateOutput : Pointer to function : state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialCrossMapLRN_updateOutput"
  p_SpatialCrossMapLRN_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_SpatialCrossMapLRN_updateGradInput : Pointer to function : state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialCrossMapLRN_updateGradInput"
  p_SpatialCrossMapLRN_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDilatedConvolution_updateOutput"
  p_SpatialDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDilatedConvolution_updateGradInput"
  p_SpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialDilatedConvolution_accGradParameters"
  p_SpatialDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullDilatedConvolution_updateOutput"
  p_SpatialFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullDilatedConvolution_updateGradInput"
  p_SpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullDilatedConvolution_accGradParameters"
  p_SpatialFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullConvolution_updateOutput"
  p_SpatialFullConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullConvolution_updateGradInput"
  p_SpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialFullConvolution_accGradParameters"
  p_SpatialFullConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialReflectionPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialReflectionPadding_updateOutput"
  p_SpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialReflectionPadding_updateGradInput"
  p_SpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialReplicationPadding_updateOutput"
  p_SpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialReplicationPadding_updateGradInput"
  p_SpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialSubSampling_updateOutput"
  p_SpatialSubSampling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialSubSampling_updateGradInput"
  p_SpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialSubSampling_accGradParameters"
  p_SpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialUpSamplingBilinear_updateOutput"
  p_SpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialUpSamplingBilinear_updateGradInput"
  p_SpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialUpSamplingNearest_updateGradInput"
  p_SpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialUpSamplingNearest_updateOutput"
  p_SpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialGridSamplerBilinear_updateOutput"
  p_SpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSpatialGridSamplerBilinear_updateGradInput"
  p_SpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricGridSamplerBilinear_updateOutput"
  p_VolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricGridSamplerBilinear_updateGradInput"
  p_VolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatRReLU_updateOutput"
  p_RReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ())

-- | p_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatRReLU_updateGradInput"
  p_RReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- | p_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSigmoid_updateOutput"
  p_Sigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSigmoid_updateGradInput"
  p_Sigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftMarginCriterion_updateOutput"
  p_SoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftMarginCriterion_updateGradInput"
  p_SoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CBool -> CBool -> IO ())

-- | p_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftMax_updateOutput"
  p_SoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftMax_updateGradInput"
  p_SoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftPlus_updateOutput"
  p_SoftPlus_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftPlus_updateGradInput"
  p_SoftPlus_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftShrink_updateOutput"
  p_SoftShrink_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ())

-- | p_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSoftShrink_updateGradInput"
  p_SoftShrink_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ())

-- | p_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSquare_updateOutput"
  p_Square_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSquare_updateGradInput"
  p_Square_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSqrt_updateOutput"
  p_Sqrt_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> IO ())

-- | p_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatSqrt_updateGradInput"
  p_Sqrt_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTanh_updateOutput"
  p_Tanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTanh_updateGradInput"
  p_Tanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalConvolution_updateOutput"
  p_TemporalConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalConvolution_updateGradInput"
  p_TemporalConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalConvolution_accGradParameters"
  p_TemporalConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalRowConvolution_updateOutput"
  p_TemporalRowConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalRowConvolution_updateGradInput"
  p_TemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalRowConvolution_accGradParameters"
  p_TemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- | p_TemporalReflectionPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalReflectionPadding_updateOutput"
  p_TemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalReflectionPadding_updateGradInput"
  p_TemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalReplicationPadding_updateOutput"
  p_TemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalReplicationPadding_updateGradInput"
  p_TemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalUpSamplingLinear_updateOutput"
  p_TemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalUpSamplingLinear_updateGradInput"
  p_TemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalUpSamplingNearest_updateGradInput"
  p_TemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatTemporalUpSamplingNearest_updateOutput"
  p_TemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatThreshold_updateOutput"
  p_Threshold_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatThreshold_updateGradInput"
  p_Threshold_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricAveragePooling_updateOutput"
  p_VolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricAveragePooling_updateGradInput"
  p_VolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricConvolution_updateOutput"
  p_VolumetricConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricConvolution_updateGradInput"
  p_VolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH padT padW padH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricConvolution_accGradParameters"
  p_VolumetricConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricDilatedConvolution_updateOutput"
  p_VolumetricDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricDilatedConvolution_updateGradInput"
  p_VolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricDilatedConvolution_accGradParameters"
  p_VolumetricDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullDilatedConvolution_updateOutput"
  p_VolumetricFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullDilatedConvolution_updateGradInput"
  p_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullDilatedConvolution_accGradParameters"
  p_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullConvolution_updateOutput"
  p_VolumetricFullConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullConvolution_updateGradInput"
  p_VolumetricFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricFullConvolution_accGradParameters"
  p_VolumetricFullConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricAdaptiveAveragePooling_updateOutput"
  p_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricAdaptiveAveragePooling_updateGradInput"
  p_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricReplicationPadding_updateOutput"
  p_VolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricReplicationPadding_updateGradInput"
  p_VolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricUpSamplingNearest_updateGradInput"
  p_VolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricUpSamplingNearest_updateOutput"
  p_VolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricUpSamplingTrilinear_updateOutput"
  p_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaFloatVolumetricUpSamplingTrilinear_updateGradInput"
  p_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())
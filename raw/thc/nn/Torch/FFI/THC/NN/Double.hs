{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.NN.Double where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_Abs_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleAbs_updateOutput"
  c_Abs_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Abs_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleAbs_updateGradInput"
  c_Abs_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_AbsCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleAbsCriterion_updateOutput"
  c_AbsCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_AbsCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleAbsCriterion_updateGradInput"
  c_AbsCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_BatchNormalization_updateOutput :  state input_ output_ weight_ bias_ runningMean_ runningVar_ saveMean_ saveStd_ train momentum eps -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleBatchNormalization_updateOutput"
  c_BatchNormalization_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BatchNormalization_backward :  state input_ gradOutput_ gradInput_ gradWeight_ gradBias_ weight_ runningMean_ runningVar_ saveMean_ saveStd_ train scale eps -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleBatchNormalization_backward"
  c_BatchNormalization_backward :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BCECriterion_updateOutput :  state input target output sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleBCECriterion_updateOutput"
  c_BCECriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> CBool -> IO ()

-- | c_BCECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleBCECriterion_updateGradInput"
  c_BCECriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> CBool -> IO ()

-- | c_ClassNLLCriterion_updateOutput :  state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleClassNLLCriterion_updateOutput"
  c_ClassNLLCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ()

-- | c_ClassNLLCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleClassNLLCriterion_updateGradInput"
  c_ClassNLLCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleDistKLDivCriterion_updateOutput"
  c_DistKLDivCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleDistKLDivCriterion_updateGradInput"
  c_DistKLDivCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_ELU_updateOutput :  state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleELU_updateOutput"
  c_ELU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_ELU_updateGradInput :  state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleELU_updateGradInput"
  c_ELU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_FeatureLPPooling_updateOutput :  state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleFeatureLPPooling_updateOutput"
  c_FeatureLPPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_FeatureLPPooling_updateGradInput :  state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleFeatureLPPooling_updateGradInput"
  c_FeatureLPPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_HardTanh_updateOutput :  state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleHardTanh_updateOutput"
  c_HardTanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_HardTanh_updateGradInput :  state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleHardTanh_updateGradInput"
  c_HardTanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_GatedLinear_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleGatedLinear_updateOutput"
  c_GatedLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_GatedLinear_updateGradInput :  state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleGatedLinear_updateGradInput"
  c_GatedLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_Im2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIm2Col_updateOutput"
  c_Im2Col_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Im2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIm2Col_updateGradInput"
  c_Im2Col_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleCol2Im_updateOutput"
  c_Col2Im_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateGradInput :  state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleCol2Im_updateGradInput"
  c_Col2Im_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_LeakyReLU_updateOutput :  state input output negval inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLeakyReLU_updateOutput"
  c_LeakyReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_LeakyReLU_updateGradInput :  state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLeakyReLU_updateGradInput"
  c_LeakyReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_GRUFused_updateOutput :  state input hidden bias1 bias2 hx hy storage -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleGRUFused_updateOutput"
  c_GRUFused_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_GRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleGRUFused_updateGradInput"
  c_GRUFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_LSTMFused_updateOutput :  state input hidden bias1 bias2 cx hy cy -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLSTMFused_updateOutput"
  c_LSTMFused_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_LSTMFused_updateGradInput :  state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLSTMFused_updateGradInput"
  c_LSTMFused_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_LogSigmoid_updateOutput :  state input output buffer -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLogSigmoid_updateOutput"
  c_LogSigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_LogSigmoid_updateGradInput :  state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLogSigmoid_updateGradInput"
  c_LogSigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_LogSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLogSoftMax_updateOutput"
  c_LogSoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_LogSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLogSoftMax_updateGradInput"
  c_LogSoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_LookupTable_accGradParameters :  state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLookupTable_accGradParameters"
  c_LookupTable_accGradParameters :: Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CBool -> CInt -> CDouble -> IO ()

-- | c_LookupTable_renorm :  state idx weight maxNorm normType -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLookupTable_renorm"
  c_LookupTable_renorm :: Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_LookupTableBag_updateOutput :  state input offsets weight output offset2bag mode seq_length -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLookupTableBag_updateOutput"
  c_LookupTableBag_updateOutput :: Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> Ptr C'THCIndexTensor -> IO ()

-- | c_LookupTableBag_accGradParameters :  state input gradOutput gradWeight offset2bag count sortedIndices origIndices scaleGradByFreq mode seq_length scale_ -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleLookupTableBag_accGradParameters"
  c_LookupTableBag_accGradParameters :: Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CBool -> CInt -> Ptr C'THCIndexTensor -> CDouble -> IO ()

-- | c_L1Cost_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleL1Cost_updateOutput"
  c_L1Cost_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_L1Cost_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleL1Cost_updateGradInput"
  c_L1Cost_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_MarginCriterion_updateOutput :  state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMarginCriterion_updateOutput"
  c_MarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> IO ()

-- | c_MarginCriterion_updateGradInput :  state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMarginCriterion_updateGradInput"
  c_MarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> IO ()

-- | c_MSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMSECriterion_updateOutput"
  c_MSECriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMSECriterion_updateGradInput"
  c_MSECriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MultiLabelMarginCriterion_updateOutput :  state input target output istarget sizeaverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput"
  c_MultiLabelMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MultiLabelMarginCriterion_updateGradInput :  state input target gradOutput gradInput istarget sizeaverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput"
  c_MultiLabelMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MultiMarginCriterion_updateOutput :  state input target output sizeAverage p weights margin reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMultiMarginCriterion_updateOutput"
  c_MultiMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CInt -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_MultiMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage p weights margin reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleMultiMarginCriterion_updateGradInput"
  c_MultiMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CInt -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_PReLU_updateOutput :  state input output weight -> void
foreign import ccall "THCUNN.h THNN_CudaDoublePReLU_updateOutput"
  c_PReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_PReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h THNN_CudaDoublePReLU_updateGradInput"
  c_PReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_PReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoublePReLU_accGradParameters"
  c_PReLU_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_SmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSmoothL1Criterion_updateOutput"
  c_SmoothL1Criterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSmoothL1Criterion_updateGradInput"
  c_SmoothL1Criterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_updateOutput"
  c_SparseLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_accGradParameters"
  c_SparseLinear_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_legacyUpdateOutput"
  c_SparseLinear_legacyUpdateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_legacyAccGradParameters"
  c_SparseLinear_legacyAccGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_zeroGradParameters"
  c_SparseLinear_zeroGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSparseLinear_updateParameters"
  c_SparseLinear_updateParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_IndexLinear_updateOutput :  state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIndexLinear_updateOutput"
  c_IndexLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_IndexLinear_accGradParameters :  state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIndexLinear_accGradParameters"
  c_IndexLinear_accGradParameters :: Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_IndexLinear_accUpdateGradParameters :  state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIndexLinear_accUpdateGradParameters"
  c_IndexLinear_accUpdateGradParameters :: Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_IndexLinear_updateParameters :  state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleIndexLinear_updateParameters"
  c_IndexLinear_updateParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | c_SpatialAdaptiveMaxPooling_updateOutput :  state input output indices osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateOutput"
  c_SpatialAdaptiveMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveMaxPooling_updateGradInput :  state input gradOutput gradInput indices -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateGradInput"
  c_SpatialAdaptiveMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateOutput"
  c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateGradInput"
  c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAveragePooling_updateOutput"
  c_SpatialAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialAveragePooling_updateGradInput"
  c_SpatialAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialClassNLLCriterion_updateOutput :  state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput"
  c_SpatialClassNLLCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ()

-- | c_SpatialClassNLLCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput"
  c_SpatialClassNLLCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ()

-- | c_SpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionLocal_updateOutput"
  c_SpatialConvolutionLocal_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionLocal_updateGradInput"
  c_SpatialConvolutionLocal_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionLocal_accGradParameters"
  c_SpatialConvolutionLocal_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_SpatialConvolutionMM_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionMM_updateOutput"
  c_SpatialConvolutionMM_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionMM_updateGradInput"
  c_SpatialConvolutionMM_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialConvolutionMM_accGradParameters"
  c_SpatialConvolutionMM_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialDepthwiseConvolution_updateOutput :  state input output weight bias kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput"
  c_SpatialDepthwiseConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDepthwiseConvolution_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput"
  c_SpatialDepthwiseConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDepthwiseConvolution_accGradParameters :  state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters"
  c_SpatialDepthwiseConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialCrossMapLRN_updateOutput :  state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialCrossMapLRN_updateOutput"
  c_SpatialCrossMapLRN_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_SpatialCrossMapLRN_updateGradInput :  state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialCrossMapLRN_updateGradInput"
  c_SpatialCrossMapLRN_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_SpatialDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDilatedConvolution_updateOutput"
  c_SpatialDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDilatedConvolution_updateGradInput"
  c_SpatialDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDilatedConvolution_accGradParameters"
  c_SpatialDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullDilatedConvolution_updateOutput"
  c_SpatialFullDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullDilatedConvolution_updateGradInput"
  c_SpatialFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullDilatedConvolution_accGradParameters"
  c_SpatialFullDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialDilatedMaxPooling_updateOutput :  state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDilatedMaxPooling_updateOutput"
  c_SpatialDilatedMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialDilatedMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialDilatedMaxPooling_updateGradInput"
  c_SpatialDilatedMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialFractionalMaxPooling_updateOutput :  state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFractionalMaxPooling_updateOutput"
  c_SpatialFractionalMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SpatialFractionalMaxPooling_updateGradInput :  state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFractionalMaxPooling_updateGradInput"
  c_SpatialFractionalMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> IO ()

-- | c_SpatialFullConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullConvolution_updateOutput"
  c_SpatialFullConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullConvolution_updateGradInput"
  c_SpatialFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialFullConvolution_accGradParameters"
  c_SpatialFullConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialMaxPooling_updateOutput :  state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialMaxPooling_updateOutput"
  c_SpatialMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialMaxPooling_updateGradInput"
  c_SpatialMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialMaxUnpooling_updateOutput :  state input output indices owidth oheight -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialMaxUnpooling_updateOutput"
  c_SpatialMaxUnpooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialMaxUnpooling_updateGradInput :  state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialMaxUnpooling_updateGradInput"
  c_SpatialMaxUnpooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialReflectionPadding_updateOutput"
  c_SpatialReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialReflectionPadding_updateGradInput"
  c_SpatialReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateOutput :  state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialReplicationPadding_updateOutput"
  c_SpatialReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialReplicationPadding_updateGradInput"
  c_SpatialReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialSubSampling_updateOutput"
  c_SpatialSubSampling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialSubSampling_updateGradInput"
  c_SpatialSubSampling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialSubSampling_accGradParameters"
  c_SpatialSubSampling_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialUpSamplingBilinear_updateOutput :  state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialUpSamplingBilinear_updateOutput"
  c_SpatialUpSamplingBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialUpSamplingBilinear_updateGradInput"
  c_SpatialUpSamplingBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialUpSamplingNearest_updateGradInput"
  c_SpatialUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialUpSamplingNearest_updateOutput"
  c_SpatialUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialGridSamplerBilinear_updateOutput"
  c_SpatialGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSpatialGridSamplerBilinear_updateGradInput"
  c_SpatialGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricGridSamplerBilinear_updateOutput"
  c_VolumetricGridSamplerBilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricGridSamplerBilinear_updateGradInput"
  c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_RReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleRReLU_updateOutput"
  c_RReLU_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ()

-- | c_RReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleRReLU_updateGradInput"
  c_RReLU_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- | c_Sigmoid_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSigmoid_updateOutput"
  c_Sigmoid_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Sigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSigmoid_updateGradInput"
  c_Sigmoid_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_SoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftMarginCriterion_updateOutput"
  c_SoftMarginCriterion_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftMarginCriterion_updateGradInput"
  c_SoftMarginCriterion_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftMax_updateOutput"
  c_SoftMax_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_SoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftMax_updateGradInput"
  c_SoftMax_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_SoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftPlus_updateOutput"
  c_SoftPlus_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftPlus_updateGradInput"
  c_SoftPlus_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftShrink_updateOutput"
  c_SoftShrink_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_SoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSoftShrink_updateGradInput"
  c_SoftShrink_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_Square_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSquare_updateOutput"
  c_Square_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Square_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSquare_updateGradInput"
  c_Square_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Sqrt_updateOutput :  state input output eps -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSqrt_updateOutput"
  c_Sqrt_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_Sqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleSqrt_updateGradInput"
  c_Sqrt_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Tanh_updateOutput :  state input output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTanh_updateOutput"
  c_Tanh_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_Tanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTanh_updateGradInput"
  c_Tanh_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_TemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalConvolution_updateOutput"
  c_TemporalConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalConvolution_updateGradInput"
  c_TemporalConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalConvolution_accGradParameters"
  c_TemporalConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalMaxPooling_updateOutput :  state input output indices kW dW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalMaxPooling_updateOutput"
  c_TemporalMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ()

-- | c_TemporalMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalMaxPooling_updateGradInput"
  c_TemporalMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ()

-- | c_TemporalRowConvolution_updateOutput :  state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalRowConvolution_updateOutput"
  c_TemporalRowConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalRowConvolution_updateGradInput"
  c_TemporalRowConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalRowConvolution_accGradParameters"
  c_TemporalRowConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- | c_TemporalReflectionPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalReflectionPadding_updateOutput"
  c_TemporalReflectionPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalReflectionPadding_updateGradInput"
  c_TemporalReflectionPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateOutput :  state input output padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalReplicationPadding_updateOutput"
  c_TemporalReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalReplicationPadding_updateGradInput"
  c_TemporalReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateOutput :  state input output outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalUpSamplingLinear_updateOutput"
  c_TemporalUpSamplingLinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalUpSamplingLinear_updateGradInput"
  c_TemporalUpSamplingLinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalUpSamplingNearest_updateGradInput"
  c_TemporalUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleTemporalUpSamplingNearest_updateOutput"
  c_TemporalUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_Threshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleThreshold_updateOutput"
  c_Threshold_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Threshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleThreshold_updateGradInput"
  c_Threshold_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAveragePooling_updateOutput"
  c_VolumetricAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAveragePooling_updateGradInput"
  c_VolumetricAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricConvolution_updateOutput :  state input output weight bias finput fgradInput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricConvolution_updateOutput"
  c_VolumetricConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricConvolution_updateGradInput"
  c_VolumetricConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput dT dW dH padT padW padH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricConvolution_accGradParameters"
  c_VolumetricConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricDilatedConvolution_updateOutput :  state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricDilatedConvolution_updateOutput"
  c_VolumetricDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricDilatedConvolution_updateGradInput"
  c_VolumetricDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricDilatedConvolution_accGradParameters"
  c_VolumetricDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullDilatedConvolution_updateOutput"
  c_VolumetricFullDilatedConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullDilatedConvolution_updateGradInput"
  c_VolumetricFullDilatedConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullDilatedConvolution_accGradParameters"
  c_VolumetricFullDilatedConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricDilatedMaxPooling_updateOutput :  state input output indices kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricDilatedMaxPooling_updateOutput"
  c_VolumetricDilatedMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricDilatedMaxPooling_updateGradInput :  state input gradOutput gradInput indices kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricDilatedMaxPooling_updateGradInput"
  c_VolumetricDilatedMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricFractionalMaxPooling_updateOutput :  state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFractionalMaxPooling_updateOutput"
  c_VolumetricFractionalMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_VolumetricFractionalMaxPooling_updateGradInput :  state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFractionalMaxPooling_updateGradInput"
  c_VolumetricFractionalMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> IO ()

-- | c_VolumetricFullConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullConvolution_updateOutput"
  c_VolumetricFullConvolution_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullConvolution_updateGradInput"
  c_VolumetricFullConvolution_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricFullConvolution_accGradParameters"
  c_VolumetricFullConvolution_accGradParameters :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricMaxPooling_updateOutput :  state input output indices kT kW kH dT dW dH padT padW padH ceilMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricMaxPooling_updateOutput"
  c_VolumetricMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricMaxPooling_updateGradInput :  state input gradOutput gradInput indices kT kW kH dT dW dH padT padW padH ceilMode -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricMaxPooling_updateGradInput"
  c_VolumetricMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricMaxUnpooling_updateOutput :  state input output indices outputTime outputWidth outputHeight dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricMaxUnpooling_updateOutput"
  c_VolumetricMaxUnpooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricMaxUnpooling_updateGradInput :  state input gradOutput gradInput indices outputTime outputWidth outputHeight dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricMaxUnpooling_updateGradInput"
  c_VolumetricMaxUnpooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveMaxPooling_updateOutput :  state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateOutput"
  c_VolumetricAdaptiveMaxPooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveMaxPooling_updateGradInput :  state input gradOutput gradInput indices -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateGradInput"
  c_VolumetricAdaptiveMaxPooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateOutput"
  c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_VolumetricReplicationPadding_updateOutput :  state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricReplicationPadding_updateOutput"
  c_VolumetricReplicationPadding_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricReplicationPadding_updateGradInput"
  c_VolumetricReplicationPadding_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricUpSamplingNearest_updateGradInput"
  c_VolumetricUpSamplingNearest_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricUpSamplingNearest_updateOutput"
  c_VolumetricUpSamplingNearest_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateOutput :  state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateOutput"
  c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateGradInput"
  c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | p_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleAbs_updateOutput"
  p_Abs_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleAbs_updateGradInput"
  p_Abs_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleAbsCriterion_updateOutput"
  p_AbsCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleAbsCriterion_updateGradInput"
  p_AbsCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_BatchNormalization_updateOutput : Pointer to function : state input_ output_ weight_ bias_ runningMean_ runningVar_ saveMean_ saveStd_ train momentum eps -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleBatchNormalization_updateOutput"
  p_BatchNormalization_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BatchNormalization_backward : Pointer to function : state input_ gradOutput_ gradInput_ gradWeight_ gradBias_ weight_ runningMean_ runningVar_ saveMean_ saveStd_ train scale eps -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleBatchNormalization_backward"
  p_BatchNormalization_backward :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleBCECriterion_updateOutput"
  p_BCECriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> CBool -> IO ())

-- | p_BCECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleBCECriterion_updateGradInput"
  p_BCECriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> CBool -> IO ())

-- | p_ClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleClassNLLCriterion_updateOutput"
  p_ClassNLLCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ())

-- | p_ClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleClassNLLCriterion_updateGradInput"
  p_ClassNLLCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleDistKLDivCriterion_updateOutput"
  p_DistKLDivCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleDistKLDivCriterion_updateGradInput"
  p_DistKLDivCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_ELU_updateOutput : Pointer to function : state input output alpha scale inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleELU_updateOutput"
  p_ELU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleELU_updateGradInput"
  p_ELU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_FeatureLPPooling_updateOutput : Pointer to function : state inputTH outputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleFeatureLPPooling_updateOutput"
  p_FeatureLPPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutputTH inputTH outputTH gradInputTH power width stride batchMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleFeatureLPPooling_updateGradInput"
  p_FeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleHardTanh_updateOutput"
  p_HardTanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleHardTanh_updateGradInput"
  p_HardTanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleGatedLinear_updateOutput"
  p_GatedLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleGatedLinear_updateGradInput"
  p_GatedLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_Im2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIm2Col_updateOutput"
  p_Im2Col_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Im2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIm2Col_updateGradInput"
  p_Im2Col_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleCol2Im_updateOutput"
  p_Col2Im_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateGradInput : Pointer to function : state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleCol2Im_updateGradInput"
  p_Col2Im_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLeakyReLU_updateOutput"
  p_LeakyReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLeakyReLU_updateGradInput"
  p_LeakyReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx hy storage -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleGRUFused_updateOutput"
  p_GRUFused_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleGRUFused_updateGradInput"
  p_GRUFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cx hy cy -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLSTMFused_updateOutput"
  p_LSTMFused_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates prevC cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLSTMFused_updateGradInput"
  p_LSTMFused_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLogSigmoid_updateOutput"
  p_LogSigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLogSigmoid_updateGradInput"
  p_LogSigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLogSoftMax_updateOutput"
  p_LogSoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLogSoftMax_updateGradInput"
  p_LogSoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_LookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLookupTable_accGradParameters"
  p_LookupTable_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CBool -> CInt -> CDouble -> IO ())

-- | p_LookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLookupTable_renorm"
  p_LookupTable_renorm :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_LookupTableBag_updateOutput : Pointer to function : state input offsets weight output offset2bag mode seq_length -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLookupTableBag_updateOutput"
  p_LookupTableBag_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> Ptr C'THCIndexTensor -> IO ())

-- | p_LookupTableBag_accGradParameters : Pointer to function : state input gradOutput gradWeight offset2bag count sortedIndices origIndices scaleGradByFreq mode seq_length scale_ -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleLookupTableBag_accGradParameters"
  p_LookupTableBag_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CBool -> CInt -> Ptr C'THCIndexTensor -> CDouble -> IO ())

-- | p_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleL1Cost_updateOutput"
  p_L1Cost_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleL1Cost_updateGradInput"
  p_L1Cost_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMarginCriterion_updateOutput"
  p_MarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> IO ())

-- | p_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMarginCriterion_updateGradInput"
  p_MarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CDouble -> IO ())

-- | p_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMSECriterion_updateOutput"
  p_MSECriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMSECriterion_updateGradInput"
  p_MSECriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output istarget sizeaverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput"
  p_MultiLabelMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput istarget sizeaverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput"
  p_MultiLabelMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMultiMarginCriterion_updateOutput"
  p_MultiMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CInt -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_MultiMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage p weights margin reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleMultiMarginCriterion_updateGradInput"
  p_MultiMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CInt -> Ptr C'THCudaDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoublePReLU_updateOutput"
  p_PReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoublePReLU_updateGradInput"
  p_PReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoublePReLU_accGradParameters"
  p_PReLU_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSmoothL1Criterion_updateOutput"
  p_SmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSmoothL1Criterion_updateGradInput"
  p_SmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_updateOutput"
  p_SparseLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_accGradParameters"
  p_SparseLinear_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_legacyUpdateOutput"
  p_SparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_legacyAccGradParameters"
  p_SparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_zeroGradParameters"
  p_SparseLinear_zeroGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSparseLinear_updateParameters"
  p_SparseLinear_updateParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_IndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIndexLinear_updateOutput"
  p_IndexLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_IndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIndexLinear_accGradParameters"
  p_IndexLinear_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_IndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIndexLinear_accUpdateGradParameters"
  p_IndexLinear_accUpdateGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCIndexTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_IndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleIndexLinear_updateParameters"
  p_IndexLinear_updateParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCIndexTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_SpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateOutput"
  p_SpatialAdaptiveMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateGradInput"
  p_SpatialAdaptiveMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateOutput"
  p_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateGradInput"
  p_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAveragePooling_updateOutput"
  p_SpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialAveragePooling_updateGradInput"
  p_SpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput"
  p_SpatialClassNLLCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ())

-- | p_SpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput"
  p_SpatialClassNLLCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> CBool -> IO ())

-- | p_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionLocal_updateOutput"
  p_SpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionLocal_updateGradInput"
  p_SpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionLocal_accGradParameters"
  p_SpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionMM_updateOutput"
  p_SpatialConvolutionMM_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns ones kW kH dW dH padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionMM_updateGradInput"
  p_SpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialConvolutionMM_accGradParameters"
  p_SpatialConvolutionMM_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialDepthwiseConvolution_updateOutput : Pointer to function : state input output weight bias kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput"
  p_SpatialDepthwiseConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDepthwiseConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput"
  p_SpatialDepthwiseConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDepthwiseConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters"
  p_SpatialDepthwiseConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialCrossMapLRN_updateOutput : Pointer to function : state input output scale size alpha beta k -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialCrossMapLRN_updateOutput"
  p_SpatialCrossMapLRN_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_SpatialCrossMapLRN_updateGradInput : Pointer to function : state input gradOutput gradInput scale output size alpha beta k -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialCrossMapLRN_updateGradInput"
  p_SpatialCrossMapLRN_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDilatedConvolution_updateOutput"
  p_SpatialDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDilatedConvolution_updateGradInput"
  p_SpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDilatedConvolution_accGradParameters"
  p_SpatialDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullDilatedConvolution_updateOutput"
  p_SpatialFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullDilatedConvolution_updateGradInput"
  p_SpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullDilatedConvolution_accGradParameters"
  p_SpatialFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDilatedMaxPooling_updateOutput"
  p_SpatialDilatedMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialDilatedMaxPooling_updateGradInput"
  p_SpatialDilatedMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFractionalMaxPooling_updateOutput"
  p_SpatialFractionalMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH poolSizeW poolSizeH indices -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFractionalMaxPooling_updateGradInput"
  p_SpatialFractionalMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> IO ())

-- | p_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullConvolution_updateOutput"
  p_SpatialFullConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullConvolution_updateGradInput"
  p_SpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialFullConvolution_accGradParameters"
  p_SpatialFullConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialMaxPooling_updateOutput"
  p_SpatialMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialMaxPooling_updateGradInput"
  p_SpatialMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialMaxUnpooling_updateOutput"
  p_SpatialMaxUnpooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialMaxUnpooling_updateGradInput"
  p_SpatialMaxUnpooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialReflectionPadding_updateOutput"
  p_SpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialReflectionPadding_updateGradInput"
  p_SpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateOutput : Pointer to function : state input output padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialReplicationPadding_updateOutput"
  p_SpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR padT padB -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialReplicationPadding_updateGradInput"
  p_SpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialSubSampling_updateOutput"
  p_SpatialSubSampling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialSubSampling_updateGradInput"
  p_SpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialSubSampling_accGradParameters"
  p_SpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialUpSamplingBilinear_updateOutput"
  p_SpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputHeight inputWidth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialUpSamplingBilinear_updateGradInput"
  p_SpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialUpSamplingNearest_updateGradInput"
  p_SpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialUpSamplingNearest_updateOutput"
  p_SpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialGridSamplerBilinear_updateOutput"
  p_SpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSpatialGridSamplerBilinear_updateGradInput"
  p_SpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricGridSamplerBilinear_updateOutput"
  p_VolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricGridSamplerBilinear_updateGradInput"
  p_VolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleRReLU_updateOutput"
  p_RReLU_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr () -> IO ())

-- | p_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleRReLU_updateGradInput"
  p_RReLU_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- | p_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSigmoid_updateOutput"
  p_Sigmoid_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSigmoid_updateGradInput"
  p_Sigmoid_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftMarginCriterion_updateOutput"
  p_SoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftMarginCriterion_updateGradInput"
  p_SoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftMax_updateOutput"
  p_SoftMax_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftMax_updateGradInput"
  p_SoftMax_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftPlus_updateOutput"
  p_SoftPlus_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftPlus_updateGradInput"
  p_SoftPlus_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftShrink_updateOutput"
  p_SoftShrink_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSoftShrink_updateGradInput"
  p_SoftShrink_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSquare_updateOutput"
  p_Square_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSquare_updateGradInput"
  p_Square_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSqrt_updateOutput"
  p_Sqrt_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleSqrt_updateGradInput"
  p_Sqrt_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTanh_updateOutput"
  p_Tanh_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTanh_updateGradInput"
  p_Tanh_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalConvolution_updateOutput"
  p_TemporalConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalConvolution_updateGradInput"
  p_TemporalConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalConvolution_accGradParameters"
  p_TemporalConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalMaxPooling_updateOutput"
  p_TemporalMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ())

-- | p_TemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalMaxPooling_updateGradInput"
  p_TemporalMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> IO ())

-- | p_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalRowConvolution_updateOutput"
  p_TemporalRowConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalRowConvolution_updateGradInput"
  p_TemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalRowConvolution_accGradParameters"
  p_TemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- | p_TemporalReflectionPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalReflectionPadding_updateOutput"
  p_TemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalReflectionPadding_updateGradInput"
  p_TemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateOutput : Pointer to function : state input output padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalReplicationPadding_updateOutput"
  p_TemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput padL padR -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalReplicationPadding_updateGradInput"
  p_TemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalUpSamplingLinear_updateOutput"
  p_TemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputWidth outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalUpSamplingLinear_updateGradInput"
  p_TemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalUpSamplingNearest_updateGradInput"
  p_TemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleTemporalUpSamplingNearest_updateOutput"
  p_TemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleThreshold_updateOutput"
  p_Threshold_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleThreshold_updateGradInput"
  p_Threshold_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAveragePooling_updateOutput"
  p_VolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAveragePooling_updateGradInput"
  p_VolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricConvolution_updateOutput"
  p_VolumetricConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricConvolution_updateGradInput"
  p_VolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH padT padW padH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricConvolution_accGradParameters"
  p_VolumetricConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricDilatedConvolution_updateOutput"
  p_VolumetricDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricDilatedConvolution_updateGradInput"
  p_VolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricDilatedConvolution_accGradParameters"
  p_VolumetricDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullDilatedConvolution_updateOutput"
  p_VolumetricFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullDilatedConvolution_updateGradInput"
  p_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullDilatedConvolution_accGradParameters"
  p_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricDilatedMaxPooling_updateOutput"
  p_VolumetricDilatedMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricDilatedMaxPooling_updateGradInput"
  p_VolumetricDilatedMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFractionalMaxPooling_updateOutput"
  p_VolumetricFractionalMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_VolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFractionalMaxPooling_updateGradInput"
  p_VolumetricFractionalMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THCIndexTensor -> IO ())

-- | p_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullConvolution_updateOutput"
  p_VolumetricFullConvolution_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullConvolution_updateGradInput"
  p_VolumetricFullConvolution_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH padT padW padH adjT adjW adjH scale -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricFullConvolution_accGradParameters"
  p_VolumetricFullConvolution_accGradParameters :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH padT padW padH ceilMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricMaxPooling_updateOutput"
  p_VolumetricMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH padT padW padH ceilMode -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricMaxPooling_updateGradInput"
  p_VolumetricMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices outputTime outputWidth outputHeight dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricMaxUnpooling_updateOutput"
  p_VolumetricMaxUnpooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices outputTime outputWidth outputHeight dT dW dH padT padW padH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricMaxUnpooling_updateGradInput"
  p_VolumetricMaxUnpooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateOutput"
  p_VolumetricAdaptiveMaxPooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateGradInput"
  p_VolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCIndexTensor -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateOutput"
  p_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  p_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricReplicationPadding_updateOutput"
  p_VolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pleft pright ptop pbottom pfront pback -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricReplicationPadding_updateGradInput"
  p_VolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricUpSamplingNearest_updateGradInput"
  p_VolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricUpSamplingNearest_updateOutput"
  p_VolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateOutput"
  p_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput nbatch nchannels inputDepth inputHeight inputWidth outputDepth outputHeight outputWidth -> void
foreign import ccall "THCUNN.h &THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateGradInput"
  p_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())
{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.NN.Float where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_Abs_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_FloatAbs_updateOutput"
  c_Abs_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Abs_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatAbs_updateGradInput"
  c_Abs_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_AbsCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatAbsCriterion_updateOutput"
  c_AbsCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_AbsCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatAbsCriterion_updateGradInput"
  c_AbsCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_BCECriterion_updateOutput :  state input target output sizeAverage weights reduce -> void
foreign import ccall "THNN.h THNN_FloatBCECriterion_updateOutput"
  c_BCECriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> CBool -> IO ()

-- | c_BCECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THNN.h THNN_FloatBCECriterion_updateGradInput"
  c_BCECriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> CBool -> IO ()

-- | c_ClassNLLCriterion_updateOutput :  state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatClassNLLCriterion_updateOutput"
  c_ClassNLLCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ()

-- | c_ClassNLLCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatClassNLLCriterion_updateGradInput"
  c_ClassNLLCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ()

-- | c_SpatialClassNLLCriterion_updateOutput :  state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatSpatialClassNLLCriterion_updateOutput"
  c_SpatialClassNLLCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ()

-- | c_SpatialClassNLLCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h THNN_FloatSpatialClassNLLCriterion_updateGradInput"
  c_SpatialClassNLLCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ()

-- | c_ELU_updateOutput :  state input output alpha scale inplace -> void
foreign import ccall "THNN.h THNN_FloatELU_updateOutput"
  c_ELU_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_ELU_updateGradInput :  state gradOutput gradInput output alpha scale -> void
foreign import ccall "THNN.h THNN_FloatELU_updateGradInput"
  c_ELU_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_DistKLDivCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatDistKLDivCriterion_updateOutput"
  c_DistKLDivCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatDistKLDivCriterion_updateGradInput"
  c_DistKLDivCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_GatedLinear_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_FloatGatedLinear_updateOutput"
  c_GatedLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_GatedLinear_updateGradInput :  state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THNN_FloatGatedLinear_updateGradInput"
  c_GatedLinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_HardShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THNN.h THNN_FloatHardShrink_updateOutput"
  c_HardShrink_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_HardShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_FloatHardShrink_updateGradInput"
  c_HardShrink_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_HardTanh_updateOutput :  state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_FloatHardTanh_updateOutput"
  c_HardTanh_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_HardTanh_updateGradInput :  state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_FloatHardTanh_updateGradInput"
  c_HardTanh_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Im2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_FloatIm2Col_updateOutput"
  c_Im2Col_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Im2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_FloatIm2Col_updateGradInput"
  c_Im2Col_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_FloatCol2Im_updateOutput"
  c_Col2Im_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateGradInput :  state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_FloatCol2Im_updateGradInput"
  c_Col2Im_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_L1Cost_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_FloatL1Cost_updateOutput"
  c_L1Cost_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_L1Cost_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatL1Cost_updateGradInput"
  c_L1Cost_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LeakyReLU_updateOutput :  state input output negval inplace -> void
foreign import ccall "THNN.h THNN_FloatLeakyReLU_updateOutput"
  c_LeakyReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ()

-- | c_LeakyReLU_updateGradInput :  state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THNN_FloatLeakyReLU_updateGradInput"
  c_LeakyReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ()

-- | c_GRUFused_updateOutput :  state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THNN_FloatGRUFused_updateOutput"
  c_GRUFused_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_GRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THNN_FloatGRUFused_updateGradInput"
  c_GRUFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LSTMFused_updateOutput :  state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THNN_FloatLSTMFused_updateOutput"
  c_LSTMFused_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LSTMFused_updateGradInput :  state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THNN_FloatLSTMFused_updateGradInput"
  c_LSTMFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LogSigmoid_updateOutput :  state input output buffer -> void
foreign import ccall "THNN.h THNN_FloatLogSigmoid_updateOutput"
  c_LogSigmoid_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LogSigmoid_updateGradInput :  state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THNN_FloatLogSigmoid_updateGradInput"
  c_LogSigmoid_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_LogSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_FloatLogSoftMax_updateOutput"
  c_LogSoftMax_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | c_LogSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_FloatLogSoftMax_updateGradInput"
  c_LogSoftMax_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | c_LookupTable_accGradParameters :  state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h THNN_FloatLookupTable_accGradParameters"
  c_LookupTable_accGradParameters :: Ptr C'THNNState -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIntegerTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CBool -> CInt -> CDouble -> IO ()

-- | c_LookupTable_renorm :  state idx weight maxNorm normType -> void
foreign import ccall "THNN.h THNN_FloatLookupTable_renorm"
  c_LookupTable_renorm :: Ptr C'THNNState -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_MarginCriterion_updateOutput :  state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THNN_FloatMarginCriterion_updateOutput"
  c_MarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> IO ()

-- | c_MarginCriterion_updateGradInput :  state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THNN_FloatMarginCriterion_updateGradInput"
  c_MarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> IO ()

-- | c_SoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSoftMarginCriterion_updateOutput"
  c_SoftMarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_SoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSoftMarginCriterion_updateGradInput"
  c_SoftMarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMSECriterion_updateOutput"
  c_MSECriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMSECriterion_updateGradInput"
  c_MSECriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_MultiLabelMarginCriterion_updateOutput :  state input target output isTarget sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMultiLabelMarginCriterion_updateOutput"
  c_MultiLabelMarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_MultiLabelMarginCriterion_updateGradInput :  state input target gradOutput gradInput isTarget sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatMultiLabelMarginCriterion_updateGradInput"
  c_MultiLabelMarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_MultiMarginCriterion_updateOutput :  state input target output sizeAverage p weights margin reduce -> void
foreign import ccall "THNN.h THNN_FloatMultiMarginCriterion_updateOutput"
  c_MultiMarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> CInt -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ()

-- | c_MultiMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage p weights margin reduce -> void
foreign import ccall "THNN.h THNN_FloatMultiMarginCriterion_updateGradInput"
  c_MultiMarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CInt -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ()

-- | c_PReLU_updateOutput :  state input output weight -> void
foreign import ccall "THNN.h THNN_FloatPReLU_updateOutput"
  c_PReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_PReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_FloatPReLU_updateGradInput"
  c_PReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_PReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THNN_FloatPReLU_accGradParameters"
  c_PReLU_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_Linear_updateOutput :  state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THNN_FloatLinear_updateOutput"
  c_Linear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Linear_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_FloatLinear_updateGradInput"
  c_Linear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Linear_accGradParameters :  state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THNN_FloatLinear_accGradParameters"
  c_Linear_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_RReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THNN_FloatRReLU_updateOutput"
  c_RReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr C'THGenerator -> IO ()

-- | c_RReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THNN_FloatRReLU_updateGradInput"
  c_RReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- | c_Sigmoid_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_FloatSigmoid_updateOutput"
  c_Sigmoid_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Sigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatSigmoid_updateGradInput"
  c_Sigmoid_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSmoothL1Criterion_updateOutput"
  c_SmoothL1Criterion_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_SmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_FloatSmoothL1Criterion_updateGradInput"
  c_SmoothL1Criterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ()

-- | c_SoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_FloatSoftMax_updateOutput"
  c_SoftMax_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | c_SoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_FloatSoftMax_updateGradInput"
  c_SoftMax_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | c_SoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THNN.h THNN_FloatSoftPlus_updateOutput"
  c_SoftPlus_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THNN_FloatSoftPlus_updateGradInput"
  c_SoftPlus_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THNN.h THNN_FloatSoftShrink_updateOutput"
  c_SoftShrink_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_SoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_FloatSoftShrink_updateGradInput"
  c_SoftShrink_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_IndexLinear_updateOutput :  state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_updateOutput"
  c_IndexLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_IndexLinear_accGradParameters :  state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_accGradParameters"
  c_IndexLinear_accGradParameters :: Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_IndexLinear_accUpdateGradParameters :  state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_accUpdateGradParameters"
  c_IndexLinear_accUpdateGradParameters :: Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_IndexLinear_updateParameters :  state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h THNN_FloatIndexLinear_updateParameters"
  c_IndexLinear_updateParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_updateOutput"
  c_SparseLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_accGradParameters"
  c_SparseLinear_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_zeroGradParameters"
  c_SparseLinear_zeroGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_updateParameters"
  c_SparseLinear_updateParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_SparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyUpdateOutput"
  c_SparseLinear_legacyUpdateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyAccGradParameters"
  c_SparseLinear_legacyAccGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_legacyZeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyZeroGradParameters"
  c_SparseLinear_legacyZeroGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SparseLinear_legacyUpdateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_FloatSparseLinear_legacyUpdateParameters"
  c_SparseLinear_legacyUpdateParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_Sqrt_updateOutput :  state input output eps -> void
foreign import ccall "THNN.h THNN_FloatSqrt_updateOutput"
  c_Sqrt_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_Sqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatSqrt_updateGradInput"
  c_Sqrt_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Square_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_FloatSquare_updateOutput"
  c_Square_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Square_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatSquare_updateGradInput"
  c_Square_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Tanh_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_FloatTanh_updateOutput"
  c_Tanh_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Tanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_FloatTanh_updateGradInput"
  c_Tanh_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_Threshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THNN.h THNN_FloatThreshold_updateOutput"
  c_Threshold_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Threshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THNN_FloatThreshold_updateGradInput"
  c_Threshold_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_TemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_updateOutput"
  c_TemporalConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_updateGradInput"
  c_TemporalConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalConvolution_accGradParameters"
  c_TemporalConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalMaxPooling_updateOutput :  state input output indices kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalMaxPooling_updateOutput"
  c_TemporalMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ()

-- | c_TemporalMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalMaxPooling_updateGradInput"
  c_TemporalMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_updateOutput :  state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_updateOutput"
  c_TemporalSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_updateGradInput"
  c_TemporalSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalSubSampling_accGradParameters"
  c_TemporalSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalRowConvolution_updateOutput :  state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_updateOutput"
  c_TemporalRowConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_updateGradInput"
  c_TemporalRowConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THNN_FloatTemporalRowConvolution_accGradParameters"
  c_TemporalRowConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- | c_TemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingNearest_updateOutput"
  c_TemporalUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingNearest_updateGradInput"
  c_TemporalUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateOutput :  state input output osizeW -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingLinear_updateOutput"
  c_TemporalUpSamplingLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h THNN_FloatTemporalUpSamplingLinear_updateGradInput"
  c_TemporalUpSamplingLinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_BatchNormalization_updateOutput :  state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THNN_FloatBatchNormalization_updateOutput"
  c_BatchNormalization_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BatchNormalization_backward :  state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THNN_FloatBatchNormalization_backward"
  c_BatchNormalization_backward :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_SpatialConvolutionMap_updateOutput :  state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_updateOutput"
  c_SpatialConvolutionMap_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMap_updateGradInput :  state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_updateGradInput"
  c_SpatialConvolutionMap_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMap_accGradParameters :  state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMap_accGradParameters"
  c_SpatialConvolutionMap_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialConvolutionMM_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_updateOutput"
  c_SpatialConvolutionMM_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_updateGradInput"
  c_SpatialConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionMM_accGradParameters"
  c_SpatialConvolutionMM_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_updateOutput"
  c_SpatialConvolutionLocal_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_updateGradInput"
  c_SpatialConvolutionLocal_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialConvolutionLocal_accGradParameters"
  c_SpatialConvolutionLocal_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_SpatialAdaptiveMaxPooling_updateOutput :  state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveMaxPooling_updateOutput"
  c_SpatialAdaptiveMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveMaxPooling_updateGradInput :  state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput"
  c_SpatialAdaptiveMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveAveragePooling_updateOutput"
  c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput"
  c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatSpatialAveragePooling_updateOutput"
  c_SpatialAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatSpatialAveragePooling_updateGradInput"
  c_SpatialAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialFractionalMaxPooling_updateOutput :  state input output outputW outputH kW kH indices randomSamples -> void
foreign import ccall "THNN.h THNN_FloatSpatialFractionalMaxPooling_updateOutput"
  c_SpatialFractionalMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_SpatialFractionalMaxPooling_updateGradInput :  state input gradOutput gradInput outputW outputH kW kH indices -> void
foreign import ccall "THNN.h THNN_FloatSpatialFractionalMaxPooling_updateGradInput"
  c_SpatialFractionalMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> IO ()

-- | c_SpatialFullConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_updateOutput"
  c_SpatialFullConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_updateGradInput"
  c_SpatialFullConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolution_accGradParameters"
  c_SpatialFullConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullConvolutionMap_updateOutput :  state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_updateOutput"
  c_SpatialFullConvolutionMap_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolutionMap_updateGradInput :  state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_updateGradInput"
  c_SpatialFullConvolutionMap_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolutionMap_accGradParameters :  state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullConvolutionMap_accGradParameters"
  c_SpatialFullConvolutionMap_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_updateOutput"
  c_SpatialDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_updateGradInput"
  c_SpatialDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedConvolution_accGradParameters"
  c_SpatialDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_updateOutput"
  c_SpatialFullDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_updateGradInput"
  c_SpatialFullDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialFullDilatedConvolution_accGradParameters"
  c_SpatialFullDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialMaxPooling_updateOutput :  state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxPooling_updateOutput"
  c_SpatialMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxPooling_updateGradInput"
  c_SpatialMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialDilatedMaxPooling_updateOutput :  state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedMaxPooling_updateOutput"
  c_SpatialDilatedMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialDilatedMaxPooling_updateGradInput :  state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialDilatedMaxPooling_updateGradInput"
  c_SpatialDilatedMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_SpatialMaxUnpooling_updateOutput :  state input output indices owidth oheight -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxUnpooling_updateOutput"
  c_SpatialMaxUnpooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialMaxUnpooling_updateGradInput :  state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h THNN_FloatSpatialMaxUnpooling_updateGradInput"
  c_SpatialMaxUnpooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_updateOutput"
  c_SpatialSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_updateGradInput"
  c_SpatialSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THNN_FloatSpatialSubSampling_accGradParameters"
  c_SpatialSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingNearest_updateOutput"
  c_SpatialUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingNearest_updateGradInput"
  c_SpatialUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateOutput :  state input output osizeH osizeW -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingBilinear_updateOutput"
  c_SpatialUpSamplingBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h THNN_FloatSpatialUpSamplingBilinear_updateGradInput"
  c_SpatialUpSamplingBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialGridSamplerBilinear_updateOutput"
  c_SpatialGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_FloatSpatialGridSamplerBilinear_updateGradInput"
  c_SpatialGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricGridSamplerBilinear_updateOutput"
  c_VolumetricGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricGridSamplerBilinear_updateGradInput"
  c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_unfolded_acc :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h THNN_Floatunfolded_acc"
  c_unfolded_acc :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_unfolded_copy :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Floatunfolded_copy"
  c_unfolded_copy :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAveragePooling_updateOutput"
  c_VolumetricAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAveragePooling_updateGradInput"
  c_VolumetricAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricConvolution_updateOutput :  state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_updateOutput"
  c_VolumetricConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_updateGradInput"
  c_VolumetricConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolution_accGradParameters"
  c_VolumetricConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricConvolutionMM_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_updateOutput"
  c_VolumetricConvolutionMM_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_updateGradInput"
  c_VolumetricConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricConvolutionMM_accGradParameters"
  c_VolumetricConvolutionMM_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFractionalMaxPooling_updateOutput :  state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFractionalMaxPooling_updateOutput"
  c_VolumetricFractionalMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_VolumetricFractionalMaxPooling_updateGradInput :  state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFractionalMaxPooling_updateGradInput"
  c_VolumetricFractionalMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> IO ()

-- | c_VolumetricFullConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_updateOutput"
  c_VolumetricFullConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_updateGradInput"
  c_VolumetricFullConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullConvolution_accGradParameters"
  c_VolumetricFullConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricDilatedConvolution_updateOutput :  state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_updateOutput"
  c_VolumetricDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_updateGradInput"
  c_VolumetricDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedConvolution_accGradParameters"
  c_VolumetricDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_updateOutput"
  c_VolumetricFullDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_updateGradInput"
  c_VolumetricFullDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_FloatVolumetricFullDilatedConvolution_accGradParameters"
  c_VolumetricFullDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricMaxPooling_updateOutput :  state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxPooling_updateOutput"
  c_VolumetricMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricMaxPooling_updateGradInput :  state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxPooling_updateGradInput"
  c_VolumetricMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricDilatedMaxPooling_updateOutput :  state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedMaxPooling_updateOutput"
  c_VolumetricDilatedMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricDilatedMaxPooling_updateGradInput :  state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h THNN_FloatVolumetricDilatedMaxPooling_updateGradInput"
  c_VolumetricDilatedMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricMaxUnpooling_updateOutput :  state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxUnpooling_updateOutput"
  c_VolumetricMaxUnpooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricMaxUnpooling_updateGradInput :  state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricMaxUnpooling_updateGradInput"
  c_VolumetricMaxUnpooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput"
  c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput"
  c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_VolumetricAdaptiveMaxPooling_updateOutput :  state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput"
  c_VolumetricAdaptiveMaxPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveMaxPooling_updateGradInput :  state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput"
  c_VolumetricAdaptiveMaxPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> IO ()

-- | c_SpatialReflectionPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_FloatSpatialReflectionPadding_updateOutput"
  c_SpatialReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_FloatSpatialReflectionPadding_updateGradInput"
  c_SpatialReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_FloatSpatialReplicationPadding_updateOutput"
  c_SpatialReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_FloatSpatialReplicationPadding_updateGradInput"
  c_SpatialReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_FeatureLPPooling_updateOutput :  state input output power width stride batchMode -> void
foreign import ccall "THNN.h THNN_FloatFeatureLPPooling_updateOutput"
  c_FeatureLPPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_FeatureLPPooling_updateGradInput :  state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THNN_FloatFeatureLPPooling_updateGradInput"
  c_FeatureLPPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNN_FloatVolumetricReplicationPadding_updateOutput"
  c_VolumetricReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNN_FloatVolumetricReplicationPadding_updateGradInput"
  c_VolumetricReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingNearest_updateOutput"
  c_VolumetricUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingNearest_updateGradInput"
  c_VolumetricUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateOutput :  state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingTrilinear_updateOutput"
  c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput"
  c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNN_FloatTemporalReflectionPadding_updateOutput"
  c_TemporalReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNN_FloatTemporalReflectionPadding_updateGradInput"
  c_TemporalReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNN_FloatTemporalReplicationPadding_updateOutput"
  c_TemporalReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNN_FloatTemporalReplicationPadding_updateGradInput"
  c_TemporalReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | p_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatAbs_updateOutput"
  p_Abs_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatAbs_updateGradInput"
  p_Abs_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatAbsCriterion_updateOutput"
  p_AbsCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatAbsCriterion_updateGradInput"
  p_AbsCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights reduce -> void
foreign import ccall "THNN.h &THNN_FloatBCECriterion_updateOutput"
  p_BCECriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> CBool -> IO ())

-- | p_BCECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THNN.h &THNN_FloatBCECriterion_updateGradInput"
  p_BCECriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> CBool -> IO ())

-- | p_ClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatClassNLLCriterion_updateOutput"
  p_ClassNLLCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ())

-- | p_ClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatClassNLLCriterion_updateGradInput"
  p_ClassNLLCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ())

-- | p_SpatialClassNLLCriterion_updateOutput : Pointer to function : state input target output sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatSpatialClassNLLCriterion_updateOutput"
  p_SpatialClassNLLCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ())

-- | p_SpatialClassNLLCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights total_weight ignore_index reduce -> void
foreign import ccall "THNN.h &THNN_FloatSpatialClassNLLCriterion_updateGradInput"
  p_SpatialClassNLLCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CBool -> IO ())

-- | p_ELU_updateOutput : Pointer to function : state input output alpha scale inplace -> void
foreign import ccall "THNN.h &THNN_FloatELU_updateOutput"
  p_ELU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha scale -> void
foreign import ccall "THNN.h &THNN_FloatELU_updateGradInput"
  p_ELU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatDistKLDivCriterion_updateOutput"
  p_DistKLDivCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatDistKLDivCriterion_updateGradInput"
  p_DistKLDivCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatGatedLinear_updateOutput"
  p_GatedLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THNN_FloatGatedLinear_updateGradInput"
  p_GatedLinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_HardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_FloatHardShrink_updateOutput"
  p_HardShrink_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_HardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_FloatHardShrink_updateGradInput"
  p_HardShrink_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_FloatHardTanh_updateOutput"
  p_HardTanh_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_FloatHardTanh_updateGradInput"
  p_HardTanh_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Im2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_FloatIm2Col_updateOutput"
  p_Im2Col_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Im2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_FloatIm2Col_updateGradInput"
  p_Im2Col_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_FloatCol2Im_updateOutput"
  p_Col2Im_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateGradInput : Pointer to function : state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_FloatCol2Im_updateGradInput"
  p_Col2Im_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatL1Cost_updateOutput"
  p_L1Cost_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatL1Cost_updateGradInput"
  p_L1Cost_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THNN_FloatLeakyReLU_updateOutput"
  p_LeakyReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ())

-- | p_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THNN_FloatLeakyReLU_updateGradInput"
  p_LeakyReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ())

-- | p_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THNN_FloatGRUFused_updateOutput"
  p_GRUFused_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THNN_FloatGRUFused_updateGradInput"
  p_GRUFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THNN_FloatLSTMFused_updateOutput"
  p_LSTMFused_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THNN_FloatLSTMFused_updateGradInput"
  p_LSTMFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THNN_FloatLogSigmoid_updateOutput"
  p_LogSigmoid_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THNN_FloatLogSigmoid_updateGradInput"
  p_LogSigmoid_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatLogSoftMax_updateOutput"
  p_LogSoftMax_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | p_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_FloatLogSoftMax_updateGradInput"
  p_LogSoftMax_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | p_LookupTable_accGradParameters : Pointer to function : state input gradOutput gradWeight count sorted indices scaleGradByFreq paddingValue scale -> void
foreign import ccall "THNN.h &THNN_FloatLookupTable_accGradParameters"
  p_LookupTable_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIntegerTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CBool -> CInt -> CDouble -> IO ())

-- | p_LookupTable_renorm : Pointer to function : state idx weight maxNorm normType -> void
foreign import ccall "THNN.h &THNN_FloatLookupTable_renorm"
  p_LookupTable_renorm :: FunPtr (Ptr C'THNNState -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_FloatMarginCriterion_updateOutput"
  p_MarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> IO ())

-- | p_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_FloatMarginCriterion_updateGradInput"
  p_MarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> IO ())

-- | p_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSoftMarginCriterion_updateOutput"
  p_SoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSoftMarginCriterion_updateGradInput"
  p_SoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMSECriterion_updateOutput"
  p_MSECriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMSECriterion_updateGradInput"
  p_MSECriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_MultiLabelMarginCriterion_updateOutput : Pointer to function : state input target output isTarget sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMultiLabelMarginCriterion_updateOutput"
  p_MultiLabelMarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_MultiLabelMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput isTarget sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatMultiLabelMarginCriterion_updateGradInput"
  p_MultiLabelMarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_MultiMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage p weights margin reduce -> void
foreign import ccall "THNN.h &THNN_FloatMultiMarginCriterion_updateOutput"
  p_MultiMarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> CBool -> CInt -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ())

-- | p_MultiMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage p weights margin reduce -> void
foreign import ccall "THNN.h &THNN_FloatMultiMarginCriterion_updateGradInput"
  p_MultiMarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CInt -> Ptr C'THFloatTensor -> CDouble -> CBool -> IO ())

-- | p_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_updateOutput"
  p_PReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_updateGradInput"
  p_PReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THNN_FloatPReLU_accGradParameters"
  p_PReLU_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_Linear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THNN_FloatLinear_updateOutput"
  p_Linear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Linear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_FloatLinear_updateGradInput"
  p_Linear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Linear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THNN_FloatLinear_accGradParameters"
  p_Linear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THNN_FloatRReLU_updateOutput"
  p_RReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr C'THGenerator -> IO ())

-- | p_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THNN_FloatRReLU_updateGradInput"
  p_RReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- | p_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatSigmoid_updateOutput"
  p_Sigmoid_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatSigmoid_updateGradInput"
  p_Sigmoid_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSmoothL1Criterion_updateOutput"
  p_SmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_FloatSmoothL1Criterion_updateGradInput"
  p_SmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CBool -> IO ())

-- | p_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_FloatSoftMax_updateOutput"
  p_SoftMax_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | p_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_FloatSoftMax_updateGradInput"
  p_SoftMax_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | p_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THNN_FloatSoftPlus_updateOutput"
  p_SoftPlus_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THNN_FloatSoftPlus_updateGradInput"
  p_SoftPlus_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_FloatSoftShrink_updateOutput"
  p_SoftShrink_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_FloatSoftShrink_updateGradInput"
  p_SoftShrink_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_IndexLinear_updateOutput : Pointer to function : state keys keysOffset values sizes cumSumSizes output weight bias normalizedValues train -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_updateOutput"
  p_IndexLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_IndexLinear_accGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput gradWeight gradBias weight bias valuesBuffer weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_accGradParameters"
  p_IndexLinear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_IndexLinear_accUpdateGradParameters : Pointer to function : state keys keysOffset values sizes cumSumSizes gradOutput weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_accUpdateGradParameters"
  p_IndexLinear_accUpdateGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THIndexTensor -> CLLong -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_IndexLinear_updateParameters : Pointer to function : state gradWeight gradBias weight bias runningKeys cumSumSizes keysOffset weightDecay learningRate -> void
foreign import ccall "THNN.h &THNN_FloatIndexLinear_updateParameters"
  p_IndexLinear_updateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> Ptr C'THIndexTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_updateOutput"
  p_SparseLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_accGradParameters"
  p_SparseLinear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_zeroGradParameters"
  p_SparseLinear_zeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_updateParameters"
  p_SparseLinear_updateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyUpdateOutput"
  p_SparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyAccGradParameters"
  p_SparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyZeroGradParameters"
  p_SparseLinear_legacyZeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_FloatSparseLinear_legacyUpdateParameters"
  p_SparseLinear_legacyUpdateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THNN_FloatSqrt_updateOutput"
  p_Sqrt_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatSqrt_updateGradInput"
  p_Sqrt_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatSquare_updateOutput"
  p_Square_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatSquare_updateGradInput"
  p_Square_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_FloatTanh_updateOutput"
  p_Tanh_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_FloatTanh_updateGradInput"
  p_Tanh_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THNN_FloatThreshold_updateOutput"
  p_Threshold_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THNN_FloatThreshold_updateGradInput"
  p_Threshold_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_updateOutput"
  p_TemporalConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_updateGradInput"
  p_TemporalConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalConvolution_accGradParameters"
  p_TemporalConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalMaxPooling_updateOutput : Pointer to function : state input output indices kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalMaxPooling_updateOutput"
  p_TemporalMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ())

-- | p_TemporalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalMaxPooling_updateGradInput"
  p_TemporalMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_updateOutput"
  p_TemporalSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_updateGradInput"
  p_TemporalSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalSubSampling_accGradParameters"
  p_TemporalSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_updateOutput"
  p_TemporalRowConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_updateGradInput"
  p_TemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THNN_FloatTemporalRowConvolution_accGradParameters"
  p_TemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- | p_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingNearest_updateOutput"
  p_TemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingNearest_updateGradInput"
  p_TemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output osizeW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingLinear_updateOutput"
  p_TemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h &THNN_FloatTemporalUpSamplingLinear_updateGradInput"
  p_TemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_BatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THNN_FloatBatchNormalization_updateOutput"
  p_BatchNormalization_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THNN_FloatBatchNormalization_backward"
  p_BatchNormalization_backward :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_SpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_updateOutput"
  p_SpatialConvolutionMap_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_updateGradInput"
  p_SpatialConvolutionMap_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMap_accGradParameters"
  p_SpatialConvolutionMap_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_updateOutput"
  p_SpatialConvolutionMM_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_updateGradInput"
  p_SpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionMM_accGradParameters"
  p_SpatialConvolutionMM_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_updateOutput"
  p_SpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_updateGradInput"
  p_SpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialConvolutionLocal_accGradParameters"
  p_SpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_SpatialAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveMaxPooling_updateOutput"
  p_SpatialAdaptiveMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput"
  p_SpatialAdaptiveMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveAveragePooling_updateOutput"
  p_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput"
  p_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAveragePooling_updateOutput"
  p_SpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatSpatialAveragePooling_updateGradInput"
  p_SpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialFractionalMaxPooling_updateOutput : Pointer to function : state input output outputW outputH kW kH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFractionalMaxPooling_updateOutput"
  p_SpatialFractionalMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_SpatialFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputW outputH kW kH indices -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFractionalMaxPooling_updateGradInput"
  p_SpatialFractionalMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> IO ())

-- | p_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_updateOutput"
  p_SpatialFullConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_updateGradInput"
  p_SpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolution_accGradParameters"
  p_SpatialFullConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_updateOutput"
  p_SpatialFullConvolutionMap_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_updateGradInput"
  p_SpatialFullConvolutionMap_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullConvolutionMap_accGradParameters"
  p_SpatialFullConvolutionMap_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_updateOutput"
  p_SpatialDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_updateGradInput"
  p_SpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedConvolution_accGradParameters"
  p_SpatialDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_updateOutput"
  p_SpatialFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_updateGradInput"
  p_SpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialFullDilatedConvolution_accGradParameters"
  p_SpatialFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxPooling_updateOutput"
  p_SpatialMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxPooling_updateGradInput"
  p_SpatialMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedMaxPooling_updateOutput"
  p_SpatialDilatedMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kW kH dW dH padW padH dilationW dilationH ceil_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialDilatedMaxPooling_updateGradInput"
  p_SpatialDilatedMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_SpatialMaxUnpooling_updateOutput : Pointer to function : state input output indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxUnpooling_updateOutput"
  p_SpatialMaxUnpooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices owidth oheight -> void
foreign import ccall "THNN.h &THNN_FloatSpatialMaxUnpooling_updateGradInput"
  p_SpatialMaxUnpooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_updateOutput"
  p_SpatialSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_updateGradInput"
  p_SpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THNN_FloatSpatialSubSampling_accGradParameters"
  p_SpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingNearest_updateOutput"
  p_SpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingNearest_updateGradInput"
  p_SpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingBilinear_updateOutput"
  p_SpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_FloatSpatialUpSamplingBilinear_updateGradInput"
  p_SpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialGridSamplerBilinear_updateOutput"
  p_SpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatSpatialGridSamplerBilinear_updateGradInput"
  p_SpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricGridSamplerBilinear_updateOutput"
  p_VolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricGridSamplerBilinear_updateGradInput"
  p_VolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_unfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h &THNN_Floatunfolded_acc"
  p_unfolded_acc :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_unfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Floatunfolded_copy"
  p_unfolded_copy :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAveragePooling_updateOutput"
  p_VolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAveragePooling_updateGradInput"
  p_VolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_updateOutput"
  p_VolumetricConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_updateGradInput"
  p_VolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolution_accGradParameters"
  p_VolumetricConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_updateOutput"
  p_VolumetricConvolutionMM_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_updateGradInput"
  p_VolumetricConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricConvolutionMM_accGradParameters"
  p_VolumetricConvolutionMM_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFractionalMaxPooling_updateOutput : Pointer to function : state input output outputT outputW outputH poolSizeT poolSizeW poolSizeH indices randomSamples -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFractionalMaxPooling_updateOutput"
  p_VolumetricFractionalMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_VolumetricFractionalMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput outputT outputW outputH poolSizeT poolSizeW poolSizeH indices -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFractionalMaxPooling_updateGradInput"
  p_VolumetricFractionalMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr C'THIndexTensor -> IO ())

-- | p_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_updateOutput"
  p_VolumetricFullConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_updateGradInput"
  p_VolumetricFullConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullConvolution_accGradParameters"
  p_VolumetricFullConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_updateOutput"
  p_VolumetricDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_updateGradInput"
  p_VolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedConvolution_accGradParameters"
  p_VolumetricDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_updateOutput"
  p_VolumetricFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_updateGradInput"
  p_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricFullDilatedConvolution_accGradParameters"
  p_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxPooling_updateOutput"
  p_VolumetricMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxPooling_updateGradInput"
  p_VolumetricMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricDilatedMaxPooling_updateOutput : Pointer to function : state input output indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedMaxPooling_updateOutput"
  p_VolumetricDilatedMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricDilatedMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH ceilMode -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricDilatedMaxPooling_updateGradInput"
  p_VolumetricDilatedMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricMaxUnpooling_updateOutput : Pointer to function : state input output indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxUnpooling_updateOutput"
  p_VolumetricMaxUnpooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricMaxUnpooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices oT oW oH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricMaxUnpooling_updateGradInput"
  p_VolumetricMaxUnpooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput"
  p_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput"
  p_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_VolumetricAdaptiveMaxPooling_updateOutput : Pointer to function : state input output indices osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput"
  p_VolumetricAdaptiveMaxPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveMaxPooling_updateGradInput : Pointer to function : state input gradOutput gradInput indices -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput"
  p_VolumetricAdaptiveMaxPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIndexTensor -> IO ())

-- | p_SpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReflectionPadding_updateOutput"
  p_SpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReflectionPadding_updateGradInput"
  p_SpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReplicationPadding_updateOutput"
  p_SpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_FloatSpatialReplicationPadding_updateGradInput"
  p_SpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_FeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_FloatFeatureLPPooling_updateOutput"
  p_FeatureLPPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_FloatFeatureLPPooling_updateGradInput"
  p_FeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricReplicationPadding_updateOutput"
  p_VolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricReplicationPadding_updateGradInput"
  p_VolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingNearest_updateOutput"
  p_VolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingNearest_updateGradInput"
  p_VolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingTrilinear_updateOutput"
  p_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput"
  p_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReflectionPadding_updateOutput"
  p_TemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReflectionPadding_updateGradInput"
  p_TemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReplicationPadding_updateOutput"
  p_TemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_FloatTemporalReplicationPadding_updateGradInput"
  p_TemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

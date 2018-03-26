{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.NN.Double where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_Abs_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_DoubleAbs_updateOutput"
  c_Abs_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Abs_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleAbs_updateGradInput"
  c_Abs_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_AbsCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleAbsCriterion_updateOutput"
  c_AbsCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_AbsCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleAbsCriterion_updateGradInput"
  c_AbsCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_BCECriterion_updateOutput :  state input target output sizeAverage weights reduce -> void
foreign import ccall "THNN.h THNN_DoubleBCECriterion_updateOutput"
  c_BCECriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> Ptr C'THDoubleTensor -> CBool -> IO ()

-- | c_BCECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THNN.h THNN_DoubleBCECriterion_updateGradInput"
  c_BCECriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> Ptr C'THDoubleTensor -> CBool -> IO ()

-- | c_ELU_updateOutput :  state input output alpha scale inplace -> void
foreign import ccall "THNN.h THNN_DoubleELU_updateOutput"
  c_ELU_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_ELU_updateGradInput :  state gradOutput gradInput output alpha scale -> void
foreign import ccall "THNN.h THNN_DoubleELU_updateGradInput"
  c_ELU_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_DistKLDivCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleDistKLDivCriterion_updateOutput"
  c_DistKLDivCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_DistKLDivCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleDistKLDivCriterion_updateGradInput"
  c_DistKLDivCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_GatedLinear_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleGatedLinear_updateOutput"
  c_GatedLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_GatedLinear_updateGradInput :  state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h THNN_DoubleGatedLinear_updateGradInput"
  c_GatedLinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_HardShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THNN.h THNN_DoubleHardShrink_updateOutput"
  c_HardShrink_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_HardShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_DoubleHardShrink_updateGradInput"
  c_HardShrink_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_HardTanh_updateOutput :  state input output min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_DoubleHardTanh_updateOutput"
  c_HardTanh_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_HardTanh_updateGradInput :  state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h THNN_DoubleHardTanh_updateGradInput"
  c_HardTanh_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Im2Col_updateOutput :  state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_DoubleIm2Col_updateOutput"
  c_Im2Col_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Im2Col_updateGradInput :  state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_DoubleIm2Col_updateGradInput"
  c_Im2Col_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateOutput :  state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_DoubleCol2Im_updateOutput"
  c_Col2Im_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_Col2Im_updateGradInput :  state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h THNN_DoubleCol2Im_updateGradInput"
  c_Col2Im_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_L1Cost_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_DoubleL1Cost_updateOutput"
  c_L1Cost_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_L1Cost_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleL1Cost_updateGradInput"
  c_L1Cost_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LeakyReLU_updateOutput :  state input output negval inplace -> void
foreign import ccall "THNN.h THNN_DoubleLeakyReLU_updateOutput"
  c_LeakyReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_LeakyReLU_updateGradInput :  state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h THNN_DoubleLeakyReLU_updateGradInput"
  c_LeakyReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CBool -> IO ()

-- | c_GRUFused_updateOutput :  state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h THNN_DoubleGRUFused_updateOutput"
  c_GRUFused_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_GRUFused_updateGradInput :  state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h THNN_DoubleGRUFused_updateGradInput"
  c_GRUFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LSTMFused_updateOutput :  state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h THNN_DoubleLSTMFused_updateOutput"
  c_LSTMFused_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LSTMFused_updateGradInput :  state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h THNN_DoubleLSTMFused_updateGradInput"
  c_LSTMFused_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LogSigmoid_updateOutput :  state input output buffer -> void
foreign import ccall "THNN.h THNN_DoubleLogSigmoid_updateOutput"
  c_LogSigmoid_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LogSigmoid_updateGradInput :  state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h THNN_DoubleLogSigmoid_updateGradInput"
  c_LogSigmoid_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_LogSoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleLogSoftMax_updateOutput"
  c_LogSoftMax_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | c_LogSoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_DoubleLogSoftMax_updateGradInput"
  c_LogSoftMax_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | c_MarginCriterion_updateOutput :  state input target output sizeAverage margin -> void
foreign import ccall "THNN.h THNN_DoubleMarginCriterion_updateOutput"
  c_MarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> IO ()

-- | c_MarginCriterion_updateGradInput :  state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h THNN_DoubleMarginCriterion_updateGradInput"
  c_MarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> IO ()

-- | c_SoftMarginCriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSoftMarginCriterion_updateOutput"
  c_SoftMarginCriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SoftMarginCriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSoftMarginCriterion_updateGradInput"
  c_SoftMarginCriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleMSECriterion_updateOutput"
  c_MSECriterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_MSECriterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleMSECriterion_updateGradInput"
  c_MSECriterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_PReLU_updateOutput :  state input output weight -> void
foreign import ccall "THNN.h THNN_DoublePReLU_updateOutput"
  c_PReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_PReLU_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_DoublePReLU_updateGradInput"
  c_PReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_PReLU_accGradParameters :  state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h THNN_DoublePReLU_accGradParameters"
  c_PReLU_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_Linear_updateOutput :  state input output weight bias addBuffer -> void
foreign import ccall "THNN.h THNN_DoubleLinear_updateOutput"
  c_Linear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Linear_updateGradInput :  state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h THNN_DoubleLinear_updateGradInput"
  c_Linear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Linear_accGradParameters :  state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h THNN_DoubleLinear_accGradParameters"
  c_Linear_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_RReLU_updateOutput :  state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h THNN_DoubleRReLU_updateOutput"
  c_RReLU_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr C'THGenerator -> IO ()

-- | c_RReLU_updateGradInput :  state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h THNN_DoubleRReLU_updateGradInput"
  c_RReLU_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ()

-- | c_Sigmoid_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_DoubleSigmoid_updateOutput"
  c_Sigmoid_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Sigmoid_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleSigmoid_updateGradInput"
  c_Sigmoid_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SmoothL1Criterion_updateOutput :  state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSmoothL1Criterion_updateOutput"
  c_SmoothL1Criterion_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SmoothL1Criterion_updateGradInput :  state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h THNN_DoubleSmoothL1Criterion_updateGradInput"
  c_SmoothL1Criterion_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ()

-- | c_SoftMax_updateOutput :  state input output dim -> void
foreign import ccall "THNN.h THNN_DoubleSoftMax_updateOutput"
  c_SoftMax_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | c_SoftMax_updateGradInput :  state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h THNN_DoubleSoftMax_updateGradInput"
  c_SoftMax_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | c_SoftPlus_updateOutput :  state input output beta threshold -> void
foreign import ccall "THNN.h THNN_DoubleSoftPlus_updateOutput"
  c_SoftPlus_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftPlus_updateGradInput :  state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h THNN_DoubleSoftPlus_updateGradInput"
  c_SoftPlus_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SoftShrink_updateOutput :  state input output lambda -> void
foreign import ccall "THNN.h THNN_DoubleSoftShrink_updateOutput"
  c_SoftShrink_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_SoftShrink_updateGradInput :  state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h THNN_DoubleSoftShrink_updateGradInput"
  c_SoftShrink_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_SparseLinear_updateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_updateOutput"
  c_SparseLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SparseLinear_accGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_accGradParameters"
  c_SparseLinear_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_zeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_zeroGradParameters"
  c_SparseLinear_zeroGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SparseLinear_updateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_updateParameters"
  c_SparseLinear_updateParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_SparseLinear_legacyUpdateOutput :  state input output weight bias -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyUpdateOutput"
  c_SparseLinear_legacyUpdateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SparseLinear_legacyAccGradParameters :  state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyAccGradParameters"
  c_SparseLinear_legacyAccGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_SparseLinear_legacyZeroGradParameters :  state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyZeroGradParameters"
  c_SparseLinear_legacyZeroGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SparseLinear_legacyUpdateParameters :  state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h THNN_DoubleSparseLinear_legacyUpdateParameters"
  c_SparseLinear_legacyUpdateParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_Sqrt_updateOutput :  state input output eps -> void
foreign import ccall "THNN.h THNN_DoubleSqrt_updateOutput"
  c_Sqrt_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | c_Sqrt_updateGradInput :  state input gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleSqrt_updateGradInput"
  c_Sqrt_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Square_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_DoubleSquare_updateOutput"
  c_Square_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Square_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleSquare_updateGradInput"
  c_Square_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Tanh_updateOutput :  state input output -> void
foreign import ccall "THNN.h THNN_DoubleTanh_updateOutput"
  c_Tanh_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Tanh_updateGradInput :  state gradOutput gradInput output -> void
foreign import ccall "THNN.h THNN_DoubleTanh_updateGradInput"
  c_Tanh_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_Threshold_updateOutput :  state input output threshold val inplace -> void
foreign import ccall "THNN.h THNN_DoubleThreshold_updateOutput"
  c_Threshold_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_Threshold_updateGradInput :  state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h THNN_DoubleThreshold_updateGradInput"
  c_Threshold_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ()

-- | c_TemporalConvolution_updateOutput :  state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_updateOutput"
  c_TemporalConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_updateGradInput"
  c_TemporalConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalConvolution_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalConvolution_accGradParameters"
  c_TemporalConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalSubSampling_updateOutput :  state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_updateOutput"
  c_TemporalSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_updateGradInput :  state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_updateGradInput"
  c_TemporalSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalSubSampling_accGradParameters"
  c_TemporalSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CDouble -> IO ()

-- | c_TemporalRowConvolution_updateOutput :  state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_updateOutput"
  c_TemporalRowConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_updateGradInput"
  c_TemporalRowConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ()

-- | c_TemporalRowConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h THNN_DoubleTemporalRowConvolution_accGradParameters"
  c_TemporalRowConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ()

-- | c_TemporalUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingNearest_updateOutput"
  c_TemporalUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingNearest_updateGradInput"
  c_TemporalUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateOutput :  state input output osizeW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingLinear_updateOutput"
  c_TemporalUpSamplingLinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_TemporalUpSamplingLinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h THNN_DoubleTemporalUpSamplingLinear_updateGradInput"
  c_TemporalUpSamplingLinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_BatchNormalization_updateOutput :  state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h THNN_DoubleBatchNormalization_updateOutput"
  c_BatchNormalization_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_BatchNormalization_backward :  state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h THNN_DoubleBatchNormalization_backward"
  c_BatchNormalization_backward :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> CDouble -> IO ()

-- | c_SpatialConvolutionMap_updateOutput :  state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_updateOutput"
  c_SpatialConvolutionMap_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMap_updateGradInput :  state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_updateGradInput"
  c_SpatialConvolutionMap_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMap_accGradParameters :  state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMap_accGradParameters"
  c_SpatialConvolutionMap_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialConvolutionMM_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_updateOutput"
  c_SpatialConvolutionMM_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_updateGradInput"
  c_SpatialConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionMM_accGradParameters"
  c_SpatialConvolutionMM_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialConvolutionLocal_updateOutput :  state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_updateOutput"
  c_SpatialConvolutionLocal_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_updateGradInput"
  c_SpatialConvolutionLocal_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_SpatialConvolutionLocal_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialConvolutionLocal_accGradParameters"
  c_SpatialConvolutionLocal_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateOutput :  state input output osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput"
  c_SpatialAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_SpatialAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput"
  c_SpatialAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SpatialAveragePooling_updateOutput :  state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAveragePooling_updateOutput"
  c_SpatialAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialAveragePooling_updateGradInput :  state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleSpatialAveragePooling_updateGradInput"
  c_SpatialAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_SpatialFullConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_updateOutput"
  c_SpatialFullConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_updateGradInput"
  c_SpatialFullConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolution_accGradParameters"
  c_SpatialFullConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullConvolutionMap_updateOutput :  state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_updateOutput"
  c_SpatialFullConvolutionMap_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolutionMap_updateGradInput :  state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_updateGradInput"
  c_SpatialFullConvolutionMap_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullConvolutionMap_accGradParameters :  state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullConvolutionMap_accGradParameters"
  c_SpatialFullConvolutionMap_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_updateOutput"
  c_SpatialDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_updateGradInput"
  c_SpatialDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialDilatedConvolution_accGradParameters"
  c_SpatialDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialFullDilatedConvolution_updateOutput :  state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_updateOutput"
  c_SpatialFullDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_updateGradInput"
  c_SpatialFullDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialFullDilatedConvolution_accGradParameters"
  c_SpatialFullDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialSubSampling_updateOutput :  state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_updateOutput"
  c_SpatialSubSampling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_updateGradInput :  state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_updateGradInput"
  c_SpatialSubSampling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialSubSampling_accGradParameters :  state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h THNN_DoubleSpatialSubSampling_accGradParameters"
  c_SpatialSubSampling_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_SpatialUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingNearest_updateOutput"
  c_SpatialUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingNearest_updateGradInput"
  c_SpatialUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateOutput :  state input output osizeH osizeW -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingBilinear_updateOutput"
  c_SpatialUpSamplingBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_SpatialUpSamplingBilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h THNN_DoubleSpatialUpSamplingBilinear_updateGradInput"
  c_SpatialUpSamplingBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialGridSamplerBilinear_updateOutput"
  c_SpatialGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_SpatialGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleSpatialGridSamplerBilinear_updateGradInput"
  c_SpatialGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateOutput :  state input grid output padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricGridSamplerBilinear_updateOutput"
  c_VolumetricGridSamplerBilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_VolumetricGridSamplerBilinear_updateGradInput :  state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricGridSamplerBilinear_updateGradInput"
  c_VolumetricGridSamplerBilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_unfolded_acc :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h THNN_Doubleunfolded_acc"
  c_unfolded_acc :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_unfolded_copy :  finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h THNN_Doubleunfolded_copy"
  c_unfolded_copy :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAveragePooling_updateOutput :  state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAveragePooling_updateOutput"
  c_VolumetricAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricAveragePooling_updateGradInput :  state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAveragePooling_updateGradInput"
  c_VolumetricAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ()

-- | c_VolumetricConvolution_updateOutput :  state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_updateOutput"
  c_VolumetricConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_updateGradInput :  state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_updateGradInput"
  c_VolumetricConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolution_accGradParameters"
  c_VolumetricConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricConvolutionMM_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_updateOutput"
  c_VolumetricConvolutionMM_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolutionMM_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_updateGradInput"
  c_VolumetricConvolutionMM_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricConvolutionMM_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricConvolutionMM_accGradParameters"
  c_VolumetricConvolutionMM_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_updateOutput"
  c_VolumetricFullConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_updateGradInput"
  c_VolumetricFullConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullConvolution_accGradParameters"
  c_VolumetricFullConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricDilatedConvolution_updateOutput :  state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_updateOutput"
  c_VolumetricDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_updateGradInput"
  c_VolumetricDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricDilatedConvolution_accGradParameters"
  c_VolumetricDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateOutput :  state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_updateOutput"
  c_VolumetricFullDilatedConvolution_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_updateGradInput :  state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput"
  c_VolumetricFullDilatedConvolution_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricFullDilatedConvolution_accGradParameters :  state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters"
  c_VolumetricFullDilatedConvolution_accGradParameters :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateOutput :  state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput"
  c_VolumetricAdaptiveAveragePooling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricAdaptiveAveragePooling_updateGradInput :  state input gradOutput gradInput -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  c_VolumetricAdaptiveAveragePooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_SpatialReflectionPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReflectionPadding_updateOutput"
  c_SpatialReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReflectionPadding_updateGradInput"
  c_SpatialReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReplicationPadding_updateOutput"
  c_SpatialReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_SpatialReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h THNN_DoubleSpatialReplicationPadding_updateGradInput"
  c_SpatialReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_FeatureLPPooling_updateOutput :  state input output power width stride batchMode -> void
foreign import ccall "THNN.h THNN_DoubleFeatureLPPooling_updateOutput"
  c_FeatureLPPooling_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_FeatureLPPooling_updateGradInput :  state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h THNN_DoubleFeatureLPPooling_updateGradInput"
  c_FeatureLPPooling_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ()

-- | c_VolumetricReplicationPadding_updateOutput :  state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricReplicationPadding_updateOutput"
  c_VolumetricReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricReplicationPadding_updateGradInput"
  c_VolumetricReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateOutput :  state input output scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingNearest_updateOutput"
  c_VolumetricUpSamplingNearest_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingNearest_updateGradInput :  state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingNearest_updateGradInput"
  c_VolumetricUpSamplingNearest_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateOutput :  state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput"
  c_VolumetricUpSamplingTrilinear_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_VolumetricUpSamplingTrilinear_updateGradInput :  state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput"
  c_VolumetricUpSamplingTrilinear_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReflectionPadding_updateOutput"
  c_TemporalReflectionPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReflectionPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReflectionPadding_updateGradInput"
  c_TemporalReflectionPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateOutput :  state input output pad_left pad_right -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReplicationPadding_updateOutput"
  c_TemporalReplicationPadding_updateOutput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | c_TemporalReplicationPadding_updateGradInput :  state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h THNN_DoubleTemporalReplicationPadding_updateGradInput"
  c_TemporalReplicationPadding_updateGradInput :: Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | p_Abs_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleAbs_updateOutput"
  p_Abs_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Abs_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleAbs_updateGradInput"
  p_Abs_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_AbsCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleAbsCriterion_updateOutput"
  p_AbsCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_AbsCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleAbsCriterion_updateGradInput"
  p_AbsCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_BCECriterion_updateOutput : Pointer to function : state input target output sizeAverage weights reduce -> void
foreign import ccall "THNN.h &THNN_DoubleBCECriterion_updateOutput"
  p_BCECriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> Ptr C'THDoubleTensor -> CBool -> IO ())

-- | p_BCECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage weights reduce -> void
foreign import ccall "THNN.h &THNN_DoubleBCECriterion_updateGradInput"
  p_BCECriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> Ptr C'THDoubleTensor -> CBool -> IO ())

-- | p_ELU_updateOutput : Pointer to function : state input output alpha scale inplace -> void
foreign import ccall "THNN.h &THNN_DoubleELU_updateOutput"
  p_ELU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_ELU_updateGradInput : Pointer to function : state gradOutput gradInput output alpha scale -> void
foreign import ccall "THNN.h &THNN_DoubleELU_updateGradInput"
  p_ELU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_DistKLDivCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleDistKLDivCriterion_updateOutput"
  p_DistKLDivCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_DistKLDivCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleDistKLDivCriterion_updateGradInput"
  p_DistKLDivCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_GatedLinear_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleGatedLinear_updateOutput"
  p_GatedLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_GatedLinear_updateGradInput : Pointer to function : state input gradOutput gradInput dim -> void
foreign import ccall "THNN.h &THNN_DoubleGatedLinear_updateGradInput"
  p_GatedLinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_HardShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_DoubleHardShrink_updateOutput"
  p_HardShrink_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_HardShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_DoubleHardShrink_updateGradInput"
  p_HardShrink_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_HardTanh_updateOutput : Pointer to function : state input output min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleHardTanh_updateOutput"
  p_HardTanh_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_HardTanh_updateGradInput : Pointer to function : state input gradOutput gradInput min_val max_val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleHardTanh_updateGradInput"
  p_HardTanh_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Im2Col_updateOutput : Pointer to function : state input output kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_DoubleIm2Col_updateOutput"
  p_Im2Col_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Im2Col_updateGradInput : Pointer to function : state gradOutput gradInput inputHeight inputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_DoubleIm2Col_updateGradInput"
  p_Im2Col_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateOutput : Pointer to function : state input output outputHeight outputWidth kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_DoubleCol2Im_updateOutput"
  p_Col2Im_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_Col2Im_updateGradInput : Pointer to function : state gradOutput gradInput kH kW dH dW padH padW sH sW -> void
foreign import ccall "THNN.h &THNN_DoubleCol2Im_updateGradInput"
  p_Col2Im_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_L1Cost_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleL1Cost_updateOutput"
  p_L1Cost_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_L1Cost_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleL1Cost_updateGradInput"
  p_L1Cost_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LeakyReLU_updateOutput : Pointer to function : state input output negval inplace -> void
foreign import ccall "THNN.h &THNN_DoubleLeakyReLU_updateOutput"
  p_LeakyReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_LeakyReLU_updateGradInput : Pointer to function : state input gradOutput gradInput negval inplace -> void
foreign import ccall "THNN.h &THNN_DoubleLeakyReLU_updateGradInput"
  p_LeakyReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CBool -> IO ())

-- | p_GRUFused_updateOutput : Pointer to function : state input hidden bias1 bias2 hx output storage -> void
foreign import ccall "THNN.h &THNN_DoubleGRUFused_updateOutput"
  p_GRUFused_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_GRUFused_updateGradInput : Pointer to function : state gradInInput gradInHidden gradOutput gradInputHx storage -> void
foreign import ccall "THNN.h &THNN_DoubleGRUFused_updateGradInput"
  p_GRUFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LSTMFused_updateOutput : Pointer to function : state input hidden bias1 bias2 cell output outputCell -> void
foreign import ccall "THNN.h &THNN_DoubleLSTMFused_updateOutput"
  p_LSTMFused_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LSTMFused_updateGradInput : Pointer to function : state storage gradInGates cx cy gradOutput gradOutputCell gradInputCx -> void
foreign import ccall "THNN.h &THNN_DoubleLSTMFused_updateGradInput"
  p_LSTMFused_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LogSigmoid_updateOutput : Pointer to function : state input output buffer -> void
foreign import ccall "THNN.h &THNN_DoubleLogSigmoid_updateOutput"
  p_LogSigmoid_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LogSigmoid_updateGradInput : Pointer to function : state input gradOutput gradInput buffer -> void
foreign import ccall "THNN.h &THNN_DoubleLogSigmoid_updateGradInput"
  p_LogSigmoid_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_LogSoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleLogSoftMax_updateOutput"
  p_LogSoftMax_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_LogSoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_DoubleLogSoftMax_updateGradInput"
  p_LogSoftMax_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_MarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_DoubleMarginCriterion_updateOutput"
  p_MarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> IO ())

-- | p_MarginCriterion_updateGradInput : Pointer to function : state input target gradInput sizeAverage margin -> void
foreign import ccall "THNN.h &THNN_DoubleMarginCriterion_updateGradInput"
  p_MarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> IO ())

-- | p_SoftMarginCriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMarginCriterion_updateOutput"
  p_SoftMarginCriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SoftMarginCriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMarginCriterion_updateGradInput"
  p_SoftMarginCriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleMSECriterion_updateOutput"
  p_MSECriterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_MSECriterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleMSECriterion_updateGradInput"
  p_MSECriterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_PReLU_updateOutput : Pointer to function : state input output weight -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_updateOutput"
  p_PReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_PReLU_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_updateGradInput"
  p_PReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_PReLU_accGradParameters : Pointer to function : state input gradOutput gradInput weight gradWeight scale -> void
foreign import ccall "THNN.h &THNN_DoublePReLU_accGradParameters"
  p_PReLU_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_Linear_updateOutput : Pointer to function : state input output weight bias addBuffer -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_updateOutput"
  p_Linear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Linear_updateGradInput : Pointer to function : state input gradOutput gradInput weight -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_updateGradInput"
  p_Linear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Linear_accGradParameters : Pointer to function : state input gradOutput gradInput weight bias gradWeight gradBias addBuffer scale -> void
foreign import ccall "THNN.h &THNN_DoubleLinear_accGradParameters"
  p_Linear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_RReLU_updateOutput : Pointer to function : state input output noise lower upper train inplace generator -> void
foreign import ccall "THNN.h &THNN_DoubleRReLU_updateOutput"
  p_RReLU_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> Ptr C'THGenerator -> IO ())

-- | p_RReLU_updateGradInput : Pointer to function : state input gradOutput gradInput noise lower upper train inplace -> void
foreign import ccall "THNN.h &THNN_DoubleRReLU_updateGradInput"
  p_RReLU_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> CBool -> IO ())

-- | p_Sigmoid_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleSigmoid_updateOutput"
  p_Sigmoid_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Sigmoid_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleSigmoid_updateGradInput"
  p_Sigmoid_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SmoothL1Criterion_updateOutput : Pointer to function : state input target output sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSmoothL1Criterion_updateOutput"
  p_SmoothL1Criterion_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SmoothL1Criterion_updateGradInput : Pointer to function : state input target gradOutput gradInput sizeAverage reduce -> void
foreign import ccall "THNN.h &THNN_DoubleSmoothL1Criterion_updateGradInput"
  p_SmoothL1Criterion_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CBool -> IO ())

-- | p_SoftMax_updateOutput : Pointer to function : state input output dim -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMax_updateOutput"
  p_SoftMax_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_SoftMax_updateGradInput : Pointer to function : state input gradOutput gradInput output dim -> void
foreign import ccall "THNN.h &THNN_DoubleSoftMax_updateGradInput"
  p_SoftMax_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_SoftPlus_updateOutput : Pointer to function : state input output beta threshold -> void
foreign import ccall "THNN.h &THNN_DoubleSoftPlus_updateOutput"
  p_SoftPlus_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftPlus_updateGradInput : Pointer to function : state input gradOutput gradInput output beta threshold -> void
foreign import ccall "THNN.h &THNN_DoubleSoftPlus_updateGradInput"
  p_SoftPlus_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SoftShrink_updateOutput : Pointer to function : state input output lambda -> void
foreign import ccall "THNN.h &THNN_DoubleSoftShrink_updateOutput"
  p_SoftShrink_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_SoftShrink_updateGradInput : Pointer to function : state input gradOutput gradInput lambda -> void
foreign import ccall "THNN.h &THNN_DoubleSoftShrink_updateGradInput"
  p_SoftShrink_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_SparseLinear_updateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_updateOutput"
  p_SparseLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SparseLinear_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_accGradParameters"
  p_SparseLinear_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_zeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_zeroGradParameters"
  p_SparseLinear_zeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SparseLinear_updateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_updateParameters"
  p_SparseLinear_updateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_SparseLinear_legacyUpdateOutput : Pointer to function : state input output weight bias -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyUpdateOutput"
  p_SparseLinear_legacyUpdateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SparseLinear_legacyAccGradParameters : Pointer to function : state input gradOutput gradWeight gradBias weight bias weightDecay scale -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyAccGradParameters"
  p_SparseLinear_legacyAccGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_SparseLinear_legacyZeroGradParameters : Pointer to function : state gradWeight gradBias lastInput -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyZeroGradParameters"
  p_SparseLinear_legacyZeroGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SparseLinear_legacyUpdateParameters : Pointer to function : state weight bias gradWeight gradBias lastInput learningRate -> void
foreign import ccall "THNN.h &THNN_DoubleSparseLinear_legacyUpdateParameters"
  p_SparseLinear_legacyUpdateParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_Sqrt_updateOutput : Pointer to function : state input output eps -> void
foreign import ccall "THNN.h &THNN_DoubleSqrt_updateOutput"
  p_Sqrt_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_Sqrt_updateGradInput : Pointer to function : state input gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleSqrt_updateGradInput"
  p_Sqrt_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Square_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleSquare_updateOutput"
  p_Square_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Square_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleSquare_updateGradInput"
  p_Square_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Tanh_updateOutput : Pointer to function : state input output -> void
foreign import ccall "THNN.h &THNN_DoubleTanh_updateOutput"
  p_Tanh_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Tanh_updateGradInput : Pointer to function : state gradOutput gradInput output -> void
foreign import ccall "THNN.h &THNN_DoubleTanh_updateGradInput"
  p_Tanh_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_Threshold_updateOutput : Pointer to function : state input output threshold val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleThreshold_updateOutput"
  p_Threshold_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_Threshold_updateGradInput : Pointer to function : state input gradOutput gradInput threshold val inplace -> void
foreign import ccall "THNN.h &THNN_DoubleThreshold_updateGradInput"
  p_Threshold_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CBool -> IO ())

-- | p_TemporalConvolution_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize outputFrameSize -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_updateOutput"
  p_TemporalConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_updateGradInput"
  p_TemporalConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalConvolution_accGradParameters"
  p_TemporalConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalSubSampling_updateOutput : Pointer to function : state input output weight bias kW dW inputFrameSize -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_updateOutput"
  p_TemporalSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW dW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_updateGradInput"
  p_TemporalSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW dW scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalSubSampling_accGradParameters"
  p_TemporalSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CDouble -> IO ())

-- | p_TemporalRowConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_updateOutput"
  p_TemporalRowConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW dW padW featFirst -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_updateGradInput"
  p_TemporalRowConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> IO ())

-- | p_TemporalRowConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW dW padW featFirst scale -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalRowConvolution_accGradParameters"
  p_TemporalRowConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CBool -> CDouble -> IO ())

-- | p_TemporalUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingNearest_updateOutput"
  p_TemporalUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingNearest_updateGradInput"
  p_TemporalUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateOutput : Pointer to function : state input output osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingLinear_updateOutput"
  p_TemporalUpSamplingLinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_TemporalUpSamplingLinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeW osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalUpSamplingLinear_updateGradInput"
  p_TemporalUpSamplingLinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_BatchNormalization_updateOutput : Pointer to function : state input output weight bias running_mean running_var save_mean save_std train momentum eps -> void
foreign import ccall "THNN.h &THNN_DoubleBatchNormalization_updateOutput"
  p_BatchNormalization_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_BatchNormalization_backward : Pointer to function : state input gradOutput gradInput gradWeight gradBias weight running_mean running_var save_mean save_std train scale eps -> void
foreign import ccall "THNN.h &THNN_DoubleBatchNormalization_backward"
  p_BatchNormalization_backward :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CBool -> CDouble -> CDouble -> IO ())

-- | p_SpatialConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_updateOutput"
  p_SpatialConvolutionMap_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_updateGradInput"
  p_SpatialConvolutionMap_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMap_accGradParameters"
  p_SpatialConvolutionMap_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_updateOutput"
  p_SpatialConvolutionMM_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_updateGradInput"
  p_SpatialConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionMM_accGradParameters"
  p_SpatialConvolutionMM_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialConvolutionLocal_updateOutput : Pointer to function : state input output weight bias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_updateOutput"
  p_SpatialConvolutionLocal_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_updateGradInput"
  p_SpatialConvolutionLocal_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_SpatialConvolutionLocal_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kW kH dW dH padW padH inputWidth inputHeight outputWidth outputHeight scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialConvolutionLocal_accGradParameters"
  p_SpatialConvolutionLocal_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput"
  p_SpatialAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_SpatialAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput"
  p_SpatialAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SpatialAveragePooling_updateOutput : Pointer to function : state input output kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAveragePooling_updateOutput"
  p_SpatialAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kW kH dW dH padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialAveragePooling_updateGradInput"
  p_SpatialAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_SpatialFullConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_updateOutput"
  p_SpatialFullConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_updateGradInput"
  p_SpatialFullConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolution_accGradParameters"
  p_SpatialFullConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullConvolutionMap_updateOutput : Pointer to function : state input output weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_updateOutput"
  p_SpatialFullConvolutionMap_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolutionMap_updateGradInput : Pointer to function : state input gradOutput gradInput weight bias connTable nInputPlane nOutputPlane dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_updateGradInput"
  p_SpatialFullConvolutionMap_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullConvolutionMap_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias connTable nInputPlane nOutputPlane dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullConvolutionMap_accGradParameters"
  p_SpatialFullConvolutionMap_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_updateOutput"
  p_SpatialDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_updateGradInput"
  p_SpatialDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialDilatedConvolution_accGradParameters"
  p_SpatialDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_updateOutput"
  p_SpatialFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kW kH dW dH padW padH dilationW dilationH adjW adjH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_updateGradInput"
  p_SpatialFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kW kH dW dH padW padH dilationW dilationH adjW adjH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialFullDilatedConvolution_accGradParameters"
  p_SpatialFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialSubSampling_updateOutput : Pointer to function : state input output weight bias kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_updateOutput"
  p_SpatialSubSampling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_updateGradInput : Pointer to function : state input gradOutput gradInput weight kW kH dW dH -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_updateGradInput"
  p_SpatialSubSampling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialSubSampling_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias kW kH dW dH scale -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialSubSampling_accGradParameters"
  p_SpatialSubSampling_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_SpatialUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingNearest_updateOutput"
  p_SpatialUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingNearest_updateGradInput"
  p_SpatialUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateOutput : Pointer to function : state input output osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingBilinear_updateOutput"
  p_SpatialUpSamplingBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_SpatialUpSamplingBilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeH isizeW osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialUpSamplingBilinear_updateGradInput"
  p_SpatialUpSamplingBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialGridSamplerBilinear_updateOutput"
  p_SpatialGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_SpatialGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialGridSamplerBilinear_updateGradInput"
  p_SpatialGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateOutput : Pointer to function : state input grid output padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricGridSamplerBilinear_updateOutput"
  p_VolumetricGridSamplerBilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_VolumetricGridSamplerBilinear_updateGradInput : Pointer to function : state input gradInput grid gradGrid gradOutput padding_mode -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricGridSamplerBilinear_updateGradInput"
  p_VolumetricGridSamplerBilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_unfolded_acc : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight osizeW outputHeight -> void
foreign import ccall "THNN.h &THNN_Doubleunfolded_acc"
  p_unfolded_acc :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_unfolded_copy : Pointer to function : finput input kW kH dW dH padW padH nInputPlane inputWidth inputHeight outputWidth outputHeight -> void
foreign import ccall "THNN.h &THNN_Doubleunfolded_copy"
  p_unfolded_copy :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAveragePooling_updateOutput : Pointer to function : state input output kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAveragePooling_updateOutput"
  p_VolumetricAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput kT kW kH dT dW dH padT padW padH ceil_mode count_include_pad -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAveragePooling_updateGradInput"
  p_VolumetricAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CBool -> CBool -> IO ())

-- | p_VolumetricConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_updateOutput"
  p_VolumetricConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_updateGradInput"
  p_VolumetricConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolution_accGradParameters"
  p_VolumetricConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricConvolutionMM_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_updateOutput"
  p_VolumetricConvolutionMM_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolutionMM_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_updateGradInput"
  p_VolumetricConvolutionMM_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricConvolutionMM_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricConvolutionMM_accGradParameters"
  p_VolumetricConvolutionMM_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_updateOutput"
  p_VolumetricFullConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_updateGradInput"
  p_VolumetricFullConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullConvolution_accGradParameters"
  p_VolumetricFullConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricDilatedConvolution_updateOutput : Pointer to function : state input output weight bias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_updateOutput"
  p_VolumetricDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight columns kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_updateGradInput"
  p_VolumetricDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias columns ones kT kW kH dT dW dH padT padW padH dilationT dilationW dilationH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricDilatedConvolution_accGradParameters"
  p_VolumetricDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateOutput : Pointer to function : state input output weight bias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_updateOutput"
  p_VolumetricFullDilatedConvolution_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_updateGradInput : Pointer to function : state input gradOutput gradInput weight finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput"
  p_VolumetricFullDilatedConvolution_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricFullDilatedConvolution_accGradParameters : Pointer to function : state input gradOutput gradWeight gradBias finput fgradInput kT kW kH dT dW dH pT pW pH dilationT dilationW dilationH aT aW aH scale -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters"
  p_VolumetricFullDilatedConvolution_accGradParameters :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CDouble -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateOutput : Pointer to function : state input output osizeT osizeW osizeH -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput"
  p_VolumetricAdaptiveAveragePooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricAdaptiveAveragePooling_updateGradInput : Pointer to function : state input gradOutput gradInput -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput"
  p_VolumetricAdaptiveAveragePooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_SpatialReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReflectionPadding_updateOutput"
  p_SpatialReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReflectionPadding_updateGradInput"
  p_SpatialReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReplicationPadding_updateOutput"
  p_SpatialReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_SpatialReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom -> void
foreign import ccall "THNN.h &THNN_DoubleSpatialReplicationPadding_updateGradInput"
  p_SpatialReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_FeatureLPPooling_updateOutput : Pointer to function : state input output power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_DoubleFeatureLPPooling_updateOutput"
  p_FeatureLPPooling_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_FeatureLPPooling_updateGradInput : Pointer to function : state gradOutput input output gradInput power width stride batchMode -> void
foreign import ccall "THNN.h &THNN_DoubleFeatureLPPooling_updateGradInput"
  p_FeatureLPPooling_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> CBool -> IO ())

-- | p_VolumetricReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricReplicationPadding_updateOutput"
  p_VolumetricReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right pad_top pad_bottom pad_front pad_back -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricReplicationPadding_updateGradInput"
  p_VolumetricReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateOutput : Pointer to function : state input output scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingNearest_updateOutput"
  p_VolumetricUpSamplingNearest_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingNearest_updateGradInput : Pointer to function : state input gradOutput gradInput scale_factor -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingNearest_updateGradInput"
  p_VolumetricUpSamplingNearest_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateOutput : Pointer to function : state input output osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput"
  p_VolumetricUpSamplingTrilinear_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_VolumetricUpSamplingTrilinear_updateGradInput : Pointer to function : state gradOutput gradInput isizeB isizeC isizeT isizeH isizeW osizeT osizeH osizeW -> void
foreign import ccall "THNN.h &THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput"
  p_VolumetricUpSamplingTrilinear_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReflectionPadding_updateOutput"
  p_TemporalReflectionPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReflectionPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReflectionPadding_updateGradInput"
  p_TemporalReflectionPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateOutput : Pointer to function : state input output pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReplicationPadding_updateOutput"
  p_TemporalReplicationPadding_updateOutput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_TemporalReplicationPadding_updateGradInput : Pointer to function : state input gradOutput gradInput pad_left pad_right -> void
foreign import ccall "THNN.h &THNN_DoubleTemporalReplicationPadding_updateGradInput"
  p_TemporalReplicationPadding_updateGradInput :: FunPtr (Ptr C'THNNState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())
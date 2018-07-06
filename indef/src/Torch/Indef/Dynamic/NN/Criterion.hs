module Torch.Indef.Dynamic.NN.Criterion where

import Foreign.C.Types
import Foreign.Ptr
import Torch.Sig.Types.NN
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.NN as Sig
import Torch.Indef.Types

-- c_ClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
-- c_ClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
-- c_SpatialClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
-- c_SpatialClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
-- 
-- c_MultiLabelMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
-- c_MultiLabelMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
-- c_MultiMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()
-- c_MultiMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()

-- | l1Cost forward pass (updates the output tensor)
_l1Cost_updateOutput :: Dynamic -> Dynamic -> IO ()
_l1Cost_updateOutput t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_L1Cost_updateOutput s' t0' t1'

-- | l1Cost backward-update (updates the layer and bias tensors)
_l1Cost_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_l1Cost_updateGradInput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_L1Cost_updateGradInput s' t0' t1' t2'

-- | smoothL1Criterion forward pass (updates the output tensor)
_smoothL1Criterion_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_smoothL1Criterion_updateOutput t0 t1 t2 b0 b1 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SmoothL1Criterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | smoothL1Criterion backward-update (updates the layer and bias tensors)
_smoothL1Criterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_smoothL1Criterion_updateGradInput t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_SmoothL1Criterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | absCriterion forward pass (updates the output tensor)
_absCriterion_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_absCriterion_updateOutput t0 t1 t2 b0 b1 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_AbsCriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | absCriterion backward-update (updates the layer and bias tensors)
_absCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_absCriterion_updateGradInput t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_AbsCriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | bCECriterion forward pass (updates the output tensor)
_bCECriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Maybe Dynamic -> Bool -> IO ()
_bCECriterion_updateOutput t0 t1 t2 b0 mt3 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    case mt3 of
      Nothing ->
        Sig.c_BCECriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) nullPtr (toEnum $ fromEnum b1)
      Just t3 ->
        withDynamicState t3 $ \_ t3' ->
          Sig.c_BCECriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) t3' (toEnum $ fromEnum b1)

-- | bCECriterion backward-update (updates the layer and bias tensors)
_bCECriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Maybe Dynamic -> Bool -> IO ()
_bCECriterion_updateGradInput t0 t1 t2 t3 b0 mt4 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' -> 
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      case mt4 of
        Nothing ->
          Sig.c_BCECriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) nullPtr (toEnum $ fromEnum b1)
        Just t4 ->
          withDynamicState t4 $ \_ t4' ->
            Sig.c_BCECriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) t4' (toEnum $ fromEnum b1)


-- | distKLDivCriterion forward pass (updates the output tensor)
_distKLDivCriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_distKLDivCriterion_updateOutput t0 t1 t2 b0 b1 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_DistKLDivCriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | distKLDivCriterion backward-update (updates the layer and bias tensors)
_distKLDivCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_distKLDivCriterion_updateGradInput t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' -> 
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_DistKLDivCriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | marginCriterion forward pass (updates the output tensor)
_marginCriterion_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> IO ()
_marginCriterion_updateOutput t0 t1 t2 b0 d0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_MarginCriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (realToFrac d0)

-- | marginCriterion backward-update (updates the layer and bias tensors)
_marginCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> IO ()
_marginCriterion_updateGradInput t0 t1 t2 b0 d0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_MarginCriterion_updateGradInput s' t0' t1' t2' (toEnum $ fromEnum b0) (realToFrac d0)

-- | softMarginCriterion forward pass (updates the output tensor)
_softMarginCriterion_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_softMarginCriterion_updateOutput t0 t1 t2 b0 b1 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SoftMarginCriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | softMarginCriterion backward-update (updates the layer and bias tensors)
_softMarginCriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_softMarginCriterion_updateGradInput t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' -> 
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_SoftMarginCriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | mSECriterion forward pass (updates the output tensor)
_mSECriterion_updateOutput    :: Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_mSECriterion_updateOutput t0 t1 t2 b0 b1 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_MSECriterion_updateOutput s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

-- | mSECriterion backward-update (updates the layer and bias tensors)
_mSECriterion_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Bool -> IO ()
_mSECriterion_updateGradInput t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_MSECriterion_updateGradInput s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)





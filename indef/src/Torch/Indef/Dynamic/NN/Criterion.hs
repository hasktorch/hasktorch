module Torch.Indef.Dynamic.NN.Criterion where

c_ClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_ClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_SpatialClassNLLCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()
c_SpatialClassNLLCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> Ptr CTensor -> Ptr CTensor -> CLLong -> CBool -> IO ()

c_MultiLabelMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MultiLabelMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CBool -> IO ()
c_MultiMarginCriterion_updateOutput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()
c_MultiMarginCriterion_updateGradInput :: Ptr CNNState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> Ptr CTensor -> CBool -> CInt -> Ptr CTensor -> CDouble -> CBool -> IO ()

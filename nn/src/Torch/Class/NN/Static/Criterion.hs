module Torch.Class.NN.Static.Criterion where

import Torch.Dimensions

class Criterion (t :: [Nat] -> *) where
  _absCriterion_updateOutput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ output
    -> Bool    -- ^ size average
    -> Bool    -- ^ reduce
    -> IO ()

  _absCriterion_updateGradInput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ gradOutput
    -> t d''   -- ^ gradInput
    -> Bool    -- ^ size average
    -> Bool    -- ^ reduce
    -> IO ()


  _bCECriterion_updateOutput
    :: t d     -- ^ input
    -> t d'    -- ^ target
    -> t d''   -- ^ output
    -> Bool    -- ^ sizeAverage
    -> t d'''  -- ^ weights
    -> Bool    -- ^ reduce
    -> IO ()

  _bCECriterion_updateGradInput
    :: t d      -- ^ input
    -> t d'     -- ^ target
    -> t d''    -- ^ grad output
    -> t d'''   -- ^ grad input
    -> Bool     -- ^  sizeAvreage
    -> t d''''  -- ^ weights
    -> Bool     -- ^ reduce
    -> IO ()


  _marginCriterion_updateOutput        :: t d -> t d -> t d -> Bool -> Double -> IO ()
  _marginCriterion_updateGradInput     :: t d -> t d -> t d -> Bool -> Double -> IO ()

  _softMarginCriterion_updateOutput    :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  _softMarginCriterion_updateGradInput :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()

  _mSECriterion_updateOutput           :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  _mSECriterion_updateGradInput        :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()

  _distKLDivCriterion_updateOutput    :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  _distKLDivCriterion_updateGradInput :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()

  _smoothL1Criterion_updateOutput    :: t d -> t d -> t d -> Bool -> Bool -> IO ()
  _smoothL1Criterion_updateGradInput :: t d -> t d -> t d -> t d -> Bool -> Bool -> IO ()

  _l1Cost_updateOutput          :: t d -> t d -> IO ()
  _l1Cost_updateGradInput       :: t d -> t d -> t d -> IO ()

-- In latest ATen master. Need to bump hasktorch's fork to this
-- class ClassNLLCriterion (t :: [Nat] -> *) where

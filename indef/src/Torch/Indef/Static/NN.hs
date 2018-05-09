module Torch.Indef.Static.NN
  ( module X
  , _batchNormalization_updateOutput
  , _batchNormalization_backward
  , _col2Im_updateOutput
  , _col2Im_updateGradInput
  , _im2Col_updateOutput
  , _im2Col_updateGradInput
  ) where

import Torch.Dimensions

import Torch.Indef.Types

import qualified Torch.Indef.Dynamic.NN as Dynamic
import Torch.Indef.Static.Tensor

import Torch.Indef.Static.NN.Activation as X
import Torch.Indef.Static.NN.Conv1d as X hiding (weights)
import Torch.Indef.Static.NN.Conv2d as X
-- import Torch.Indef.Static.NN.Conv3d as X
import Torch.Indef.Static.NN.Criterion as X
import Torch.Indef.Static.NN.Layers as X
import Torch.Indef.Static.NN.Math as X
import Torch.Indef.Static.NN.Padding as X
import Torch.Indef.Static.NN.Pooling as X
import Torch.Indef.Static.NN.Sampling as X
import Torch.Indef.Static.NN.Backprop as X

_batchNormalization_updateOutput 
  :: Tensor d    -- ^ input
  -> Tensor d    -- ^ output
  -> Tensor d    -- ^ weight
  -> Tensor d    -- ^ bias
  -> Tensor d    -- ^ running mean
  -> Tensor d    -- ^ running var
  -> Tensor d    -- ^ save mean
  -> Tensor d    -- ^ save std
  -> Bool   -- ^ train
  -> Double -- ^ momentum
  -> Double -- ^ eps
  -> IO ()
_batchNormalization_updateOutput t0 t1 t2 t3 t4 t5 t6 t7 = Dynamic._batchNormalization_updateOutput
  (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
  (asDynamic t5) (asDynamic t6) (asDynamic t7)

_batchNormalization_backward
  :: Tensor d      -- ^ input
  -> Tensor d      -- ^ grad output
  -> Tensor d      -- ^ grad input
  -> Tensor d      -- ^ grad weight
  -> Tensor d      -- ^ grad bias
  -> Tensor d      -- ^ weight
  -> Tensor d      -- ^ running mean
  -> Tensor d      -- ^ running var
  -> Tensor d      -- ^ save mean
  -> Tensor d      -- ^ save std
  -> Bool     -- ^ train
  -> Double   -- ^ momentum
  -> Double   -- ^ eps
  -> IO ()
_batchNormalization_backward t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 = Dynamic._batchNormalization_backward
  (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
  (asDynamic t5) (asDynamic t6) (asDynamic t7) (asDynamic t8) (asDynamic t9)

_col2Im_updateOutput
  :: Tensor d -- ^ input
  -> Tensor d -- ^ output
  -> Int -- ^ output Height
  -> Int -- ^ output Width
  -> Int -- ^ kH
  -> Int -- ^ kW
  -> Int -- ^ dH
  -> Int -- ^ dW
  -> Int -- ^ padH
  -> Int -- ^ padW
  -> Int -- ^ sH
  -> Int -- ^ sW
  -> IO ()
_col2Im_updateOutput t0 t1 = Dynamic._col2Im_updateOutput (asDynamic t0) (asDynamic t1)

_col2Im_updateGradInput
  :: Tensor d -- ^ grad output
  -> Tensor d -- ^ grad input
  -> Int -- ^ kH
  -> Int -- ^ kW
  -> Int -- ^ dH
  -> Int -- ^ dW
  -> Int -- ^ padH
  -> Int -- ^ padW
  -> Int -- ^ sH
  -> Int -- ^ sW
  -> IO ()
_col2Im_updateGradInput g0 g1 = Dynamic._col2Im_updateGradInput (asDynamic g0) (asDynamic g1)

_im2Col_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_im2Col_updateOutput g0 g1 = Dynamic._im2Col_updateOutput (asDynamic g0) (asDynamic g1)

_im2Col_updateGradInput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_im2Col_updateGradInput g0 g1 = Dynamic._im2Col_updateGradInput (asDynamic g0) (asDynamic g1)

{-
class CPUNN t d where
  unfolded_acc  :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  unfolded_copy :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_updateOutput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_updateGradInput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricConvolutionMM_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
  temporalSubSampling_updateOutput :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> IO ()
  temporalSubSampling_updateGradInput :: t d -> t d -> t d -> t d -> Int -> Int -> IO ()
  temporalSubSampling_accGradParameters :: t d -> t d -> t d -> t d -> Int -> Int -> Double -> IO ()
  spatialFullConvolutionMap_updateOutput :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolutionMap_updateGradInput :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolutionMap_accGradParameters :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Double -> IO ()
  hardShrink_updateOutput      :: t d -> t d -> Double -> IO ()
  hardShrink_updateGradInput   :: t d -> t d -> t d -> Double -> IO ()
  linear_updateOutput      :: t d -> t d -> t d -> t d -> t d -> IO ()
  linear_updateGradInput   :: t d -> t d -> t d -> t d -> IO ()
  linear_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> t d -> Double -> IO ()
  sparseLinear_legacyZeroGradParameters :: t d -> t d -> t d -> IO ()
  sparseLinear_legacyUpdateParameters   :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()
-}

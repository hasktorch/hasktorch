module Torch.Class.NN.Static
  ( module X
  , BatchNormalization(..)
  , Col2Im(..)
  , NN(..)
  , CPUNN(..)
  ) where

import Control.Monad.Trans.Class
import Control.Monad.IO.Class

import Torch.Dimensions
import Torch.Class.Types
import Torch.Class.Tensor.Static

import Torch.Class.NN.Static.Math       as X
import Torch.Class.NN.Static.Criterion  as X
import Torch.Class.NN.Static.Pooling    as X
import Torch.Class.NN.Static.Padding    as X
import Torch.Class.NN.Static.Activation as X
import Torch.Class.NN.Static.Layers     as X
import Torch.Class.NN.Static.Conv       as X

class BatchNormalization (t :: [Nat] -> *) where
  _batchNormalization_updateOutput 
    :: t d    -- ^ input
    -> t d    -- ^ output
    -> t d    -- ^ weight
    -> t d    -- ^ bias
    -> t d    -- ^ running mean
    -> t d    -- ^ running var
    -> t d    -- ^ save mean
    -> t d    -- ^ save std
    -> Bool   -- ^ train
    -> Double -- ^ momentum
    -> Double -- ^ eps
    -> IO ()

  _batchNormalization_backward
    :: t d      -- ^ input
    -> t d      -- ^ grad output
    -> t d      -- ^ grad input
    -> t d      -- ^ grad weight
    -> t d      -- ^ grad bias
    -> t d      -- ^ weight
    -> t d      -- ^ running mean
    -> t d      -- ^ running var
    -> t d      -- ^ save mean
    -> t d      -- ^ save std
    -> Bool     -- ^ train
    -> Double   -- ^ momentum
    -> Double   -- ^ eps
    -> IO ()


class Col2Im (t :: [Nat] -> *) where
  _col2Im_updateOutput
    :: t d -- ^ input
    -> t d -- ^ output
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
  _col2Im_updateGradInput
    :: t d -- ^ grad output
    -> t d -- ^ grad input
    -> Int -- ^ kH
    -> Int -- ^ kW
    -> Int -- ^ dH
    -> Int -- ^ dW
    -> Int -- ^ padH
    -> Int -- ^ padW
    -> Int -- ^ sH
    -> Int -- ^ sW
    -> IO ()

  _im2Col_updateOutput    :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  _im2Col_updateGradInput :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()

class
  ( Math t
  , Criterion t
  , Activation t
  , FusedLayers t
  , Pooling t
  , Padding t
  , Convolutions t

  , BatchNormalization t
  , Col2Im t

  , IsTensor t
  ) => NN (t :: [Nat] -> *) where


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

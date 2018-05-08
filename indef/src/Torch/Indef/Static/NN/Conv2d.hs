-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Conv2d
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Spatial (2D) Convolutions.
--
-- Complete types and documentation with https://github.com/torch/nn/blob/master/doc/convolution.md#spatial-modules
-------------------------------------------------------------------------------
module Torch.Indef.Static.NN.Conv2d where

import Torch.Indef.Types
import Data.Kind (Type)
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- | Applies a 2D convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D tensor (nInputPlane x height x width).
_spatialConvolutionMM_updateOutput
  :: Tensor d  -- ^ input
  -> Tensor d  -- ^ output
  -> Tensor d  -- ^ weight
  -> Tensor d  -- ^ bias
  -> Tensor d  -- ^ finput
  -> Tensor d  -- ^ fgradInput
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.
  -> IO ()
_spatialConvolutionMM_updateOutput t0 t1 t2 t3 t4 t5 (kW, kH) (dW, dH) (pW, pH) =
  Dynamic._spatialConvolutionMM_updateOutput
    (asDynamic t0) (asDynamic t1)
    (asDynamic t2) (asDynamic t3)
    (asDynamic t4) (asDynamic t5)
    kW kH dW dH pW pH

_spatialConvolutionMM_updateGradInput
  :: Tensor d   -- ^ input
  -> Tensor d   -- ^ gradOutput
  -> Tensor d   -- ^ gradInput
  -> Tensor d   -- ^ weight
  -> Tensor d   -- ^ finput
  -> Tensor d   -- ^ fgradInput
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> IO ()
_spatialConvolutionMM_updateGradInput t0 t1 t2 t3 t4 t5 (kW, kH) (dW, dH) (pW, pH) =
  Dynamic._spatialConvolutionMM_updateGradInput
    (asDynamic t0) (asDynamic t1)
    (asDynamic t2) (asDynamic t3)
    (asDynamic t4) (asDynamic t5)
    kW kH dW dH pW pH

_spatialConvolutionMM_accGradParameters
  :: Tensor d   -- ^ input
  -> Tensor d   -- ^ gradOutput
  -> Tensor d   -- ^ gradInput
  -> Tensor d   -- ^ weight
  -> Tensor d   -- ^ finput
  -> Tensor d   -- ^ fgradInput
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double
  -> IO ()
_spatialConvolutionMM_accGradParameters t0 t1 t2 t3 t4 t5 (kW, kH) (dW, dH) (pW, pH) d =
  Dynamic._spatialConvolutionMM_accGradParameters
    (asDynamic t0) (asDynamic t1)
    (asDynamic t2) (asDynamic t3)
    (asDynamic t4) (asDynamic t5)
    kW kH dW dH pW pH d


-- Applies a 2D locally-connected layer over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D or 4D tensor. A locally-connected layer is similar to a convolution layer but without weight-sharing.
-- _spatialConvolutionLocal_updateOutput      :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> IO ()
-- _spatialConvolutionLocal_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> IO ()
-- _spatialConvolutionLocal_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> Double -> IO ()

-- Applies a 2D full convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D or 4D tensor. Note that instead of setting adjW and adjH, SpatialFullConvolution also accepts a table input with two tensors: {convInput, sizeTensor} where convInput is the standard input on which the full convolution is applied, and the size of sizeTensor is used to set the size of the output. Using the two-input version of forward will ignore the adjW and adjH values used to construct the module. The layer can be used without a bias by module:noBias().
-- spatialFullConvolution_updateOutput      :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolution_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolution_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

-- Also sometimes referred to as atrous convolution. Applies a 2D dilated convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D or 4D tensor.
-- spatialDilatedConvolution_updateOutput      :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialDilatedConvolution_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialDilatedConvolution_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- 
-- spatialFullDilatedConvolution_updateOutput      :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullDilatedConvolution_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullDilatedConvolution_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()



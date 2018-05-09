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
-- Excluding an optional batch dimension, spatial layers expect a 3D Tensor as
-- input. The first dimension is the number of features (e.g. frameSize), the
-- last two dimensions are spatial (e.g. height x width). These are commonly
-- used for processing images.
--
-- Complete types and documentation at https://github.com/torch/nn/blob/master/doc/convolution.md#spatial-modules
-------------------------------------------------------------------------------
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Indef.Static.NN.Conv2d where

import Torch.Indef.Types
import Data.Kind (Type)
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.NN as Dynamic


-- | Applies a 2D convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 3D tensor (nInputPlane x height x width).
_spatialConvolutionMM_updateOutput
  :: Tensor d  -- ^ input
  -> Tensor d  -- ^ output
  -> Tensor d  -- ^ 3D weight tensor (connTable:size(1) x kH x kW) 
  -> Tensor d  -- ^ 1D bias tensor (nOutputPlane) 
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

-- ========================================================================= --

newtype Conv2d f o kW kH
  = Conv2d { getTensors :: (Tensor '[o, f, kH, kW], Tensor '[o]) }

weights :: Conv2d f o kW kH -> Tensor '[o, f, kH, kW]
weights (Conv2d (w, _)) = w

bias :: Conv2d f o kW kH -> Tensor '[o]
bias (Conv2d (_, b)) = b

featureSize :: forall f o kW dW . KnownNat f => Conv2d f o kW dW -> Int
featureSize _ = fromIntegral (natVal (Proxy :: Proxy f))

outputSize :: forall f o kW dW . KnownNat o => Conv2d f o kW dW -> Int
outputSize _ = fromIntegral (natVal (Proxy :: Proxy o))

-- | kW: The kernel width of the convolution
kernelWidth :: forall i f o kW kH . (Integral i, KnownNat kW) => Conv2d f o kW kH -> i
kernelWidth _ = fromIntegral (natVal (Proxy :: Proxy kW))

-- | kH: The kernel width of the convolution
kernelHeight :: forall i f o kW kH . (Integral i, KnownNat kH) => Conv2d f o kW kH -> i
kernelHeight _ = fromIntegral (natVal (Proxy :: Proxy kH))

-- ========================================================================= --

data Param2d (w::Nat) (h::Nat) = Param2d

paramW :: forall w h i . (KnownNat w, Integral i) => Param2d w h -> i
paramW _ = fromIntegral $ natVal (Proxy :: Proxy w)

paramH :: forall w h i . (KnownNat h, Integral i) => Param2d w h -> i
paramH _ = fromIntegral $ natVal (Proxy :: Proxy h)

-- ========================================================================= --
type SpatialConvolutionC h w kW kH dW dH pW pH =
  ( KnownNat2 kW kH, KnownNat2 dW dH, KnownNat2 pW pH
  , KnownNat2 h w
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  , (kH > 0) ~ 'True
  , (dH > 0) ~ 'True
  , ((Div (h + 2*pH - kH) dH) + 1) > 0 ~ 'True
  , ((Div (w + 2*pW - kW) dW) + 1) > 0 ~ 'True
  )

conv2dMM_forward
  :: SpatialConvolutionC h w kW kH dW dH pW pH
  => oh ~ ((Div (h + 2*pH - kH) dH) + 1)
  => ow ~ ((Div (w + 2*pW - kW) dW) + 1)
  => Tensor '[f,h,w]     -- ^ input: f stands for "features" or "input plane"
  -> Conv2d f o kW kH    -- ^ conv2d state
  -> Tensor d            -- ^ finput
  -> Tensor d            -- ^ fgradInput
  -> Param2d dW dH -- ^ step of the convolution in width and height dimensions. C-default is 1 for both.
  -> Param2d pW pH -- ^ zero padding to the input plane for width and height. (kW-1)/2 is often used.
                   -- C-default is 0 for both.
  -> IO (Tensor '[o, oh, ow])
conv2dMM_forward inp conv t3 t4 step pad = do
  out <- empty
  Dynamic._spatialConvolutionMM_updateOutput
    (asDynamic inp) (asDynamic out)
    (asDynamic (weights conv)) (asDynamic (bias conv))
    (asDynamic t3) (asDynamic t4)
    (kernelWidth conv) (kernelHeight conv)
    (paramW step) (paramH step)
    (paramW pad) (paramH pad)
  pure out


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



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
{-# LANGUAGE FlexibleContexts #-}
{- LANGUAGE AllowAmbiguousTypes #-}
module Torch.Indef.Static.NN.Conv2d where

import Control.Arrow
import Data.Kind (Type)
import Numeric.Backprop
import System.IO.Unsafe


import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Types
import Numeric.Backprop
import qualified Torch.Indef.Dynamic.NN as Dynamic


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

-- | The kernel width of the convolution
kernelWidth :: forall i f o kW kH . (Integral i, KnownNat kW) => Conv2d f o kW kH -> i
kernelWidth _ = fromIntegral (natVal (Proxy :: Proxy kW))

-- | The kernel height of the convolution
kernelHeight :: forall i f o kW kH . (Integral i, KnownNat kH) => Conv2d f o kW kH -> i
kernelHeight _ = fromIntegral (natVal (Proxy :: Proxy kH))

kernel2d :: (Integral i, KnownNat kH, KnownNat kW) => Conv2d f o kW kH -> (i, i)
kernel2d = kernelWidth &&& kernelHeight

-------------------------------------------------------------------------------

data Param2d (w::Nat) (h::Nat) = Param2d

paramW :: forall w h i . (KnownNat w, Integral i) => Param2d w h -> i
paramW _ = fromIntegral $ natVal (Proxy :: Proxy w)

paramH :: forall w h i . (KnownNat h, Integral i) => Param2d w h -> i
paramH _ = fromIntegral $ natVal (Proxy :: Proxy h)

param2d :: (KnownNat h, KnownNat w, Integral i) => Param2d w h -> (i, i)
param2d = paramW &&& paramH

-- ========================================================================= --

type SpatialConvolutionC h w kW kH dW dH pW pH =
  ( KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH
  , KnownNatDim2 h w
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  , (kH > 0) ~ 'True
  , (dH > 0) ~ 'True
  , ((Div (h + (2*pH) - kH) dH) + 1) > 0 ~ 'True
  , ((Div (w + (2*pW) - kW) dW) + 1) > 0 ~ 'True
  )

-- ========================================================================= --

-- -- | Backprop convolution function
-- conv2dMM
--   :: Reifies s W
--   => SpatialConvolutionC h w kW kH dW dH pW pH
--   => oh ~ ((Div (h + (2*pH) - kH) dH) + 1)
--   => ow ~ ((Div (w + (2*pW) - kW) dW) + 1)
--   => Param2d dW dH               -- ^ step of the convolution in width and height dimensions.
--                                  --   C-default is 1 for both.
--                                  --
--   -> Param2d pW pH               -- ^ zero padding to the input plane for width and height.
--                                  --   (kW-1)/2 is often used. C-default is 0 for both.
--                                  --
--   -> BVar s (Conv2d f o kW kH)   -- ^ conv2d state
--   -> BVar s (Tensor '[f,h,w])    -- ^ input: f stands for "features" or "input plane")
--   -> BVar s (Tensor '[o,oh,ow])
-- conv2dMM s p = liftOp2 . op2 $ \c inp ->
--   (fst . unsafePerformIO . conv2dMM_forward $ inp c s p, \out ->
--     fst . unsafePerformIO . conv2dMM_backward $ inp out c s p)

-- ========================================================================= --

-- | Applies a 2D convolution over an input image composed of several input
-- planes. The input tensor in forward(input) is expected to be a 3D tensor
-- (nInputPlane x height x width).
--
conv2dMM_forward
  :: SpatialConvolutionC h w kW kH dW dH pW pH
  => oh ~ ((Div (h + (2*pH) - kH) dH) + 1)
  => ow ~ ((Div (w + (2*pW) - kW) dW) + 1)
  => Tensor '[f,h,w]     -- ^ input: f stands for "features" or "input plane"
  -> Conv2d f o kW kH    -- ^ conv2d state
  -> Param2d dW dH       -- ^ step of the convolution in width and height dimensions.
                         --   C-default is 1 for both.
  -> Param2d pW pH       -- ^ zero padding to the input plane for width and height.
                         --   (kW-1)/2 is often used. C-default is 0 for both.
  -> IO (Tensor '[o, oh, ow])
conv2dMM_forward = _conv2dMM_forward

conv2dMM_backwardGradInput
  :: SpatialConvolutionC h w kW kH dW dH pW pH
  => oh ~ ((Div (h + (2*pH) - kH) dH) + 1)
  => ow ~ ((Div (w + (2*pW) - kW) dW) + 1)
  => Tensor '[f,h,w]      -- ^ input
  -> Tensor '[o, oh, ow]  -- ^ gradOutput
  -> Conv2d f o kW kH     -- ^ conv2d state
  -> Param2d dW dH        -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Param2d pW pH        -- ^ (pW, pH) zero padding to the input plane for width and height.
  -> IO (Tensor '[f,h,w])
conv2dMM_backwardGradInput = _conv2dMM_backwardGradInput

-- ========================================================================= --

-- | 'conv2dMM_forward' with a batch dimension
conv2dMM_forwardBatch
  :: SpatialConvolutionC h w kW kH dW dH pW pH
  => oh ~ ((Div (h + (2*pH) - kH) dH) + 1)
  => ow ~ ((Div (w + (2*pW) - kW) dW) + 1)
  => Tensor '[b,f,h,w]     -- ^ input: f stands for "features" or "input plane"
  -> Conv2d f o kW kH      -- ^ conv2d state
  -> Param2d dW dH         -- ^ step of the convolution in width and height dimensions.
                           --   C-default is 1 for both.
  -> Param2d pW pH         -- ^ zero padding to the input plane for width and height.
                           --   (kW-1)/2 is often used. C-default is 0 for both.
  -> IO (Tensor '[b,o,oh,ow])
conv2dMM_forwardBatch = _conv2dMM_forward

conv2dMM_backwardGradInputBatch
  :: SpatialConvolutionC h w kW kH dW dH pW pH
  => oh ~ ((Div (h + (2*pH) - kH) dH) + 1)
  => ow ~ ((Div (w + (2*pW) - kW) dW) + 1)
  => Tensor '[b,f,h,w]    -- ^ input
  -> Tensor '[o, oh, ow]  -- ^ gradOutput
  -> Conv2d f o kW kH     -- ^ conv2d state
  -> Param2d dW dH        -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Param2d pW pH        -- ^ (pW, pH) zero padding to the input plane for width and height.
  -> IO (Tensor '[b,f,h,w])
conv2dMM_backwardGradInputBatch = _conv2dMM_backwardGradInput


-- ========================================================================= --

-- | helper of forward functions with unspecified dimensions
_conv2dMM_forward
  :: (KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH)
  => Tensor din
  -> Conv2d f o kW kH
  -> Param2d dW dH
  -> Param2d pW pH
  -> IO (Tensor dout)
_conv2dMM_forward inp conv step pad =
  asStatic <$> Dynamic.spatialConvolutionMM_updateOutput
    (asDynamic inp) (asDynamic (weights conv)) (asDynamic (bias conv))
    (kernel2d conv)
    (param2d step)
    (param2d pad)

-- | helper of backward update to compute gradient input with unspecified dimensions
_conv2dMM_backwardGradInput
  :: (KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH)
  => Tensor inp
  -> Tensor gout
  -> Conv2d f o kW kH
  -> Param2d dW dH
  -> Param2d pW pH
  -> IO (Tensor d)
_conv2dMM_backwardGradInput inp gout conv step pad = do
  asStatic <$> Dynamic.spatialConvolutionMM_updateGradInput
    (asDynamic inp) (asDynamic gout)
    (asDynamic (weights conv))
    (kernel2d conv)
    (param2d step)
    (param2d pad)

_conv2dMM_updGradParameters
  :: (KnownNat2 kW kH, KnownNat2 dW dH, KnownNat2 pW pH)
  => Tensor d   -- ^ input
  -> Tensor d   -- ^ gradOutput
  -> Conv2d f o kW kH
  -- -> Tensor d   -- ^ gradWeight
  -- -> Tensor d   -- ^ gradBias
  -- -> Tensor d   -- ^ columns
  -- -> Tensor d   -- ^ ones
  -> Param2d dW dH -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Param2d pW pH -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double        -- ^ scale / learning rate
  -> IO ()
_conv2dMM_updGradParameters inp gout conv step pad lr = do
  w' <- copy (weights conv)
  b' <- copy (bias conv)
  Dynamic._spatialConvolutionMM_accGradParameters
    (asDynamic inp) (asDynamic gout)
    (asDynamic w')  (asDynamic b')
    (kernel2d conv)
    (param2d step)
    (param2d pad)
    lr


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



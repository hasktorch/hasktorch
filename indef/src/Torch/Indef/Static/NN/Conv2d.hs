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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Indef.Static.NN.Conv2d where

import Control.Arrow
import Data.Kind (Type)
import Data.List (intercalate)
import Numeric.Backprop
import System.IO.Unsafe


import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.NN.Backprop ()
import Torch.Indef.Types
import Numeric.Backprop
import qualified Torch.Indef.Dynamic.NN as Dynamic


-- ========================================================================= --

newtype Conv2d f o kW kH
  = Conv2d { getTensors :: (Tensor '[o, f, kH, kW], Tensor '[o]) }

instance KnownNat4 f o kW kH => Show (Conv2d f o kW kH) where
  show c = intercalate ","
    [ "Conv2d ("
    ++ "features: " ++ show (featureSize c)
    , " output: "   ++ show (outputSize c)
    , " kernelWidth: "   ++ show (kernelWidth c)
    , " kernelHeight:  "     ++ show (kernelHeight c)
    ++ ")"
    ]

instance KnownNat4 f o kH kW => Backprop (Conv2d f o kW kH) where
  one  = const $ Conv2d (constant 1, constant 1)
  zero = const $ Conv2d (constant 0, constant 0)
  add c0 c1 = Conv2d (weights c0 + weights c1, bias c0 + bias c1)

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

class Param2d p where
  paramW :: forall w h i . (KnownNat w, Integral i) => p w h -> i
  paramW _ = fromIntegral $ natVal (Proxy :: Proxy w)

  paramH :: forall w h i . (KnownNat h, Integral i) => p w h -> i
  paramH _ = fromIntegral $ natVal (Proxy :: Proxy h)

  param2d :: (KnownNat h, KnownNat w, Integral i) => p w h -> (i, i)
  param2d = paramW &&& paramH

data Step2d    (w::Nat) (h::Nat) = Step2d
data Padding2d (w::Nat) (h::Nat) = Padding2d

instance Param2d Step2d
instance Param2d Padding2d
instance Param2d (Conv2d f o) where

-- ========================================================================= --

type SpatialConvolutionC f h w kW kH dW dH pW pH oW oH =
  ( KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH
  , KnownNatDim2 (f * kH * kW) (oH * oW)
  , KnownNatDim2 h w
  , KnownNatDim2 oH oW
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  , (kH > 0) ~ 'True
  , (dH > 0) ~ 'True
  , ((Div (h + (2*pH) - kH) dH) + 1) > 0 ~ 'True
  , ((Div (w + (2*pW) - kW) dW) + 1) > 0 ~ 'True
  , oH ~ ((Div (h + (2*pH) - kH) dH) + 1)
  , oW ~ ((Div (w + (2*pW) - kW) dW) + 1)
  )

-- ========================================================================= --

-- | Backprop convolution function
conv2dMM
  :: Reifies s W
  => SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => KnownNatDim2 f o
  => KnownNatDim2 oH oW
  => Step2d dW dH                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d pW pH             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o kW kH)   -- ^ conv2d state
  -> BVar s (Tensor '[f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor '[o,oH,oW])
conv2dMM s p lr = liftOp2 . op2 $ \c inp ->
  (conv2dMM_forward s p c inp, \gout ->
    ( conv2dMM_updGradParameters s p lr c inp gout
    , conv2dMM_updGradInput s p c inp gout
    ))

-- | Backprop convolution function with batching
conv2dMMBatch
  :: Reifies s W
  => SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => KnownNatDim3 f o b
  => Step2d dW dH                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d pW pH             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o kW kH)   -- ^ conv2d state
  -> BVar s (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor '[b,o,oH,oW])
conv2dMMBatch s p lr = liftOp2 . op2 $ \c inp ->
  (conv2dMM_forwardBatch s p c inp, \gout ->
    ( conv2dMM_updGradParametersBatch s p lr c inp gout
    , conv2dMM_updGradInputBatch s p c inp gout
    ))


-- ========================================================================= --

-- | Applies a 2D convolution over an input image composed of several input
-- planes. The input tensor in forward(input) is expected to be a 3D tensor
-- (nInputPlane x height x width).
--
conv2dMM_forward
  :: SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => Step2d dW dH        -- ^ step of the convolution in width and height dimensions.
                         --   C-default is 1 for both.
  -> Padding2d pW pH     -- ^ zero padding to the input plane for width and height.
                         --   (kW-1)/2 is often used. C-default is 0 for both.
  -> Conv2d f o kW kH    -- ^ conv2d state
  -> Tensor '[f,h,w]     -- ^ input: f stands for "features" or "input plane"
  -> Tensor '[o, oH, oW]
conv2dMM_forward = _conv2dMM_forward

conv2dMM_updGradInput
  :: forall f h w kW kH dW dH pW pH oW oH o
  .  SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => Step2d dW dH         -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH      -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Conv2d f o kW kH     -- ^ conv2d state
  -> Tensor '[f,h,w]      -- ^ input
  -> Tensor '[o, oH, oW]  -- ^ gradOutput
  -> Tensor '[f,h,w]
conv2dMM_updGradInput =
  -- for dim1, see THNN/generic/SpatialConvolutionMM.c#L85: https://bit.ly/2KRQhsa
  -- for dim2, see THNN/generic/SpatialConvolutionMM.c#L233: https://bit.ly/2G8Dvlw
  _conv2dMM_updGradInput (new :: IO (Tensor '[f * kH * kW, oH * oW]))

conv2dMM_updGradParameters
  :: forall f h w kW kH dW dH pW pH oW oH o
  .  SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => Step2d dW dH     -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH  -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double           -- ^ scale / learning rate

  -> Conv2d f o kW kH
  -> Tensor '[f,h,w]      -- ^ input
  -> Tensor '[o, oH, oW]  -- ^ gradOutput
  -> Conv2d f o kW kH
conv2dMM_updGradParameters =
  _conv2dMM_updGradParameters (new :: IO (Tensor '[f * kH * kW, oH * oW]))


-- ========================================================================= --

-- | 'conv2dMM_forward' with a batch dimension
conv2dMM_forwardBatch
  :: forall f h w kW kH dW dH pW pH oW oH b o
  .  SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => Step2d dW dH          -- ^ step of the convolution in width and height dimensions.
                           --   C-default is 1 for both.
  -> Padding2d pW pH       -- ^ zero padding to the input plane for width and height.
                           --   (kW-1)/2 is often used. C-default is 0 for both.
  -> Conv2d f o kW kH      -- ^ conv2d state
  -> Tensor '[b,f,h,w]     -- ^ input: f stands for "features" or "input plane"
  -> Tensor '[b,o,oH,oW]
conv2dMM_forwardBatch = _conv2dMM_forward

-- | 'conv2dMM_updGradInputBatch' with batch dimension
conv2dMM_updGradInputBatch
  :: forall f h w kW kH dW dH pW pH oW oH o b
  .  SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => KnownDim b
  => Step2d dW dH         -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH      -- ^ (pW, pH) zero padding to the input plane for width and height.
  -> Conv2d f o kW kH     -- ^ conv2d state
  -> Tensor '[b,f,h,w]    -- ^ input
  -> Tensor '[b,o,oH,oW]    -- ^ gradOutput
  -> Tensor '[b,f,h,w]
conv2dMM_updGradInputBatch =
  _conv2dMM_updGradInput (new :: IO (Tensor '[b, f * kH * kW, oH * oW]))

-- | 'conv2dMM_updGradParameters' with batch dimension
conv2dMM_updGradParametersBatch
  :: forall f h w kW kH dW dH pW pH oW oH o b
  .  SpatialConvolutionC f h w kW kH dW dH pW pH oW oH
  => KnownDim b
  => Step2d dW dH     -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH  -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double           -- ^ scale / learning rate

  -> Conv2d f o kW kH
  -> Tensor '[b,f,h,w]      -- ^ input
  -> Tensor '[b,o, oH, oW]  -- ^ gradOutput
  -> Conv2d f o kW kH
conv2dMM_updGradParametersBatch =
  _conv2dMM_updGradParameters (new :: IO (Tensor '[b, f * kH * kW, oH * oW]))


-- ========================================================================= --

-- | helper of forward functions with unspecified dimensions
_conv2dMM_forward
  :: (KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH)
  => Step2d dW dH
  -> Padding2d pW pH
  -> Conv2d f o kW kH
  -> Tensor din
  -> Tensor dout
_conv2dMM_forward step pad conv inp = unsafePerformIO $
  asStatic <$> Dynamic.spatialConvolutionMM_updateOutput
    (asDynamic inp) (asDynamic (weights conv)) (asDynamic (bias conv))
    (kernel2d conv)
    (param2d step)
    (param2d pad)

-- | helper of backward update to compute gradient input with unspecified dimensions
_conv2dMM_updGradInput
  :: forall f o oH oW kW kH dW dH pW pH inp gout fgin
  . (KnownNatDim2 kW kH, KnownNatDim2 dW dH, KnownNatDim2 pW pH)
  => IO (Tensor fgin)
  -> Step2d dW dH
  -> Padding2d pW pH
  -> Conv2d f o kW kH
  -> Tensor inp
  -> Tensor gout
  -> Tensor inp
_conv2dMM_updGradInput mkGradIBuffer step pad conv inp gout = unsafePerformIO $ do
  (gin, ones) <- (,) <$> empty <*> empty
  gradInputBuffer <- mkGradIBuffer
  Dynamic._spatialConvolutionMM_updateGradInput
    (asDynamic inp) (asDynamic gout)
    (asDynamic gin) (asDynamic (weights conv))
    (asDynamic gradInputBuffer) (asDynamic ones)
    (kernel2d conv)
    (param2d step)
    (param2d pad)
  pure gin

_conv2dMM_accGradParameters
  :: forall f o oH oW kW kH dW dH pW pH inp gout finput
  .  (KnownNat2 kW kH, KnownNat2 dW dH, KnownNat2 pW pH)
  => Dimensions finput
  => IO (Tensor finput)
  -> Step2d dW dH    -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double          -- ^ scale / learning rate

  -> Conv2d f o kW kH      -- ^ weights and bias which will be mutated in-place
  -> Tensor inp            -- ^ input
  -> Tensor gout           -- ^ gradOutput
  -> IO ()
_conv2dMM_accGradParameters mkGradIBuffer step pad lr conv inp gout = do
  ones <- empty
  gradInputBuffer <- mkGradIBuffer
  Dynamic._spatialConvolutionMM_accGradParameters
    (asDynamic inp) (asDynamic gout)
    (asDynamic (weights conv)) (asDynamic (bias conv))
    (asDynamic gradInputBuffer) (asDynamic ones)
    (kernel2d conv)
    (param2d step)
    (param2d pad)
    lr

_conv2dMM_updGradParameters
  :: forall f o oH oW kW kH dW dH pW pH inp gout finput
  .  (KnownNat2 kW kH, KnownNat2 dW dH, KnownNat2 pW pH)
  => Dimensions finput
  => IO (Tensor finput)
  -> Step2d dW dH     -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> Padding2d pW pH  -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double           -- ^ scale / learning rate

  -> Conv2d f o kW kH      -- ^ weights and bias which will be mutated in-place
  -> Tensor inp            -- ^ input
  -> Tensor gout           -- ^ gradOutput
  -> Conv2d f o kW kH
_conv2dMM_updGradParameters mkGradIBuffer step pad lr conv inp gout = unsafePerformIO $ do
  conv' <- Conv2d <$> ((,) <$> copy (weights conv) <*> copy (bias conv))
  _conv2dMM_accGradParameters mkGradIBuffer step pad lr conv' inp gout
  pure conv'

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



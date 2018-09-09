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
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Conv2d where

import Control.Arrow
import Data.Kind (Type)
import Data.List (intercalate)
import Numeric.Backprop
import Numeric.Dimensions
import System.IO.Unsafe
import Data.Singletons.Prelude (type (>), type (<), Fst, Snd)
import GHC.TypeLits (Div) -- (type Div)

import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pairwise ((*^))
import Torch.Indef.Static.NN.Backprop ()
import Torch.Indef.Types
import Numeric.Backprop
import qualified Torch.Indef.Dynamic.NN as Dynamic


-- ========================================================================= --

-- | ADT representation of a convolutional 2d layer.
--
-- FIXME: the type is a bit of a hiccup: can we remove the kernel dimensions or
-- move pad/stride into the phantoms?
--
-- possibly something like @Conv2d i o (kH, kW) (dH, dW) (pH, pW)@ or
-- @Conv2d i o (kH, kW) (Maybe (dH, dW)) (Maybe (pH, pW))@
newtype Conv2d i o step
  = Conv2d { getTensors :: (Tensor '[o, i, Fst step, Snd step], Tensor '[o]) }

instance (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
  => Show (Conv2d i o '(kH, kW)) where
  show c = intercalate ","
    [ "Conv2d ("
    ++ "features: " ++ show (featureSize c)
    , " output: "   ++ show (outputSize c)
    , " kernelWidth: "   ++ show (kernelWidth c)
    , " kernelHeight: "     ++ show (kernelHeight c)
    ++ ")"
    ]

instance (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
  => Backprop (Conv2d i o '(kH,kW)) where
  one  = const $ Conv2d (constant 1, constant 1)
  zero = const $ Conv2d (constant 0, constant 0)
  add c0 c1 = Conv2d (weights c0 + weights c1, bias c0 + bias c1)

-- | get the weights from a 'Conv2d' ADT
weights :: Conv2d i o '(kH,kW) -> Tensor '[o, i, kH, kW]
weights (Conv2d (w, _)) = w

-- | get the bias from a 'Conv2d' ADT
bias :: Conv2d i o '(kH,kW) -> Tensor '[o]
bias (Conv2d (_, b)) = b

-- | get the featureSize from a 'Conv2d' ADT
featureSize :: forall i o kH kW . KnownDim i => Conv2d i o '(kH,kW) -> Int
featureSize _ = fromIntegral (dimVal (dim :: Dim i))

-- | get the outputSize from a 'Conv2d' ADT
outputSize :: forall f o kH kW . KnownDim o => Conv2d f o '(kH,kW) -> Int
outputSize _ = fromIntegral (dimVal (dim :: Dim o))

-- | get the kernelWidth from a 'Conv2d' ADT
kernelWidth :: forall i f o kH kW . (Integral i, KnownDim kW) => Conv2d f o '(kH,kW) -> i
kernelWidth _ = fromIntegral (dimVal (dim :: Dim kW))

-- | get the kernelHeight from a 'Conv2d' ADT
kernelHeight :: forall i f o kH kW . (Integral i, KnownDim kH) => Conv2d f o '(kH,kW) -> i
kernelHeight _ = fromIntegral (dimVal (dim :: Dim kH))

-- | get the kernel tuple as (width, height) from a 'Conv2d' ADT
--
-- FIXME: Isn't this supposed to be "height" /then/ "width"???
kernel2d :: (Integral i, KnownDim kH, KnownDim kW) => Conv2d f o '(kH,kW) -> (i, i)
kernel2d = kernelWidth &&& kernelHeight

-------------------------------------------------------------------------------

-- | Typeclass to generically pull out Width and Height information from a parameter
--
-- FIXME: this can be replaced with simple functions.
class Param2d (p :: (Nat, Nat) -> Type) where

  -- | get the width parameter
  paramW :: forall w h i . (KnownDim w, Integral i) => p '(h, w) -> i
  paramW _ = fromIntegral $ dimVal (dim :: Dim w)

  -- | get the height parameter
  paramH :: forall w h i . (KnownDim h, Integral i) => p '(h, w) -> i
  paramH _ = fromIntegral $ dimVal (dim :: Dim h)

  -- | get both parameters as a (width, height) tuple
  -- FIXME: Isn't this supposed to be "height" /then/ "width"???
  param2d :: (KnownDim h, KnownDim w, Integral i) => p '(h, w) -> (i, i)
  param2d = paramW &&& paramH

-- | Representation of how much to step in the height and width dimensions
data Step2d (hw :: (Nat, Nat)) = Step2d

-- | Representation of how much to pad in the height and width dimensions
data Padding2d (hw :: (Nat, Nat)) = Padding2d

-- | Representation of how big a kernel will be in the height and width dimensions
data Kernel2d (hw :: (Nat, Nat)) = Kernel2d

-- | Representation of how much to dilate in the height and width dimensions
data Dilation2d (hw :: (Nat, Nat)) = Dilation2d

instance Param2d Step2d
instance Param2d Padding2d
instance Param2d Kernel2d
instance Param2d Dilation2d
instance Param2d (Conv2d f o) where

-- ========================================================================= --

-- | Constraint to check both sides (height and width) of a function and
-- assert that all nessecary dimension values are 'KnownDim's.
type SpatialConvolutionC f h w kH kW dH dW pH pW oH oW =
  ( All KnownDim '[f * kH * kW, oH * oW, f]

  , SideCheck h kH dH pH oH
  , SideCheck w kW dW pW oW
  )

-- | Constraint to check valid dimensions on one side.
type SideCheck h k d p o =
  -- all of these are nats and 'dimensions' knows about them
  ( All KnownDim '[h,k,d,p,o]
  -- kernel and step size must be > 0
  , k > 0 ~ 'True
  , d > 0 ~ 'True
  -- kernel size can't be greater than actual input size
  , (h + (2*p)) < k ~ 'False

  -- output size must be greater than 0
  , o > 0 ~ 'True

  -- output forumlation:
  , o ~ ((Div ((h + (2*p)) - k) d) + 1)
  )


-- | Backprop convolution function
conv2dMM
  :: forall f h w kH kW dH dW pH pW oW oH s o
  .  Reifies s W
  => SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> BVar s (Tensor '[f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor '[o,oH,oW])
conv2dMM = _conv2dMM (new :: IO (Tensor '[f * kH * kW, oH * oW]))

-- | Backprop convolution function with batching
conv2dMMBatch
  :: forall f h w kH kW dH dW pH pW oW oH s o b
  .  Reifies s W
  => SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,b]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> BVar s (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor '[b,o,oH,oW])
conv2dMMBatch = _conv2dMM (new :: IO (Tensor '[b, f * kH * kW, oH * oW]))

-- | Backprop convolution function
_conv2dMM
  :: Reifies s W
  => All Dimensions '[din,dout,fgin]
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]
  => IO (Tensor fgin)            -- ^ make grad input buffer
  -> Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> BVar s (Tensor din)    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor dout)
_conv2dMM mkGradIBuffer s p lr = liftOp2 . op2 $ \c inp ->
  (_conv2dMM_forward s p c inp, \gout ->
    ( _conv2dMM_updGradParameters mkGradIBuffer s p lr c inp gout
    , _conv2dMM_updGradInput mkGradIBuffer s p c inp gout
    ))
 where
  -- | helper of forward functions with unspecified dimensions
  --_conv2dMM_forward
  --  :: All KnownDim '[kH,kW,dH,dW,pH,pW,f,o]
  --  => Step2d dH dW
  --  -> Padding2d pH pW
  --  -> Conv2d f o kH kW
  --  -> Tensor din
  --  -> Tensor dout
  {-# NOINLINE _conv2dMM_forward #-}
  _conv2dMM_forward step pad conv inp = unsafePerformIO $
    asStatic <$> Dynamic.spatialConvolutionMM_updateOutput
      (asDynamic inp) (asDynamic (weights conv)) (asDynamic (bias conv))
      (kernel2d conv)
      (param2d step)
      (param2d pad)

  -- |  conv2dMM updGradParameters
  -- _conv2dMM_updGradParameters
  --   :: forall f o oH oW kH kW dH dW pH pW inp gout finput
  --   .  All KnownDim '[kH,kW,dH,dW,pH,pW,f,o]
  --   => Dimensions finput
  --   => IO (Tensor finput)
  --   -> Step2d '(dH,dW)     -- ^ (dH, dW) step of the convolution in width and height dimensions
  --   -> Padding2d '(pH,pW)  -- ^ (pH, pW) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  --   -> Double           -- ^ scale / learning rate
  --
  --   -> Conv2d f o '(kH, kW)  -- ^ weights and bias which will be mutated in-place
  --   -> Tensor inp            -- ^ input
  --   -> Tensor gout           -- ^ gradOutput
  --   -> Conv2d f o '(kH, kW)
  {-# NOINLINE _conv2dMM_updGradParameters #-}
  _conv2dMM_updGradParameters mkGradIBuffer step pad lr conv inp gout = unsafePerformIO $ do
    let conv' = Conv2d (copy (weights conv), copy (bias conv))
    _conv2dMM_accGradParameters mkGradIBuffer step pad lr conv' inp gout
    pure conv'

  -- | helper of backward update to compute gradient input with unspecified dimensions
  -- _conv2dMM_updGradInput
  --   :: forall f o oH oW kH kW dH dW pH pW inp gout fgin
  --   .  All KnownDim '[kH,kW,dH,dW,pH,pW,f,o]
  --   => IO (Tensor fgin)
  --   -> Step2d dH dW
  --   -> Padding2d pH pW
  --   -> Conv2d f o kH kW
  --   -> Tensor inp
  --   -> Tensor gout
  --   -> Tensor inp
  {-# NOINLINE _conv2dMM_updGradInput #-}
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

  -- |  conv2dMM accGradParameters
  -- _conv2dMM_accGradParameters
  --   :: forall f o oH oW kH kW dH dW pH pW inp gout finput
  --   .  All KnownDim '[kH,kW,dH,dW,pH,pW,f,o]
  --   => Dimensions finput
  --   => IO (Tensor finput)
  --   -> Step2d dH dW    -- ^ (dH, dW) step of the convolution in width and height dimensions
  --   -> Padding2d pH pW -- ^ (pH, pW) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  --   -> Double          -- ^ scale / learning rate
  --
  --   -> Conv2d f o kH kW      -- ^ weights and bias which will be mutated in-place
  --   -> Tensor inp            -- ^ input
  --   -> Tensor gout           -- ^ gradOutput
  --   -> IO ()
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



-- -- ========================================================================= --
--
-- -- | Applies a 2D convolution over an input image composed of several input
-- -- planes. The input tensor in forward(input) is expected to be a 3D tensor
-- -- (nInputPlane x height x width).
-- --
-- conv2dMM_forward
--   :: SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => KnownDim o
--   => Step2d dH dW        -- ^ step of the convolution in width and height dimensions.
--                          --   C-default is 1 for both.
--   -> Padding2d pH pW     -- ^ zero padding to the input plane for width and height.
--                          --   (kW-1)/2 is often used. C-default is 0 for both.
--   -> Conv2d f o kH kW    -- ^ conv2d state
--   -> Tensor '[f,  h,  w] -- ^ input: f stands for "features" or "input plane"
--   -> Tensor '[o, oH, oW]
-- conv2dMM_forward = _conv2dMM_forward
--
-- -- | conv2dMM updGradInput
-- conv2dMM_updGradInput
--   :: forall f h w kH kW dH dW pH pW oW oH o
--   .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => KnownDim o
--   => Step2d dH dW         -- ^ (dH, dW) step of the convolution in width and height dimensions
--   -> Padding2d pH pW      -- ^ (pH, pW) zero padding to the input plane for width and height. (kW-1)/2 is often used.
--   -> Conv2d f o kH kW     -- ^ conv2d state
--   -> Tensor '[f,h,w]      -- ^ input
--   -> Tensor '[o, oH, oW]  -- ^ gradOutput
--   -> Tensor '[f,h,w]
-- conv2dMM_updGradInput =
--   -- for dim1, see THNN/generic/SpatialConvolutionMM.c#L85: https://bit.ly/2KRQhsa
--   -- for dim2, see THNN/generic/SpatialConvolutionMM.c#L233: https://bit.ly/2G8Dvlw
--   _conv2dMM_updGradInput (new :: IO (Tensor '[f * kH * kW, oH * oW]))
--
-- -- | conv2dMM updGradParameters
-- conv2dMM_updGradParameters
--   :: forall f h w kH kW dH dW pH pW oW oH o
--   .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => KnownDim o
--   => Step2d dH dW     -- ^ (dH, dW) step of the convolution in width and height dimensions
--   -> Padding2d pH pW  -- ^ (pH, pW) zero padding to the input plane for width and height. (kW-1)/2 is often used.
--   -> Double           -- ^ scale / learning rate
--
--   -> Conv2d f o kH kW
--   -> Tensor '[f,h,w]      -- ^ input
--   -> Tensor '[o, oH, oW]  -- ^ gradOutput
--   -> Conv2d f o kH kW
-- conv2dMM_updGradParameters =
--   _conv2dMM_updGradParameters (new :: IO (Tensor '[f * kH * kW, oH * oW]))
--
--
-- -- ========================================================================= --
--
-- -- | 'conv2dMM_forward' with a batch dimension
-- conv2dMM_forwardBatch
--   :: forall f h w kH kW dH dW pH pW oW oH b o
--   .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => All KnownDim '[b,o]
--   => Step2d dH dW          -- ^ step of the convolution in width and height dimensions.
--                            --   C-default is 1 for both.
--   -> Padding2d pH pW       -- ^ zero padding to the input plane for width and height.
--                            --   (kW-1)/2 is often used. C-default is 0 for both.
--   -> Conv2d f o kH kW      -- ^ conv2d state
--   -> Tensor '[b,f,h,w]     -- ^ input: f stands for "features" or "input plane"
--   -> Tensor '[b,o,oH,oW]
-- conv2dMM_forwardBatch = _conv2dMM_forward
--
-- -- | 'conv2dMM_updGradInputBatch' with batch dimension
-- conv2dMM_updGradInputBatch
--   :: forall f h w kH kW dH dW pH pW oW oH o b
--   .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => All KnownDim '[b,o,f*kW]
--   => Step2d dH dW         -- ^ (dH, dW) step of the convolution in width and height dimensions
--   -> Padding2d pH pW      -- ^ (pH, pW) zero padding to the input plane for width and height.
--   -> Conv2d f o kH kW     -- ^ conv2d state
--   -> Tensor '[b,f,h,w]    -- ^ input
--   -> Tensor '[b,o,oH,oW]    -- ^ gradOutput
--   -> Tensor '[b,f,h,w]
-- conv2dMM_updGradInputBatch =
--   _conv2dMM_updGradInput (new :: IO (Tensor '[b, f * kH * kW, oH * oW]))
--
-- -- | 'conv2dMM_updGradParameters' with batch dimension
-- conv2dMM_updGradParametersBatch
--   :: forall f h w kH kW dH dW pH pW oW oH o b
--   .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
--   => All KnownDim '[b,o]
--   => Step2d dH dW     -- ^ (dH, dW) step of the convolution in width and height dimensions
--   -> Padding2d pH pW  -- ^ (pH, pW) zero padding to the input plane for width and height. (kW-1)/2 is often used.
--   -> Double           -- ^ scale / learning rate
--
--   -> Conv2d f o kH kW
--   -> Tensor '[b,f,h,w]      -- ^ input
--   -> Tensor '[b,o, oH, oW]  -- ^ gradOutput
--   -> Conv2d f o kH kW
-- conv2dMM_updGradParametersBatch =
--   _conv2dMM_updGradParameters (new :: IO (Tensor '[b, f * kH * kW, oH * oW]))


-- ========================================================================= --

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



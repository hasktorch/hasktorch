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
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP #-}

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Conv2d where

import Control.Arrow
import Data.Kind (Type)
import Data.List (intercalate)
import Data.Maybe
import Numeric.Backprop
import Numeric.Dimensions
import System.IO.Unsafe
import Data.Singletons.Prelude (type (>), type (<), Fst, Snd)
import GHC.TypeLits (Div) -- (type Div)

import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pairwise (Pairwise(..))
import Torch.Indef.Static.NN.Backprop ()
import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic

import qualified Torch.Indef.Dynamic.Tensor.Math as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pairwise as Dynamic

-- ========================================================================= --

-- | ADT representation of a convolutional 2d layer.
--
-- FIXME: the type is a bit of a hiccup: can we remove the kernel dimensions or
-- move pad/stride into the phantoms?
--
-- possibly something like @Conv2d i o (kH, kW) (dH, dW) (pH, pW)@ or
-- @Conv2d i o (kH, kW) (Maybe (dH, dW)) (Maybe (pH, pW))@
newtype Conv2d i o kers
  = Conv2d { getTensors :: (Tensor '[o, i, Fst kers, Snd kers], Tensor '[o]) }

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

-- instance (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
--   => Backprop (Conv2d i o '(kH,kW)) where
--

instance (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
  => Backprop (Conv2d i o '(kH,kW)) where
  one  = const $ Conv2d (constant 1, constant 1)
  zero = const $ Conv2d (constant 0, constant 0)

--   one  (Conv2d (a, b)) = unsafePerformIO $ do
--     Dynamic.onesLike_ (asDynamic a) (asDynamic a)
--     Dynamic.onesLike_ (asDynamic b) (asDynamic b)
--     pure (Conv2d (a, b))
--   {-# NOINLINE one #-}
--
--   zero (Conv2d (a, b)) = unsafePerformIO $ do
--     Dynamic.zerosLike_ (asDynamic a) (asDynamic a)
--     Dynamic.zerosLike_ (asDynamic b) (asDynamic b)
--     pure (Conv2d (a, b))
--   {-# NOINLINE zero #-}
--
  add (Conv2d (a0, b0)) (Conv2d (a1, b1)) = unsafePerformIO $ do
    Dynamic.cadd_ (asDynamic a1) 1 (asDynamic a0)
    Dynamic.cadd_ (asDynamic b1) 1 (asDynamic b0)
    pure (Conv2d (a1, b1))
  {-# NOINLINE add #-}
  -- add c0 c1 = Conv2d (weights c0 + weights c1, bias c0 + bias c1)

instance (All KnownDim '[i, o, Fst kers, Snd kers]) => Num (Conv2d i o kers) where
  (+) (Conv2d (a0, b0)) (Conv2d (a1, b1)) = Conv2d (a0+a1, b0+b1)
  (-) (Conv2d (a0, b0)) (Conv2d (a1, b1)) = Conv2d (a0-a1, b0-b1)
  (*) (Conv2d (a0, b0)) (Conv2d (a1, b1)) = Conv2d (a0*a1, b0*b1)
  abs (Conv2d (a0, b0)) = Conv2d (abs a0, abs b0)
  fromInteger i = Conv2d (fromInteger i, fromInteger i)

instance (All KnownDim '[i, o, Fst kers, Snd kers]) => Pairwise (Conv2d i o kers) HsReal where
  (Conv2d tens) ^+ v = Conv2d (tens ^+ v)
  (Conv2d tens) ^- v = Conv2d (tens ^- v)
  (Conv2d tens) ^* v = Conv2d (tens ^* v)
  (Conv2d tens) ^/ v = Conv2d (tens ^/ v)


-- | update a Conv2d layer
update
  :: (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
  => Conv2d i o '(kH, kW)  -- ^ network to update
  -> HsReal                -- ^ learning rate
  -> Conv2d i o '(kH, kW)  -- ^ gradient
  -> Conv2d i o '(kH, kW)  -- ^ updated network
update (Conv2d (w, b)) lr (Conv2d (gw, gb)) = Conv2d (w + gw ^* lr, b + gb ^* lr)

-- | update a Conv2d layer inplace
update_
  :: (KnownDim i, KnownDim o, KnownDim kH, KnownDim kW)
  => Conv2d i o '(kH, kW)  -- ^ network to update
  -> HsReal                -- ^ learning rate
  -> Conv2d i o '(kH, kW)  -- ^ gradient
  -> IO ()  -- ^ update network
update_ (Conv2d (w, b)) lr (Conv2d (gw, gb)) = do
  Dynamic.cadd_ (asDynamic w) lr (asDynamic gw)
  Dynamic.cadd_ (asDynamic b) lr (asDynamic gb)




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
  , (k > 0) ~ 'True
  , (d > 0) ~ 'True
  -- kernel size can't be greater than actual input size
  , ((h + (2*p)) < k) ~ 'False

  -- output size must be greater than 0
  , (o > 0) ~ 'True

  -- output forumlation:
  , o ~ ((Div ((h + (2*p)) - k) d) + 1)
  )


-- | Backprop convolution function with batching
conv2dBatchIO
  :: forall f h w kH kW dH dW pH pW oW oH s o b
  .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,b,kW*kH*f,oH*oW]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate
  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor '[b, o,oH,oW], (Tensor '[b,o,oH,oW] -> IO (Conv2d f o '(kH,kW), Tensor '[b,f,h,w])))
conv2dBatchIO = genericConv2dWithIO
  (Just ( constant 0 :: Tensor '[b,f,h,w]))
  (Just ( constant 0 :: Tensor '[b,kW*kH*f,oH*oW]))
  (Just ( constant 0 :: Tensor '[b,kW*kH*f,oH*oW]))
  (Just $ constant 0)
  (Just $ constant 0)
  (Just $ Conv2d (constant 0, constant 0))

-- | Backprop convolution function with batching
{-# NOINLINE conv2dBatch #-}
conv2dBatch
  :: Reifies s W
  => SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW,b]
  => All KnownDim '[kW*kH*f,oH*oW]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
                                 --   C-default is 1 for both.
                                 --
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
                                 --   (kW-1)/2 is often used. C-default is 0 for both.
                                 --
  -> Double                      -- ^ learning rate
  -> BVar s (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> BVar s (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> BVar s (Tensor '[b, o,oH,oW])
conv2dBatch step pad lr = liftOp2 . op2 $ \conv inp -> unsafePerformIO $ do
  (o, getgrad) <- conv2dBatchIO step pad lr conv inp
  pure (o, unsafePerformIO . getgrad)


-- | Backprop convolution function with batching
conv2dIO
  :: forall f h w kH kW dH dW pH pW oW oH s o
  .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,kW*kH*f,oH*oW]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate
  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor '[f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor '[o,oH,oW], (Tensor '[o,oH,oW] -> IO (Conv2d f o '(kH,kW), Tensor '[f,h,w])))
conv2dIO = genericConv2dWithIO
  (Just ( constant 0 :: Tensor '[f,h,w]))
  (Just ( constant 0 :: Tensor '[kW*kH*f,oH*oW]))
  (Just ( constant 0 :: Tensor '[kW*kH*f,oH*oW]))
  (Just $ constant 0)
  (Just $ constant 0)
  (Just $ Conv2d (constant 0, constant 0))


-- | Backprop convolution function
{-# NOINLINE conv2d #-}
conv2d
  :: Reifies s W
  => SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]
  => All KnownDim '[kW*kH*f,oH*oW]
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
conv2d step pad lr = liftOp2 . op2 $ \conv inp -> unsafePerformIO $ do
  (o, getgrad) <- conv2dIO step pad lr conv inp
  pure (o, unsafePerformIO . getgrad)

genericConv2dWithIO
  :: forall din dout fgin f o kH kW dH dW pH pW inBuff
  .  All Dimensions '[din,dout,fgin, inBuff]
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]

  -- buffers
  => Maybe (Tensor fgin)            -- ^ grad input buffer
  -> Maybe (Tensor inBuff)            -- ^ columns buffer
  -> Maybe (Tensor inBuff)            -- ^ ones buffer

  -- cacheables
  -> Maybe (Tensor dout)            -- output
  -> Maybe (Tensor din)             -- gradient input
  -> Maybe (Conv2d f o '(kH, kW))   -- gradient params

  -> Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate

  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor din)    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor dout, (Tensor dout -> IO (Conv2d f o '(kH,kW), Tensor din)))
genericConv2dWithIO
  mginbuffer mfinput mfgradInput mout mgin mgparams
  step pad lr conv inp = do
  let ginbuffer = fromMaybe new mginbuffer
  let finput = fromMaybe new mfinput
  let fgradInput = fromMaybe new mfgradInput
  let out = fromMaybe new mout

  zero_ out
  updateOutput_ finput fgradInput step pad conv inp out

  pure (copy out,
    \gout -> do
      let gin = fromMaybe new mgin
      let gparams = fromMaybe (Conv2d (new, new)) mgparams
      zero_ gin
      zero_ (weights gparams)
      zero_ (bias gparams)

      updateGradInput_ inp gout gin conv finput fgradInput step pad

      accGradParameters_ inp gout gparams finput fgradInput step pad
      print (weights gparams)

      pure (gparams, gin))
 where
  updateOutput_ finput fgradInput step pad conv inp out =
    Dynamic._spatialConvolutionMM_updateOutput
      (asDynamic inp)                      -- input
      (asDynamic out)                      -- output
      (asDynamic (weights conv))    -- 3D weight tensor (connTable:size(1) x kH x kW)
      (asDynamic (bias conv))       -- 1D bias tensor (nOutputPlane)

      (asDynamic finput)                  -- BUFFER: temporary columns -- also called "finput"
      (asDynamic fgradInput)              -- BUFFER: buffer of ones for bias accumulation  -- also called "fgradInput"

      (kernel2d conv)               -- (kW, kH) kernel height and width
      (param2d step)                       -- (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
      (param2d pad)                        -- (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.

  updateGradInput_ inp gout gin conv colsbuffer onesbuffer step pad = do
    Dynamic._spatialConvolutionMM_updateGradInput
      (asDynamic inp)                      -- input
      (asDynamic gout)                     -- gradOutput
      (asDynamic gin)                      -- gradInput
      (asDynamic (weights conv))    -- weight
      (asDynamic colsbuffer)               -- columns
      (asDynamic onesbuffer)               -- ones
      (kernel2d conv)               -- (kW, kH) kernel height and width
      (param2d step)                       -- (dW, dH) step of the convolution in width and height dimensions
      (param2d pad)                        -- (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.


  accGradParameters_ inp gout gconv columnsbuff onesbuff step pad = do
    Dynamic._spatialConvolutionMM_accGradParameters
      (asDynamic inp)    -- input
      (asDynamic gout)    -- gradOutput
      (asDynamic (weights gconv))    -- gradWeight
      (asDynamic (bias gconv))    -- gradBias
      (asDynamic columnsbuff)    -- finput/columns <<- required. This can be NULL in C if gradWeight is NULL.
      (asDynamic onesbuff)   -- ones
      (kernel2d conv) -- (kW, kH) kernel height and width
      (param2d step)         -- (dW, dH) step of the convolution in width and height dimensions
      (param2d pad)          -- (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
      lr

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



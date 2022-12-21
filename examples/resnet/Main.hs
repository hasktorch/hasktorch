{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}

module Main where

import Common
import Control.Exception.Safe
  ( SomeException (..),
    try,
  )
import GHC.Generics
import GHC.TypeLits
import System.Environment
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed
import Torch.Typed.Functional
import qualified Torch.Tensor as D
import Data.Proxy

import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Cast
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.Tuple as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Internal.Cast
import System.IO.Unsafe


type NoStrides = '(1, 1)

type NoPadding = '(0, 0)

type KernelSize = '(2, 2)

type Strides = '(2, 2)

type Padding = '(1,1)

data VggConfig = VggConfig deriving (Show, Generic, Parameterized)

instance 
  ( All KnownNat '[ inputChannelSize, outputChannelSize,
                    inputSize0, inputSize1,
                    batchSize,
                    outputSize0, outputSize1 ]
  , ConvSideCheck inputSize0 3 1 0 outputSize0 
  , ConvSideCheck inputSize1 3 1 0 outputSize1
  , StandardFloatingPointDTypeValidation device dtype
  ) =>
  HasForward
    (Conv2d inputChannelSize outputChannelSize 3 3 dtype device, VggConfig)
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1])
  where
    forward (conv,_) = relu . conv2dForward @'(1,1) @'(0,0) conv
    forwardStoch = (pure .) . forward

instance 
  ( All KnownNat '[ inputChannelSize,
                    inputSize0, inputSize1,
                    batchSize,
                    outputSize0, outputSize1 ]
  , ConvSideCheck inputSize0 2 1 0 outputSize0 
  , ConvSideCheck inputSize1 2 1 0 outputSize1
  ) =>
  HasForward
    (MaxPool2d, VggConfig)
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, inputChannelSize, outputSize0, outputSize1])
  where
    forward (conv,_) = maxPool2d @'(2,2) @'(1,1) @'(0,0)
    forwardStoch = (pure .) . forward

instance 
  ( All KnownNat '[channelSize, inputSize0, inputSize1, batchSize]
  ) =>
  HasForward
    (AdaptiveAvgPool2d, VggConfig)
    (Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, channelSize, 7, 7])
  where
    forward _ = adaptiveAvgPool2d @'(7,7)
    forwardStoch = (pure .) . forward

instance 
  ( StandardFloatingPointDTypeValidation device dtype
  ) =>
  HasForward
    Relu
    (Tensor device dtype shape)
    (Tensor device dtype shape)
  where
    forward _ = relu
    forwardStoch = (pure .) . forward

instance 
  HasForward
    Id
    (Tensor device dtype shape)
    (Tensor device dtype shape)
  where
    forward _ = id
    forwardStoch = (pure .) . forward


instance 
  ( All KnownNat '[channelSize, inputSize0, inputSize1, batchSize, outputSize]
  , channelSize * inputSize0 * inputSize1 ~ outputSize
  ) =>
  HasForward
    (Flatten, VggConfig)
    (Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, outputSize])
  where
    forward _ = reshape
    forwardStoch = (pure .) . forward

data MaxPool2d = MaxPool2d deriving (Show, Generic, Parameterized)

data AdaptiveAvgPool2d = AdaptiveAvgPool2d deriving (Show, Generic, Parameterized)

data Flatten = Flatten deriving (Show, Generic, Parameterized)

data Relu = Relu deriving (Show, Generic, Parameterized)

data Id = Id deriving (Show, Generic, Parameterized)

data (:>>>) a b = Forward a b deriving (Show, Generic)
data (:&&&) a b = Fanout a b deriving (Show, Generic)
data (:***) a b = Merge a b deriving (Show, Generic)

instance (HasForward f a b, HasForward g b c) => HasForward (f :>>> g) a c where
  forward (Forward f g) = forward g . forward f
  forwardStoch (Forward f g) a = forwardStoch f a >>= forwardStoch g

instance (HasForward f a b, HasForward g a c) => HasForward (f :&&& g) a (b,c) where
  forward (Fanout f g) a = (forward f a, forward g a)
  forwardStoch (Fanout f g) a = do
    a0 <- forwardStoch f a
    a1 <- forwardStoch g a
    return (a0,a1)

instance (HasForward f a0 b, HasForward g a1 c) => HasForward (f :*** g) (a0,a1) (b,c) where
  forward (Merge f g) (a0,a1) = (forward f a0, forward g a1)
  forwardStoch (Merge f g) (a0,a1) = do
    a0' <- forwardStoch f a0
    a1' <- forwardStoch g a1
    return (a0',a1')

type VGGClassifier dtype device numClasses =
  (Linear (512 * 7 * 7) 4096 dtype device) :>>>
  Relu :>>>
  Dropout :>>>
  (Linear 4096 4096 dtype device) :>>>
  Relu :>>>
  Dropout :>>>
  (Linear 4096 numClasses dtype device)

newtype VGG16 dtype device numClasses =
  VGG16 (
  (Conv2d 3 64 3 3 dtype device, VggConfig) :>>>
  (Conv2d 64 64 3 3 dtype device, VggConfig) :>>>
  (MaxPool2d, VggConfig) :>>>
  (Conv2d 64 128 3 3 dtype device, VggConfig) :>>>
  (Conv2d 128 128 3 3 dtype device, VggConfig) :>>>
  (MaxPool2d, VggConfig) :>>>
  (Conv2d 128 256 3 3 dtype device, VggConfig) :>>>
  (Conv2d 256 256 3 3 dtype device, VggConfig) :>>>
  (Conv2d 256 256 3 3 dtype device, VggConfig) :>>>
  (MaxPool2d, VggConfig) :>>>
  (Conv2d 256 512 3 3 dtype device, VggConfig) :>>>
  (Conv2d 512 512 3 3 dtype device, VggConfig) :>>>
  (Conv2d 512 512 3 3 dtype device, VggConfig) :>>>
  (MaxPool2d, VggConfig) :>>>
  (Conv2d 512 512 3 3 dtype device, VggConfig) :>>>
  (Conv2d 512 512 3 3 dtype device, VggConfig) :>>>
  (Conv2d 512 512 3 3 dtype device, VggConfig) :>>>
  (AdaptiveAvgPool2d, VggConfig) :>>>
  (Flatten, VggConfig) :>>>
  (VGGClassifier dtype device numClasses)
  ) -- deriving (Show, Generic, Parameterized)

forwardVgg16
  :: ( All KnownNat '[ inputSize0, inputSize1, batchSize, numClasses ]
     , StandardFloatingPointDTypeValidation device dtype
     , 31 <= inputSize0
     , 31 <= inputSize1)
  => VGG16 dtype device numClasses
  -> Tensor device dtype '[batchSize, 3, inputSize0, inputSize1]
  -> IO (Tensor device dtype '[batchSize, numClasses])
forwardVgg16 (VGG16 conv) = forwardStoch conv


data ResnetConfig = ResnetConfig

data Shortcut a = Shortcut a

instance 
  ( All KnownNat '[ inputChannelSize, outputChannelSize,
                    inputSize0, inputSize1,
                    batchSize ]
  , KnownDevice device
  , HasForward
      a
      (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
      (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
  ) =>
  HasForward
    (Shortcut a)
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
  where
    forward (Shortcut a) input = forward a input + input
    forwardStoch (Shortcut a) input = do
      output <- forwardStoch a input
      return (output + input)

data ConvConfig (stride :: Nat) (padding :: Nat) = ConvConfig

instance 
  ( All KnownNat '[ inputChannelSize, outputChannelSize,
                    kernelSize, stride, padding,
                    inputSize0, inputSize1,
                    batchSize,
                    outputSize0, outputSize1 ]
  , ConvSideCheck inputSize0 kernelSize stride padding outputSize0 
  , ConvSideCheck inputSize1 kernelSize stride padding outputSize1
  , StandardFloatingPointDTypeValidation device dtype
  ) =>
  HasForward
    (Conv2d inputChannelSize outputChannelSize kernelSize kernelSize dtype device, ConvConfig stride padding)
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1])
  where
    forward (conv,_) = relu . conv2dForward @'(stride,stride) @'(padding,padding) conv
    forwardStoch = (pure .) . forward

type BasicBlock dtype device inchannel outchennel =
  Shortcut (
    (Conv2d inchannel outchennel 7 7 dtype device, ConvConfig 2 3) :>>>
    (BatchNorm2d outchennel dtype device) :>>>
    (Relu) :>>>
    (Conv2d outchennel outchennel 7 7 dtype device, ConvConfig 2 3) :>>>
    (BatchNorm2d outchennel dtype device) :>>>
    (Relu)
  )

instance 
  ( HasForward
      a
      (Tensor device dtype shape)
      (Tensor device dtype shape)
  ) =>
  HasForward
    (ReplicateBlock num a)
    (Tensor device dtype shape)
    (Tensor device dtype shape)
  where
    forward (ReplicateBlockZero) input = input
    forward (ReplicateBlock x xs) input = forward xs (forward x input)
    forwardStoch (ReplicateBlockZero) input = pure input
    forwardStoch (ReplicateBlock x xs) input = forwardStoch x input >>= forwardStoch xs

data ReplicateBlock (num :: Nat) a where
  ReplicateBlockZero :: ReplicateBlock 0 a
  ReplicateBlock :: a -> ReplicateBlock (num - 1) a -> ReplicateBlock num a

newtype MutableTensor device dtype shape = MutableTensor { fromMutable :: Tensor device dtype shape }

instance Show (MutableTensor device dtype shape) where
  show = show . fromMutable

data
  BatchNorm2d
    (channelSize :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = BatchNorm2d
  { weight :: Parameter device dtype '[channelSize],
    bias :: Parameter device dtype '[channelSize],
    runningMean :: MutableTensor device dtype '[channelSize],
    runningVar :: MutableTensor device dtype '[channelSize]
  }
  deriving (Show, Generic)

instance 
  ( All KnownNat '[ channelSize,
                    inputSize0, inputSize1,
                    batchSize ]
  ) =>
  HasForward
    (BatchNorm2d (channelSize :: Nat) dtype device)
    (Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1])
  where
    forward params input = unsafePerformIO $ do
       tensor <-  cast9
                    ATen.batch_norm_tttttbddb
                    (toDynamic input :: D.Tensor)
                    (toDynamic (toDependent params.weight :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (toDependent params.bias :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (fromMutable params.runningMean :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (fromMutable params.runningVar :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    False -- training or not
                    (0.1 :: Double) -- momentum
                    (1e-5 :: Double) -- eps
                    True
       return (UnsafeMkTensor tensor)
    forwardStoch params input = do
       tensor <-  cast9
                    ATen.batch_norm_tttttbddb
                    (toDynamic input :: D.Tensor)
                    (toDynamic (toDependent params.weight :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (toDependent params.bias :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (fromMutable params.runningMean :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    (toDynamic (fromMutable params.runningVar :: Tensor device dtype '[channelSize]) :: D.Tensor)
                    True -- training or not
                    (0.1 :: Double) -- momentum
                    (1e-5 :: Double) -- eps
                    True
       return (UnsafeMkTensor tensor)



instance 
  ( All KnownNat '[ inputChannelSize,
                    inputSize0, inputSize1,
                    batchSize,
                    outputSize0, outputSize1 ]
  , outputSize0 ~ 2 + Div (inputSize0 - 1) 2
  , outputSize1 ~ 2 + Div (inputSize1 - 1) 2
  , ConvSideCheck inputSize0 2 1 0 outputSize0 
  , ConvSideCheck inputSize1 2 1 0 outputSize1
  ) =>
  HasForward
    (MaxPool2d, ResnetConfig)
    (Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1])
    (Tensor device dtype '[batchSize, inputChannelSize, outputSize0, outputSize1])
  where
    forward (conv,_) = maxPool2d @'(3,3) @'(2,2) @'(1,1)
    forwardStoch = (pure .) . forward

newtype Resnet dtype device (numClasses :: Nat) =
  Resnet (
  (Conv2d 3 64 7 7 dtype device, ConvConfig 2 3) :>>>
  BatchNorm2d 64 dtype device :>>>
  Relu :>>>
  (Conv2d 3 64 3 3 dtype device, ConvConfig 2 1) :>>>
--  (MaxPool2d, ResnetConfig) :>>>
  (ReplicateBlock 2 (BasicBlock dtype device 64 64))
  -- (ReplicateBlock 2 (BasicBlock dtype device 64 128)) :>>>
  -- (ReplicateBlock 2 (BasicBlock dtype device 128 256)) :>>>
  -- (ReplicateBlock 2 (BasicBlock dtype device 256 512)) :>>>
  -- BasicBlock dtype device 64 64 :>>>
  -- BasicBlock dtype device 64 128 :>>>
  -- BasicBlock dtype device 128 256 :>>>
  -- BasicBlock dtype device 256 512 :>>>
  -- (AdaptiveAvgPool2d, ResnetConfig) :>>>
  -- (Flatten, ResnetConfig) :>>>
  -- (Linear 512 numClasses dtype device, ResnetConfig)
  )
  
forwardResnet
  :: ( All KnownNat '[ inputSize0, inputSize1, batchSize, numClasses ]
     , StandardFloatingPointDTypeValidation device dtype
     , 1 <= inputSize0
     , 1 <= inputSize1)
  => Resnet dtype device numClasses
  -> Tensor device dtype '[batchSize, 3, inputSize0, inputSize1]
  -> IO (Tensor device dtype '[batchSize, 64, _, _])
forwardResnet (Resnet conv) = forwardStoch conv


data CNNSpec (dtype :: DType) (device :: (DeviceType, Nat))
  = CNNSpec
  deriving (Show, Eq)

data CNN (dtype :: DType) (device :: (DeviceType, Nat)) where
  CNN ::
    forall dtype device.
    { conv0 :: Conv2d 1 20 5 5 dtype device,
      conv1 :: Conv2d 20 50 5 5 dtype device,
      fc0 :: Linear (4 * 4 * 50) 500 dtype device,
      fc1 :: Linear 500 ClassDim dtype device
    } ->
    CNN dtype device
  deriving (Show, Generic, Parameterized)

cnn ::
  forall batchSize dtype device.
  _ =>
  CNN dtype device ->
  Tensor device dtype '[batchSize, DataDim] ->
  Tensor device dtype '[batchSize, ClassDim]
cnn CNN {..} =
  forward fc1
    . relu
    . forward fc0
    . reshape @'[batchSize, 4 * 4 * 50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2dForward @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2dForward @NoStrides @NoPadding conv0
    . unsqueeze @1
    . reshape @'[batchSize, Rows, Cols]

instance
  ( KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (CNNSpec dtype device)
    (CNN dtype device)
  where
  sample CNNSpec =
    CNN
      <$> sample (Conv2dSpec @1 @20 @5 @5)
      <*> sample (Conv2dSpec @20 @50 @5 @5)
      <*> sample (LinearSpec @(4 * 4 * 50) @500)
      <*> sample (LinearSpec @500 @10)

type BatchSize = 256

train' ::
  forall (device :: (DeviceType, Nat)).
  _ =>
  IO ()
train' = do
  let learningRate = 0.001
      numEpochs = 30
  manual_seed_L 123
--  initModel <- sample (CNNSpec @'Float @device)
  initModel <- sample (CNNSpec @'Float @device)
  let initOptim = mkAdam 0.00001 0.9 0.999 (flattenParameters initModel)
  (trainedModel,_) <- train @BatchSize @device
    initModel
    initOptim
    (\model _ input -> return $ cnn model input)
    learningRate
    "static-mnist-cnn.pt"
    numEpochs
    
  test  @BatchSize @device trainedModel (\model _ input -> return $ cnn model input)

  return ()

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu" -> train' @'( 'CPU, 0)
    Right "cuda:0" -> train' @'( 'CUDA, 0)
    Right device -> error $ "Unknown device setting: " ++ device
    _ -> train' @'( 'CPU, 0)

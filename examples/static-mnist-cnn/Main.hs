{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           GHC.Generics
import           GHC.TypeLits
import           System.Environment

import Torch.Typed
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import           Common
import Pipeline
import Control.Monad.State.Lazy

type NoStrides = '(1, 1)
type NoPadding = '(0, 0)

type KernelSize = '(2, 2)
type Strides = '(2, 2)

data CNNSpec (dtype :: DType) (device :: (DeviceType, Nat))
  = CNNSpec deriving (Show, Eq)

data CNN (dtype :: DType) (device :: (DeviceType, Nat))
 where
  CNN
    :: forall dtype device
     . { conv0 :: Conv2d 1  20 5 5 dtype device
       , conv1 :: Conv2d 20 50 5 5 dtype device
       , fc0   :: Linear (4*4*50) 500      dtype device
       , fc1   :: Linear 500      ClassDim dtype device
       }
    -> CNN dtype device
 deriving (Show, Generic)

cnn
  :: forall batchSize dtype device
   . _
  => CNN dtype device
  -> Tensor device dtype '[batchSize, DataDim]
  -> Tensor device dtype '[batchSize, ClassDim]
cnn CNN {..} =
  forward fc1
    . relu
    . forward fc0
    . reshape @'[batchSize, 4*4*50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2dForward @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2dForward @NoStrides @NoPadding conv0
    . unsqueeze @1
    . reshape @'[batchSize, Rows, Cols]

instance ( KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => Randomizable (CNNSpec dtype device)
                  (CNN     dtype device)
 where
  sample CNNSpec =
    CNN
      <$> sample (Conv2dSpec @1  @20 @5 @5)
      <*> sample (Conv2dSpec @20 @50 @5 @5)
      <*> sample (LinearSpec @(4*4*50) @500)
      <*> sample (LinearSpec @500      @10)

  -- TODO: figure out how to write the constraints so we can have CNN implement the HasForward typeclass
-- instance HasForward (CNN dtype device) (Tensor device dtype '[batchSize, I.DataDim]) (Tensor device dtype '[batchSize, I.ClassDim]) where
--   forward = cnn
  
type BatchSize = 256

train'
  :: forall (device :: (DeviceType, Nat))
   . _
  => IO ()
train' = do
  let learningRate = 0.1
  manual_seed_L 123
  initModel <- sample (CNNSpec @ 'Float @device)
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  newMain @BatchSize @device  cnn initModel initOptim 10 

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu"    -> train' @'( 'CPU, 0)
    Right "cuda:0" -> train' @'( 'CUDA, 0)
    _              -> error "Don't know what to do or how."

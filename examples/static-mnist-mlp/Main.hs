{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Common
import Control.Exception.Safe
  ( SomeException (..),
    try,
  )
import Control.Monad
  ( foldM,
    when,
  )
import Data.Proxy
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import System.Environment
import System.IO.Unsafe
import System.Random
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed
import Prelude hiding (tanh)

--------------------------------------------------------------------------------
-- MLP for MNIST
--------------------------------------------------------------------------------

data
  MLPSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  MLPSpec ::
    forall inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    {mlpDropoutProbSpec :: Double} ->
    MLPSpec inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device
  deriving (Show, Eq)

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  MLP ::
    forall inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    { mlpLayer0 :: Linear inputFeatures hiddenFeatures0 dtype device,
      mlpLayer1 :: Linear hiddenFeatures0 hiddenFeatures1 dtype device,
      mlpLayer2 :: Linear hiddenFeatures1 outputFeatures dtype device,
      mlpDropout :: Dropout
    } ->
    MLP inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device
  deriving (Show, Generic, Parameterized)

mlp ::
  forall
    batchSize
    inputFeatures
    outputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device.
  (StandardFloatingPointDTypeValidation device dtype) =>
  MLP
    inputFeatures
    outputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device ->
  Bool ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (Tensor device dtype '[batchSize, outputFeatures])
mlp MLP {..} train input =
  return
    . forward mlpLayer2
    =<< dropoutForward mlpDropout train
      . tanh
      . forward mlpLayer1
    =<< dropoutForward mlpDropout train
      . tanh
      . forward mlpLayer0
    =<< pure input

instance
  ( KnownNat inputFeatures,
    KnownNat outputFeatures,
    KnownNat hiddenFeatures0,
    KnownNat hiddenFeatures1,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
    (MLP inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
  where
  sample MLPSpec {..} =
    MLP
      <$> sample LinearSpec
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample (DropoutSpec mlpDropoutProbSpec)

type BatchSize = 64

type HiddenFeatures0 = 512

type HiddenFeatures1 = 256

train' ::
  forall (device :: (DeviceType, Nat)).
  _ =>
  IO ()
train' = do
  let dropoutProb = 0.5
      learningRate = 0.1
  manual_seed_L 123
  initModel <-
    sample
      ( MLPSpec @DataDim @ClassDim
          @HiddenFeatures0
          @HiddenFeatures1
          @'Float
          @device
          dropoutProb
      )
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  train @BatchSize @device initModel initOptim mlp learningRate "static-mnist-mlp.pt"

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu" -> train' @'( 'CPU, 0)
    Right "cuda:0" -> train' @'( 'CUDA, 0)
    Right device -> error $ "Unknown device setting: " ++ device
    _ -> train' @'( 'CPU, 0)

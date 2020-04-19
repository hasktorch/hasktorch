{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
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
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.NN as A
import qualified Torch.Serialize as D
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding
  ( dropout,
    linear,
  )
import Torch.Typed.NN
import Torch.Typed.Optim
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import qualified Torch.Typed.Vision as I
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
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  MLPSpec ::
    forall inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    {mlpDropoutProbSpec :: Double} ->
    MLPSpec inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype
      device
  deriving (Show, Eq)

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  MLP ::
    forall inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    { mlpLayer0 :: Linear inputFeatures hiddenFeatures0 dtype device,
      mlpLayer1 :: Linear hiddenFeatures0 hiddenFeatures1 dtype device,
      mlpLayer2 :: Linear hiddenFeatures1 outputFeatures dtype device,
      mlpDropout :: Dropout
    } ->
    MLP inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype
      device
  deriving (Show, Generic)

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
  MLP inputFeatures outputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device ->
  Bool ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (Tensor device dtype '[batchSize, outputFeatures])
mlp MLP {..} train input =
  return
    . linear mlpLayer2
    =<< dropout mlpDropout train
    . tanh
    . linear mlpLayer1
    =<< dropout mlpDropout train
    . tanh
    . linear mlpLayer0
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
  A.Randomizable (MLPSpec inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
    (MLP inputFeatures outputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
  where
  sample MLPSpec {..} =
    MLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (DropoutSpec mlpDropoutProbSpec)

type BatchSize = 512

type HiddenFeatures0 = 512

type HiddenFeatures1 = 256

train' ::
  forall (device :: (D.DeviceType, Nat)).
  _ =>
  IO ()
train' = do
  let dropoutProb = 0.5
      learningRate = 0.1
  ATen.manual_seed_L 123
  initModel <-
    A.sample
      ( MLPSpec @I.DataDim @I.ClassDim
          @HiddenFeatures0
          @HiddenFeatures1
          @D.Float
          @device
          dropoutProb
      )
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  train @BatchSize @device initModel initOptim mlp learningRate "static-mnist-mlp.pt"

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu" -> train' @'( 'D.CPU, 0)
    Right "cuda:0" -> train' @'( 'D.CUDA, 0)
    _ -> error "Don't know what to do or how."

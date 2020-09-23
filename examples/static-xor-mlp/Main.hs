{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Main where

import Control.Monad
  ( foldM,
    when,
  )
import Control.Monad.Cont (ContT (runContT))
import GHC.Generics
import GHC.TypeLits
import Pipes
import Pipes (ListT (enumerate))
import qualified Pipes.Prelude as P
import Torch.Data.Pipeline
import Torch.Data.StreamedPipeline
import Torch.Typed hiding (Device)
import Prelude hiding (tanh)

--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------

data
  MLPSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = MLPSpec
  deriving (Eq, Show)

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat)) = MLP
  { layer0 :: Linear inputFeatures hiddenFeatures dtype device,
    layer1 :: Linear hiddenFeatures hiddenFeatures dtype device,
    layer2 :: Linear hiddenFeatures outputFeatures dtype device
  }
  deriving (Show, Generic, Parameterized)

instance
  (StandardFloatingPointDTypeValidation device dtype) =>
  HasForward
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
    (Tensor device dtype '[batchSize, inputFeatures])
    (Tensor device dtype '[batchSize, outputFeatures])
  where
  forward MLP {..} = forward layer2 . tanh . forward layer1 . tanh . forward layer0
  forwardStoch = (pure .) . forward

instance
  ( KnownDevice device,
    KnownDType dtype,
    All KnownNat '[inputFeatures, outputFeatures, hiddenFeatures],
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
  where
  sample MLPSpec =
    MLP <$> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec

xor ::
  forall batchSize dtype device.
  KnownDevice device =>
  Tensor device dtype '[batchSize, 2] ->
  Tensor device dtype '[batchSize]
xor t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
  where
    a = select @1 @0 t
    b = select @1 @1 t

newtype Xor device batchSize = Xor {iters :: Int}

instance
  ( KnownNat batchSize,
    KnownDevice device,
    RandDTypeIsValid device 'Float,
    ComparisonDTypeIsValid device 'Float
  ) =>
  Datastream IO () (Xor device batchSize) (Tensor device 'Float '[batchSize, 2])
  where
  streamSamples Xor {..} _ = Select $ P.replicateM iters randBool
    where
      randBool =
        toDType @'Float @'Bool
          . gt (toDevice @device (0.5 :: CPUTensor 'Float '[]))
          <$> rand @'[batchSize, 2] @'Float @device

type Device = '( 'CUDA, 0)

train ::
  forall device batchSize model optim.
  (model ~ MLP 2 1 4 'Float device, _) =>
  LearningRate device 'Float ->
  (model, optim) ->
  ListT IO (Tensor device 'Float '[batchSize, 2]) ->
  IO (model, optim)
train learningRate (model, optim) = P.foldM step begin done . enumerateData
  where
    step (model, optim) (input, i) = do
      let actualOutput = squeezeAll . ((sigmoid .) . forward) model $ input
          expectedOutput = xor input
          loss = mseLoss @ReduceMean actualOutput expectedOutput

      when (i `mod` 2500 == 0) (print loss)

      runStep model optim loss learningRate
    begin = pure (model, optim)
    done = pure

main :: IO ()
main = do
  let numIters = 100000
      learningRate = 0.1
  initModel <- sample (MLPSpec :: MLPSpec 2 1 4 'Float Device)
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
      dataset = Xor @Device @256 numIters
      dataSource = streamFrom' datastreamOpts dataset [()]
  (trained, _) <- runContT dataSource $ train learningRate (initModel, initOptim)
  print trained

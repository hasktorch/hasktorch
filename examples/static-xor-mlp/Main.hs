{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
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
import Data.List
  ( foldl',
    intersperse,
    scanl',
  )
import Data.Reflection
import GHC.Generics
import GHC.TypeLits
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import qualified Torch.NN as A
import qualified Torch.Optim as A
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Autograd
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear)
import Torch.Typed.NN
import Torch.Typed.Optim
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (tanh)

--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------

data
  MLPSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = MLPSpec

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = MLP
      { layer0 :: Linear inputFeatures hiddenFeatures dtype device,
        layer1 :: Linear hiddenFeatures hiddenFeatures dtype device,
        layer2 :: Linear hiddenFeatures outputFeatures dtype device
      }
  deriving (Show, Generic)

instance
  ( KnownDevice device,
    KnownDType dtype,
    KnownNat inputFeatures,
    KnownNat outputFeatures,
    KnownNat hiddenFeatures,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
  where
  sample MLPSpec =
    MLP <$> A.sample LinearSpec <*> A.sample LinearSpec <*> A.sample LinearSpec

mlp ::
  forall batchSize inputFeatures outputFeatures hiddenFeatures dtype device.
  (StandardFloatingPointDTypeValidation device dtype) =>
  MLP inputFeatures outputFeatures hiddenFeatures dtype device ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  Tensor device dtype '[batchSize, outputFeatures]
mlp MLP {..} = linear layer2 . tanh . linear layer1 . tanh . linear layer0

forward ::
  forall batchSize inputFeatures outputFeatures hiddenFeatures dtype device.
  (StandardFloatingPointDTypeValidation device dtype) =>
  MLP inputFeatures outputFeatures hiddenFeatures dtype device ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  Tensor device dtype '[batchSize, outputFeatures]
forward = (sigmoid .) . mlp

foldLoop ::
  forall a b m. (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

xor ::
  forall batchSize dtype device.
  Tensor device dtype '[batchSize, 2] ->
  Tensor device dtype '[batchSize]
xor t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
  where
    a = select @1 @0 t
    b = select @1 @1 t

main = do
  let numIters = 100000
      learningRate = 0.1
  initModel <- A.sample (MLPSpec :: MLPSpec 2 1 4 'D.Float '( 'D.CPU, 0))
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  (trained, _) <- foldLoop (initModel, initOptim) numIters $ \(model, optim) i -> do
    input <-
      Torch.Typed.Tensor.toDType @D.Float
        . gt (0.5 :: CPUTensor 'D.Float '[])
        <$> rand @'[256, 2] @'D.Float @'( 'D.CPU, 0)
    let expectedOutput = xor input
    let actualOutput = squeezeAll . Main.forward model $ input
    let loss = mseLoss @D.ReduceMean expectedOutput actualOutput
    when (i `mod` 2500 == 0) (print loss)
    (model', optim') <- runStep model optim loss learningRate
    return (model', optim')
  print trained

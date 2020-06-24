{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Prelude hiding ( tanh )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           GHC.Generics
import           GHC.TypeLits

import Torch.Typed hiding (Device)
import           Torch.Data.Pipeline


--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------

data MLPSpec (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
             (dtype :: DType)
             (device :: (DeviceType, Nat))
  = MLPSpec

data MLP (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
         (dtype :: DType)
         (device :: (DeviceType, Nat))
  = MLP { layer0 :: Linear inputFeatures  hiddenFeatures dtype device
        , layer1 :: Linear hiddenFeatures hiddenFeatures dtype device
        , layer2 :: Linear hiddenFeatures outputFeatures dtype device
        } deriving (Show, Generic)

instance
  (StandardFloatingPointDTypeValidation device dtype) => HasForward
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
    (Tensor device dtype '[batchSize, inputFeatures])
    (Tensor device dtype '[batchSize, outputFeatures])
 where
  forward MLP {..} = forward layer2 . tanh . forward layer1 . tanh . forward layer0

instance
  ( KnownDevice device
  , KnownDType dtype
  , All KnownNat '[inputFeatures, outputFeatures, hiddenFeatures]
  , RandDTypeIsValid device dtype
  ) => Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (MLP     inputFeatures outputFeatures hiddenFeatures dtype device)
 where
  sample MLPSpec =
    MLP <$> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec

xor
  :: forall batchSize dtype device
   . KnownDevice device
  => Tensor device dtype '[batchSize, 2]
  -> Tensor device dtype '[batchSize]
xor t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
 where
  a = select @1 @0 t
  b = select @1 @1 t

newtype Xor = Xor { iters :: Int }

instance ( KnownNat batchSize
         , KnownDevice device
         , RandDTypeIsValid device 'Float
         , ComparisonDTypeIsValid device 'Float
         ) => Dataset Xor (Tensor device 'Float '[batchSize, 2]) where
  getBatch _ _ =  toDType @Float .
                  gt (toDevice @device (0.5 :: CPUTensor 'Float '[]))
                  <$> rand @'[batchSize, 2] @'Float @device
  numIters = iters 

type Device = '( 'CUDA, 0)

trainIter :: forall device batchSize model optim . (model ~ MLP 2 1 4 'Float device, _)
  => LearningRate device 'Float
  -> (model, optim, Int)
  -> Tensor device 'Float '[batchSize, 2]
  -> IO (model, optim, Int)
trainIter learningRate (model,optim, i) input = do
    let actualOutput   = squeezeAll . ((sigmoid .) . forward) model $ input
        expectedOutput = xor input
        loss           = mseLoss @ReduceMean actualOutput expectedOutput

    when (i `mod` 2500 == 0) (print loss)

    (model', optim') <- runStep model optim loss learningRate
    return (model', optim', i+1)

main :: IO ()
main = do
  let numIters = 100000
      learningRate = 0.1
  initModel <- sample (MLPSpec :: MLPSpec 2 1 4 'Float Device)
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  trainFold <- makeFold $ Xor { iters = numIters }
  (trained, _, _) <- trainFold $ FoldM (trainIter @Device @256 learningRate ) (pure (initModel, initOptim, 0)) pure
  print trained

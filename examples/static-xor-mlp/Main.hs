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

import           Prelude                 hiding ( tanh )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           GHC.Generics
import           GHC.TypeLits

import Torch.Typed hiding (Device)


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

type Device = '( 'CUDA, 0)

main :: IO ()
main = do
  model <- sample (MLPSpec :: MLPSpec 2 1 4 'Float Device)

  let learningRate = 0.1
      optim = mkAdam 0 0.9 0.999 (flattenParameters model)

  let step (model, optim) iter = do
        input <-
          toDType @'Float @'Bool
          .   gt (0.5 :: Tensor Device 'Float '[])
          <$> rand @'[256, 2] @'Float @Device
        let actualOutput   = squeezeAll . ((sigmoid .) . forward) model $ input
            expectedOutput = xor input
            loss           = mseLoss @ReduceMean actualOutput expectedOutput
        when (iter `mod` 2500 == 0) $
          putStrLn $ "iter = " <> show iter <> " " <> "loss = " <> show loss
        runStep model optim loss learningRate
      numIters = 100000 :: Int
  (trained, _) <- foldM step (model, optim) [1 .. numIters]

  print trained

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}

module Main where

import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe
import           System.Random

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified ATen.Managed.Type.Context     as ATen
import           Torch.Typed
import           Torch.Typed.Native
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D
import qualified Image                         as I
import qualified Monitoring
import           Common

type NoStrides = '(1, 1)
type NoPadding = '(0, 0)

type KernelSize = '(2, 2)
type Strides = '(2, 2)

data CNNSpec (dtype :: D.DType)
  = CNNSpec deriving (Show, Eq)

data CNN (dtype :: D.DType)
 where
  CNN
    :: forall dtype
     . { conv0 :: Conv2d dtype 1  20 5 5
       , conv1 :: Conv2d dtype 20 50 5 5
       , fc0   :: Linear dtype (4*4*50) 500
       , fc1   :: Linear dtype 500      10
       }
    -> CNN dtype
 deriving (Show, Generic)

cnn
  :: forall dtype batchSize
   . _
  => CNN dtype
  -> Tensor dtype '[batchSize, I.DataDim]
  -> Tensor dtype '[batchSize, I.ClassDim]
cnn CNN {..} =
  Torch.Typed.NN.linear fc1
    . relu
    . Torch.Typed.NN.linear fc0
    . reshape @'[batchSize, 4*4*50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Typed.NN.conv2d @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Typed.NN.conv2d @NoStrides @NoPadding conv0
    . unsqueeze @1
    . reshape @'[batchSize, I.Rows, I.Cols]

instance A.Parameterized (CNN dtype)

instance (KnownDType dtype)
  => A.Randomizable (CNNSpec dtype)
                    (CNN     dtype)
 where
  sample CNNSpec =
    CNN
      <$> A.sample (Conv2dSpec @dtype @1  @20 @5 @5)
      <*> A.sample (Conv2dSpec @dtype @20 @50 @5 @5)
      <*> A.sample (LinearSpec @dtype @(4*4*50) @500)
      <*> A.sample (LinearSpec @dtype @500      @10)

type BatchSize = 512
type TestBatchSize = 8192

main = do
  backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
  let backend = case backend' of
        Right "CUDA" -> "CUDA"
        _            -> "CPU"
      (numIters, printEvery) = (1000000, 250)
  (trainingData, testData) <- I.initMnist
  ATen.manual_seed_L 123
  init                     <- A.sample (CNNSpec @'D.Float)
  init' <- A.replaceParameters init <$> traverse
    (A.makeIndependent . toBackend backend . A.toDependent)
    (A.flattenParameters init)
  (trained, _, _) <-
    foldLoop (init', randomIndexes (I.length trainingData), []) numIters
      $ \(state, idxs, metrics) i -> do
          let (indexes, nextIndexes) =
                (take (natValI @I.DataDim) idxs, drop (natValI @I.DataDim) idxs)
          (trainingLoss, _) <- computeLossAndErrorRate @BatchSize backend
                                                                  state
                                                                  True
                                                                  indexes
                                                                  trainingData
          let flat_parameters = A.flattenParameters state
          let gradients       = A.grad (toDynamic trainingLoss) flat_parameters

          metrics' <-
            if (i `mod` printEvery == 0) then do
              (testLoss, testError) <-
                 withTestSize (I.length testData) $ \(Proxy :: Proxy testSize) ->
                   computeLossAndErrorRate @(Min TestBatchSize testSize)
                     backend
                     state
                     False
                     (randomIndexes (I.length testData))
                     testData
              let metric = (i, Monitoring.Metric trainingLoss testLoss testError)
                  metrics' = metric:metrics
              Monitoring.printLosses metric
              Monitoring.plotLosses "loss.html" metrics'
              return metrics'
            else
              return metrics

          new_flat_parameters <- mapM A.makeIndependent
            $ A.sgd 1e-01 flat_parameters gradients
          return (A.replaceParameters state new_flat_parameters,
                  nextIndexes,
                  metrics')
  print trained
 where
  computeLossAndErrorRate
    :: forall n
     . (KnownNat n)
    => String
    -> CNN 'D.Float
    -> Bool
    -> [Int]
    -> I.MnistData
    -> IO (Tensor 'D.Float '[], Tensor 'D.Float '[])
  computeLossAndErrorRate backend state train indexes data' = do
    let input  = toBackend backend $ I.getImages @n data' indexes
        target = toBackend backend $ I.getLabels @n data' indexes
        result = cnn state input
    return (crossEntropyLoss backend result target, errorRate result target)

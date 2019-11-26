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
{-# LANGUAGE UndecidableInstances #-}

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
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Native      hiding ( linear
                                                , conv2d
                                                )
import           Torch.Typed.Factories
import           Torch.Typed.NN
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.Device                  as D
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

data CNNSpec (dtype :: D.DType) (device :: (D.DeviceType, Nat))
  = CNNSpec deriving (Show, Eq)

data CNN (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  CNN
    :: forall dtype device
     . { conv0 :: Conv2d 1  20 5 5 dtype device
       , conv1 :: Conv2d 20 50 5 5 dtype device
       , fc0   :: Linear (4*4*50) 500 dtype device
       , fc1   :: Linear 500      10  dtype device
       }
    -> CNN dtype device
 deriving (Show, Generic)

cnn
  :: forall batchSize dtype device
   . _
  => CNN dtype device
  -> Tensor device dtype '[batchSize, I.DataDim]
  -> Tensor device dtype '[batchSize, I.ClassDim]
cnn CNN {..} =
  linear fc1
    . relu
    . linear fc0
    . reshape @'[batchSize, 4*4*50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2d @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . conv2d @NoStrides @NoPadding conv0
    . unsqueeze @1
    . reshape @'[batchSize, I.Rows, I.Cols]

instance A.Parameterized (CNN dtype device)

instance ( KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (CNNSpec dtype device)
                    (CNN     dtype device)
 where
  sample CNNSpec =
    CNN
      <$> A.sample (Conv2dSpec @1  @20 @5 @5)
      <*> A.sample (Conv2dSpec @20 @50 @5 @5)
      <*> A.sample (LinearSpec @(4*4*50) @500)
      <*> A.sample (LinearSpec @500      @10)

type BatchSize = 512
type TestBatchSize = 8192

train :: forall (device :: (D.DeviceType, Nat)) . _ => IO ()
train = do
  let (numIters, printEvery) = (1000000, 250)
  (trainingData, testData) <- I.initMnist
  ATen.manual_seed_L 123
  init            <- A.sample (CNNSpec @ 'D.Float @device)
  (trained, _, _) <-
    foldLoop (init, randomIndexes (I.length trainingData), []) numIters
      $ \(state, idxs, metrics) i -> do
          let (indexes, nextIndexes) =
                (take (natValI @I.DataDim) idxs, drop (natValI @I.DataDim) idxs)
          (trainingLoss, _) <- computeLossAndErrorRate @BatchSize state
                                                                  True
                                                                  indexes
                                                                  trainingData
          let flat_parameters = A.flattenParameters state
          let gradients       = A.grad (toDynamic trainingLoss) flat_parameters

          metrics' <- if (i `mod` printEvery == 0)
            then do
              (testLoss, testError) <-
                withTestSize (I.length testData)
                  $ \(Proxy :: Proxy testSize) ->
                      computeLossAndErrorRate @(Min TestBatchSize testSize)
                        state
                        False
                        (randomIndexes (I.length testData))
                        testData
              let metric =
                    (i, Monitoring.Metric (Monitoring.asFLoat trainingLoss) (Monitoring.asFLoat testLoss) (Monitoring.asFLoat testError))
                  metrics' = metric : metrics
              Monitoring.printLosses metric
              Monitoring.plotLosses "loss.html" metrics'
              return metrics'
            else return metrics

          new_flat_parameters <- mapM A.makeIndependent
            $ A.sgd 1e-01 flat_parameters gradients
          return
            ( A.replaceParameters state new_flat_parameters
            , nextIndexes
            , metrics'
            )
  print trained
 where
  computeLossAndErrorRate
    :: forall n
     . (KnownNat n)
    => CNN 'D.Float device
    -> Bool
    -> [Int]
    -> I.MnistData
    -> IO
         ( Tensor device 'D.Float '[]
         , Tensor device 'D.Float '[]
         )
  computeLossAndErrorRate state train indexes data' = do
    let input      = toDevice @device $ I.getImages @n data' indexes
        target     = toDevice @device $ I.getLabels @n data' indexes
        prediction = cnn state input
    return (crossEntropyLoss prediction target, errorRate prediction target)

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu"    -> train @'( 'D.CPU, 0)
    Right "cuda:0" -> train @'( 'D.CUDA, 0)
    _              -> error "Don't know what to do or how."

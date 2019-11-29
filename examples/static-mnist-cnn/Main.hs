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
import qualified Torch.Serialize               as D
import qualified Image                         as I
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
  let numEpochs = 1000
  (trainingData, testData) <- I.initMnist
  ATen.manual_seed_L 123
  init            <- A.sample (CNNSpec @ 'D.Float @device)
  foldLoop_ init numEpochs $ \state' epoch -> do
    let numIters = I.length trainingData `div` natValI @BatchSize
    nextState <- foldLoop state' numIters $ \state i -> do
      let from = (i-1) * natValI @BatchSize
          to = (i * natValI @BatchSize) - 1
          indexes = [from .. to]
      (trainingLoss,_) <- computeLossAndErrorCount @BatchSize state
                                                              indexes
                                                              trainingData

      let flat_parameters = A.flattenParameters state
          gradients       = A.grad (toDynamic trainingLoss) flat_parameters
      new_flat_parameters <- mapM A.makeIndependent
        $ A.sgd 1e-01 flat_parameters gradients
      return
        $ A.replaceParameters state new_flat_parameters

    (testLoss, testError) <- do
      let numIters = I.length testData `div` natValI @BatchSize
      foldLoop (0,0) numIters $ \(org_loss,org_err) i -> do
        let from = (i-1) * natValI @BatchSize
            to = (i * natValI @BatchSize) - 1
            indexes = [from .. to]
        (loss,err) <- computeLossAndErrorCount @BatchSize nextState
                                                          indexes
                                                          testData
        return (org_loss + toFloat loss,org_err + toFloat err)
    putStrLn
      $  "Epoch: "
      <> show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (I.length testData))
      <> ". Test error-rate: "
      <> show (testError / realToFrac (I.length testData))
    
    D.save (map A.toDependent $ A.flattenParameters nextState) "static-mnist-cnn.pt"
    return nextState
 where
  computeLossAndErrorCount
    :: forall n
     . (KnownNat n)
    => CNN 'D.Float device
    -> [Int]
    -> I.MnistData
    -> IO ( Tensor device 'D.Float '[], Tensor device 'D.Float '[] )
  computeLossAndErrorCount state indexes data' = do
    let input      = toDevice @device $ I.getImages @n data' indexes
        target     = toDevice @device $ I.getLabels @n data' indexes
        prediction = cnn state input
    return (crossEntropyLoss prediction target,errorCount prediction target)

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu"    -> train @'( 'D.CPU, 0)
    Right "cuda:0" -> train @'( 'D.CUDA, 0)
    _              -> error "Don't know what to do or how."

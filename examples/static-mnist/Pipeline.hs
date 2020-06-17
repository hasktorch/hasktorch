{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
module Pipeline where

import Torch.Data.Pipeline
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional               as D
import qualified Torch.TensorFactories         as D
import Torch.Typed.Tensor
import qualified Torch.Typed.Vision as I
import Torch.Typed.Functional
import Torch.Typed.Aux
import Torch.Typed.NN
import Torch.Typed.Optim
import           Torch.Typed.Factories

  -- new
import Torch.HList
import Torch.Typed.Parameter hiding (toDevice)
import Torch.Typed.Autograd
import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
  -- new


import Control.Monad.State.Lazy
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import qualified Control.Concurrent.Async.Lifted as U -- can get rid of this dependency if we don't use StateT

import Common

-- type BatchedTensor device dtype batchSize features = Tensor device dtype (batchSize ': features ': '[])

getBatchMnist :: forall batchSize model optim . _ => I.MnistData -> Int -> Int ->  IO _
getBatchMnist dataset numIters iter =  
  let from = (iter-1) * natValI @batchSize
      to = (iter * natValI @batchSize) - 1
      indexes = [from .. to]
      input  = I.getImages @batchSize dataset indexes
      target = I.getLabels @batchSize dataset indexes
  in
    if iter == numIters
    then pure ((input, target), Final)
    else pure ((input, target), KeepTrain)

runBatches 
  :: forall batchSize device model optim shape1 shape2 . 
     _ 
  => model 
  -> optim
  -> (Tensor device 'D.Float '[batchSize, 10] -> Tensor device 'D.Int64 shape1 -> Loss device 'D.Float)
  -> LearningRate device 'D.Float
  -> (model -> Tensor device 'D.Float shape2 -> Tensor device 'D.Float '[batchSize, 10])
  -> _ 
  -> _
  -> Int
  -> Tensor device 'D.Float '[]
  -> IO ()
runBatches model optim lossFn learningRate forward trainInputs evalInputs numEpochs testSetSize = do
  void $ foldLoop (model,optim) numEpochs  $ \(model,optim) epoch -> do
    liftIO $ print $ "Running epoch " <> show epoch 
    (newModel, newOptim) <- trainInputs (trainFold lossFn learningRate forward) (model, optim)
    (testLoss, testError) <- evalInputs (evaluation model forward) (0,0)
    liftIO $ putStrLn
      $  "Epoch: " <> show epoch
      <> ". Test loss: "
      <> show (testLoss / testSetSize)  
      <> ". Test error-rate: "
      <> show (testError / testSetSize)
    pure (newModel, newOptim)

trainFold :: forall device . _ => _ -> _ -> _ -> _ -> _
trainFold lossFn learningRate forward (model, optim) (trainData, trainLabels) = do 
  let prediction = forward model $ toDevice @device trainData
  let target = toDevice @device trainLabels
  let loss = lossFn prediction target
  liftIO $ runStep  model optim loss learningRate


evaluation :: forall device shape batchSize model optim . _
  => _
  -> _
  -> (Tensor device 'D.Float '[], Tensor device 'D.Float '[])
  -> (CPUTensor 'D.Float shape, CPUTensor 'D.Int64 '[batchSize])
  -> IO (Tensor device 'D.Float '[], Tensor device 'D.Float '[])
evaluation model forward (totalLoss, totalError) (testData, testLabels) = do
            let prediction = forward model $ toDevice @device testData
            let target = toDevice @device testLabels
            let (testLoss, testError) = lossAndErrorCount prediction target
            pure (testLoss + totalLoss, testError + totalError) 

newMain :: forall batchSize device . _ => _ -> _ -> _
newMain forward model optim numEpochs = do
  (rawTrainingData, rawTestData) <- liftIO $ I.initMnist "data"
  let numTrain = I.length rawTrainingData `div` natValI @batchSize
      numTest = I.length rawTestData `div` natValI @batchSize
      trainingData = DatasetMock { getBatchMock = getBatchMnist @batchSize rawTrainingData numTrain
                                   , numIters = numTrain
                                   }
      testData = DatasetMock { getBatchMock = getBatchMnist  @batchSize rawTestData numTest
                             , numIters = numTest
                             }
  trainingInputs <- makeFoldWithTransform id trainingData numEpochs
  evalInputs <- makeFoldWithTransform id testData numEpochs

  runBatches @batchSize @device model optim
    crossEntropyLoss 1e-3 forward trainingInputs evalInputs numEpochs (full $ I.length rawTestData)


lossAndErrorCount
  :: forall batchSize device shape .
     (SumDTypeIsValid device 'D.Bool,
      ComparisonDTypeIsValid device 'D.Int64,
      KnownNat batchSize,
      KnownNat shape,
      KnownDevice device,
      StandardFloatingPointDTypeValidation device 'D.Float
     )
  => Tensor device 'D.Float '[batchSize, shape]
  -> Tensor device 'D.Int64 '[batchSize]
  -> ( Tensor device 'D.Float '[]
     , Tensor device 'D.Float '[]
     )
lossAndErrorCount input target = 
    (crossEntropyLoss input target, errorCount input target)


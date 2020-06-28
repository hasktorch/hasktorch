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
import qualified Control.Concurrent.Async.Lifted as U

import Pipes
import qualified Pipes.Prelude as P
import Pipes.Concurrent
import Common



  --TODO: make dtype of tensors polymorphic

type Loader model optim = StateT (model, optim) IO 

data Mailboxes tensor tensor' = Mailboxes
  {
    transformBox :: (Output tensor, Input tensor),
    evalTransformBox :: (Output tensor, Input tensor),
    tensors :: (Output tensor', Input tensor'),
    evalBox :: (Output tensor', Input tensor')
  }
-- pipeline =
--   load stuff on iteration x
--   >-> perform transform >-> perform iteration on batch >-> evaluate loss?
--   >-> feed back into model >-> 


makeMailboxes :: IO (Mailboxes tensor tensor')
makeMailboxes = do
   tensors <- spawn (bounded 5)
   transformBox <- spawn (bounded 5)
   evalTransformBox <- spawn (bounded 5)
   evalBox <- spawn (bounded 5)
   pure Mailboxes{..}

type BatchSize = 500
type Device = '(D.CPU, 0)

readBatch
  :: forall batchSize model optim .
     KnownNat batchSize =>
     Output
       (CPUTensor 'D.Float '[batchSize, 784],
        CPUTensor 'D.Int64 '[batchSize], Int)
     -> Effect (Loader model optim) ()
readBatch outputBox = do
  (trainingData, testData) <- liftIO $ I.initMnist "data"

  let numIters = 100
  for (each [1..numIters]) (getBatch trainingData) 
    >-> toOutput outputBox

  where
    -- implement inside a dataset typeclass?
    getBatch ::  I.MnistData -> Int -> Producer _ (Loader model optim) ()
    getBatch trainingData iter = do
      let from = (iter-1) * natValI @batchSize
          to = (iter * natValI @batchSize) - 1
          indexes = [from .. to]
          input  = I.getImages @batchSize trainingData indexes
          target = I.getLabels @batchSize trainingData indexes
      yield (input, target, iter)
  
  -- TODO : try to use model forward function
runBatch
  :: forall batchSize device gradients parameters dtype1 dtype2 dtype3 dtype4 device1 device2 device3 device4 device5 model optim shape1 shape2 c1 c2. 
     (
       KnownNat batchSize,
       DTypeIsFloatingPoint device 'D.Float,
       DTypeIsNotHalf device 'D.Float,
       SumDTypeIsValid device 'D.Bool,
       ComparisonDTypeIsValid device 'D.Int64,
       HMapM' IO MakeIndependent gradients parameters,
       HUnfoldM
       IO
       TensorListUnfold
       (HUnfoldMRes IO [D.ATenTensor] gradients)
       gradients,
       Apply
        TensorListUnfold
        [D.ATenTensor]
        (HUnfoldMRes IO [D.ATenTensor] gradients),
      HFoldrM IO TensorListFold [D.ATenTensor] gradients [D.ATenTensor],
      Parameterized model parameters,
      HasGrad (HList parameters) (HList gradients),
      HMap' ToDependent parameters gradients,
      Optimizer optim gradients gradients 'D.Float device,
      StandardFloatingPointDTypeValidation device 'D.Float,
      KnownDevice device
      ) =>
     (Tensor device 'D.Float '[batchSize, 10]
     -> Tensor device 'D.Int64 shape1 -> Loss device 'D.Float)
     -> LearningRate device 'D.Float
     -> (model
         -> Tensor device 'D.Float shape2
         -> Tensor device 'D.Float '[batchSize, 10])
     -> Input
          (CPUTensor 'D.Float shape2, CPUTensor 'D.Int64 shape1, c1)
     -> Input
          (CPUTensor 'D.Float shape2,
           CPUTensor 'D.Int64 '[batchSize], c2)
     -> Effect (Loader model optim) ()
runBatch lossFn learningRate forward tensorBox evalBox =
  for (each [1..10]) $ \epoch -> do
    liftIO $ print $ "running epoch"  <> show epoch 
    -- since we are currently using StateT we have no return value in the fold
    -- but we can just get rid of StateT
    lift $ P.foldM' trainFold (pure ()) pure inputs
    (testLoss, testError) <- lift $ evalModel forward evalBox
    liftIO $ putStrLn
      $  "Epoch: "
      <>  show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (100 * natValI @batchSize))
      <> ". Test error-rate: "
      <> show (testError / realToFrac (100 * natValI @batchSize))
    liftIO $ performGC

  where
    inputs = fromInput tensorBox >-> (P.take 100)
    trainFold accum (trainData, trainLabels, ix) = do 
      (model, optim) <- get 
      let prediction = forward model $ toDevice @device trainData
      let target = toDevice @device trainLabels
      let loss = lossFn prediction target
      liftIO $ performGC
      newState <- liftIO $ runStep  model optim loss learningRate
      put newState

evalModel
  :: forall batchSize device shape model optim c.
     (KnownNat batchSize,
      SumDTypeIsValid device 'D.Bool,
      ComparisonDTypeIsValid device 'D.Int64,
      DTypeIsFloatingPoint device 'D.Float,
      DTypeIsNotHalf device 'D.Float,
      StandardFloatingPointDTypeValidation device 'D.Float,
      KnownDType 'D.Float,
      KnownDevice device
      ) =>
     (model
      -> Tensor device 'D.Float shape
      -> Tensor device 'D.Float '[batchSize, 10])
     -> Input
          (CPUTensor 'D.Float shape, CPUTensor 'D.Int64 '[batchSize],
           c)
     -> Loader model optim (Tensor device 'D.Float '[], Tensor device 'D.Float '[])
evalModel forward evalBox =  P.foldM evalError (pure (0,0)) pure inputs 

  where
    inputs = fromInput evalBox >-> P.take 100
    evalError (totalLoss, totalError) (testData, testLabels, ix) =
          do
            (model, optim) <- get
            let prediction = forward model $ toDevice @device testData
            let target = toDevice @device testLabels
            let (testLoss, testError) =  lossAndErrorCount  prediction target
            liftIO $ performGC
            pure (testLoss + totalLoss, testError + totalError)

runTransforms transforms transformBox batchBox = fromInput transformBox >-> P.map transforms >-> toOutput batchBox

  -- FIXME IMPORTANT: we might only want each effect to deal with one input and output! Then we can use the withBuffer function
trainingLoop
  :: forall dtype device batchSize gradients parameters model  optim.
     (DTypeIsFloatingPoint device dtype,
      DTypeIsNotHalf device dtype,
      SumDTypeIsValid device 'D.Bool,
      ComparisonDTypeIsValid device 'D.Int64,
      HMapM' IO MakeIndependent gradients parameters,
      HUnfoldM
        IO
        TensorListUnfold
        (HUnfoldMRes IO [D.ATenTensor] gradients)
        gradients,
      Apply
        TensorListUnfold
        [D.ATenTensor]
        (HUnfoldMRes IO [D.ATenTensor] gradients),
      HFoldrM IO TensorListFold [D.ATenTensor] gradients [D.ATenTensor],
      Parameterized model parameters,
      HasGrad (HList parameters) (HList gradients),
      HMap' ToDependent parameters gradients,
      Optimizer optim gradients gradients dtype device,
      StandardFloatingPointDTypeValidation device 'D.Float,
      KnownDType dtype,
      KnownNat batchSize,
      KnownDevice device,
      dtype ~ 'D.Float
     ) =>
     (model
      -> Tensor device 'D.Float '[batchSize, 784]
      -> Tensor device 'D.Float '[batchSize, 10])
     -> Loader model optim ()
trainingLoop forward =  do
  Mailboxes{..} <- liftIO $ makeMailboxes
  -- (trainingData, testData) <- liftIO $ I.initMnist "data"
  batchReader <-  U.async $  do runEffect $ forever $ readBatch @batchSize (fst transformBox) 
                                liftIO $ print "clean batch"
                                liftIO $ performGC
  transformer <-  U.async $ do runEffect $ runTransforms id (snd transformBox) (fst tensors)
                               liftIO $ print "clean transf"
                               liftIO $ performGC
  evalReader <-  U.async $ do runEffect $ forever $ readBatch @batchSize (fst evalTransformBox) 
                              liftIO $ print "clean eval"
                              liftIO $ performGC
  evalTransformer <-  U.async $ do runEffect $ runTransforms id (snd evalTransformBox) (fst evalBox)
                                   liftIO $ print "clean evalTransf"
                                   liftIO $ performGC

  batchRunner <- U.async $ do runEffect $ runBatch crossEntropyLoss 1e-3 forward (snd tensors) (snd evalBox)
                              liftIO $ performGC

  mapM U.wait [batchReader, evalReader, transformer,  evalTransformer, batchRunner]
  pure ()
  

lossAndErrorCount
  :: (SumDTypeIsValid device 'D.Bool,
      ComparisonDTypeIsValid device 'D.Int64,
      KnownNat n,
      KnownNat shape,
      KnownDevice device,
      StandardFloatingPointDTypeValidation device 'D.Float
     )
  =>
  Tensor device 'D.Float '[n, shape]
  -> Tensor device 'D.Int64 '[n]
  -> 
  ( Tensor device 'D.Float '[]
  , Tensor device 'D.Float '[]
  )
lossAndErrorCount input target = 
    (crossEntropyLoss input target, errorCount input target)


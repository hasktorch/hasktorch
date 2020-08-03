{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Common where

import Prelude hiding (length)
import           Control.Monad                  ( foldM
                                                , void
                                                )
import           GHC.TypeLits

import Torch.Typed
import Torch (ATenTensor)
import Torch.Internal.Class (Castable)

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

foldLoop_
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m ()
foldLoop_ = ((void .) .) . foldLoop

crossEntropyLoss
  :: forall batchSize seqLen dtype device
   . ( KnownNat batchSize
     , KnownNat seqLen
     , KnownDType dtype
     , KnownDevice device
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype '[batchSize, seqLen]
  -> Tensor device 'Int64 '[batchSize]
  -> Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @ReduceMean @batchSize @seqLen @'[]
    ones
    (-100)
    (logSoftmax @1 prediction)
    target

errorCount
  :: forall batchSize outputFeatures device
   . ( KnownNat batchSize
     , KnownNat outputFeatures
     , SumDTypeIsValid device 'Bool
     , ComparisonDTypeIsValid device 'Int64
     , StandardDTypeValidation device 'Float
     )
  => Tensor device 'Float '[batchSize, outputFeatures] -- ^ prediction
  -> Tensor device 'Int64 '[batchSize] -- ^ target
  -> Tensor device 'Float '[]
errorCount prediction = toDType @'Float @'Int64 . sumAll . ne (argmax @1 @DropDim prediction)

train
  :: forall (batchSize :: Nat) (device :: (DeviceType, Nat)) model optim gradients parameters tensors
   . ( KnownNat batchSize
     , StandardFloatingPointDTypeValidation device 'Float
     , SumDTypeIsValid device 'Bool
     , ComparisonDTypeIsValid device 'Int64
     , StandardDTypeValidation device 'Float
     , KnownDevice device
     , HasGrad (HList parameters) (HList gradients)
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , Castable (HList gradients) [ATenTensor]
     , Parameterized model parameters
     , Optimizer optim gradients tensors 'Float device
     , HMapM' IO MakeIndependent tensors parameters
     )
  => model
  -> optim
  -> (model -> Bool -> Tensor device 'Float '[batchSize, DataDim] -> IO (Tensor device 'Float '[batchSize, ClassDim]))
  -> LearningRate device 'Float
  -> String
  -> IO ()
train initModel initOptim forward learningRate ptFile = do
  let numEpochs = 1000
  (trainingData, testData) <- initMnist "data"
  foldLoop_ (initModel, initOptim) numEpochs $ \(epochModel, epochOptim) epoch -> do
    let numIters = length trainingData `div` natValI @batchSize
    (epochModel', epochOptim') <- foldLoop (epochModel, epochOptim) numIters $ \(model, optim) i -> do
      (trainingLoss,_) <- computeLossAndErrorCount @batchSize (forward model True) 
                                                              i
                                                              trainingData
      (model', optim') <- runStep model optim trainingLoss learningRate
      return (model', optim')

    (testLoss, testError) <- do
      let numIters = length testData `div` natValI @batchSize
      foldLoop (0,0) numIters $ \(org_loss,org_err) i -> do
        (loss,err) <- computeLossAndErrorCount @batchSize (forward epochModel' False)
                                                          i
                                                          testData
        return (org_loss + toFloat loss,org_err + toFloat err)
    putStrLn
      $  "Epoch: "
      <> show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (length testData))
      <> ". Test error-rate: "
      <> show (testError / realToFrac (length testData))
    
    save (hmap' ToDependent . flattenParameters $ epochModel') ptFile
    return (epochModel', epochOptim')
    
 where
  computeLossAndErrorCount
    :: forall n (device :: (DeviceType, Nat))
    . _
    => (Tensor device 'Float '[n, DataDim] -> IO (Tensor device 'Float '[n, ClassDim]))
    -> Int
    -> MnistData
    -> IO
         ( Tensor device 'Float '[]
         , Tensor device 'Float '[]
         )
  computeLossAndErrorCount forward' index_of_batch data' = do
    let from = (index_of_batch-1) * natValI @n
        to = (index_of_batch * natValI @n) - 1
        indexes = [from .. to]
        input  = toDevice @device $ getImages @n data' indexes
        target = toDevice @device $ getLabels @n data' indexes
    prediction <- forward' input
    return (crossEntropyLoss prediction target, errorCount prediction target)

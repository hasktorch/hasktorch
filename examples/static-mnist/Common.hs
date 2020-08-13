{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Common where

import           Control.Monad                  ( foldM
                                                , void
                                                )
import           GHC.TypeLits
import           Prelude hiding (length)
import qualified Pipes.Prelude as P
import           Pipes (ListT(enumerate))
import           Control.Monad.Cont (ContT(runContT))

import           Torch.Typed
import           Torch (ATenTensor)
import           Torch.Internal.Class (Castable)
import           Torch.Data.Pipeline
import           Torch.Typed.Vision (mnistData, Mnist(Mnist))

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
     , _
     )
  => model
  -> optim
  -> (model -> Bool -> Tensor device 'Float '[batchSize, DataDim] -> IO (Tensor device 'Float '[batchSize, ClassDim]))
  -> LearningRate device 'Float
  -> String
  -> IO ()
train initModel initOptim forward learningRate ptFile = do
  let numEpochs = 1000
  (trainingData, testData) <- mkMnist @device @batchSize
  foldLoop_ (initModel, initOptim) numEpochs $ \(epochModel, epochOptim) epoch -> do
    (epochModel', epochOptim') <- runContT (makeListT (mapStyleOpts 1) trainingData ) $
                                  trainStep learningRate forward  (epochModel, epochOptim)
    (testLoss, testError) <- runContT (makeListT (mapStyleOpts 1) testData ) $
                                  evalStep (forward epochModel' False)
    putStrLn
      $  "Epoch: "
      <> show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (length $ mnistData testData))
      <> ". Test error-rate: "
      <> show (testError / realToFrac (length $ mnistData testData))
    
    save (hmap' ToDependent . flattenParameters $ epochModel') ptFile
    return (epochModel', epochOptim')
    
  where     

    trainStep lr forward' init  = P.foldM step begin done . enumerate
      where step (model, optim) ((input, target), iter) = do 
              prediction <- forward' model True input
              let trainingLoss =  crossEntropyLoss prediction target
              runStep model optim trainingLoss lr
              -- pure (model, optim)
            begin = pure init
            done = pure
        
    evalStep forward' = P.foldM step begin done . enumerate
      where step (org_loss, org_err) ((input, target), _) = do 
              prediction <- forward' input
              let (loss, err) = (crossEntropyLoss prediction target, errorCount prediction target)
              return (org_loss + toFloat loss, org_err + toFloat err)
            begin = pure (0,0)
            done = pure 

mkMnist :: IO (Mnist device batchSize, Mnist device batchSize)
mkMnist = do
  (train, test) <- initMnist "data"
  return (Mnist train, Mnist test)

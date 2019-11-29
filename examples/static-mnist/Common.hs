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

module Common where

import           Control.Monad                  ( foldM
                                                , when
                                                , void
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
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Native
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
  -> Tensor device 'D.Int64 '[batchSize]
  -> Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @seqLen @'[]
    ones
    (-100)
    (logSoftmax @1 prediction)
    target

errorCount
  :: forall batchSize outputFeatures device
   . ( KnownNat batchSize
     , KnownNat outputFeatures
     , SumDTypeIsValid device 'D.Bool
     , ComparisonDTypeIsValid device 'D.Int64
     )
  => Tensor device 'D.Float '[batchSize, outputFeatures]
  -> Tensor device 'D.Int64 '[batchSize]
  -> Tensor device 'D.Float '[]
errorCount prediction target =
  toDType @D.Float . sumAll . ne (argmax @1 @DropDim prediction) $ target

train
  :: forall (batchSize :: Nat) (device :: (D.DeviceType, Nat)) state
   . _
  => state
  -> (state -> Bool -> Tensor device 'D.Float '[batchSize, I.DataDim] -> IO (Tensor device 'D.Float '[batchSize, I.ClassDim]))
  -> String
  -> IO ()
train init predictFunction ptFile = do
  let numEpochs = 1000
  (trainingData, testData) <- I.initMnist
  foldLoop_ init numEpochs $ \state' epoch -> do
    let numIters = I.length trainingData `div` natValI @batchSize
    nextState <- foldLoop state' numIters $ \state i -> do
      (trainingLoss,_) <- computeLossAndErrorCount @batchSize (predictFunction state True) 
                                                              i
                                                              trainingData

      let flat_parameters = A.flattenParameters state
          gradients       = A.grad (toDynamic trainingLoss) flat_parameters
      new_flat_parameters <- mapM A.makeIndependent
        $ A.sgd 1e-01 flat_parameters gradients
      return
        $ A.replaceParameters state new_flat_parameters

    (testLoss, testError) <- do
      let numIters = I.length testData `div` natValI @batchSize
      foldLoop (0,0) numIters $ \(org_loss,org_err) i -> do
        (loss,err) <- computeLossAndErrorCount @batchSize (predictFunction nextState False)
                                                          i
                                                          testData
        return (org_loss + toFloat loss,org_err + toFloat err)
    putStrLn
      $  "Epoch: "
      <> show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (I.length testData))
      <> ". Test error-rate: "
      <> show (testError / realToFrac (I.length testData))
    
    D.save (map A.toDependent $ A.flattenParameters nextState) ptFile
    return nextState
    
 where
  computeLossAndErrorCount
    :: forall n (device :: (D.DeviceType, Nat))
    . _
    => (Tensor device 'D.Float '[n, I.DataDim] -> IO (Tensor device 'D.Float '[n, I.ClassDim]))
    -> Int
    -> I.MnistData
    -> IO
         ( Tensor device 'D.Float '[]
         , Tensor device 'D.Float '[]
         )
  computeLossAndErrorCount mlp index_of_batch data' = do
    let from = (index_of_batch-1) * natValI @n
        to = (index_of_batch * natValI @n) - 1
        indexes = [from .. to]
        input  = toDevice @device $ I.getImages @n data' indexes
        target = toDevice @device $ I.getLabels @n data' indexes
    prediction <- mlp input
    return (crossEntropyLoss prediction target, errorCount prediction target)

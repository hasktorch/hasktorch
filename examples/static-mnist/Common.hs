{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Common where

import Control.Monad
  ( foldM,
    void,
    when,
  )
import Data.Proxy
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import System.Environment
import System.IO.Unsafe
import System.Random
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.NN as A
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Autograd
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.NN
import Torch.Typed.Optim
import Torch.Typed.Parameter
import Torch.Typed.Serialize
import Torch.Typed.Tensor
import qualified Torch.Typed.Vision as I

foldLoop ::
  forall a b m. (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

foldLoop_ ::
  forall a b m. (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m ()
foldLoop_ = ((void .) .) . foldLoop

crossEntropyLoss ::
  forall batchSize seqLen dtype device.
  ( KnownNat batchSize,
    KnownNat seqLen,
    KnownDType dtype,
    KnownDevice device,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Tensor device dtype '[batchSize, seqLen] ->
  Tensor device 'D.Int64 '[batchSize] ->
  Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @seqLen @'[]
    ones
    (-100)
    (logSoftmax @1 prediction)
    target

errorCount ::
  forall batchSize outputFeatures device.
  ( KnownNat batchSize,
    KnownNat outputFeatures,
    SumDTypeIsValid device 'D.Bool,
    ComparisonDTypeIsValid device 'D.Int64
  ) =>
  -- | prediction
  Tensor device 'D.Float '[batchSize, outputFeatures] ->
  -- | target
  Tensor device 'D.Int64 '[batchSize] ->
  Tensor device 'D.Float '[]
errorCount prediction = Torch.Typed.Tensor.toDType @D.Float . sumAll . ne (argmax @1 @DropDim prediction)

train ::
  forall (batchSize :: Nat) (device :: (D.DeviceType, Nat)) model optim gradients parameters tensors.
  ( KnownNat batchSize,
    StandardFloatingPointDTypeValidation device 'D.Float,
    SumDTypeIsValid device 'D.Bool,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDevice device,
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    ATen.Castable (HList gradients) [D.ATenTensor],
    Parameterized model parameters,
    Optimizer optim gradients tensors 'D.Float device,
    HMapM' IO MakeIndependent tensors parameters
  ) =>
  model ->
  optim ->
  (model -> Bool -> Tensor device 'D.Float '[batchSize, I.DataDim] -> IO (Tensor device 'D.Float '[batchSize, I.ClassDim])) ->
  LearningRate device 'D.Float ->
  String ->
  IO ()
train initModel initOptim forward learningRate ptFile = do
  let numEpochs = 1000
  (trainingData, testData) <- I.initMnist "data"
  foldLoop_ (initModel, initOptim) numEpochs $ \(epochModel, epochOptim) epoch -> do
    let numIters = I.length trainingData `div` natValI @batchSize
    (epochModel', epochOptim') <- foldLoop (epochModel, epochOptim) numIters $ \(model, optim) i -> do
      (trainingLoss, _) <-
        computeLossAndErrorCount @batchSize
          (forward model True)
          i
          trainingData
      (model', optim') <- runStep model optim trainingLoss learningRate
      return (model', optim')
    (testLoss, testError) <- do
      let numIters = I.length testData `div` natValI @batchSize
      foldLoop (0, 0) numIters $ \(org_loss, org_err) i -> do
        (loss, err) <-
          computeLossAndErrorCount @batchSize
            (forward epochModel' False)
            i
            testData
        return (org_loss + toFloat loss, org_err + toFloat err)
    putStrLn $
      "Epoch: "
        <> show epoch
        <> ". Test loss: "
        <> show (testLoss / realToFrac (I.length testData))
        <> ". Test error-rate: "
        <> show (testError / realToFrac (I.length testData))
    save (hmap' ToDependent . flattenParameters $ epochModel') ptFile
    return (epochModel', epochOptim')
  where
    computeLossAndErrorCount ::
      forall n (device :: (D.DeviceType, Nat)).
      _ =>
      (Tensor device 'D.Float '[n, I.DataDim] -> IO (Tensor device 'D.Float '[n, I.ClassDim])) ->
      Int ->
      I.MnistData ->
      IO
        ( Tensor device 'D.Float '[],
          Tensor device 'D.Float '[]
        )
    computeLossAndErrorCount forward' index_of_batch data' = do
      let from = (index_of_batch -1) * natValI @n
          to = (index_of_batch * natValI @n) - 1
          indexes = [from .. to]
          input = Torch.Typed.Tensor.toDevice @device $ I.getImages @n data' indexes
          target = Torch.Typed.Tensor.toDevice @device $ I.getLabels @n data' indexes
      prediction <- forward' input
      return (crossEntropyLoss prediction target, errorCount prediction target)

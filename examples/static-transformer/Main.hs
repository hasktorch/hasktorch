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
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}
{-# OPTIONS_GHC -fdefer-typed-holes #-}
{-# OPTIONS_GHC -Wno-typed-holes #-}

module Main where

import           Prelude
import           Control.Concurrent.Async
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Control.Monad                  ( foldM
                                                , when
                                                , void
                                                )
import           Torch.HList
import           Data.Proxy
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import qualified Torch.Internal.Managed.Type.Context     as ATen
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Device
import           Torch.Typed.Functional
import           Torch.Typed.Factories
import           Torch.Typed.NN
import           Torch.Typed.NN.Transformer
import           Torch.Typed.Autograd
import           Torch.Typed.Optim
import           Torch.Typed.Serialize
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.Optim                   as A
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional              as D
import qualified Torch.TensorFactories         as D

crossEntropyLoss
  :: forall paddingIdx batchSize seqLen dtype device
   . ( KnownNat paddingIdx
     , KnownNat batchSize
     , KnownNat seqLen
     , KnownDType dtype
     , KnownDevice device
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype '[batchSize, seqLen, seqLen]
  -> Tensor device 'D.Int64 '[batchSize, seqLen]
  -> Tensor device dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @seqLen @'[seqLen]
    ones
    (natValI @paddingIdx)
    (logSoftmax @1 prediction)
    target

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

foldLoop_
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m ()
foldLoop_ = ((void .) .) . foldLoop

type NumAttnLayers = 1
type NumHeads = 1
type FFNDim = 1
type PaddingIdx = 0
type NumEmbeds = 16
type EmbedDim = 8
type SeqLen = 1

type Model device
  = TransformerLM
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      NumEmbeds
      EmbedDim
      SeqLen
      'D.Float
      device

type ModelSpec device
  = TransformerLMSpec
      NumAttnLayers
      NumHeads
      FFNDim
      PaddingIdx
      NumEmbeds
      EmbedDim
      SeqLen
      'D.Float
      device

data Data

type BatchSize = 1

data CrossEntropyLoss = CrossEntropyLoss Bool

instance
  ( model ~ TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim seqLen dtype device
  , input ~ Tensor device 'D.Int64 '[batchSize, seqLen]
  , target ~ Tensor device 'D.Int64 '[batchSize, seqLen]
  , loss ~ Loss device dtype
  , (paddingIdx + 1) <= numEmbeds
  , 1 <= seqLen
  , BasicArithmeticDTypeIsValid device dtype
  , ComparisonDTypeIsValid device dtype
  , ComparisonDTypeIsValid device 'D.Int64
  , StandardFloatingPointDTypeValidation device dtype
  , All KnownNat '[paddingIdx, embedDim, seqLen, batchSize]
  , KnownDType dtype
  , KnownDevice device
  , HFoldrM
      IO
      FoldLayers
      (Tensor device 'D.Bool '[seqLen, batchSize], Tensor device dtype '[seqLen, batchSize, embedDim])
      (HReplicateR numAttnLayers (TransformerLMLayer embedDim numHeads ffnDim dtype device))
  ) => Apply' CrossEntropyLoss (model, input, target) (IO (Async loss)) where
  apply' (CrossEntropyLoss train) (modelState, input, target) = async $ do
    prediction <- logits modelState train input
    pure . crossEntropyLoss @PaddingIdx prediction $ target

data Gradients = Gradients

instance
  ( Parameterized model parameters
  , loss ~ Loss device dtype
  , HasGrad (HList parameters) gradients
  ) => Apply' Gradients (model, Async loss) (IO (Async gradients)) where
  apply' _ (modelState, asyncLoss) = async $ do
    loss <- wait asyncLoss
    let parameters = flattenParameters modelState
        gradients = grad loss parameters
    pure $ gradients

data ConvertGradients = ConvertGradients

data MergeGradients = MergeGradients

instance
  ( Num gradient
  ) => Apply' MergeGradients (gradient, gradient) gradient where
  apply' _ (gradient, gradient') = gradient + gradient

data FoldGradients = FoldGradients

instance
  ( HMap' ConvertGradients gradients gradients'
  , HZipWith MergeGradients gradients' gradients' gradients'
  ) => Apply FoldGradients (Async (HList gradients)) (Maybe (Async (HList gradients')) -> IO (Maybe (Async (HList gradients')))) where
  apply _ asyncGradients Nothing = fmap Just $ async $ do
    gradients <- wait asyncGradients
    pure $ hmap' ConvertGradients gradients
  apply _ asyncGradients (Just asyncGradients') = fmap Just $ async $ do
    gradients <- wait asyncGradients
    gradients' <- wait asyncGradients'
    pure $ hZipWith MergeGradients (hmap' ConvertGradients gradients) gradients'

gather
  :: forall foldedGradients m models inputs targets xs ys losses gradients
   . ( Monad m
     , HZipList3 models inputs targets xs
     , HMapM' m CrossEntropyLoss xs losses
     , HZip models losses ys
     , HMapM' m Gradients ys gradients
     , HFoldrM m FoldGradients (Maybe foldedGradients) gradients
     )
  => HList models
  -> Bool
  -> HList inputs
  -> HList targets
  -> m (Maybe foldedGradients)
gather modelStates train inputs targets = do
  losses <- hmapM' (CrossEntropyLoss train) (hZipList3 modelStates inputs targets)
  gradients <- hmapM' Gradients (hzip modelStates losses)
  foldedGradients <- hfoldrM FoldGradients Nothing gradients
  return foldedGradients

data ToDevice a = ToDevice a

instance
  ( HasToDevice device' device f g
  ) => Apply' (ToDevice f) (Proxy device, Proxy device') g where
  apply' (ToDevice f) (Proxy, Proxy) =
    Torch.Typed.Device.toDevice @device' @device f

scatter
  :: forall device devices model models xs
   . ( HAttach (Proxy device) devices xs
     , HMap' (ToDevice model) xs models
     )
  => model
  -> HList devices
  -> HList models
scatter modelState devices = hmap' (ToDevice modelState) $ hattach (Proxy @device) devices

runStep'
  :: forall model parameters m tensors gradients1 xs2 losses optim gradients2 dtype device shape chunks tensorChunks modelStates ys devices xs3
   . ( Parameterized model parameters
     , HMapM' m MakeIndependent tensors parameters
     , HMapM' m Gradients ys gradients1
     , HMapM' m CrossEntropyLoss xs2 losses
     , Optimizer optim gradients2 tensors dtype device
     , HFoldrM m FoldGradients (Maybe (HList gradients2)) gradients1
     , tensorChunks ~ Chunk chunks 0 shape dtype device
     , ATen.Castable (HList tensorChunks) [D.ATenTensor]
     , HZip modelStates losses ys
     , HZipList3 modelStates tensorChunks tensorChunks xs2
     , Monad m
     , HAttach (Proxy device) devices xs3
     , KnownNat chunks
     , HMap' (ToDevice model) xs3 modelStates
     , HMap' ToDependent parameters tensors
     )
  => model
  -> optim
  -> HList devices
  -> LearningRate device dtype
  -> Tensor device dtype shape
  -> Tensor device dtype shape
  -> m (model, optim)
runStep' modelState optimState devices learningRate input target = do
  let modelStates = scatter @device modelState devices
      inputs = chunk @chunks @0 input
      targets = chunk @chunks @0 target
  foldedGradients <- gather @(HList gradients2) modelStates True inputs targets
  case foldedGradients of
    Just gradients -> do
                        let parameters = flattenParameters modelState
                            tensors = hmap' ToDependent parameters
                            (tensors', optimState') = step learningRate gradients tensors optimState
                        parameters' <- hmapM' MakeIndependent tensors'
                        let modelState' = replaceParameters modelState parameters'
                        return (modelState', optimState')
    Nothing -> return (modelState, optimState)

train
  :: forall (batchSize :: Nat) (seqLen :: Nat) (device :: (D.DeviceType, Nat)) model optim gradients parameters tensors
   . ( All KnownNat '[batchSize, seqLen]
     , StandardFloatingPointDTypeValidation device 'D.Float
     , SumDTypeIsValid device 'D.Bool
     , ComparisonDTypeIsValid device 'D.Int64
     , KnownDevice device
     , HasGrad (HList parameters) (HList gradients)
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     , Parameterized model parameters
     , Optimizer optim gradients tensors 'D.Float device
     , HMapM' IO MakeIndependent tensors parameters
     )
  => model
  -> optim
  -> (model -> Bool -> Tensor device 'D.Int64 '[batchSize, seqLen] -> IO (Tensor device 'D.Float '[batchSize, seqLen, seqLen]))
  -> LearningRate device 'D.Float
  -> String
  -> IO ()
train initModel initOptim forward learningRate ptFile = do
  let numEpochs = 1000
  foldLoop_ (initModel, initOptim) numEpochs $ \(epochModel, epochOptim) epoch -> do
    let numIters = _lengthTrainingData `div` natValI @batchSize
    (epochModel', epochOptim') <-
      foldLoop (epochModel, epochOptim) numIters $ \(model, optim) i -> do
        trainingLoss <- computeLoss @batchSize (forward model True) i
        (model', optim') <- runStep model optim trainingLoss learningRate
        return (model', optim')
    
    testLoss <- do
      let numIters = _lengthTestData `div` natValI @batchSize
      foldLoop 0 numIters $ \aggTestLoss i -> do
        testLoss' <- computeLoss @batchSize (forward epochModel' False) i
        return $ aggTestLoss + toFloat testLoss'
    
    putStrLn
      $  "Epoch: "
      <> show epoch
      <> ". Test loss: "
      <> show (testLoss / realToFrac (_lengthTestData))
    
    save (hmap' ToDependent . flattenParameters $ epochModel') ptFile
    return (epochModel', epochOptim')
 where
  computeLoss
    :: forall (n :: Nat)
     . KnownNat n
    => (Tensor device 'D.Int64 '[n, seqLen] -> IO (Tensor device 'D.Float '[n, seqLen, seqLen]))
    -> Int
    -> IO (Tensor device 'D.Float '[])
  computeLoss forward' index_of_batch = do
    let from    = (index_of_batch-1) * natValI @n
        to      = (index_of_batch * natValI @n) - 1
        indexes = [from .. to]
        input   = getInput @n indexes
        target  = getTarget @n indexes
    prediction <- forward' input
    return $ crossEntropyLoss @PaddingIdx prediction (Torch.Typed.Tensor.toDevice target)
  getInput :: forall (n :: Nat) . [Int] -> Tensor device 'D.Int64 '[n, seqLen]
  getInput = undefined
  getTarget :: forall (n :: Nat) . [Int] -> Tensor device 'D.Int64 '[n, seqLen]
  getTarget = undefined

train'
  :: forall (device :: (D.DeviceType, Nat))
   . _
  => IO ()
train' = do
  let dropoutProb  = 0.5
      learningRate = 0.1
  -- ATen.manual_seed_L 123
  initModel <- A.sample
    (TransformerLMSpec
          (DropoutSpec 0.2)
          (TransformerLMLayerSpec
            (MultiheadAttentionSpec
              (DropoutSpec 0.2)
            )
            (DropoutSpec 0.2)
            0.001
            (TransformerLMMLPSpec
              (DropoutSpec 0.2)
              (DropoutSpec 0.2)
              (Activation Torch.Typed.Functional.relu)
              (Activation Torch.Typed.Functional.relu)
            )
          ) :: ModelSpec device
    )
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  train @BatchSize @SeqLen @device initModel initOptim logits learningRate "transformer.pt"

main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  case deviceStr of
    Right "cpu"    -> train' @'( 'D.CPU, 0)
    Right "cuda:0" -> train' @'( 'D.CUDA, 0)
    _              -> error "Don't know what to do or how."

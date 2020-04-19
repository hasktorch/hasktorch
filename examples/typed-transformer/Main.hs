{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

-- {-# OPTIONS_GHC -fdefer-typed-holes #-}
-- {-# OPTIONS_GHC -Wno-typed-holes #-}

module Main where

import Control.Concurrent.Async
import qualified Control.Foldl as L
import Data.Constraint
import Data.Kind
import qualified Data.List as List
import qualified Data.Maybe as Maybe
import Data.Proxy
import Data.Set.Ordered as OSet
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified GHC.Exts as Exts
import GHC.TypeLits
import qualified GHC.TypeNats as N
import Lens.Family
import Pipes
import Pipes.Group
import qualified Pipes.Prelude as P
import qualified Pipes.Random as Random
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import qualified Pipes.Text as Text
import qualified Pipes.Text.IO as Text
import qualified System.IO as IO
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC)
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.NN as A
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Autograd
import Torch.Typed.Aux
import Torch.Typed.Device
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.NN
import Torch.Typed.NN.DataParallel
import Torch.Typed.NN.Transformer
import Torch.Typed.Optim
import Torch.Typed.Parameter
import Torch.Typed.Serialize
import Torch.Typed.Tensor
import Unsafe.Coerce (unsafeCoerce)
import Prelude

type WorkerDevices = '[ '( 'D.CUDA, 0)]

type ModelDevice = '( 'D.CUDA, 0)

type DataDevice = '( 'D.CPU, 0)

type BatchSize = 1

type SeqLen = 512

type NumAttnLayers = 2

type NumHeads = 12

type FFNDim = 3072

type PaddingIdx = 0

type EmbedDim = 768

type Model numEmbeds modelDevice =
  TransformerLM
    NumAttnLayers
    NumHeads
    FFNDim
    PaddingIdx
    numEmbeds
    EmbedDim
    'D.Float
    modelDevice

type ModelSpec numEmbeds modelDevice =
  TransformerLMSpec
    NumAttnLayers
    NumHeads
    FFNDim
    PaddingIdx
    numEmbeds
    EmbedDim
    'D.Float
    modelDevice

main :: IO ()
main = program 100 "trainingFile.txt" 1000 "evaluationFile.txt" 1

program ::
  -- | number of epochs
  Int ->
  -- | training file path
  FilePath ->
  -- | number batches taken from training file per epoch
  Int ->
  -- | evaluation file path
  FilePath ->
  -- | number batches taken from evaluation file per epoch
  Int ->
  IO ()
program numEpochs trainingFile trainingLen evaluationFile evaluationLen = Safe.runSafeT . runEffect $ do
  vocab <- liftIO $ L.fold (L.Fold (OSet.|<>) (OSet.singleton "[PAD]") id) <$> traverse buildVocabFromFile [trainingFile, evaluationFile]
  liftIO . print . size $ vocab
  let vocabLen = N.someNatVal . fromIntegral . size $ vocab
  case vocabLen of
    (SomeNat proxy) -> case mkNumEmbedsProof proxy of
      Just dict -> go dict vocab
      Nothing -> pure ()
  where
    go ::
      forall (numEmbeds :: Nat).
      KnownNat numEmbeds =>
      Dict ((1 <=? numEmbeds) ~ 'True) ->
      OSet.OSet Text.Text ->
      Effect (Safe.SafeT IO) ()
    go Dict vocab =
      let trainingData = readData @SeqLen @DataDevice @BatchSize @(Safe.SafeT IO) trainingFile vocab >-> P.take trainingLen
          evaluationData = readData @SeqLen @DataDevice @BatchSize @(Safe.SafeT IO) evaluationFile vocab >-> P.take evaluationLen
          learning' = do
            let learningRate = 0.01
            -- ATen.manual_seed_L 123
            model <-
              liftIO $
                A.sample
                  ( TransformerLMSpec
                      (DropoutSpec 0.2)
                      ( TransformerLayerSpec
                          ( MultiheadAttentionSpec
                              (DropoutSpec 0.2)
                          )
                          (DropoutSpec 0.2)
                          0.001
                          ( TransformerMLPSpec
                              (DropoutSpec 0.2)
                              (DropoutSpec 0.2)
                          )
                      ) ::
                      ModelSpec numEmbeds ModelDevice
                  )
            let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
            learning @WorkerDevices @ModelDevice @DataDevice @numEmbeds @BatchSize @SeqLen numEpochs learningRate (model, optim) trainingData evaluationData
       in learning' >-> P.map (\(loss, _, _) -> loss) >-> P.print

mkNumEmbedsProof ::
  forall (numEmbeds :: Nat).
  KnownNat numEmbeds =>
  Data.Proxy.Proxy numEmbeds ->
  Maybe (Dict ((1 <=? numEmbeds) ~ 'True))
mkNumEmbedsProof Proxy =
  let numEmbeds = natValI @numEmbeds
   in if numEmbeds > 0
        then Just (unsafeCoerce (Dict :: Dict ('True ~ 'True)))
        else Nothing

data FlattenParametersF = FlattenParametersF

instance
  ( Parameterized model parameters
  ) =>
  Apply' FlattenParametersF model (HList parameters)
  where
  apply' _ = flattenParameters

training ::
  forall workerDevices modelDevice dataDevice dtype model models optim input inputs target targets inputTargets losses parameters' gradients parameters tensors m.
  ( HasScatter workerDevices dataDevice input inputs,
    HasScatter workerDevices dataDevice target targets,
    HasReplicate workerDevices modelDevice model models,
    HZip inputs targets inputTargets,
    HZipWithM Concurrently ForwardConcurrentlyF models inputTargets losses,
    HMap' FlattenParametersF models parameters',
    HasGradConcurrently modelDevice workerDevices parameters' losses gradients,
    Parameterized model parameters,
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    Optimizer optim gradients tensors dtype modelDevice,
    HMapM' IO MakeIndependent tensors parameters,
    KnownDType dtype,
    KnownDevice modelDevice,
    MonadIO m
  ) =>
  LearningRate modelDevice dtype ->
  (model, optim) ->
  Producer (input, target) m () ->
  m (model, optim)
training learningRate (model, optim) = P.foldM step begin done
  where
    step (model', optim') (input, target) = do
      let models' = Torch.Typed.Device.replicate @workerDevices @modelDevice @model @models model'
          inputs = scatter @workerDevices @dataDevice @input @inputs input
          targets = scatter @workerDevices @dataDevice @target @targets target
      losses <- liftIO . runConcurrently . forwardConcurrentlyStoch @models @inputTargets models' $ hzip inputs targets
      let parameters' = hmap' FlattenParametersF models'
      gradients <- liftIO . runConcurrently $ gradConcurrently @modelDevice @workerDevices @parameters' @losses @gradients parameters' losses
      liftIO performGC -- force cleanup after every batch
      liftIO $ runStep' model' optim' learningRate gradients
    begin = pure (model, optim)
    done = pure

evaluation ::
  forall workerDevices modelDevice dataDevice numEmbeds batchSize seqLen dtype model models input inputs output outputs target m.
  ( HasScatter workerDevices dataDevice input inputs,
    HasReplicate workerDevices modelDevice model models,
    HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs,
    HasGather dataDevice workerDevices outputs output,
    input ~ Tensor dataDevice 'D.Int64 '[batchSize, seqLen],
    output ~ Tensor dataDevice dtype '[batchSize, seqLen, numEmbeds],
    target ~ Tensor dataDevice 'D.Int64 '[batchSize, seqLen],
    StandardFloatingPointDTypeValidation dataDevice dtype,
    Torch.Typed.Tensor.All KnownNat '[batchSize, seqLen, numEmbeds],
    KnownDType dtype,
    KnownDevice dataDevice,
    MonadIO m
  ) =>
  model ->
  Producer (input, target) m () ->
  m Float
evaluation model = P.foldM step begin done
  where
    step aggLoss (input, target) = do
      let models = Torch.Typed.Device.replicate @workerDevices @modelDevice @model @models model
          inputs = scatter @workerDevices @dataDevice @input @inputs input
      outputs <- liftIO . runConcurrently $ forwardConcurrently @models @inputs models inputs
      let prediction = gather @dataDevice @workerDevices @outputs @output outputs
      let loss = crossEntropyLoss @PaddingIdx prediction $ target
      liftIO performGC -- force cleanup after every batch
      pure $ aggLoss + toFloat (Torch.Typed.Tensor.toDType @'D.Float loss)
    begin = pure 0
    done = pure

learning ::
  forall workerDevices modelDevice dataDevice numEmbeds batchSize seqLen dtype model models parameters parameters' input inputs target targets inputTargets output outputs losses optim gradients tensors m.
  ( HasScatter workerDevices dataDevice input inputs,
    HasScatter workerDevices dataDevice target targets,
    HasReplicate workerDevices modelDevice model models,
    HZip inputs targets inputTargets,
    HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs,
    HZipWithM Concurrently ForwardConcurrentlyF models inputTargets losses,
    HMap' FlattenParametersF models parameters',
    HasGradConcurrently modelDevice workerDevices parameters' losses gradients,
    HasGather dataDevice workerDevices outputs output,
    Parameterized model parameters,
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    ATen.Castable (HList gradients) [D.ATenTensor],
    Optimizer optim gradients tensors dtype modelDevice,
    HMapM' IO MakeIndependent tensors parameters,
    input ~ Tensor dataDevice 'D.Int64 '[batchSize, seqLen],
    output ~ Tensor dataDevice dtype '[batchSize, seqLen, numEmbeds],
    target ~ Tensor dataDevice 'D.Int64 '[batchSize, seqLen],
    StandardFloatingPointDTypeValidation dataDevice dtype,
    Torch.Typed.Tensor.All KnownNat '[batchSize, seqLen, numEmbeds],
    KnownDType dtype,
    Torch.Typed.Tensor.All KnownDevice '[modelDevice, dataDevice],
    MonadIO m
  ) =>
  Int ->
  LearningRate modelDevice dtype ->
  (model, optim) ->
  Producer (input, target) m () ->
  Producer (input, target) m () ->
  Producer (Float, model, optim) m ()
learning numEpochs learningRate (model, optim) trainingData evaluationData =
  for (each [1 .. numEpochs]) $ \_epoch -> do
    (model', optim') <- lift $ training @workerDevices @modelDevice @dataDevice @dtype @model @models @optim @input @inputs @target @targets @inputTargets @losses @parameters' @gradients @parameters @tensors learningRate (model, optim) trainingData
    evalLoss' <- lift $ evaluation @workerDevices @modelDevice @dataDevice @numEmbeds @batchSize @seqLen @dtype @model @models @input @inputs @output @outputs @target model' evaluationData
    yield (evalLoss', model', optim')

readData ::
  forall seqLen modelDevice batchSize m.
  (KnownNat seqLen, KnownDevice modelDevice, KnownNat batchSize, Safe.MonadSafe m) =>
  FilePath ->
  OSet.OSet Text.Text ->
  Producer (Tensor modelDevice 'D.Int64 '[batchSize, seqLen], Tensor modelDevice 'D.Int64 '[batchSize, seqLen]) m ()
readData file vocab = raw >-> pipe
  where
    raw = Safe.withFile file IO.ReadMode $
      \h ->
        batching
          . L.purely folds L.list
          . chain (sequencing . readHandleEndlesslyFromOffset h)
          $ randomOffsets
    sequencing =
      (>-> applyVocab vocab)
        . L.purely folds L.mconcat
        . takes (natValI @(seqLen + 1))
        . drops 1
        . view Text.words
    batching = L.purely folds L.list . view (chunksOf (natValI @batchSize))
    pipe = for Pipes.cat $ \x -> case f x of
      Nothing -> return ()
      Just y -> yield y
    f xs = do
      let xs' = Maybe.catMaybes <$> xs
      input <- Exts.fromList $ take (natValI @seqLen) <$> xs'
      target <- Exts.fromList $ drop 1 <$> xs'
      pure (Torch.Typed.Device.toDevice @modelDevice @'( 'D.CPU, 0) input, Torch.Typed.Device.toDevice @modelDevice @'( 'D.CPU, 0) target)

randomOffsets :: MonadIO m => Producer Integer m ()
randomOffsets = hoist liftIO $ Random.uniform @Int >-> P.map toInteger

chain :: forall a b m r. Monad m => (a -> Producer b m r) -> Producer a m r -> FreeT (Producer b m) m r
chain f = go
  where
    go p = FreeT $ do
      x <- next p
      return $ case x of
        Left r -> Pure r
        Right (a, p') -> Free $ fmap go (f a >> return p')

readHandleEndlesslyFromOffset :: forall m. MonadIO m => IO.Handle -> Integer -> Producer Text.Text m ()
readHandleEndlesslyFromOffset h offset = do
  fileSize <- liftIO $ IO.hFileSize h
  let offset' = offset `mod` fileSize
  liftIO $ IO.hSeek h IO.AbsoluteSeek offset'
  fromHandleEndlessly $ h

fromHandleEndlessly :: forall m. MonadIO m => IO.Handle -> Producer Text.Text m ()
fromHandleEndlessly h = go
  where
    go = do
      txt <- liftIO (T.hGetChunk h)
      if T.null txt
        then liftIO (IO.hSeek h IO.AbsoluteSeek 0)
        else yield txt
      go

buildVocab :: forall a m. (Ord a, Monad m) => Producer a m () -> m (OSet.OSet a)
buildVocab = L.purely P.fold oSet

oSet :: forall a. Ord a => L.Fold a (OSet.OSet a)
oSet = L.Fold (\as a -> as |> a) OSet.empty id

buildVocabFromFile :: FilePath -> IO (OSet.OSet Text.Text)
buildVocabFromFile file = IO.withFile file IO.ReadMode $ buildVocab . L.purely folds L.mconcat . view Text.words . Text.fromHandle

applyVocab :: forall a m. (Ord a, Functor m) => OSet.OSet a -> Pipe a (Maybe OSet.Index) m ()
applyVocab = P.map . flip OSet.findIndex

crossEntropyLoss ::
  forall paddingIdx batchSize seqLen numEmbeds dtype modelDevice.
  ( KnownNat paddingIdx,
    KnownNat batchSize,
    KnownNat seqLen,
    KnownNat numEmbeds,
    KnownDType dtype,
    KnownDevice modelDevice,
    StandardFloatingPointDTypeValidation modelDevice dtype
  ) =>
  Tensor modelDevice dtype '[batchSize, seqLen, numEmbeds] ->
  Tensor modelDevice 'D.Int64 '[batchSize, seqLen] ->
  Tensor modelDevice dtype '[]
crossEntropyLoss prediction target =
  nllLoss @D.ReduceMean @batchSize @numEmbeds @'[seqLen]
    ones
    (natValI @paddingIdx)
    (logSoftmax @1 . transpose @1 @2 $ prediction)
    target

instance
  ( HasForward (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype modelDevice) (Tensor modelDevice 'D.Int64 '[batchSize, seqLen]) (Tensor modelDevice dtype '[batchSize, seqLen, numEmbeds]),
    StandardFloatingPointDTypeValidation modelDevice dtype,
    KnownNat batchSize,
    KnownNat seqLen,
    KnownNat numEmbeds,
    KnownDType dtype,
    KnownDevice modelDevice
  ) =>
  HasForward (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds embedDim dtype modelDevice) (Tensor modelDevice 'D.Int64 '[batchSize, seqLen], Tensor modelDevice 'D.Int64 '[batchSize, seqLen]) (Loss modelDevice dtype)
  where
  forward model (input, target) =
    let prediction = forward model input
     in crossEntropyLoss @PaddingIdx prediction target
  forwardStoch model (input, target) = do
    prediction <- forwardStoch model input
    return $ crossEntropyLoss @PaddingIdx prediction target

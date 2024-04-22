{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

-- {-# OPTIONS_GHC -fdefer-typed-holes #-}
-- {-# OPTIONS_GHC -Wno-typed-holes #-}

module Main where

import Control.Concurrent.Async
import qualified Control.Foldl as L
import Control.Monad.Base (MonadBase, liftBase)
import Control.Monad.Cont (ContT (runContT))
import Control.Monad.Trans.Control (MonadBaseControl (..), control)
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
import qualified GHC.TypeNats
import Lens.Family hiding (All)
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
import Torch (ATenTensor)
import Torch.Data.Dataset
import Torch.Data.StreamedPipeline
import Torch.Internal.Class (Castable)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed
import Unsafe.Coerce (unsafeCoerce)
import Prelude hiding (replicate)

type WorkerDevices = '[ '( 'CPU, 0)]

type ModelDevice = '( 'CPU, 0)

type DataDevice = '( 'CPU, 0)

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
    'Float
    modelDevice

type ModelSpec numEmbeds modelDevice =
  TransformerLMSpec
    NumAttnLayers
    NumHeads
    FFNDim
    PaddingIdx
    numEmbeds
    EmbedDim
    'Float
    modelDevice

data TransformerData seqLen = TransformerData
  { length :: Int,
    filePath :: FilePath,
    vocab :: OSet.OSet Text.Text
  }

instance
  (KnownNat seqLen, Safe.MonadSafe m, MonadBase IO m) =>
  Datastream m () (TransformerData seqLen) [Maybe Int]
  where
  streamSamples TransformerData {..} _ =
    Select $
      readData @seqLen filePath length vocab

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
program numEpochs trainingFile trainingLen evaluationFile evaluationLen =
  Safe.runSafeT . runEffect $ do
    vocab <-
      liftIO $
        L.fold (L.Fold (OSet.|<>) (OSet.singleton "[PAD]") id)
          <$> traverse buildVocabFromFile [trainingFile, evaluationFile]
    liftIO . print . size $ vocab
    let vocabLen = GHC.TypeNats.someNatVal . fromIntegral . size $ vocab
    case vocabLen of
      (SomeNat (proxy :: Data.Proxy.Proxy numEmbeds)) -> case mkNumEmbedsProof @numEmbeds proxy of
        Just dict -> go @numEmbeds dict vocab
        Nothing -> pure ()
  where
    go ::
      forall (numEmbeds :: Nat).
      KnownNat numEmbeds =>
      Dict ((1 <=? numEmbeds) ~ 'True) ->
      OSet.OSet Text.Text ->
      Effect (Safe.SafeT IO) ()
    go Dict vocab =
      let trainingData =
            TransformerData
              { length = trainingLen,
                filePath = trainingFile,
                vocab = vocab
              }
          evaluationData =
            TransformerData
              { length = evaluationLen,
                filePath = evaluationFile,
                vocab = vocab
              }
          learning' = do
            let learningRate = 0.01
            -- manual_seed_L 123
            model <-
              liftIO $
                sample
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
                              0.001
                          )
                      ) ::
                      ModelSpec numEmbeds ModelDevice
                  )
            let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
            learning @WorkerDevices @ModelDevice @DataDevice @numEmbeds @BatchSize @SeqLen
              numEpochs
              learningRate
              (model, optim)
              trainingData
              evaluationData
       in learning' >-> P.map (\(loss, _, _) -> loss) >-> P.print

mkNumEmbedsProof ::
  forall (numEmbeds :: Nat).
  KnownNat numEmbeds  =>
  Data.Proxy.Proxy numEmbeds ->
  Maybe (Dict ((1 <=? numEmbeds) ~ 'True))
mkNumEmbedsProof Proxy =
  let numEmbeds = natValI @numEmbeds
   in if numEmbeds > 0
        then Just (unsafeCoerce (Dict :: Dict ('True ~ 'True)))
        else Nothing

data FlattenParametersF = FlattenParametersF

instance
  ( parameters ~ Parameters model,
    Parameterized model
  ) =>
  Apply' FlattenParametersF model (HList parameters)
  where
  apply' _ = flattenParameters

training ::
  forall
    workerDevices
    modelDevice
    dataDevice
    dtype
    model
    models
    optim
    input
    inputs
    target
    targets
    inputTargets
    losses
    parameters'
    gradients
    parameters
    tensors
    m.
  ( HasScatter workerDevices dataDevice input inputs,
    HasScatter workerDevices dataDevice target targets,
    HasReplicate workerDevices modelDevice model models,
    HZip inputs targets inputTargets,
    HZipWithM Concurrently ForwardConcurrentlyF models inputTargets losses,
    HMap' FlattenParametersF models parameters',
    HasGradConcurrently modelDevice workerDevices parameters' losses gradients,
    parameters ~ Parameters model,
    Parameterized model,
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
  ListT m (input, target) ->
  m (model, optim)
training learningRate (model, optim) = P.foldM step begin done . enumerateData
  where
    step (model', optim') ((input, target), iter) = do
      let models' = replicate @workerDevices @modelDevice @model @models model'
          inputs = scatter @workerDevices @dataDevice @input @inputs input
          targets = scatter @workerDevices @dataDevice @target @targets target
      losses <-
        liftIO . runConcurrently
          . forwardConcurrentlyStoch @models @inputTargets models'
          $ hzip inputs targets
      let parameters' = hmap' FlattenParametersF models'
      gradients <-
        liftIO
          . runConcurrently
          $ gradConcurrently @modelDevice @workerDevices @parameters' @losses @gradients
            parameters'
            losses
      liftIO performGC -- force cleanup after every batch
      liftIO $ runStep' model' optim' learningRate gradients
    begin = pure (model, optim)
    done = pure

evaluation ::
  forall
    workerDevices
    modelDevice
    dataDevice
    numEmbeds
    batchSize
    seqLen
    dtype
    model
    models
    input
    inputs
    output
    outputs
    target
    m.
  ( 1 <= seqLen,
    1 <= numEmbeds,
    workerDevices ~ WorkerDevices,
    modelDevice ~ ModelDevice,
    dtype ~ 'Float,
    inputs ~ '[Tensor '( 'CPU, 0) 'Int64 '[batchSize, seqLen]],
    outputs ~ '[Tensor '( 'CPU, 0) dtype '[batchSize, seqLen, numEmbeds]],
    models ~ '[Model numEmbeds modelDevice],
    model ~ Model numEmbeds modelDevice,
    HasScatter workerDevices dataDevice input inputs,
    HasReplicate workerDevices modelDevice model models,
    HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs,
    HasGather dataDevice workerDevices outputs output,
    input ~ Tensor dataDevice 'Int64 '[batchSize, seqLen],
    output ~ Tensor dataDevice dtype '[batchSize, seqLen, numEmbeds],
    target ~ Tensor dataDevice 'Int64 '[batchSize, seqLen],
    StandardFloatingPointDTypeValidation dataDevice dtype,
    All KnownNat '[batchSize, seqLen, numEmbeds],
    KnownDType dtype,
    KnownDevice dataDevice,
    MonadIO m
  ) =>
  model ->
  ListT m (input, target) ->
  m _
evaluation model = P.foldM step begin done . enumerateData
  where
    step aggLoss ((input, target), iter) = do
      let models = replicate @workerDevices @modelDevice @model @models model
          inputs = scatter @workerDevices @dataDevice @input @inputs input
      outputs <- liftIO . runConcurrently $ forwardConcurrently @models @inputs models inputs
      let prediction = gather @dataDevice @workerDevices @outputs @output outputs
      let loss = crossEntropyLoss @PaddingIdx prediction $ target
      liftIO performGC -- force cleanup after every batch
      pure $ aggLoss + toFloat (toDType @'Float @dtype loss)
    begin = pure 0
    done = pure

learning ::
  forall
    workerDevices
    modelDevice
    dataDevice
    numEmbeds
    batchSize
    seqLen
    dtype
    model
    models
    parameters
    parameters'
    input
    inputs
    target
    targets
    inputTargets
    output
    outputs
    losses
    optim
    gradients
    tensors
    m.
  ( 1 <= numEmbeds,
    1 <= seqLen,
    dtype ~ 'Float,
    workerDevices ~ WorkerDevices,
    modelDevice ~ ModelDevice,
    dataDevice ~ DataDevice,
    inputs ~ '[Tensor '( 'CPU, 0) 'Int64 '[batchSize, seqLen]],
    outputs ~ '[Tensor '( 'CPU, 0) dtype '[batchSize, seqLen, numEmbeds]],
    models ~ '[Model numEmbeds modelDevice],
    model ~ Model numEmbeds modelDevice,
    HasScatter workerDevices dataDevice input inputs,
    HasScatter workerDevices dataDevice target targets,
    HasReplicate workerDevices modelDevice model models,
    HZip inputs targets inputTargets,
    HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs,
    HZipWithM Concurrently ForwardConcurrentlyF models inputTargets losses,
    HMap' FlattenParametersF models parameters',
    HasGradConcurrently modelDevice workerDevices parameters' losses gradients,
    HasGather dataDevice workerDevices outputs output,
    parameters ~ Parameters model,
    Parameterized model,
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    Castable (HList gradients) [ATenTensor],
    Optimizer optim gradients tensors dtype modelDevice,
    HMapM' IO MakeIndependent tensors parameters,
    input ~ Tensor dataDevice 'Int64 '[batchSize, seqLen],
    output ~ Tensor dataDevice dtype '[batchSize, seqLen, numEmbeds],
    target ~ Tensor dataDevice 'Int64 '[batchSize, seqLen],
    StandardFloatingPointDTypeValidation dataDevice dtype,
    All KnownNat '[batchSize, seqLen, numEmbeds],
    KnownDType dtype,
    All KnownDevice '[modelDevice, dataDevice],
    MonadIO m,
    MonadBaseControl IO m,
    Safe.MonadSafe m
  ) =>
  Int ->
  LearningRate modelDevice dtype ->
  (model, optim) ->
  TransformerData seqLen ->
  TransformerData seqLen ->
  Producer (Float, model, optim) m ()
learning numEpochs learningRate (model, optim) trainingData evaluationData =
  let collatedTrain ::
        CollatedDataset
          m
          (TransformerData seqLen)
          [Maybe Int]
          (input, target)
      collatedTrain =
        CollatedDataset
          { set = trainingData,
            chunkSize = natValI @batchSize,
            collateFn = collation
          }
      collatedEval ::
        CollatedDataset
          m
          (TransformerData seqLen)
          [Maybe Int]
          (input, target)
      collatedEval =
        CollatedDataset
          { set = evaluationData,
            chunkSize = natValI @batchSize,
            collateFn = collation
          }
   in void $ P.foldM (step collatedTrain collatedEval) begin done $ each [1 .. numEpochs]
  where
    step trainSet testSet (model, optim) epoch = do
      (model', optim') <-
        lift $
          runContT (streamFrom' datastreamOpts trainSet [()]) $
            training
              @workerDevices
              @modelDevice
              @dataDevice
              @dtype
              @model
              @models
              @optim
              @input
              @inputs
              @target
              @targets
              @inputTargets
              @losses
              @parameters'
              @gradients
              @parameters
              @tensors
              learningRate
              (model, optim)
      evalLoss' <-
        lift $
          runContT (streamFrom' datastreamOpts testSet [()]) $
            evaluation
              @workerDevices
              @modelDevice
              @dataDevice
              @numEmbeds
              @batchSize
              @seqLen
              @dtype
              @model
              @models
              @input
              @inputs
              @output
              @outputs
              @target
              model'
      yield (evalLoss', model', optim')
      pure (model', optim')
    begin = pure (model, optim)
    done = pure

readData ::
  forall seqLen m.
  (Safe.MonadSafe m, KnownNat seqLen) =>
  FilePath ->
  Int ->
  OSet.OSet Text.Text ->
  Producer [Maybe Int] m ()
readData file length vocab = raw >-> P.take length
  where
    raw = Safe.withFile file IO.ReadMode $
      \h ->
        L.purely folds L.list
          . chain (sequencing . readHandleEndlesslyFromOffset h)
          $ randomOffsets
    sequencing =
      (>-> applyVocab vocab)
        . L.purely folds L.mconcat
        . takes (natValI @(seqLen + 1))
        . drops 1
        . view Text.words

collation ::
  forall modelDevice batchSize seqLen m.
  ( KnownNat seqLen,
    KnownDevice modelDevice,
    KnownNat batchSize,
    Safe.MonadSafe m
  ) =>
  Pipe
    [[Maybe Int]]
    ( Tensor modelDevice 'Int64 '[batchSize, seqLen],
      Tensor modelDevice 'Int64 '[batchSize, seqLen]
    )
    m
    ()
collation = for Pipes.cat $ \x -> case f x of
  Nothing -> return ()
  Just y -> yield y
  where
    f xs = do
      let xs' = Maybe.catMaybes <$> xs
      input <- Exts.fromList $ take (natValI @seqLen) <$> xs'
      target <- Exts.fromList $ drop 1 <$> xs'
      pure
        ( toDevice @modelDevice @'( 'CPU, 0) input,
          toDevice @modelDevice @'( 'CPU, 0) target
        )

randomOffsets :: MonadIO m => Producer Integer m ()
randomOffsets = hoist liftIO $ Random.uniform @Int >-> P.map toInteger

chain ::
  forall a b m r.
  Monad m =>
  (a -> Producer b m r) ->
  Producer a m r ->
  FreeT (Producer b m) m r
chain f = go
  where
    go p = FreeT $ do
      x <- next p
      return $ case x of
        Left r -> Pure r
        Right (a, p') -> Free $ fmap go (f a >> return p')

readHandleEndlesslyFromOffset ::
  forall m.
  MonadIO m =>
  IO.Handle ->
  Integer ->
  Producer Text.Text m ()
readHandleEndlesslyFromOffset h offset = do
  fileSize <- liftIO $ IO.hFileSize h
  let offset' = offset `mod` fileSize
  liftIO $ IO.hSeek h IO.AbsoluteSeek offset'
  fromHandleEndlessly $ h

fromHandleEndlessly ::
  forall m.
  MonadIO m =>
  IO.Handle ->
  Producer Text.Text m ()
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
buildVocabFromFile file =
  IO.withFile file IO.ReadMode $
    buildVocab
      . L.purely folds L.mconcat
      . view Text.words
      . Text.fromHandle

applyVocab ::
  forall a m.
  (Ord a, Functor m) =>
  OSet.OSet a ->
  Pipe a (Maybe OSet.Index) m ()
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
  Tensor modelDevice 'Int64 '[batchSize, seqLen] ->
  Tensor modelDevice dtype '[]
crossEntropyLoss prediction target =
  nllLoss @ReduceMean @batchSize @numEmbeds @'[seqLen]
    ones
    (natValI @paddingIdx)
    (logSoftmax @1 . transpose @1 @2 $ prediction)
    target

instance
  ( HasForward
      ( TransformerLM
          numAttnLayers
          numHeads
          ffnDim
          paddingIdx
          numEmbeds
          embedDim
          dtype
          modelDevice
      )
      (Tensor modelDevice 'Int64 '[batchSize, seqLen])
      (Tensor modelDevice dtype '[batchSize, seqLen, numEmbeds]),
    StandardFloatingPointDTypeValidation modelDevice dtype,
    KnownNat batchSize,
    KnownNat seqLen,
    KnownNat numEmbeds,
    KnownDType dtype,
    KnownDevice modelDevice
  ) =>
  HasForward
    ( TransformerLM
        numAttnLayers
        numHeads
        ffnDim
        paddingIdx
        numEmbeds
        embedDim
        dtype
        modelDevice
    )
    ( Tensor modelDevice 'Int64 '[batchSize, seqLen],
      Tensor modelDevice 'Int64 '[batchSize, seqLen]
    )
    (Loss modelDevice dtype)
  where
  forward model (input, target) =
    let prediction = forward model input
     in crossEntropyLoss @PaddingIdx prediction target
  forwardStoch model (input, target) = do
    prediction <- forwardStoch model input
    return $ crossEntropyLoss @PaddingIdx prediction target

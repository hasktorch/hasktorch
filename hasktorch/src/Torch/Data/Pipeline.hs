{-# LANGUAGE TupleSections #-}
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
module Torch.Data.Pipeline ( takeBatch
                           , readBatches
                           , readBatchesConcurrently
                           , makeFold
                           , makeFoldWithTransform
                           , makeConcurrentFold
                           , foldFromProducer
                           , L.FoldM(..)
                           , Dataset(..)
                           , DatasetMock(..)
                           , ConcurrentDataset(..)
                           ) where

import           Control.Arrow (first)
import           Control.Concurrent.Async
import qualified Control.Foldl as L
import           Control.Monad
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P


type Iter = Int
type WorkerId = Int

data DatasetMock m tensor = DatasetMock { getBatchMock :: Int -> m tensor
                                        , numItersMock :: Iter
                                        }

class Dataset dataset batch  where
  getBatch :: dataset -> Iter -> IO batch
  numIters :: dataset -> Int

class Dataset dataset batch => ConcurrentDataset dataset batch where
  getBatchConcurrently :: WorkerId -> dataset -> Iter -> IO batch

data RunBatch = Final | KeepTrain  deriving (Eq, Show)

takeBatch :: MonadIO m => Input (batch, RunBatch) -> Producer batch m ()  
takeBatch input = fromInput input >-> P.takeWhile ((/=) Final . snd) >-> P.map fst

readBatchesConcurrently :: forall dataset batch m .
  (MonadIO m, ConcurrentDataset dataset batch) => Int -> dataset -> Output (batch, RunBatch)  -> Effect m () 
readBatchesConcurrently workerId dataset transformBox = 
  for (each [1..numIters @dataset @batch dataset ]) (\iter -> yieldBatch iter >-> toOutput transformBox)
    where yieldBatch iter =
             (liftIO $ getBatchConcurrently workerId dataset iter) >>= yield . (, runBatch iter)
          runBatch iter = if numIters @dataset @batch dataset == iter then Final else KeepTrain

readBatches :: forall dataset batch m.
  (MonadIO m, Dataset dataset batch) => dataset -> Output (batch, RunBatch)  -> Effect m () 
readBatches dataset outputBox = 
  for (each [1..numIters @dataset @batch dataset]) (\iter -> yieldBatch iter >-> toOutput outputBox)
    where yieldBatch iter = (liftIO $ getBatch dataset iter) >>= yield . (, runBatch iter)
          runBatch iter = if numIters @dataset @batch dataset == iter then Final else KeepTrain


runTransforms :: MonadIO m => (tensor -> tensor') -> Input (tensor, RunBatch) -> Output (tensor', RunBatch) -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput transformBox >->  P.map (first transforms) >-> toOutput trainBox

makeFoldWithTransform :: (MonadIO m, Dataset dataset batch)  
  => (batch -> batch')
  -> dataset
  -> IO (L.FoldM m batch' b -> m b)
makeFoldWithTransform transforms dataset = do
          -- TODO: we can allow different buffer sizes
          -- which would be necessary for data echoing
            (toTransformBox, fromTransformBox, sealTransform) <- spawn' (bounded 1)
            (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
            async $ runEffect $ forever $ readBatches dataset toTransformBox 
            async $ runEffect $ runTransforms transforms fromTransformBox toBatches
            pure $ foldFromProducer (takeBatch fromBatches)

makeFold :: (Dataset dataset batch, MonadIO m)
    => dataset
    -> IO (L.FoldM m batch b -> m b)
makeFold dataset = do
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
  async $ runEffect $ forever $ readBatches dataset toBatches
  pure $ foldFromProducer (takeBatch fromBatches)

makeConcurrentFold :: (ConcurrentDataset dataset batch', MonadIO m)
  => (batch' -> batch)
  -> dataset
  -> Int
  -> IO (L.FoldM m batch b -> m b)
makeConcurrentFold transforms dataset numWorkers = do
  -- Buffer size is equal to numWorkers so that each thread can yield a batch.
  -- This is not actually the enforced behaviour, one thread may fill the buffer with multiple batches,
  -- but it should be better than a buffer size of 1 in this multithreaded case.
  (toTransformBox, fromTransformBox, sealTransform) <- spawn' (bounded numWorkers)
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded numWorkers)
  forM_ [1..numWorkers] $ \workerId -> async $ runEffect $ forever $ readBatchesConcurrently workerId dataset toTransformBox
  async $ runEffect $ runTransforms transforms fromTransformBox toBatches
  pure $ foldFromProducer (takeBatch fromBatches)
  
foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
foldFromProducer prod fold = (L.impurely P.foldM) fold prod

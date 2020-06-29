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
                           , makeFold'
                           , makeFoldWithTransform'
                           , makeConcurrentFold'
                           , makeFold
                           , makeFoldWithTransform
                           , makeConcurrentFold
                           , foldFromProducer
                           , L.FoldM(..)
                           , Dataset(..)
                           , DatasetMock(..)
                           , ConcurrentDataset(..)
                           ) where

import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (isJust)
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P

import           Control.Monad.Trans.Control (MonadBaseControl(..))


type Iter = Int
type WorkerId = Int

data DatasetMock m tensor = DatasetMock { getBatchMock :: Int -> m tensor
                                        , numItersMock :: Iter
                                        }

class (MonadPlus m, MonadIO m, MonadBaseControl IO m) => Dataset m dataset batch  where
  getBatch :: dataset -> Iter -> m batch
  numIters :: dataset -> Int

class Dataset m dataset batch => ConcurrentDataset m dataset batch where
  getBatchConcurrently :: WorkerId -> dataset -> Iter -> m batch

takeBatch :: MonadIO m => Input (Maybe batch) -> Producer batch m ()  
takeBatch input = fromInput input >-> P.takeWhile isJust >-> yieldMore
  where yieldMore = forever $ await >>= \case
          Just batch -> yield batch
          Nothing -> return ()

readBatchesConcurrently :: forall dataset batch m .
  (ConcurrentDataset m dataset batch) => Int -> dataset -> Output (Maybe batch)  -> Effect m () 
readBatchesConcurrently workerId dataset transformBox = 
  for (each [1..numIters @m @dataset @batch dataset + 1]) (\iter -> yieldBatch iter >-> toOutput transformBox)
    where yieldBatch iter =
             runBatch iter >>= yield
          runBatch iter = if numIters @m @dataset @batch dataset + 1 == iter then pure Nothing else Just <$> getBatch iter
          getBatch iter = lift $ getBatchConcurrently workerId dataset iter

readBatches :: forall dataset batch m.
  (Dataset m dataset batch) => dataset -> Output (Maybe batch)  -> Effect m () 
readBatches dataset outputBox = 
  for (each [1..numIters @m @dataset @batch dataset + 1]) (\iter -> yieldBatch iter >-> toOutput outputBox)
    where yieldBatch iter = runBatch iter >>= yield
          runBatch iter = if numIters @m @dataset @batch dataset + 1 == iter then pure Nothing else Just <$> batch iter
          batch iter = lift $ getBatch dataset iter

runTransforms :: MonadIO m => (batch -> batch') -> Input (Maybe batch) -> Output (Maybe batch') -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput transformBox >->  P.map (fmap transforms) >-> toOutput trainBox

makeFold' :: (Dataset m2 dataset batch, MonadIO m, MonadIO m2)
    => dataset
    -> m2 (L.FoldM m batch b -> m b, Async (StM m2 ()))
makeFold' dataset = do
  (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
  batchThread <- async $ void $ runEffect $ readBatches dataset toBatches
  pure (foldFromProducer (takeBatch fromBatches), batchThread)


makeConcurrentFold' :: (MonadIO m2, ConcurrentDataset m2 dataset batch', MonadIO m)
  => (batch' -> batch)
  -> dataset
  -> Int
  -> m2 (L.FoldM m batch b -> m b, [Async (StM m2 ())])
makeConcurrentFold' transforms dataset numWorkers = do
  -- Buffer size is equal to numWorkers so that each thread can yield a batch.
  -- This is not actually the enforced behaviour, one thread may fill the buffer with multiple batches,
  -- but it should be better than a buffer size of 1 in this multithreaded case.
  (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded numWorkers)
  (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded numWorkers)
  batchThreads <- forM [1..numWorkers] $ \workerId -> async $ void $ runEffect $ readBatchesConcurrently workerId dataset toTransformBox
  async $ runEffect $ runTransforms transforms fromTransformBox toBatches
  pure  $ (foldFromProducer (takeBatch fromBatches), batchThreads)

  
makeFoldWithTransform' :: (MonadIO m, MonadIO m2, Dataset m2 dataset batch)  
  => (batch -> batch')
  -> dataset
  -> m2 (L.FoldM m batch' b -> m b, Async (StM m2 ()))
makeFoldWithTransform' transforms dataset = do
          -- TODO: we can allow different buffer sizes
          -- which would be necessary for data echoing
            (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded 1)
            (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
            batchThread <- async $ void $ runEffect $ forever $ readBatches dataset toTransformBox 
            async $ runEffect $ runTransforms transforms fromTransformBox toBatches
            pure $ (foldFromProducer (takeBatch fromBatches), batchThread)

makeFold :: (Dataset m2 dataset batch, MonadIO m, MonadIO m2)
    => dataset
    -> m2 (L.FoldM m batch b -> m b)
makeFold = fmap fst . makeFold' 

makeFoldWithTransform :: (MonadIO m, MonadIO m2, Dataset m2 dataset batch)  
  => (batch -> batch')
  -> dataset
  -> m2 (L.FoldM m batch' b -> m b)
makeFoldWithTransform transf = fmap fst . makeFoldWithTransform' transf 

makeConcurrentFold :: (MonadIO m2, MonadIO m, ConcurrentDataset m2 dataset batch')
  => (batch' -> batch)
  -> dataset
  -> Int
  -> m2 (L.FoldM m batch b -> m b)
makeConcurrentFold transforms dataset = fmap fst . makeConcurrentFold' transforms dataset
  
foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
foldFromProducer prod fold = (L.impurely P.foldM) fold prod

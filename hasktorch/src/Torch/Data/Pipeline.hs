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
module Torch.Data.Pipeline where

import Pipes
import qualified Pipes.Prelude as P
import Pipes.Concurrent
import Control.Concurrent.Async
import Control.Monad
import Control.Arrow (first)


type Iter = Int
type WorkerId = Int

data DatasetMock m tensor = DatasetMock { getBatchMock :: Int -> m tensor
                                        , numItersMock :: Iter
                                        }

class Dataset dataset batch  where
  getBatch :: dataset -> Iter -> IO batch
  numIters :: dataset -> Int

class Dataset dataset batch => ConcurrentDataset dataset batch where
  getBatchConcur :: WorkerId -> dataset -> Iter -> IO batch

data RunBatch = Final | KeepTrain  deriving (Eq, Show)

takeBatch :: MonadIO m => Input (batch, RunBatch) -> Producer batch m ()  
takeBatch input = fromInput input >-> P.takeWhile ((/=) Final . snd) >-> P.map fst

readBatchesConcurrently :: forall dataset batch m .
  (MonadIO m, Dataset dataset batch, ConcurrentDataset dataset batch) => Int -> dataset -> Output (batch, RunBatch)  -> Effect m () 
readBatchesConcurrently workerId dataset transformBox = 
  for (each [1..numIters @dataset @batch dataset ]) (\iter -> yieldBatch iter >-> toOutput transformBox)
    where yieldBatch iter =
             (liftIO $ getBatchConcur workerId dataset iter) >>= yield . (, runBatch iter)
          runBatch iter = if numIters @dataset @batch dataset == iter then Final else KeepTrain

readBatches :: forall dataset batch m.
  (MonadIO m, Dataset dataset batch) => dataset -> Output (batch, RunBatch)  -> Effect m () 
readBatches dataset outputBox = 
  for (each [1..numIters @dataset @batch dataset]) (\iter -> yieldBatch iter >-> toOutput outputBox)
    where yieldBatch iter = (liftIO $ getBatch dataset iter) >>= yield . (, runBatch iter)
          runBatch iter = if numIters @dataset @batch dataset == iter then Final else KeepTrain


runTransforms :: MonadIO m => (tensor -> tensor') -> Input (tensor, RunBatch) -> Output (tensor', RunBatch) -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput transformBox >->  P.map (first transforms) >-> toOutput trainBox

makeFoldWithTransform :: _
  => (batch -> batch')
  -> dataset
  -> IO ((b -> batch' -> m b) -> b -> m b) 
makeFoldWithTransform transforms dataset = do
          -- TODO: we can allow different buffer sizes
          -- which would be necessary for data echoing
            (toTransformBox, fromTransformBox, sealTransform) <- spawn' (bounded 1)
            (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
            async $ runEffect $ forever $ readBatches dataset toTransformBox 
            async $ runEffect $ runTransforms transforms fromTransformBox toBatches

            pure (\foldFn initial -> P.foldM foldFn (pure initial) pure (takeBatch fromBatches))

makeFold :: (Dataset dataset batch, _)
    => dataset
    -> IO ((b -> batch -> m b) -> b -> m b)
makeFold dataset = do
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
  async $ runEffect $ forever $ readBatches dataset toBatches
  pure (\foldFn initial -> P.foldM foldFn (pure initial) pure (takeBatch fromBatches))

makeConcurrentFold :: (ConcurrentDataset dataset batch, _)
  => (batch -> batch')
  -> dataset
  -> Int -> IO ((b -> batch' -> m b) -> b -> m b)
makeConcurrentFold transforms dataset numWorkers = do
  (toTransformBox, fromTransformBox, sealTransform) <- spawn' (bounded 1)
  (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
  forM_ [1..numWorkers] $ \workerId -> async $ runEffect $ readBatchesConcurrently workerId dataset toTransformBox
  async $ runEffect $ runTransforms transforms fromTransformBox toBatches
  pure (\foldFn initial -> P.foldM foldFn (pure initial) pure (takeBatch fromBatches))
  

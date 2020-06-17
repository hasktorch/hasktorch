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
                                        , numIters :: Iter
                                        }

class ConcurrentDataset dataset numWorkers where
  getBatchConcur :: Iter -> WorkerId -> IO tensor

class Dataset dataset tensor where
  getBatch :: Iter -> IO tensor

data RunBatch = Final | KeepTrain  deriving (Eq, Show)

takeBatch :: MonadIO m => Input (batch, RunBatch) -> Producer batch m ()  
takeBatch input = fromInput input >-> P.takeWhile ((/=) Final . snd) >-> P.map fst

readBatches ::
  MonadIO m => DatasetMock m (tensor, RunBatch) -> Output (tensor, RunBatch)  -> Effect m () 
readBatches DatasetMock{..} transformBox = do
  -- TODO: might want to pair the RunBatch value in this function
  for (each [1..numIters]) (\iter -> yieldBatch iter >-> toOutput transformBox)
    where yieldBatch iter = (lift $ getBatchMock iter) >>= yield

runTransforms :: MonadIO m => (tensor -> tensor') -> Input (tensor, RunBatch) -> Output (tensor', RunBatch) -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput transformBox >->  P.map (first transforms) >-> toOutput trainBox

makeFoldWithTransform ::
  _ => (batch -> batch')
  -> DatasetMock IO (batch, RunBatch)
  -> Int
  -> IO ((b -> batch' -> m b) -> b -> m b) 
makeFoldWithTransform transforms dataset numEpochs = createTrainLoop
  where createTrainLoop  =  do
          -- TODO: we can allow different buffer sizes
          -- which would be necessary for data echoing
            (toTransformBox, fromTransformBox, sealTransform) <- spawn' (bounded 1)
            (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
            batchReader <-  async $ do runEffect $ replicateM_ numEpochs $ readBatches dataset toTransformBox 
            transformer <-  async $ do runEffect $ runTransforms transforms fromTransformBox toBatches
                                       atomically sealTransform
                                       atomically sealBatch
            pure (\foldFn initial -> P.foldM foldFn (pure initial) pure (takeBatch fromBatches))

makeFold :: 
    _ => DatasetMock IO (batch, RunBatch)
    -> Int
    -> IO ((b -> batch -> m b) -> b -> m b)
makeFold dataset numEpochs = createTrainLoop
  where createTrainLoop  =  do
            (toBatches, fromBatches, sealBatch) <- spawn' (bounded 1)
            batchReader <-  async $ do runEffect $ replicateM_ numEpochs $ readBatches dataset toBatches
                                       atomically sealBatch

            pure (\foldFn initial -> P.foldM foldFn (pure initial) pure (takeBatch fromBatches))

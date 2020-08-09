{-# LANGUAGE FunctionalDependencies #-}
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
module Torch.Data.Pipeline (
  -- takeBatch
                           -- , readBatches
                           -- , makeFoldWithTransform'
                           -- , makeFold
                           -- , makeFoldWithTransform
                           -- , L.FoldM(..)
                            Dataset(..)
                           , Sampler(..)
                           , readBatches
                           ) where

import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import Control.Applicative (Alternative, (<|>))
import           Data.Maybe (isJust)
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P

import           Control.Monad.Trans.Control (MonadBaseControl(..))
import Control.Monad.Cont (ContT)
import Torch.Data.Internal

type Iter = Int
-- type WorkerId = Int

-- data DatasetMock m tensor = DatasetMock { getBatchMock :: Int -> m tensor
--                                         , numItersMock :: Iter
--                                         }

data MapStyleOptions = MapStyleOptions {
  -- dropLast :: Bool
                                        bufferSize :: Int
                                       , numWorkers :: Int
                                       }
class Dataset dataset batch | dataset -> batch  where
  getItem :: dataset -> Int -> batch
  size :: dataset -> Int

-- class Sampler (sampler :: * -> *) where
--   generateIx :: Int -> Int

type Sampler m = (Int, Int) -> Producer Int m ()
type CollateFn sample batch m = Pipe sample batch m ()

readBatches :: forall sampler dataset sample batch m b. (Dataset dataset sample, MonadBaseControl IO m) =>
  MapStyleOptions -> dataset -> Sampler m -> CollateFn sample batch m -> ContT b m (ListT m (batch, Int))
readBatches MapStyleOptions{..} dataset sampler collateFn = runWithBuffer bufferSize streamBatches
  where
    streamBatches output = forConcurrently_ [0..numWorkers - 1]  (runWorker output)
    -- runWorker output workerId = runEffect $ for (each $ indices workerId) $ yieldItem output
    runWorker output = runEffect . yieldItem output
    -- yieldItem output ix = yield (getItem dataset (generateIx @sampler ix)) >-> collateFn >->  toOutput' output
    yieldItem output workerId = sampler (indices workerId) >-> pipeItems  >-> collateFn >->  toOutput' output
    pipeItems = for cat (yield . getItem dataset)
    -- indices workerId = [workerId * size dataset `div` numWorkers .. (workerId + 1) * size dataset `div` numWorkers]
    indices workerId = (workerId * size dataset `div` numWorkers, (workerId + 1) * size dataset `div` numWorkers)
  

-- class Dataset m dataset batch => ConcurrentDataset m dataset batch where
--   getBatchConcurrently :: WorkerId -> dataset -> Iter -> m batch

-- takeBatch :: MonadIO m => Input (Maybe batch) -> Producer batch m ()  
-- takeBatch input = fromInput input >-> P.takeWhile isJust >-> yieldMore
--   where yieldMore = forever $ await >>= \case
--           Just batch -> yield batch
--           Nothing -> return ()
-- readBatches'
--   :: (MonadIO m)
--   => Int
--   -> (Int -> m (Maybe batch))
--   -> Output (Maybe batch)
--   -> Effect m ()
-- readBatches' numIters getBatch outputBox = 
--   for (each [0..numIters]) (\iter -> yieldBatch iter >-> toOutput outputBox)
--     where yieldBatch iter = do
            -- this is a workaround to using MonadPlus in the Proxy monad, since
            -- it doesn't have an instance. Instead we implement failure with
            -- MonadPlus in the getBatch function, and have it yield Maybe values
            -- which marks failure here
          --   thing <- lift $ runBatch iter
          --   case thing of
          --     Nothing -> yield Nothing
          --     Just (Just batch) -> yield (Just batch)
          --     Just Nothing -> return ()
          -- runBatch iter = if numIters == iter
          --                 then pure Nothing
          --                 else Just <$> getBatch iter

-- readBatchesConcurrently :: forall dataset batch m .
--   (MonadPlus m, ConcurrentDataset m dataset batch) => Int -> dataset -> Output (Maybe batch)  -> Effect m () 
-- readBatchesConcurrently workerId dataset outputBox = readBatches' iter getBatchOrFail outputBox
--   where iter = numIters @m @dataset @batch dataset
--         getBatchOrFail = (\iter -> (Just <$> (getBatchConcurrently workerId dataset iter)) <|> (pure Nothing))

-- readBatches :: forall dataset batch m.
--   (Dataset dataset batch) => dataset -> Output (Maybe batch)  -> Effect m () 
-- readBatches dataset outputBox = readBatches' iter getBatchOrFail outputBox
--   where iter = (iters  dataset)
--         getBatchOrFail = (\iter -> (Just <$> getBatch dataset iter) <|> (pure Nothing))

-- runTransforms :: MonadIO m => (batch -> batch') -> Input (Maybe batch) -> Output (Maybe batch') -> Effect m ()
-- runTransforms transforms transformBox trainBox = fromInput transformBox >->  P.map (fmap transforms) >-> toOutput trainBox

-- makeFold' :: (Dataset dataset batch, MonadIO m)
--     => dataset
--     -> m (L.FoldM m batch b -> m b, Async (StM m ()))
-- makeFold' dataset = do
--   (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
--   batchThread <- async $ void $ runEffect $ readBatches dataset toBatches
--   pure (foldFromProducer (takeBatch fromBatches), batchThread)


-- makeConcurrentFold' :: (MonadIO m, ConcurrentDataset m dataset batch')
--   => (batch' -> batch)
--   -> dataset
--   -> Int
--   -> m (L.FoldM m batch b -> m b, [Async (StM m ())])
-- makeConcurrentFold' transforms dataset numWorkers = do
--   -- Buffer size is equal to numWorkers so that each thread can yield a batch.
--   -- This is not actually the enforced behaviour, one thread may fill the buffer with multiple batches,
--   -- but it should be better than a buffer size of 1 in this multithreaded case.
--   (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded numWorkers)
--   (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded numWorkers)
--   batchThreads <- forM [1..numWorkers] $ \workerId -> async $ void $ runEffect $ readBatchesConcurrently workerId dataset toTransformBox
--   async $ runEffect $ runTransforms transforms fromTransformBox toBatches
--   pure  $ (foldFromProducer (takeBatch fromBatches), batchThreads)

  
-- makeFoldWithTransform' :: (MonadIO m, Dataset  dataset batch)  
--   => (batch -> batch')
--   -> dataset
--   -> m (L.FoldM m batch' b -> m b, Async (StM m ()))
-- makeFoldWithTransform' transforms dataset = do
--           -- TODO: we can allow different buffer sizes
--           -- which would be necessary for data echoing
--             (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded 1)
--             (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
--             batchThread <- async $ void $ runEffect $ forever $ readBatches dataset toTransformBox 
--             async $ runEffect $ runTransforms transforms fromTransformBox toBatches
--             pure $ (foldFromProducer (takeBatch fromBatches), batchThread)

-- makeFold :: (Dataset  dataset batch, MonadIO m)
--     => dataset
--     -> m (L.FoldM m batch b -> m b)
-- makeFold = fmap fst . makeFold' 

-- makeFoldWithTransform :: (MonadIO m, Dataset  dataset batch)  
--   => (batch -> batch')
--   -> dataset
--   -> m (L.FoldM m batch' b -> m b)
-- makeFoldWithTransform transf = fmap fst . makeFoldWithTransform' transf 

-- makeConcurrentFold :: (MonadIO m, ConcurrentDataset m dataset batch')
--   => (batch' -> batch)
--   -> dataset
--   -> Int
--   -> m (L.FoldM m batch b -> m b)
-- makeConcurrentFold transforms dataset = fmap fst . makeConcurrentFold' transforms dataset
  
-- foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
-- foldFromProducer prod fold = (L.impurely P.foldM) fold prod


-- sequentialSampler = 

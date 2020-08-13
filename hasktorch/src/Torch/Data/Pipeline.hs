{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}
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
                           , makeListT
                           , mapStyleOpts
                           , runContT
                           ) where

import           Control.Applicative (Alternative, (<|>))
import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (fromJust, isJust)
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P

import           Control.Monad.Trans.Control (MonadBaseControl(..))
import           Control.Monad.Cont (runContT, ContT)
import           Torch.Data.Internal
import qualified Torch as Torch
import Lens.Family (view)
import Pipes.Group
import Data.Vector (Vector)
import qualified Data.Map.Strict as M
import Data.Set
import Control.Concurrent.STM
import qualified Data.Vector as V
import Control.Monad.Base (MonadBase)

data MapStyleOptions = MapStyleOptions { bufferSize :: Int
                                       , numWorkers :: Int
                                       }

mapStyleOpts numWorkers = MapStyleOptions { bufferSize = numWorkers
                                          , numWorkers = numWorkers
                                          }



  
class (Ord k) => Dataset m dataset k sample | dataset -> sample, dataset -> k  where
  getItem :: dataset -> k -> m sample
  keys :: dataset -> Set k

---------------------- Workflow --------------------
-- - make a new map of keys to TVars of samples, possibly shuffled keys, tracking which keys have been sampled
-- - create a TQueue of keys (using pipes-concurrency wrapper)
-- - fork off workers which pull from all pull from the TQueue and sample that key,
--   then update the map of TVars with the sample
-- have a worker waiting for each successive key to be updated in the map of TVars

makeListT :: forall m dataset k sample r .
  (MonadIO m, MonadBaseControl IO m,  Dataset m dataset k sample) =>
  MapStyleOptions ->
  dataset ->
  ContT r m  (ListT m (sample, Int))
makeListT MapStyleOptions{..} dataset = do
  (keyOutput, keyInput, seal) <- liftIO $ spawn' unbounded
  -- TODO optionally permute set before creating queue and TVar map
  (runEffect $ keyQueue keyOutput $ keys @m dataset) 
  liftIO $ atomically seal
  tvars <- liftIO $ newTVarMap (keys @m dataset) 
  lift $ runWorkers numWorkers dataset tvars keyInput
  runWithBuffer bufferSize $ (awaitNextItem tvars)

runWorkers ::
  (Dataset m dataset k sample, MonadIO m, MonadBaseControl IO m) =>
  Int ->
  dataset ->
  M.Map k (TVar (Maybe sample)) ->
  Input k -> 
  m ()
runWorkers numWorkers dataset tvars keyInput = replicateConcurrently_ numWorkers (runEffect $ fromInput' keyInput >-> runWorker)
    where 
      runWorker = forever $ do key <- await
                               -- liftIO $ print key
                               let tvar = fromJust $ M.lookup key tvars 
                               item <- lift $ getItem dataset key
                               liftIO $ atomically $ writeTVar tvar (Just item) 
awaitNextItem ::
  (MonadBase IO m, MonadIO m) =>
  M.Map k (TVar (Maybe sample)) ->
  Output sample ->
  m ()
awaitNextItem tvars output  = runEffect $ each (M.toList tvars) >-> readNextItem >-> toOutput' output
  where readNextItem  = forever $ do
          (key, tvar) <- await
          item <- liftIO $ atomically $ do
            val <- readTVar tvar 
            case val of
              Nothing -> retry
              Just item -> writeTVar tvar Nothing >> pure item -- reset the tvar once we get the sample out of it to save memory
          yield item 

newTVarMap :: Set k -> IO (M.Map k (TVar (Maybe sample)))
newTVarMap = atomically . sequence . M.fromSet (\_ -> newTVar Nothing)

keyQueue keyOutput set = each (toList set) >-> toOutput' keyOutput

  


-- makeListT :: forall k sampler dataset sample batch m b. (Dataset k dataset sample, MonadBaseControl IO m) =>
--   MapStyleOptions -> dataset -> Sampler k m -> ContT b m (ListT m (batch, Int))
-- makeListT MapStyleOptions{..} dataset sampler = runWithBuffer bufferSize streamBatches
--   where
--     streamBatches output = forConcurrently_ [0..numWorkers - 1]  (runWorker output)
--     runWorker output = runEffect . yieldItem output
--     -- yieldItem output workerId = collateFn (sampler (indices workerId) (size dataset) >-> pipeItems) >->  toOutput' output
--     yieldItem output workerId = sampler (indices workerId) (size dataset) >-> pipeItems >->  toOutput' output
--     pipeItems = for cat (yield . getItem dataset)
--     -- indices workerId = (workerId * (size dataset `div` numWorkers), (workerId + 1) * (size dataset `div` numWorkers) - 1)

-- -- | A sampler takes the range of indexes it needs to produce,
-- -- and the size of the dataset it is processing, and returns
-- -- indexes to process. The range of indexes is determined by
-- -- the number of workers, so with 1 worker the range is the entire dataset,
-- -- with 2 each workers range is half the size of the dataset.
-- -- type Sampler k m = (Int, Int) -> Int -> Producer Int m ()
-- -- type Sampler k m = (Int, Int) -> Int -> Producer Int m ()
-- type Sampler k m = Int -> Producer k m ()

-- -- | A collation pipe that batches a sequence of processed samples.
-- type CollateFn sample batch m = Producer sample m () -> Producer batch m ()

-- vectorBatch :: Monad m => Int -> CollateFn sample (Vector sample) m
-- vectorBatch batchSize =  L.purely folds L.vector . view (chunksOf batchSize)

-- sequentialSampler :: Functor m => Sampler k m 
-- sequentialSampler _ = each [low .. high]

-- randomSampler :: MonadIO m => Int -> Sampler k m 
-- randomSampler seed size =
--   for (each [0..size]) $ \_ -> (liftIO $ (Torch.toInt <$> Torch.randintIO' 0 size [])) >>= yield 


--  fold the set, and build a set for each worker!
-- splitSet = L.fold

-- toWorker = workerId % numWorkers



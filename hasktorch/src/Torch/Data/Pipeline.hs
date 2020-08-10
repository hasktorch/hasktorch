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
                           , makeListT
                           , mapStyleOpts
                           , sequentialSampler
                           ) where

import           Control.Applicative (Alternative, (<|>))
import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (isJust)
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P

import           Control.Monad.Trans.Control (MonadBaseControl(..))
import           Control.Monad.Cont (ContT)
import           Torch.Data.Internal
import qualified Torch as Torch
import Lens.Family (view)
import Pipes.Group
import Data.Vector (Vector)

data MapStyleOptions = MapStyleOptions { bufferSize :: Int
                                       , numWorkers :: Int
                                       }

mapStyleOpts numWorkers = MapStyleOptions { bufferSize = numWorkers
                                          , numWorkers = numWorkers
                                          }

class Dataset dataset sample | dataset -> sample  where
  getItem :: dataset -> Int -> sample
  size :: dataset -> Int

-- | A sampler takes the range of indexes it needs to produce,
-- and the size of the dataset it is processing, and returns
-- indexes to process. The range of indexes is determined by
-- the number of workers, so with 1 worker the range is the entire dataset,
-- with 2 each workers range is half the size of the dataset.
type Sampler m = (Int, Int) -> Int -> Producer Int m ()

-- | A collation pipe that batches a sequence of processed samples.
type CollateFn sample batch m = Producer sample m () -> Producer batch m ()

makeListT :: forall sampler dataset sample batch m b. (Dataset dataset sample, MonadBaseControl IO m) =>
  MapStyleOptions -> dataset -> Sampler m -> CollateFn sample batch m -> ContT b m (ListT m (batch, Int))
makeListT MapStyleOptions{..} dataset sampler collateFn = runWithBuffer bufferSize streamBatches
  where
    streamBatches output = forConcurrently_ [0..numWorkers - 1]  (runWorker output)
    runWorker output = runEffect . yieldItem output
    -- yieldItem output workerId = sampler (indices workerId) (size dataset) >-> pipeItems  >-> collateFn >->  toOutput' output
    yieldItem output workerId = collateFn (sampler (indices workerId) (size dataset) >-> pipeItems) >->  toOutput' output
    pipeItems = for cat (yield . getItem dataset)
    indices workerId = (workerId * (size dataset `div` numWorkers), (workerId + 1) * (size dataset `div` numWorkers) - 1)

vectorBatch :: Monad m => Int -> CollateFn sample (Vector sample) m
vectorBatch batchSize =  L.purely folds L.vector . view (chunksOf batchSize)

sequentialSampler :: Functor m => Sampler m 
sequentialSampler (low, high) _ = each [low .. high]

randomSampler :: MonadIO m => Sampler m 
randomSampler (low, high) size = for (each [low..high]) $ \_ -> (liftIO $ (Torch.toInt <$> Torch.randintIO' 0 size [])) >>= yield 

-- (workerId + 1) * ((size dataset `div` numWorkers) - 1)

--   2 * 

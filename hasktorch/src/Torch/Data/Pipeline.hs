{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeOperators #-}

module Torch.Data.Pipeline
  ( -- * Defining a Dataset
    -- $dataset

    -- * Dataset
    Dataset (..),
    DatasetOptions (..),
    datasetOpts,
    Sample (..),

    -- * Dataloading
    streamFromMap,
  )
where

import Control.Concurrent.Async.Lifted
import Control.Concurrent.STM hiding (atomically)
import Control.Monad
import Control.Monad.Base (MonadBase)
import Control.Monad.Cont (ContT)
import Control.Monad.Trans.Control (MonadBaseControl (..))
import Data.IntMap (IntMap)
import qualified Data.IntMap as I
import Data.Set
import Pipes
import Pipes.Concurrent hiding (atomically)
import System.Random
import Torch.Data.Internal

-- $dataset
-- See the 'Torch.Vision' module which implements the MNIST dataset for a good example of how to define a dataset.

-- | The base dataset class. A dataset is capable of returning a sample
-- for a given key, and every 'Dataset' has a known set of keys.
class (Ord k) => Dataset m dataset k sample | dataset -> m, dataset -> sample, dataset -> k where
  getItem :: dataset -> k -> m sample
  keys :: dataset -> Set k

-- | Dataset options used when loading datasets. Specify shuffling behavior, the number of
-- threads to use, and the buffer size used to store retrieved samples in each thread.
data DatasetOptions = DatasetOptions
  { -- | Max number of samples stored in each buffer at a given time.
    dataBufferSize :: Int,
    -- | Number of threads retrieving samples.
    numWorkers :: Int,
    -- | The ordering of samples streamed.
    shuffle :: Sample
  }

-- | Default 'DatasetOptions'. The 'Int' parameter specifies the
-- number of workers, and sets the buffer size equal to the number of workers.
-- Sampling is sequential.
datasetOpts :: Int -> DatasetOptions
datasetOpts numWorkers =
  DatasetOptions
    { dataBufferSize = numWorkers,
      numWorkers = numWorkers,
      shuffle = Sequential
    }

-- | A 'Sample' determines the ordering of samples streamed out of a dataset.
-- You can either order sequentially, or supply a random generator to shuffle samples.
data Sample where
  Sequential :: Sample
  Shuffle :: RandomGen g => g -> Sample

---------------------- Workflow --------------------
-- - make a new map of keys to TVars of samples, possibly shuffled keys, tracking which keys have been sampled
-- - create a TQueue of keys (using pipes-concurrency wrapper)
-- - fork off workers which all pull from the TQueue and sample that key using the dataset,
--   then update the TVar associated with that key
-- have a worker waiting for each successive key to be updated in the list of (key, TVar)

-- | Return a stream of samples from the given dataset, along with a new 'Sample' value.
-- The returned stream contains every sample returned by @'getItem'@ for every key in the set of keys
-- associated with the given dataset. The returned 'Sample' value returns an updated 'Sample' value,
-- this will be identical to the original 'Sample' value if sampling is 'Sequential' but will return a new random number generator
-- if sampling is 'Shuffle'.
streamFromMap ::
  forall m dataset k sample r.
  (Dataset m dataset k sample, MonadIO m, MonadBaseControl IO m) =>
  DatasetOptions ->
  dataset ->
  ContT r m (ListT m sample, Sample)
streamFromMap DatasetOptions {..} dataset = do
  (keyOutput, keyInput, seal) <- liftIO $ spawn' unbounded

  let retrieveSet = liftIO $ keyTVarSet $ keys dataset
  (keyTVarSet, updatedSample) <- case shuffle of
    Sequential -> (,Sequential) <$> retrieveSet
    Shuffle g -> fmap Shuffle . fisherYates g <$> retrieveSet

  -- fill the queue with each key and associated TVar then seal it
  keyQueue keyOutput keyTVarSet
  liftIO $ atomically seal

  let workers = runWorkers numWorkers dataset keyInput
      datastream = awaitNextItem keyTVarSet
  listT <- runWithBuffer dataBufferSize $ \output -> concurrently_ workers (datastream output)
  pure (listT, updatedSample)

runWorkers ::
  (Dataset m dataset k sample, MonadIO m, MonadBaseControl IO m) =>
  Int ->
  dataset ->
  Input (k, TVar (Maybe sample)) ->
  m ()
runWorkers numWorkers dataset keyInput = replicateConcurrently_ numWorkers (runEffect $ fromInput' keyInput >-> runWorker)
  where
    runWorker = forever $ do
      (key, tvar) <- await
      item <- lift $ getItem dataset key
      atomically $ writeTVar tvar (Just item)

awaitNextItem ::
  (MonadBase IO m, MonadIO m) =>
  [(k, TVar (Maybe sample))] ->
  Output sample ->
  m ()
awaitNextItem tvars output = runEffect $ each tvars >-> readNextItem >-> toOutput' output
  where
    readNextItem = forever $ do
      (_, tvar) <- await
      item <- atomically $ do
        val <- readTVar tvar
        case val of
          Nothing -> retry
          Just item -> writeTVar tvar Nothing >> pure item -- reset the tvar once we get the sample out of it to save memory
      yield item

keyTVarSet :: MonadIO m => Set k -> m [(k, TVar (Maybe sample))]
keyTVarSet = atomically . mapM (\k -> (,) k <$> newTVar Nothing) . toList

keyQueue :: MonadBase IO m => Output (k, TVar (Maybe sample)) -> [(k, TVar (Maybe sample))] -> m ()
keyQueue keyOutput keyTVarSet = runEffect $ each keyTVarSet >-> toOutput' keyOutput

fisherYatesStep :: RandomGen g => (IntMap a, g) -> (Int, a) -> (IntMap a, g)
fisherYatesStep (m, gen) (i, x) = ((I.insert j x . I.insert i (m I.! j)) m, gen')
  where
    (j, gen') = randomR (0, i) gen

fisherYates :: RandomGen g => g -> [a] -> ([a], g)
fisherYates gen [] = ([], gen)
fisherYates gen l =
  toElems $ Prelude.foldl fisherYatesStep (initial (head l) gen) (numerate (tail l))
  where
    toElems (x, y) = (I.elems x, y)
    numerate = zip [1 ..]
    initial x gen = (I.singleton 0 x, gen)

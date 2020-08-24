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
module Torch.Data.Pipeline ( Dataset(..)
                           , DatasetOptions(..)
                           , Sample(..)
                           , streamFromMap
                           , datasetOpts
                           ) where

import           Control.Applicative (Alternative, (<|>))
import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (fromJust, isJust)
import           Pipes
import           Pipes.Concurrent hiding (atomically)
import qualified Pipes.Prelude as P

import           Control.Arrow (Arrow(first))
import           Control.Concurrent.STM hiding (atomically)
import           Control.Monad.Base (MonadBase)
import           Control.Monad.Cont (runContT, ContT)
import           Control.Monad.Trans.Control (MonadBaseControl(..))
import           Data.IntMap (IntMap)
import qualified Data.IntMap as I
import qualified Data.Map.Strict as M
import           Data.Set
import           Data.Vector (Vector)
import qualified Data.Vector as V
import           Lens.Family (view)
import           Pipes.Group
import           System.Random
import           Torch.Data.Internal

data DatasetOptions = DatasetOptions { bufferSize :: Int
                               , numWorkers :: Int
                               , shuffle :: Sample
                               }

datasetOpts numWorkers = DatasetOptions { bufferSize = numWorkers
                                         , numWorkers = numWorkers
                                         , shuffle = Sequential
                                         }

data Sample where
   Sequential :: Sample
   Shuffle ::  RandomGen g => g -> Sample 

class (Ord k) => Dataset m dataset k sample | dataset -> m, dataset -> sample, dataset -> k  where
  getItem :: dataset -> k -> m sample
  keys :: dataset -> Set k

---------------------- Workflow --------------------
-- - make a new map of keys to TVars of samples, possibly shuffled keys, tracking which keys have been sampled
-- - create a TQueue of keys (using pipes-concurrency wrapper)
-- - fork off workers which all pull from the TQueue and sample that key using the dataset,
--   then update the TVar associated with that key
-- have a worker waiting for each successive key to be updated in the list of (key, TVar)

streamFromMap :: forall m dataset k sample r .
  (Dataset m dataset k sample, MonadIO m, MonadBaseControl IO m) =>
  DatasetOptions ->
  dataset ->
  ContT r m  (ListT m sample, Sample)
streamFromMap DatasetOptions{..} dataset = do
  (keyOutput, keyInput, seal) <- liftIO $ spawn' unbounded

  let retrieveSet = liftIO $ keyTVarSet $ keys dataset
  (keyTVarSet, updatedSample) <- case shuffle of
    Sequential -> (, Sequential) <$> retrieveSet
    Shuffle g -> (fmap Shuffle  . fisherYates g) <$> retrieveSet

  -- fill the queue with each key and associated TVar then seal it
  keyQueue keyOutput $ keyTVarSet
  liftIO $ atomically seal

  let workers = runWorkers numWorkers dataset keyInput
      datastream = awaitNextItem keyTVarSet
  listT <- runWithBuffer bufferSize $ \output -> concurrently_ workers (datastream output)
  pure (listT, updatedSample)


runWorkers ::
  (Dataset m dataset k sample, MonadIO m, MonadBaseControl IO m) =>
  Int ->
  dataset ->
  Input (k, TVar (Maybe sample)) ->
  m ()
runWorkers numWorkers dataset keyInput = replicateConcurrently_ numWorkers (runEffect $ fromInput' keyInput >-> runWorker)
    where 
      runWorker = forever $ do (key, tvar) <- await
                               item <- lift $ getItem dataset key
                               atomically $ writeTVar tvar (Just item) 
awaitNextItem ::
  (MonadBase IO m, MonadIO m) =>
  [(k, TVar (Maybe sample))] ->
  Output sample ->
  m ()
awaitNextItem tvars output  = runEffect $ each tvars >-> readNextItem >-> toOutput' output
  where readNextItem  = forever $ do
          (key, tvar) <- await
          item <- atomically $ do
            val <- readTVar tvar 
            case val of
              Nothing -> retry
              Just item -> writeTVar tvar Nothing >> pure item -- reset the tvar once we get the sample out of it to save memory
          yield item 

keyTVarSet :: MonadIO m => Set k -> m [(k, TVar (Maybe sample))]
keyTVarSet = atomically . mapM (\k -> newTVar Nothing >>= pure . (,) k)  . toList 

keyQueue :: MonadBase IO m => Output (k, TVar (Maybe sample)) ->  [(k, TVar (Maybe sample))] -> m ()
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
    numerate = zip [1..]
    initial x gen = (I.singleton 0 x, gen)


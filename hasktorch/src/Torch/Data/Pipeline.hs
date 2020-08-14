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
                           , MapStyleOptions(..)
                           , Sample(..)
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
import System.Random
import qualified Data.IntMap as I
import Data.IntMap (IntMap)
import Control.Arrow (Arrow(first))

data MapStyleOptions = MapStyleOptions { bufferSize :: Int
                                       , numWorkers :: Int
                                       , shuffle :: Sample
                                       }

mapStyleOpts numWorkers = MapStyleOptions { bufferSize = numWorkers
                                          , numWorkers = numWorkers
                                          , shuffle = Sequential
                                          }

data Sample = Sequential | Shuffle StdGen

class (Ord k) => Dataset m dataset k sample | dataset -> sample, dataset -> k  where
  getItem :: dataset -> k -> m sample
  keys :: dataset -> Set k

---------------------- Workflow --------------------
-- - make a new map of keys to TVars of samples, possibly shuffled keys, tracking which keys have been sampled
-- - create a TQueue of keys (using pipes-concurrency wrapper)
-- - fork off workers which all pull from the TQueue and sample that key using the dataset,
--   then update the TVar associated with that key
-- have a worker waiting for each successive key to be updated in the list of (key, TVar)

makeListT :: forall m dataset k sample r .
  (MonadIO m, MonadBaseControl IO m,  Dataset m dataset k sample) =>
  MapStyleOptions ->
  dataset ->
  ContT r m  (ListT m (sample, Int))
makeListT MapStyleOptions{..} dataset = do
  (keyOutput, keyInput, seal) <- liftIO $ spawn' unbounded

  let retreiveSet = liftIO $ keyTVarSet $ keys @m dataset
  keyTVarSet <- case shuffle of
    Sequential -> retreiveSet
    Shuffle g -> fst . fisherYates g <$> retreiveSet
   
  -- fill the queue with each key and associated TVar then seal it
  keyQueue keyOutput $ keyTVarSet
  liftIO $ atomically seal

  let workers = runWorkers numWorkers dataset keyInput
      datastream = awaitNextItem keyTVarSet
  runWithBuffer bufferSize $ \output -> concurrently_ workers (datastream output)

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
                               liftIO $ atomically $ writeTVar tvar (Just item) 
awaitNextItem ::
  (MonadBase IO m, MonadIO m) =>
  [(k, TVar (Maybe sample))] ->
  Output sample ->
  m ()
awaitNextItem tvars output  = runEffect $ each tvars >-> readNextItem >-> toOutput' output
  where readNextItem  = forever $ do
          (key, tvar) <- await
          item <- liftIO $ atomically $ do
            val <- readTVar tvar 
            case val of
              Nothing -> retry
              Just item -> writeTVar tvar Nothing >> pure item -- reset the tvar once we get the sample out of it to save memory
          yield item 

keyTVarSet :: Set k -> IO [(k, TVar (Maybe sample))]
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


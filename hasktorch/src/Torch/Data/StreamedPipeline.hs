{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Data.StreamedPipeline
  (
    runMockData,
    makeListT,
    makeListT',
    pmap,
    pmap',
    dataloaderOpts,
    Datastream (..),
    MonadBase (..),
    MonadBaseControl (..),
    module Pipes,
    module Pipes.Safe
  )
where

import           Control.Arrow (second)
import           Control.Concurrent.Async.Lifted
import           Control.Concurrent.STM hiding (atomically)
import           Control.Exception.Safe (bracket, finally)
import           Control.Foldl (Fold, FoldM (FoldM))
import qualified Control.Foldl as L
import           Control.Monad
import           Control.Monad.Base (MonadBase, liftBase)
import           Control.Monad.Cont (ContT (..), runContT)
import           Control.Monad.Trans.Control
import           Data.Maybe (isJust)
import qualified Data.Vector as V
import           Lens.Family
import           Pipes
import           Pipes.Concurrent hiding (atomically)
import qualified Pipes.Prelude as P
import           Pipes.Safe (MonadSafe (Base))
import qualified Pipes.Safe as Safe
import           Torch.Data.Internal

class Monad m => Datastream m seed dataset sample | dataset -> sample where
  streamBatch :: dataset -> seed -> ListT m sample 

data DataloaderOptions = DataloaderOptions
  { bufferSize :: Int -- ^ Number of inputs stored in a buffer.
  }

-- | Default dataloader options, you should override the fields in this record.
dataloaderOpts = DataloaderOptions { bufferSize = 4 } -- 4 is relatively arbitrary

-- | Run a parallel map over the given ListT (TODO: with the given number of workers)
pmap :: (MonadIO m, MonadBaseControl IO m) => Int -> (a -> b) -> ListT m a -> ContT r m (ListT m b)
pmap n f prod = ContT $ \cont ->
  snd
    <$> withBufferLifted
      (bounded n)
      (\output -> runEffect $ enumerate prod >-> P.map f >-> toOutput output)
      (\input -> cont . Select $ fromInput input)

-- | Run a parallel pipe over the given ListT. (TODO: with the given number of workers)
pmap' :: (MonadIO m, MonadBaseControl IO m) => Int -> Pipe a b m () ->  ListT m a -> ContT r m (ListT m b)
pmap' n f  prod = ContT $ \cont ->
  snd
    <$> withBufferLifted
      (bounded n)
      (\output -> runEffect $ enumerate prod >-> f >-> toOutput output)
      (\input -> cont . Select $ fromInput input)


makeListT ::
  forall sample m dataset seed b r.
  (Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m) =>
  DataloaderOptions -> 
  dataset ->
  ListT m seed ->
  ContT b m (ListT m (sample, Int))
makeListT DataloaderOptions{..} dataset seeds = runWithBuffer bufferSize $ readSamples dataset seeds

-- makeListT' ::
--   forall sample m f dataset seed b.
--   (Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m, Foldable f) =>
--   DataloaderOptions -> 
--   dataset ->
--   f seed ->
--   ContT b m (ListT m (sample, Int))
-- makeListT' DataloaderOptions{..} dataset seeds = runWithBuffer bufferSize $ readSamples' dataset seeds

readSamples ::
  forall m seed dataset sample.
  (Datastream m seed dataset sample, MonadBaseControl IO m) =>
  dataset ->
  ListT m seed ->
  Output sample ->
  m ()
readSamples dataset seeds outputBox =
  let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset @sample dataset
   in join . P.fold this mempty runConcurrently $ enumerate seeds

-- readSamples' ::
--   forall m seed f dataset sample.
--   (Datastream m seed dataset sample, MonadBaseControl IO m, Foldable f) =>
--   dataset ->
--   f seed ->
--   Output sample ->
--   m ()
-- readSamples' dataset seeds outputBox =
--   let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset @sample dataset
--    in L.fold (L.Fold this mempty runConcurrently) seeds

readSamplesDeterministic :: 
  forall m seed f dataset sample.
  (Datastream m seed dataset sample, MonadBaseControl IO m ,MonadIO m, Foldable f) =>
  dataset ->
  f (seed, Output sample) -> 
  m ()
readSamplesDeterministic dataset seeds = 
  let this c (seed, outputBox) =
        mappend c . Concurrently . runEffect .  (>-> toOutput' outputBox) . enumerate $ streamBatch @m @seed @dataset @sample dataset seed
   in L.fold (L.Fold this mempty runConcurrently) seeds
  
makeListT' ::
  forall sample m f dataset seed b.
  (Show sample, Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m, MonadIO m, Foldable f) =>
  DataloaderOptions -> 
  dataset ->
  f seed ->
  ContT b m (ListT m (sample, Int))
makeListT' DataloaderOptions{..} dataset seeds = do
  workerTracker <- atomically $ newTVar 0
  let
      consumeSeeds mailboxes o = do
        for (each mailboxes) $ \(output, input, _) -> fromInputOnce workerTracker input >->  toOutput' o
        keepReading <- lift $ atomically $ (\x -> x < V.length mailboxes) <$> readTVar workerTracker
        when keepReading $ consumeSeeds mailboxes o
  runWithBuffer bufferSize $ \o -> do
    liftedBracket 
      (L.foldM pairSeedWithBuffer seeds)
      (mapM_ (atomically . third . snd))
      (\a ->
         let mailboxes = snd <$> a
             seedAndOutput = second fst3 <$> a
         in concurrently_
            (readSamplesDeterministic dataset seedAndOutput `liftedFinally` (mapM_ (atomically . third) mailboxes))
            ((runEffect $ consumeSeeds mailboxes o) `liftedFinally` (mapM_ (atomically . third) mailboxes))
      )
  where
    fst3 (a,b,c) = a
    snd3 (a,b,c) = b
    third (a,b,c) = c

pairSeedWithBuffer :: MonadIO m => FoldM m seed (V.Vector (seed, (Output a, Input a, STM ()) ) ) 
pairSeedWithBuffer = L.premapM (\a -> (a, ) <$> makeMailbox) $ L.generalize L.vector
  where makeMailbox = liftIO $ spawn' (bounded 1)

fromInputOnce workerTracker input = do
  ma <- atomically $ recv input
  case ma of
    Nothing -> do
      atomically $ readTVar workerTracker >>= writeTVar workerTracker. (+) 1
      return ()
    Just a  -> do
      yield a
      return ()


data MockData = MockData 
instance Datastream IO Int MockData Int where
  streamBatch _ seed = Select $ P.replicateM 100 (pure seed)

runMockData :: IO ()
runMockData = runContT (makeListT' dataloaderOpts MockData [1 :: Int,2,3,4,5])
  (\x -> runEffect $ enumerate x >-> P.chain (liftIO . print) >-> P.drain)


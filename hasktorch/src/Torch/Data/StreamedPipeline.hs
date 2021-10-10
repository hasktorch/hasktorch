{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Data.StreamedPipeline
  ( -- * Defining a Datastream
    -- $dataset

    -- * Datastream
    Datastream (..),
    DatastreamOptions (..),
    datastreamOpts,

    -- * Dataloading
    streamFrom,
    streamFrom',

    -- * Reexports
    MonadBase (..),
    MonadBaseControl (..),
  )
where

import Control.Arrow (second)
import Control.Concurrent.Async.Lifted
import Control.Concurrent.STM hiding (atomically)
import Control.Foldl (FoldM)
import qualified Control.Foldl as L
import Control.Monad
import Control.Monad.Base (MonadBase, liftBase)
import Control.Monad.Cont (ContT (..))
import Control.Monad.Trans.Control
import qualified Data.Vector as V
import Pipes
import Pipes.Concurrent hiding (atomically)
import qualified Pipes.Prelude as P
import Torch.Data.Internal

-- $dataset
-- We will show how to retrieve the IMDB dataset as an example datastream.
-- The dataset used here can be found at https://ai.stanford.edu/~amaas/data/sentiment/
--
-- > import Pipes
-- > import qualified Pipes.Safe as Safe
-- > import qualified Pipes.Prelude as P
-- > import System.Directory
-- >
-- > newtype Imdb = Imdb { dataDir :: String }
-- >
-- > data Sentiment = Positive | Negative
-- >
-- > instance (MonadBaseControl IO m, MonadSafe m) => Datastream m Sentiment Imdb (Text, Sentiment) where
-- >   streamSamples Imdb{..} sent = Select $ do
-- >     rawFilePaths <- zip (repeat sent) <$> (liftIO $ listDirectory (dataDir </> sentToPath sent))
-- >     let filePaths = fmap (second $ mappend (dataDir </> sentToPath sent)) rawFilePaths
-- >     for (each filePaths) $ \(rev, fp) -> Safe.withFile fp ReadMode $ \fh ->
-- >       P.zip (PT.fromHandleLn fh) (yield rev)
-- >         where sentToPath Pos = "pos" ++ pure pathSeparator
-- >               sentToPath Neg = "neg" ++ pure pathSeparator
--
-- This streams in movie reviews from each file in either the positive review directory or
-- the negative review directory, depending on the seed value used.
--
-- This highlights a use of seed values that is more interesting than just specifying the thread count, but also has some problems.
-- When running this datastream with either 'streamFrom' or 'streamFrom\'', you need to supply both 'Positive' and 'Negative' values as seeds
-- to retrieve the entire IMDB dataset, and in this case positive and negative reviews will be streamed in concurrently.
-- The problem with designing a datastream in this fashion is you limit the amount of concurrency (2 threads in this case) without
-- duplicating data. Ultimately though seeds should be quite flexible and allow you to design the concurrency how you see fit. Be careful
-- not to use duplicate seed values unless you want duplicate data.

-- | The base datastream class. A dataset returns a stream of samples
-- based on a seed value.
class Monad m => Datastream m seed dataset sample | dataset -> sample where
  streamSamples :: dataset -> seed -> ListT m sample

-- | Datastream options used when looding datastreams. Currently only buffer size is configurable,
-- since thread count is controlled by the number of seeds (see @'streamFrom'@ functions).
newtype DatastreamOptions = DatastreamOptions
  { -- | Max number of samples stored in each buffer at a given time.
    bufferSize :: Int
  }

-- | Default dataloader options, you should override the fields in this record.
datastreamOpts :: DatastreamOptions
datastreamOpts = DatastreamOptions {bufferSize = 4} -- 4 is relatively arbitrary

-- | Return a stream of samples from the given dataset as a continuation.
-- A stream of samples is generated for every seed in the given stream of seeds, and all of these streams are merged
-- into the output stream in a non-deterministic order (if you need determinism see 'streamFrom\'').
-- Every stream created for each seed value is made in its own thread.
streamFrom ::
  forall sample m dataset seed b.
  (Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m) =>
  DatastreamOptions ->
  dataset ->
  ListT m seed ->
  ContT b m (ListT m sample)
streamFrom DatastreamOptions {..} dataset seeds = runWithBuffer bufferSize $ readSamples dataset seeds

-- | This function is the same as 'streamFrom' except the seeds are specified as
-- a 'Foldable', and the stream returned has a deterministic ordering. The results
-- from each given seed are interspersed in the order defined by the @'Foldable'@ of seeds.
streamFrom' ::
  forall sample m f dataset seed b.
  (Show sample, Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m, MonadIO m, Foldable f) =>
  DatastreamOptions ->
  dataset ->
  f seed ->
  ContT b m (ListT m sample)
streamFrom' DatastreamOptions {..} dataset seeds = do
  workerTracker <- atomically $ newTVar 0
  let consumeSeeds mailboxes o = do
        for (each mailboxes) $ \(_, input, _) -> fromInputOnce workerTracker input >-> toOutput' o
        keepReading <- lift $ atomically $ (\x -> x < V.length mailboxes) <$> readTVar workerTracker
        when keepReading $ consumeSeeds mailboxes o
  runWithBuffer bufferSize $ \o ->
    liftedBracket
      (L.foldM pairSeedWithBuffer seeds)
      (mapM_ (atomically . third . snd))
      ( \a ->
          let mailboxes = snd <$> a
              seedAndOutput = second fst3 <$> a
           in concurrently_
                (readSamplesDeterministic dataset seedAndOutput `liftedFinally` mapM_ (atomically . third) mailboxes)
                (runEffect (consumeSeeds mailboxes o) `liftedFinally` mapM_ (atomically . third) mailboxes)
      )
  where
    fst3 (a, _, _) = a
    third (_, _, c) = c

readSamples ::
  forall m seed dataset sample.
  (Datastream m seed dataset sample, MonadBaseControl IO m) =>
  dataset ->
  ListT m seed ->
  Output sample ->
  m ()
readSamples dataset seeds outputBox =
  let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamSamples @m @seed @dataset @sample dataset
   in join . P.fold this mempty runConcurrently $ enumerate seeds

readSamplesDeterministic ::
  forall m seed f dataset sample.
  (Datastream m seed dataset sample, MonadBaseControl IO m, MonadIO m, Foldable f) =>
  dataset ->
  f (seed, Output sample) ->
  m ()
readSamplesDeterministic dataset seeds =
  let this c (seed, outputBox) =
        mappend c . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate $ streamSamples @m @seed @dataset @sample dataset seed
   in L.fold (L.Fold this mempty runConcurrently) seeds

pairSeedWithBuffer :: MonadIO m => FoldM m seed (V.Vector (seed, (Output a, Input a, STM ())))
pairSeedWithBuffer = L.premapM (\a -> (a,) <$> makeMailbox) $ L.generalize L.vector
  where
    makeMailbox = liftIO $ spawn' (bounded 1)

fromInputOnce :: MonadIO m => TVar Int -> Input a -> Producer a m ()
fromInputOnce workerTracker input = do
  ma <- atomically $ recv input
  case ma of
    Nothing -> do
      atomically $ readTVar workerTracker >>= writeTVar workerTracker . (+) 1
      return ()
    Just a -> do
      yield a
      return ()

{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Data.StreamedPipeline
  ( makeListT,
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

import           Control.Concurrent.Async.Lifted
import           Control.Exception.Safe (bracket, finally)
import           Control.Foldl (Fold, FoldM (FoldM))
import qualified Control.Foldl as L
import           Control.Monad
import           Control.Monad.Base (MonadBase, liftBase)
import           Control.Monad.Cont (ContT (..), runContT)
import           Control.Monad.Trans.Control
import           Data.Maybe (isJust)
import           Lens.Family
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P
import           Pipes.Safe (MonadSafe (Base))
import qualified Pipes.Safe as Safe
import Torch.Data.Internal


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

makeListT' ::
  forall sample m f dataset seed b.
  (Datastream m seed dataset sample, MonadBaseControl IO m, MonadBase IO m, Foldable f) =>
  DataloaderOptions -> 
  dataset ->
  f seed ->
  ContT b m (ListT m (sample, Int))
makeListT' DataloaderOptions{..} dataset seeds = runWithBuffer bufferSize $ readSamples' dataset seeds

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

readSamples' ::
  forall m seed f dataset sample.
  (Datastream m seed dataset sample, MonadBaseControl IO m, Foldable f) =>
  dataset ->
  f seed ->
  Output sample ->
  m ()
readSamples' dataset seeds outputBox =
  let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset @sample dataset
   in L.fold (L.Fold this mempty runConcurrently) seeds


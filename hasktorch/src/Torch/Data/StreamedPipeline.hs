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
    Datastream (..),
    MonadBase (..),
    MonadBaseControl (..),
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
import           Torch.Typed


class (MonadBase IO m) => Datastream m seed dataset batch | dataset -> batch where
  streamBatch :: dataset -> seed -> ListT m batch

-- TODO : incorporate these options
data DataloaderOpts = DataloaderOpts
  { echoData :: Bool,
    bufferSize :: Int
  }

dataloaderOpts = DataloaderOpts {echoData = False, bufferSize = 4} -- 4 is relatively arbitrary

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
  forall batch m dataset seed b r.
  (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m) =>
  dataset ->
  ListT m seed ->
  ContT b m (ListT m (batch, Int))
makeListT dataset seeds = ContT $ \f ->
  snd
    <$> withBufferLifted
      (bounded 10)
      (\batchOutput -> readBatches dataset seeds batchOutput)
      (\input -> f . Select $ P.zip (fromInput' input) iters)
  where
    iters :: Producer Int m ()
    iters = each [0 ..]

makeListT' ::
  forall batch m f dataset seed b.
  (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m, Foldable f) =>
  dataset ->
  f seed ->
  ContT b m (ListT m (batch, Int))
makeListT' dataset seeds = ContT $ \f ->
  snd
    <$> withBufferLifted
      (bounded 10)
      (\batchOutput -> readBatches' dataset seeds batchOutput)
      (\input -> f . Select $ P.zip (fromInput' input) iters)
  where
    iters :: Producer Int m ()
    iters = each [0 ..]

readBatches ::
  forall m seed dataset batch.
  (Datastream m seed dataset batch, MonadBaseControl IO m) =>
  dataset ->
  ListT m seed ->
  Output batch ->
  m ()
readBatches dataset seeds outputBox =
  let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset @batch dataset
   in join . P.fold this mempty runConcurrently $ enumerate seeds

readBatches' ::
  forall m seed f dataset batch.
  (Datastream m seed dataset batch, MonadBaseControl IO m, Foldable f) =>
  dataset ->
  f seed ->
  Output batch ->
  m ()
readBatches' dataset seeds outputBox =
  let this = flip $ mappend . Concurrently . runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset @batch dataset
   in L.fold (L.Fold this mempty runConcurrently) seeds

-- foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
-- foldFromProducer prod fold = (L.impurely P.foldM) fold prod

liftedBracket :: MonadBaseControl IO m => m a -> (a -> m b) -> (a -> m c) -> m c
liftedBracket acquire release action = control $ \runInIO ->
  bracket
    (runInIO acquire)
    (\saved -> runInIO (restoreM saved >>= release))
    (\saved -> runInIO (restoreM saved >>= action))

withBufferLifted ::
  (MonadBaseControl IO m) =>
  Buffer a ->
  (Output a -> m l) ->
  (Input a -> m r) ->
  m (l, r)
withBufferLifted buffer fOutput fInput =
  liftedBracket
    (liftBase $ spawn' buffer)
    (\(_, _, seal) -> liftBase $ atomically seal)
    ( \(output, input, seal) ->
        concurrently
          (fOutput output `liftedFinally` (liftBase $ atomically seal))
          (fInput input `liftedFinally` (liftBase $ atomically seal))
    )

fromInput' :: (MonadBase IO m) => Input a -> Producer' a m ()
fromInput' input = loop
  where
    loop = do
      ma <- liftBase $ atomically $ recv input
      case ma of
        Nothing -> return ()
        Just a -> do
          yield a
          loop

toOutput' :: (MonadBase IO m) => Output a -> Consumer' a m ()
toOutput' output = loop
  where
    loop = do
      a <- await
      alive <- liftBase $ atomically $ send output a
      when alive loop

liftedFinally :: MonadBaseControl IO m => m a -> m b -> m a
liftedFinally a sequel = control $ \runInIO ->
  finally
    (runInIO a)
    (runInIO sequel)

instance (MonadBase IO m) => MonadBase IO (Proxy a' a b' b m) where
  liftBase = lift . liftBase

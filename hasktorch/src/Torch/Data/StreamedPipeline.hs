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
    makeListTCont,
    foldOverWith,
    foldOverWith',
    pmap,
    pmap',
    pmapCont,
    Datastream (..),
    -- defaultDataloaderOpts,
    MonadBase (..),
    MonadBaseControl (..),
  )
where

import Control.Applicative (Alternative, (<|>))
-- import           Control.Monad.Trans.Control (MonadBaseControl(..), control)

import Control.Concurrent.Async.Lifted
import Control.Exception.Safe (MonadMask, bracket, finally, bracketOnError)
import Control.Foldl (Fold, FoldM (FoldM))
import qualified Control.Foldl as L
import Control.Monad
import Control.Monad.Base (MonadBase, liftBase)
import Control.Monad.Trans.Control
import Data.Maybe (isJust)
import Lens.Family
import Pipes
import Pipes.Concurrent
import qualified Pipes.Prelude as P
import Pipes.Safe (MonadSafe (Base))
import qualified Pipes.Safe as Safe
import Torch.Typed
import Control.Monad.Cont (ContT(..), runContT )

-- data DataloaderOpts = DataloaderOpts {numWorkers :: Int}

class (MonadPlus m, MonadBase IO m) => Datastream m seed dataset batch where
  streamBatch :: dataset -> seed -> ListT m batch

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

pmapCont :: (MonadIO m, MonadBaseControl IO m) =>  (a -> b) -> Int -> ListT m a -> ContT r m (ListT m b)
pmapCont f n prod = ContT $ \cont -> snd <$> withBufferLifted
  (bounded 10)
  (\output -> runEffect $ enumerate prod >-> P.map f >-> toOutput  output)
  (\input -> cont . Select $ fromInput input )



pmap :: (Monad m, MonadBaseControl IO m) =>  (a -> b) -> Int -> ListT m a -> m (ListT m b)
pmap f n prod  = Select <$> withBufferInput unbounded unwrap
                 (\input -> withBufferInput unbounded (mapConcur input)
                   (\input2 -> pure $ fromInput' input2)
                 )
  where
    unwrap output = runEffect $ enumerate prod >-> toOutput' output
    mapConcur input output = replicateConcurrently_ n $ runEffect $ fromInput' input >-> P.map f >-> toOutput' output

-- can we get rid of this m?
pmap' :: (Monad m, MonadBaseControl IO m) =>  (ListT m a -> ListT m b) -> Int -> ListT m a -> m (ListT m b)
pmap' func n prod  = withBufferInput unbounded unwrap (\input -> pure $ func $ (Select $ fromInput' input))
  where unwrap output = runEffect $ enumerate prod >-> toOutput' output

streamBatches fold inputBox = foldFromProducer inputs fold
  where
    inputs = fromInput' inputBox

foldOverWith' ::
  forall m dataset seed batch f b.
  (Datastream m seed dataset batch, MonadBaseControl IO m, Foldable f) =>
  dataset ->
  f seed ->
  FoldM m (batch, Int) b ->
  m b
foldOverWith' dataset seeds fold = join $ fmap (flip foldFromProducer fold . enumerate) $ makeListT' dataset seeds

foldOverWith ::
  forall m dataset seed batch b.
  (Datastream m seed dataset batch, MonadBaseControl IO m) =>
  -- DataloaderOpts ->
  dataset ->
  ListT m seed ->
  FoldM m (batch, Int) b ->
  m b
foldOverWith  dataset seeds fold = do
  join $ fmap (flip foldFromProducer fold . enumerate) $ makeListT dataset seeds


makeListTCont ::
  forall batch m dataset seed b r.
  (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m) =>
  dataset ->
  ListT m seed ->
  ContT b m (ListT m (batch, Int))
makeListTCont  dataset seeds  = ContT $ \f ->  snd <$> withBufferLifted
      (bounded 10) 
      (\batchOutput -> readBatches dataset seeds batchOutput)
      (\input ->  f . Select $ P.zip (fromInput' input) iters )
    where
      iters ::  Producer Int m ()
      iters = each [0..]

makeListT ::
  forall batch m dataset seed b.
  (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m) =>
  -- DataloaderOpts ->
  dataset ->
  ListT m seed ->
  m (ListT m (batch, Int))
makeListT  dataset seeds = do
  Select <$>
    withBufferInput
      (bounded 10) -- FIXME : what bound
      (\batchOutput -> readBatches dataset seeds batchOutput)
      (\input -> pure $ P.zip (fromInput' input) iters)
    where
      iters ::  Producer Int m ()
      iters = each [0..]

makeListT' ::
  forall batch m f dataset seed b.
  (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m, Foldable f) =>
  dataset ->
  f seed ->
  m (ListT m (batch, Int))
makeListT'  dataset seeds = do
  Select <$>
    withBufferInput
      (bounded 10)
      (\batchOutput ->  readBatches' dataset seeds batchOutput)
      (\input -> pure $ P.zip (fromInput' input) iters)
    where
      iters ::  Producer Int m ()
      iters = each [0..]


foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
foldFromProducer prod fold = (L.impurely P.foldM) fold prod

liftedBracket :: MonadBaseControl IO m => m a -> (a -> m b) -> (a -> m c) -> m c
liftedBracket acquire release action = control $ \runInIO ->
  bracket
    (runInIO acquire)
    (\saved -> runInIO (restoreM saved >>= release))
    (\saved -> runInIO (restoreM saved >>= action))

liftedBracketOnError :: MonadBaseControl IO m => m a -> (a -> m b) -> (a -> m c) -> m c
liftedBracketOnError acquire release action = control $ \runInIO ->
  bracketOnError
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

-- We need to cleanup exception handling here more, but withBuffer is too restrictive
-- for how we use input and output as it ends up sealing too quickly
-- | Forks a thread for the output producing function and returns
-- | the value of the input consuming function.  
withBufferInput ::
  (MonadBaseControl IO m) =>
  Buffer a ->
  (Output a -> m l) ->
  (Input a -> m r) ->
  m r
withBufferInput buffer fOutput fInput =
  liftedBracketOnError
    (liftBase $ spawn' buffer)
    ( \(_, _, seal) -> do liftBase $ atomically seal
    )
    ( \(output, input, seal) -> do
        async (fOutput output `liftedFinally` (liftBase $ atomically seal))
        fInput input
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


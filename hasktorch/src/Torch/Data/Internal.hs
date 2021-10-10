{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Data.Internal where

import Control.Concurrent.Async.Lifted (concurrently)
import qualified Control.Concurrent.STM as STM
import Control.Exception.Safe (bracket, finally)
import Control.Monad (when)
import Control.Monad.Base (MonadBase (..))
import Control.Monad.Cont (ContT (ContT))
import Control.Monad.Trans.Control
import Pipes
import Pipes.Concurrent hiding (atomically)
import qualified Pipes.Prelude as P

runWithBuffer ::
  forall a m b.
  (MonadBaseControl IO m) =>
  Int ->
  (Output a -> m ()) ->
  -- ContT b m (ListT m (a, Int))
  ContT b m (ListT m a)
runWithBuffer bufferSize batchHandler = ContT $ \f ->
  snd
    <$> withBufferLifted
      (bounded bufferSize)
      (\batchOutput -> batchHandler batchOutput)
      -- (\input -> f . Select $ P.zip (fromInput' input) iters)
      (\input -> f . Select $ fromInput' input)

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

atomically :: MonadIO m => STM a -> m a
atomically = liftIO . STM.atomically

instance (MonadBase IO m) => MonadBase IO (Proxy a' a b' b m) where
  liftBase = lift . liftBase

---- make a runData function which just does runContT but zips that
---- the listT with the iteration! This is much better

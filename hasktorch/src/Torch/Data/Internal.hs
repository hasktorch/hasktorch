{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Data.Internal where

import Control.Monad.Base (MonadBase(..))
import Control.Monad.Trans.Control 
import Pipes
import Pipes.Concurrent
import Control.Exception.Safe (finally, bracket)
import Control.Concurrent.Async.Lifted (concurrently)
import Control.Monad (when)
import qualified Pipes.Prelude as P
import Control.Monad.Cont (ContT(ContT))

runWithBuffer :: forall batch m b .
  (MonadBaseControl IO m) =>
  Int ->
  (Output batch -> m ()) ->
  ContT b m (ListT m (batch, Int))
runWithBuffer bufferSize batchHandler = ContT $ \f ->
  snd
    <$> withBufferLifted
      (bounded bufferSize)
      (\batchOutput -> batchHandler batchOutput)
      (\input -> f . Select $ P.zip (fromInput' input) iters)
  where
    iters :: Producer Int m ()
    iters = each [0..]

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

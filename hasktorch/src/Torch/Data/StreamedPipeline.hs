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
module Torch.Data.StreamedPipeline ( makeListT
                                   , makeListT'
                                   , foldOverWith
                                   , foldOverWith'
                                   , Datastream(..)
                                   , defaultDataloaderOpts
                                   , MonadBase(..)
                                   ) where



import           Control.Applicative (Alternative, (<|>))
import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (isJust)
import           Pipes
import           Pipes.Concurrent
-- import           Pipes.Group
import qualified Pipes.Prelude as P
import           Torch.Typed

import           Lens.Family
import           Control.Monad.Trans.Control (MonadBaseControl(..), control)
import           Control.Monad.Base (MonadBase, liftBase)

import           Control.Foldl (Fold, FoldM(FoldM))
import           Control.Concurrent.Async.Lifted
import           Control.Exception.Safe (MonadMask, finally, bracket)
import qualified Pipes.Safe as Safe
import Pipes.Safe (MonadSafe(Base))

data DataloaderOpts = DataloaderOpts { numWorkers :: Int }

class (MonadPlus m, MonadBase IO m) => Datastream m seed dataset batch  where
  streamBatch :: dataset -> seed -> ListT m batch 

readBatches :: forall m seed dataset batch .
  (Datastream m seed dataset batch, MonadBaseControl IO m)
  => dataset
  -> ListT m seed 
  -> Output batch
  -> m ()
readBatches dataset seeds outputBox =
  let this = flip $ mappend . Concurrently .  runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset  @batch dataset
  in join . P.fold this mempty runConcurrently $ enumerate seeds

readBatches' :: forall m seed f dataset batch .
  (Datastream m seed dataset batch, MonadBaseControl IO m, Foldable f)
  => dataset
  -> f seed 
  -> Output batch
  -> m ()
readBatches' dataset seeds outputBox =
  let this = flip $ mappend . Concurrently .  runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @seed @dataset  @batch dataset
  in L.fold (L.Fold this mempty runConcurrently) seeds 

runTransforms :: MonadBase IO m => (batch -> batch') -> Input (batch) -> Output (batch') -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput' transformBox >-> P.map transforms >-> toOutput' trainBox

-- pMap p f n =
  
--   where firstBuffer = withBufferLifted unbounded (\output -> enumerate p >-> toOutput' output) (\input -> replicateConcurrently_ n $ runEffect $ fromInput' input  >>= yield . f >-> toOutput outputBox )
--         secondBuffer = 

   -- >->  do
  -- val <- await 
  -- trans <- lift $ liftBase $ forkIO $ f val 
  -- yield val
  

streamBatches fold inputBox = foldFromProducer inputs fold
  where inputs = fromInput' inputBox
                          
  -- NOTE: TODO: these transformation functions aren't really that functional in spirit
  -- it would be nicer to be able to just specify a parallel transformation over the ListT produced by makeListT
  
foldOverWith' :: forall m dataset seed batch' batch b. (Datastream m seed dataset batch, MonadBaseControl IO m)
  => dataset
  -> ListT m seed
  -> FoldM m batch b
  -> m b
foldOverWith' = foldOverWith @m @dataset @seed defaultDataloaderOpts

foldOverWith :: forall m dataset seed batch' batch b. (Datastream m seed dataset batch, MonadBaseControl IO m)
  => DataloaderOpts
  -> dataset
  -> ListT m seed
  -> FoldM m batch b
  -> m b
foldOverWith opts@DataloaderOpts{..} dataset seeds fold = do
  join $ fmap  (flip foldFromProducer fold . enumerate) $ makeListT @m @dataset @seed opts dataset id seeds

makeListT :: forall m dataset seed batch' batch b. (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m)
  => DataloaderOpts
  -> dataset
  -> (batch' -> batch) 
  -> ListT m seed
  -> m (ListT m batch)
makeListT DataloaderOpts{..} dataset transforms seeds = do
   -- fmap (Select . snd) $ withBufferLifted (bounded numWorkers)
   --   (\batchOutput -> readBatches @m @seed dataset seeds batchOutput) (\input -> pure $ fromInput' input)
   fmap Select $ withOne (bounded numWorkers)
     (\batchOutput -> readBatches @m @seed dataset seeds batchOutput) (\input -> pure $ fromInput' input)

makeListT' :: forall m f dataset seed batch' batch b. (Datastream m seed dataset batch, MonadBaseControl IO m, MonadBase IO m, Foldable f)
  => DataloaderOpts
  -> dataset
  -> (batch' -> batch) 
  -> f seed
  -> m (ListT m batch)
makeListT' DataloaderOpts{..} dataset transforms seeds = do
   fmap (Select . snd) $ withBufferLifted (bounded numWorkers)
     (\batchOutput -> readBatches' @m @seed dataset seeds batchOutput) (\input -> (pure $ fromInput' input ))

defaultDataloaderOpts = DataloaderOpts { numWorkers = 1 }


foldFromProducer :: Monad m => Producer batch m () -> L.FoldM m batch b -> m b
foldFromProducer prod fold = (L.impurely P.foldM) fold prod

liftedBracket :: MonadBaseControl IO m => m a -> (a -> m b) -> (a -> m c) -> m c
liftedBracket acquire release action = control $ \runInIO ->
    bracket (runInIO acquire)
            (\saved -> runInIO (restoreM saved >>= release))
            (\saved -> runInIO (restoreM saved >>= action))
  
withBufferLifted 
    :: (MonadBaseControl IO m)
    => Buffer a
    -> (Output a -> m l)
    -> (Input  a -> m r)
    -> m (l, r)
withBufferLifted buffer fOutput fInput = liftedBracket
  (liftBase $ spawn' buffer)
  (\(_, _, seal) -> liftBase $ atomically seal)
  (\(output, input, seal) ->
    concurrently
      (fOutput output `liftedFinally` (liftBase $ atomically seal))
      (fInput  input  `liftedFinally` (liftBase $ atomically seal))
  )

-- We need to cleanup exception handling here more, but withBuffer is too restrictive
-- for how we use input and output as it ends up sealing too quickly
withOne 
    :: (MonadBaseControl IO m)
    => Buffer a
    -> (Output a -> m l)
    -> (Input a -> m r)
    -> m r
withOne buffer fOutput fInput = liftedBracket
  (liftBase $ spawn' buffer)
  (\(_, _, seal) -> do liftBase $ print "here"
  )
  (\(output, input, seal) -> do
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
            Just a  -> do
                yield a
                loop
toOutput' :: (MonadBase IO m) => Output a -> Consumer' a m ()
toOutput' output = loop
  where
    loop = do
        a     <- await
        alive <- liftBase $ atomically $ send output a
        when alive loop

liftedFinally :: MonadBaseControl IO m => m a -> m b -> m a
liftedFinally a sequel = control $ \runInIO ->
                           finally (runInIO a)
                                   (runInIO sequel)
instance (MonadBase IO m) => MonadBase IO (Proxy a' a b' b m) where
  liftBase = lift . liftBase


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
module Torch.Data.StreamedPipeline where



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

type Iter = Int
type WorkerId = Int

data DataloaderOpts = DataloaderOpts { numWorkers :: Int }

class (MonadPlus m, MonadBase IO m) => Datastream m dataset batch  where
  streamBatch :: dataset -> seed -> ListT m batch 

readBatches' :: forall m seed dataset batch .
  (Datastream m dataset batch, MonadBaseControl IO m)
  => dataset
  -> ListT m seed 
  -> Output batch
  ->  m ()
readBatches' dataset seeds outputBox =
  let this = flip $ mappend . Concurrently .  runEffect . (>-> toOutput' outputBox) . enumerate . streamBatch @m @dataset  @batch dataset
  in join . P.fold this mempty runConcurrently $ enumerate seeds

runTransforms :: MonadBase IO m => (batch -> batch') -> Input (batch) -> Output (batch') -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput' transformBox >-> P.map transforms >-> toOutput' trainBox

streamBatches fold inputBox = foldFromProducer inputs fold
  where inputs = fromInput' inputBox
                          
foldOverWith' :: forall m dataset seed batch' batch b. (Datastream m dataset batch', MonadBaseControl IO m)
  => dataset
  -> (batch' -> batch) 
  -- -> seed
  -> FoldM m batch b
  -> m b
foldOverWith' = foldOverWith defaultDataloaderOpts

foldOverWith :: forall m dataset seed batch' batch b. (Datastream m dataset batch', MonadBaseControl IO m)
  => DataloaderOpts
  -> dataset
  -> (batch' -> batch) 
  -- -> seed
  -> FoldM m batch b
  -> m b
foldOverWith DataloaderOpts{..} dataset transforms fold = do
   fmap snd $ withBufferLifted
    (bounded numWorkers)
    (\batchOutput -> withBufferLifted (bounded numWorkers)
      (\transformOutput ->  readBatches' dataset seeds transformOutput)
      (\transformInput -> replicateConcurrently_ numWorkers $ runEffect $ runTransforms transforms transformInput batchOutput))
    (\input -> streamBatches fold input)

defaultDataloaderOpts = DataloaderOpts { numWorkers = 1 }

seeds :: ListT m (Concurrently m ())
seeds = error "not implemented"


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


{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Data.Utils ( pmap
                        , pmap'
                        , pmapGroup
                        , enumerateData
                        , CachedDataset
                        , cache
                        ) where

import           Control.Monad.Trans.Control
import           Control.Monad.Cont
import           Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as M
import qualified Data.Set as S
import Lens.Family
import qualified Control.Foldl as L
import           Pipes
import Pipes.Group

import qualified Pipes.Prelude as P
import           Pipes.Concurrent

import           Torch.Data.Internal
import           Torch.Data.Pipeline

-- | Run a parallel map over the given stream. 
pmap :: (MonadIO m, MonadBaseControl IO m) => (a -> b) -> ListT m a -> ContT r m (ListT m b)
pmap f prod = ContT $ \cont ->
  snd
    <$>
    withBufferLifted
    unbounded
      (\output -> runEffect $ enumerate prod >-> P.map f >-> toOutput' output)
      (\input -> cont . Select $ fromInput' input)

-- | Run a parallel pipe over the given stream.
pmap' :: (MonadIO m, MonadBaseControl IO m) => Pipe a b m () -> ListT m a -> ContT r m (ListT m b)
pmap' f prod = ContT $ \cont ->
  snd
    <$>
    withBufferLifted
    unbounded
      (\output -> runEffect $ enumerate prod >-> f >-> toOutput' output)
      (\input -> cont . Select $ fromInput' input)

-- | Map a ListT transform over the given the stream in parallel. This should be useful
-- for using functions which groups elements of a stream and yields them downstream. 
pmapGroup :: (MonadIO m, MonadBaseControl IO m) => (ListT m a -> ListT m b) -> ListT m a -> ContT r m (ListT m b)
pmapGroup f prod = ContT $ \cont ->
  snd
    <$> withBufferLifted
      unbounded 
      (\output -> runEffect $ enumerate (f prod) >-> toOutput' output )
      (\input -> cont . Select $ fromInput' input)


-- | Enumerate the given stream, zipping each element with an index.
enumerateData :: Monad m => ListT m a -> Producer (a, Int) m ()
enumerateData input = P.zip (enumerate input) (each [0..])  


bufferedCollate :: (MonadIO m, MonadBaseControl IO m) => Int -> ([sample] -> Maybe batch) -> ListT m sample -> ContT r m (ListT m batch)
bufferedCollate batchSize collateFn = pmapGroup (collate batchSize collateFn) 

collate :: Monad m => Int -> ([sample] -> Maybe batch) -> ListT m sample -> ListT m batch
collate batchSize collateFn  = Select . (>-> P.mapFoldable collateFn) . L.purely folds L.list . view (chunksOf batchSize) . enumerate

newtype CachedDataset (m :: * -> *) sample = CachedDataset { cached :: IntMap sample }

cache datastream = P.fold step begin done . enumerate
  where step (cacheMap, ix) sample = (M.insert ix sample cacheMap, ix + 1) 
        begin = (M.empty, 0)
        done = id

instance Applicative m => Dataset m (CachedDataset m sample) Int sample where
  getItem CachedDataset{..} key = pure $ cached M.! key
  keys CachedDataset{..} = S.fromAscList [0..M.size cached]

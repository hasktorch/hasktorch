{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Data.Utils ( pmap
                        , pmap'
                        , pmapGroup
                        , bufferedCollate
                        , collate
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

-- | Run a map function in parallel over the given stream. 
pmap :: (MonadIO m, MonadBaseControl IO m) => (a -> b) -> ListT m a -> ContT r m (ListT m b)
pmap f prod = ContT $ \cont ->
  snd
    <$>
    withBufferLifted
    unbounded
      (\output -> runEffect $ enumerate prod >-> P.map f >-> toOutput' output)
      (cont . Select . fromInput')

-- | Run a pipe in parallel over the given stream.
pmap' :: (MonadIO m, MonadBaseControl IO m) => Pipe a b m () -> ListT m a -> ContT r m (ListT m b)
pmap' f prod = ContT $ \cont ->
  snd
    <$>
    withBufferLifted
    unbounded
      (\output -> runEffect $ enumerate prod >-> f >-> toOutput' output)
      (cont . Select . fromInput')

-- | Map a ListT transform over the given the stream in parallel. This should be useful
-- for using functions which groups elements of a stream and yields them downstream. 
pmapGroup :: (MonadIO m, MonadBaseControl IO m) => (ListT m a -> ListT m b) -> ListT m a -> ContT r m (ListT m b)
pmapGroup f prod = ContT $ \cont ->
  snd
    <$> withBufferLifted
      unbounded 
      (\output -> runEffect $ enumerate (f prod) >-> toOutput' output )
      (cont . Select . fromInput')


-- | Enumerate the given stream, zipping each element with an index.
enumerateData :: Monad m => ListT m a -> Producer (a, Int) m ()
enumerateData input = P.zip (enumerate input) (each [0..])  


-- | Run a given batching function in parallel. See 'collate' for how the
-- given samples are batched.
bufferedCollate :: (MonadIO m, MonadBaseControl IO m) => Int -> ([sample] -> Maybe batch) -> ListT m sample -> ContT r m (ListT m batch)
bufferedCollate batchSize collateFn = pmapGroup (collate batchSize collateFn) 

-- | Run a batching function with integer batch size over the given stream. The elements of the stream are
-- split into lists of the given batch size and are collated with the given function. Only Just values are yielded 
-- downstream. If the last chunk of samples is less than the given batch size then the batching function will be passed a list
-- of length less than batch size.
collate :: Monad m => Int -> ([sample] -> Maybe batch) -> ListT m sample -> ListT m batch
collate batchSize collateFn  = Select . (>-> P.mapFoldable collateFn) . L.purely folds L.list . view (chunksOf batchSize) . enumerate

-- | An In-Memory cached dataset. See the 'cache' function for
-- how to create a cached dataset.  
newtype CachedDataset (m :: * -> *) sample = CachedDataset { cached :: IntMap sample }

-- | Enumerate a given stream and store it as a 'CachedDataset'. This function should
-- be used after a time consuming preprocessing pipeline and used in subsequent epochs
-- to avoid repeating the preprocessing pipeline.
cache :: Monad m => ListT m sample -> m (CachedDataset m sample)
cache datastream = P.fold step begin done . enumerate $ datastream
  where step (cacheMap, ix) sample = (M.insert ix sample cacheMap, ix + 1) 
        begin = (M.empty, 0)
        done = CachedDataset . fst

instance Applicative m => Dataset m (CachedDataset m sample) Int sample where
  getItem CachedDataset{..} key = pure $ cached M.! key
  keys CachedDataset{..} = S.fromAscList [0..M.size cached]

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module Torch.Data.Dataset where

import Torch.Typed
import Torch.Data.Pipeline
import Torch.Data.StreamedPipeline
import Pipes ((>->), Pipe, enumerate, ListT(Select), Producer)
import qualified Control.Foldl as L
import Pipes.Group (folds, chunksOf)
import Lens.Family (view)
import Data.Vector


-- | This class is actually not very useful.
-- | It would actually be better to define a transform
-- | on top of another dataset, since then we can do this in parallel
data CollatedDataset m dataset batch collatedBatch = CollatedDataset { set       :: dataset
                                                                     , chunkSize :: Int
                                                                     , collateFn :: Pipe [batch] collatedBatch m ()
                                                                   } 

instance Datastream m seed dataset batch => Datastream m seed (CollatedDataset m dataset batch collatedBatch) collatedBatch where
  streamBatch CollatedDataset{..} = Select
                                    . (>-> collateFn)
                                    . L.purely folds L.list
                                    . view (chunksOf chunkSize)
                                    . enumerate
                                    . streamBatch @m @seed @dataset @batch set 

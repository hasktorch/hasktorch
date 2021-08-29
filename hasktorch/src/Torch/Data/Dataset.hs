{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Data.Dataset where

import qualified Control.Foldl as L
import Lens.Family (view)
import Pipes (ListT (Select), Pipe, Producer, enumerate, (>->))
import Pipes.Group (chunksOf, folds)
import Torch.Data.StreamedPipeline

-- | This type is actually not very useful.
-- | It would actually be better to define a transform
-- | on top of another dataset, since then we can do this in parallel
data CollatedDataset m dataset batch collatedBatch = CollatedDataset
  { set :: dataset,
    chunkSize :: Int,
    collateFn :: Pipe [batch] collatedBatch m ()
  }

instance Datastream m seed dataset batch => Datastream m seed (CollatedDataset m dataset batch collatedBatch) collatedBatch where
  streamSamples CollatedDataset {..} =
    Select
      . (>-> collateFn)
      . L.purely folds L.list
      . view (chunksOf chunkSize)
      . enumerate
      . streamSamples set

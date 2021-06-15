-- |
-- Modules for defining datasets and how to efficiently iterate over them.
-- If you have an indexable (fixed-size) dataset, see "Torch.Data.Pipeline". If you
-- want to stream in your data then see "Torch.Data.StreamedPipeline". The "Torch.Data.Utils" module
-- provides some convienient functions for both indexable and streamed datasets.
--
-- The mnist examples show how to run data for a predefined dataset.
module Torch.Data
  ( -- * Running data
    -- $data
    module Torch.Data.Pipeline,
    module Torch.Data.StreamedPipeline,
    module Torch.Data.Utils,
  )
where

import Torch.Data.Pipeline
import Torch.Data.StreamedPipeline
import Torch.Data.Utils

-- $data
--
--
--  The preferred method for running data is the same for both 'Dataset' and 'Datastream'. The intended use is to
-- use the 'streamFrom' family of functions and run the continuation returned by those functions with a function
-- that specifies what to do with the given stream. Datasets are then a [pipes](https://hackage.haskell.org/package/pipes)
-- stream of samples, so anything that you can with a pipes stream you can do with a 'Dataset' or 'Datastream'. As
-- such you should have some basic familiarity with pipes streams, though
-- typically you'll want to a fold over the dataset, where "Pipes.Prelude" has convenient functions for folding streams.
--
--
-- > import qualified Pipes.Prelude as P
-- > import Pipes
-- >
-- > -- Take a model and a stream of data from a Dataset or Datastream,
-- > -- and train the model.
-- > train :: model -> ListT m sample -> m model
-- > train model = runEffect . P.foldM step begin done . enumerate
-- >   where
-- >       -- run a training step over a given sample from the dataset
-- >       step model batch = undefined
-- >       begin = pure model
-- >       done = pure
-- >
-- > runData = runContT (train model) $ streamFromMap (datasetOptions 1) myDataset
--
-- See the [foldl](https://hackage.haskell.org/package/foldl) library for the style of fold used here.

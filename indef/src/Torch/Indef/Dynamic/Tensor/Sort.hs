-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Sort
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Sort where

import Torch.Indef.Types
import qualified Torch.Sig.Tensor.Sort as Sig
import qualified Torch.Indef.Index as Ix

-- | Returns a tensor and index where all entries are sorted along the given
-- dimension, in the chosen sort order. The index corresponds to the original
-- indices.
_sort :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> SortOrder -> IO ()
_sort (r, ix) t1 i0 i1 = with2DynamicState r t1 $ \s' r' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_sort s' r' ix' t1' (fromIntegral i0) (fromIntegral $ fromEnum i1)


-- THC Only:
-- sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()

-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Sort
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Sort where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Sort as Dynamic

-- | Static call to 'Dynamic._sort'
_sort :: (Tensor d', IndexTensor '[n]) -> Tensor d -> DimVal -> SortOrder -> IO ()
_sort (r, ix) t = Dynamic._sort (asDynamic r, longAsDynamic ix) (asDynamic t)

-- GPU only:
--   sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()

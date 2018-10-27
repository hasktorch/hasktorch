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

import Foreign
import Control.Monad.Managed
import Torch.Indef.Types
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.Sort as Sig
import qualified Torch.Indef.Index as Ix

-- | Returns a tensor and index where all entries are sorted along the given
-- dimension, in the chosen sort order. The index corresponds to the original
-- indices.
_sort
  :: (Dynamic, IndexDynamic)
  -> Dynamic
  -> Word           -- ^ dimension to operate on
  -> SortOrder
  -> IO ()
_sort (r, ix) t1 i0 i1 = withLift $ Sig.c_sort
  <$> managedState
  <*> managedTensor r
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t1
  <*> pure (fromIntegral i0)
  <*> pure (fromIntegral $ fromEnum i0)


-- THC Only:
-- sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()

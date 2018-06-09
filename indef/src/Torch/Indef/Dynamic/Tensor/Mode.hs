-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Mode
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Mode where

import qualified Torch.Sig.Tensor.Mode as Sig
import qualified Torch.Indef.Index as Ix

import Torch.Indef.Types

-- | Mutate the tuple of a tensor and index tensor to contain the most frequent element in the specified dimension.
--
-- FIXME: if KeepDim is False, we need to return Nothing for the index tensor -- otherwise bad things may happen.
_mode :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_mode (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_mode s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)



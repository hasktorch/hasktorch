-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.TopK
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.TopK where

import qualified Torch.Sig.Tensor.TopK as Sig

import Torch.Indef.Types
import qualified Torch.Indef.Index as Ix

-- | returns all @k@ smallest elements in a tensor over a given dimension, including their indices, in unsorted order.
_topk :: (Dynamic, IndexDynamic) -> Dynamic -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO ()
_topk (t0, ix) t1 l i0 o sorted = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_topk s' t0' ix' t1' (fromIntegral l) (fromIntegral i0) (fromIntegral $ fromEnum o) (fromKeepDim sorted)


-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.ScatterGather
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.ScatterGather where

import Torch.Indef.Types
import qualified Torch.Indef.Index as Ix
import qualified Torch.Sig.Tensor.ScatterGather as Sig

-- | From the Lua docs:
--
-- Creates a new Tensor from the original tensor by gathering a number of values from each "row", where the rows are along the dimension dim. The values in a LongTensor, passed as index, specify which values to take from each row. Specifically, the resulting Tensor, which will have the same size as the index tensor, is given by
-- 
-- @
--   -- dim = 1
--   result[i][j][k]... = src[index[i][j][k]...][j][k]...
--
--   -- dim = 2
--   result[i][j][k]... = src[i][index[i][j][k]...][k]...
--
--   -- etc.
-- @
--
-- where src is the original Tensor.
--
-- The same number of values are selected from each row, and the same value cannot be selected from a row more than once. The values in the index tensor must not be larger than the length of the row, that is they must be between 1 and src:size(dim) inclusive. It can be somewhat confusing to ensure that the index tensor has the correct shape. Viewed pictorially:
_gather :: Dynamic -> Dynamic -> DimVal -> IndexDynamic -> IO ()
_gather r src d ix = with2DynamicState r src $ \s' r' src' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_gather s' r' src' (fromIntegral d) ix'

-- | From the Lua docs:
--
-- Writes all values from tensor src or the scalar val into self at the
-- specified indices. The indices are specified with respect to the given
-- dimension, dim, in the manner described in gather. Note that, as for gather,
-- the values of index must be between 1 and self:size(dim) inclusive and all
-- values in a row along the specified dimension must be unique.
_scatter :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
_scatter r d ix src = with2DynamicState r src $ \s' r' src' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_scatter s' r' (fromIntegral d) ix' src'

-- | TODO
_scatterAdd   :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
_scatterAdd r d ix src = with2DynamicState r src $ \s' r' src' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_scatterAdd s' r' (fromIntegral d) ix' src'

-- | TODO
_scatterFill  :: Dynamic -> DimVal -> IndexDynamic -> HsReal -> IO ()
_scatterFill r d ix v = withDynamicState r $ \s' r' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_scatterFill s' r' (fromIntegral d) ix' (hs2cReal v)


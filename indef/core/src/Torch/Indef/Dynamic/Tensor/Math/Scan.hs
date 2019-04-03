-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Scan
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Math.Scan where

import Torch.Indef.Types

import qualified Torch.Sig.Tensor.Math.Scan as Sig

-- | Mutate the first tensor to contain the cumulative sum of the elements in the second, performing the operation over the specified dimension.
_cumsum
  :: Dynamic     -- ^ return tensor to mutate, inplace
  -> Dynamic     -- ^ source tensor
  -> Word        -- ^ dimension to operate on
  -> IO ()
_cumsum t0 t1 i0 = withLift $ Sig.c_cumsum
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral i0)

-- | Mutate the first tensor to contain the cumulative product of the elements in the second, performing the operation over the specified dimension.
_cumprod
  :: Dynamic     -- ^ return tensor to mutate, inplace
  -> Dynamic     -- ^ source tensor
  -> Word        -- ^ dimension to operate on
  -> IO ()
_cumprod t0 t1 i0 = withLift $ Sig.c_cumprod
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (fromIntegral i0)


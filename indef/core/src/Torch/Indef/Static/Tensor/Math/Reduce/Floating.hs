-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Reduce.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Reduce.Floating where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating as Dynamic

-- | Returns the @p@-norm of @x - y@.
dist
  :: Tensor d      -- ^ tensor @x@
  -> Tensor d      -- ^ tensor @y@
  -> HsReal        -- ^ @p@
  -> HsAccReal
dist r t = Dynamic.dist (asDynamic r) (asDynamic t)

-- | Get the variance over a tensor in the specified dimension. The 'Bool'
-- parameter specifies whether the standard deviation should be used with
-- @n-1@ or @n@. 'False' normalizes by @n-1@, while 'True' normalizes @n@.
var :: Tensor d
  -> Word -- ^ dimension to operate over
  -> KeepDim -> Bool -> (Tensor d')
var r a b c = asStatic $ Dynamic.var (asDynamic r) a b c

-- | Returns the variance of all elements.
varall  :: Tensor d -> Int -> (HsAccReal)
varall t = Dynamic.varall (asDynamic t)

-- | Performs the @std@ operation over the specified dimension. The 'Bool'
-- parameter specifies whether the standard deviation should be used with
-- @n-1@ or @n@. 'False' normalizes by @n-1@, while 'True' normalizes @n@.
std
  :: (Tensor d)
  -> Word -- ^ dimension to operate over
  -> KeepDim
  -> Bool
  -> (Tensor d')
std t a b c = asStatic $ Dynamic.std (asDynamic t) a b c

-- | Returns the standard deviation of all elements.
stdall  :: Tensor d -> Int -> (HsAccReal)
stdall t = Dynamic.stdall (asDynamic t)

-- | Static 'Dynamic.renorm'
renorm  :: Tensor d -> HsReal -> Int -> HsReal -> (Tensor d')
renorm t a b c = asStatic $ Dynamic.renorm (asDynamic t) a b c

-- | Static 'Dynamic.norm'
norm :: Tensor d -> HsReal
  -> Word -- ^ dimension to operate over
  -> (Tensor d')
norm t a b = asStatic $ Dynamic.norm (asDynamic t) a b

-- | Returns the @p@-norm of all elements.
normall
  :: Tensor d  -- ^ tensor of values to norm over
  -> HsReal   -- ^ @p@
  -> HsAccReal
normall t = Dynamic.normall (asDynamic t)

-- | Static 'Dynamic.mean
mean :: Tensor d
  -> Word -- ^ dimension to operate over
  -> (Tensor d')
mean r a = asStatic $ Dynamic.mean (asDynamic r) a

-- | Returns the mean of all elements.
meanall :: Tensor d -> HsAccReal
meanall t = Dynamic.meanall (asDynamic t)



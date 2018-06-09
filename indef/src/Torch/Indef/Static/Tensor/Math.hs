-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math where

import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Data.Singletons.Prelude.List hiding (All)
import qualified Torch.Types.TH as TH
import qualified Torch.Indef.Dynamic.Tensor.Math as Dynamic

-- | Static call to 'Dynamic.fill_'
fill_ r = Dynamic.fill_ (asDynamic r)
-- | Static call to 'Dynamic.zero_'
zero_ r = Dynamic.zero_ (asDynamic r)

-- | mutate a tensor, inplace, resizing the tensor to the given IndexStorage
-- size and replacing its value with zeros.
zeros_ :: Tensor d -> IndexStorage -> IO ()
zeros_ t0 ix = Dynamic.zeros_ (asDynamic t0) ix

-- | mutate a tensor, inplace, resizing the tensor to the same shape as the second tensor argument
-- and replacing the first tensor's values with zeros.
zerosLike_
  :: Tensor d  -- ^ tensor to mutate inplace and replace contents with zeros
  -> Tensor d'  -- ^ tensor to extract shape information from.
  -> IO ()
zerosLike_ t0 t1 = Dynamic.zerosLike_ (asDynamic t0) (asDynamic t1)

-- | mutate a tensor, inplace, resizing the tensor to the given IndexStorage
-- size and replacing its value with ones.
ones_ :: Tensor d -> TH.IndexStorage -> IO ()
ones_ t0 ix = Dynamic.ones_ (asDynamic t0) ix

-- | mutate a tensor, inplace, resizing the tensor to the same shape as the second tensor argument
-- and replacing the first tensor's values with ones.
onesLike_
  :: Tensor d  -- ^ tensor to mutate inplace and replace contents with ones
  -> Tensor d'  -- ^ tensor to extract shape information from.
  -> IO ()
onesLike_ t0 t1 = Dynamic.onesLike_ (asDynamic t0) (asDynamic t1)


-- | Static call to 'Dynamic.numel'
numel t = Dynamic.numel (asDynamic t)
-- | Static call to 'Dynamic._reshape'
_reshape r t = Dynamic._reshape (asDynamic r) (asDynamic t)
-- | Static call to 'Dynamic._catArray'
_catArray r = Dynamic._catArray (asDynamic r)
-- | Static call to 'Dynamic._nonzero'
_nonzero r t = Dynamic._nonzero (longAsDynamic r) (asDynamic t)
-- | Static call to 'Dynamic._tril'
_tril r t = Dynamic._tril (asDynamic r) (asDynamic t)
-- | Static call to 'Dynamic._triu'
_triu r t = Dynamic._triu (asDynamic r) (asDynamic t)
-- | Static call to 'Dynamic.eye_'
eye_ r = Dynamic.eye_ (asDynamic r)
-- | Static call to 'Dynamic.trace'
trace r = Dynamic.trace (asDynamic r)
-- | Static call to 'Dynamic._arange'
_arange r = Dynamic._arange (asDynamic r)
-- | Static call to 'Dynamic.range_'
range_ r = Dynamic.range_ (asDynamic r)

-- | Static call to 'Dynamic.constant'
constant :: forall d . Dimensions d => HsReal -> Tensor d
constant = asStatic . Dynamic.constant (dims :: Dims d)

-- | Static call to 'Dynamic.diag_'
diag_ :: All Dimensions '[d, d'] => Tensor d -> Int -> IO (Tensor d')
diag_ t d = do
  Dynamic.diag_ (asDynamic t) d
  pure $ (asStatic . asDynamic) t

-- | Static call to 'Dynamic.diag'
diag :: All Dimensions '[d, d'] => Tensor d -> Int -> IO (Tensor d')
diag t d = asStatic <$> Dynamic.diag (asDynamic t) d

-- | Create a diagonal matrix from a 1D vector
diag1d :: (KnownDim n) => Tensor '[n] -> IO (Tensor '[n, n])
diag1d t = diag t 0

-- | Static call to 'Dynamic.cat_'. Unsafely returning the resulting tensor with new dimensions.
cat_
  :: All Dimensions '[d, d', d'']
  => Tensor d -> Tensor d' -> DimVal -> IO (Tensor d'')
cat_ a b d = Dynamic._cat (asDynamic a) (asDynamic a) (asDynamic b) d >> pure (asStatic (asDynamic a))
{-# WARNING cat_ "this function is impure and the dimensions can fall out of sync with the type, if used incorrectly" #-}

-- | Static call to 'Dynamic.cat'
cat :: (All Dimensions '[d, d', d'']) => Tensor d -> Tensor d' -> DimVal -> IO (Tensor d'')
cat a b d = asStatic <$> Dynamic.cat (asDynamic a) (asDynamic b) d

-- | convenience function, specifying a type-safe 'cat' operation.
cat1d :: (All KnownDim '[n1,n2,n], n ~ Sum [n1, n2]) => Tensor '[n1] -> Tensor '[n2] -> IO (Tensor '[n])
cat1d a b = cat a b 0

-- | convenience function, specifying a type-safe 'cat' operation.
cat2d1 :: (All KnownDim '[n,m,m0,m1], m ~ Sum [m0, m1]) => Tensor '[n, m0] -> Tensor '[n, m1] -> IO (Tensor '[n, m])
cat2d1 a b = cat a b 1

-- | convenience function, specifying a type-safe 'cat' operation.
cat2d0 :: (All KnownDim '[n,m,n0,n1], n ~ Sum [n0, n1]) => Tensor '[n0, m] -> Tensor '[n1, m] -> IO (Tensor '[n, m])
cat2d0 a b = cat a b 0

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d0
  :: (All KnownDim '[x,y,x0,x1,z], x ~ Sum [x0, x1])
  => Tensor '[x0, y, z]
  -> Tensor '[x1, y, z]
  -> IO (Tensor '[x, y, z])
cat3d0 a b = cat a b 0

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d1
  :: (All KnownDim '[x,y,y0,y1,z], y ~ Sum [y0, y1])
  => Tensor '[x, y0, z]
  -> Tensor '[x, y1, z]
  -> IO (Tensor '[x, y, z])
cat3d1 a b = cat a b 1

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d2
  :: (All KnownDim '[x,y,z0,z1,z], z ~ Sum [z0, z1])
  => Tensor '[x, y, z0]
  -> Tensor '[x, y, z1]
  -> IO (Tensor '[x, y, z])
cat3d2 a b = cat a b 2

-- | Concatenate all tensors in a given list of dynamic tensors along the given dimension.
--
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
catArray :: (Dimensions d) => [Dynamic] -> DimVal -> IO (Tensor d)
catArray ts dv = empty >>= \r -> Dynamic._catArray (asDynamic r) ts dv >> pure r

-- | Static call to 'Dynamic.onesLike'
onesLike :: forall d . Dimensions d => IO (Tensor d)
onesLike = asStatic <$> Dynamic.onesLike (dims :: Dims d)

-- | Static call to 'Dynamic.zerosLike'
zerosLike :: forall d . Dimensions d => IO (Tensor d)
zerosLike = asStatic <$> Dynamic.zerosLike (dims :: Dims d)




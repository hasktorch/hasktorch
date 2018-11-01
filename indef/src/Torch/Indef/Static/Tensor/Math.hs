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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.Math where

import Numeric.Dimensions -- hiding (Length)

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import System.IO.Unsafe
import Data.Singletons.Prelude (fromSing)
import Data.List.NonEmpty (NonEmpty)
import Data.Either (fromRight)
import qualified Data.Singletons.Prelude.List as Sing hiding (All, type (++))
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

-- | Returns the trace (sum of the diagonal elements) of a matrix x. This is
-- equal to the sum of the eigenvalues of x.
--
-- Static call to 'Dynamic.ttrace'
ttrace r = Dynamic.ttrace (asDynamic r)
-- | Identical to a direct C call to the @arange@, or @range@ with special consideration for floating precision types. Static call to 'Dynamic._arange'
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
diag :: All Dimensions '[d, d'] => Tensor d -> Int -> Tensor d'
diag t d = asStatic $ Dynamic.diag (asDynamic t) d

-- | Create a diagonal matrix from a 1D vector
diag1d :: (KnownDim n) => Tensor '[n] -> Tensor '[n, n]
diag1d t = diag t 0

-- | Static call to 'Dynamic.cat_'. Unsafely returning the resulting tensor with new dimensions.
cat_
  :: All Dimensions '[d, d', d'']
  => Tensor d -> Tensor d' -> Word -> IO (Tensor d'')
cat_ a b d = do
  Dynamic._cat (asDynamic a) (asDynamic a) (asDynamic b) d
  pure (asStatic (asDynamic a))
{-# WARNING cat_ "this function is impure and the dimensions can fall out of sync with the type, if used incorrectly" #-}

-- | Static call to 'Dynamic.cat'
cat
  :: '(ls, r0:+rs) ~ Sing.SplitAt i d
  => '(ls, r1:+rs) ~ Sing.SplitAt i d'
  => Tensor d
  -> Tensor d'
  -> Dim (i::Nat)
  -> Tensor (ls ++ '[r0 + r1] ++ rs)
cat a b d = fromRight (error "impossible: cat type should not allow this branch") $
  asStatic <$> Dynamic.cat (asDynamic a) (asDynamic b) (fromIntegral $ dimVal d)

-- | convenience function, specifying a type-safe 'cat' operation.
cat1d
  :: (All KnownDim '[n1,n2,n], n ~ Sing.Sum [n1, n2])
  => Tensor '[n1] -> Tensor '[n2] -> Tensor '[n]
cat1d a b = cat a b (dim :: Dim 0)

-- | convenience function, specifying a type-safe 'cat' operation.
cat2d0 :: (All KnownDim '[n,m,n0,n1], n ~ Sing.Sum [n0, n1]) => Tensor '[n0, m] -> Tensor '[n1, m] -> Tensor '[n, m]
cat2d0 a b = cat a b (dim :: Dim 0)

-- | convenience function, stack two rank-1 tensors along the 0-dimension
stack1d0 :: KnownDim m => Tensor '[m] -> Tensor '[m] -> (Tensor '[2, m])
stack1d0 a b = cat2d0
  (unsqueeze1d (dim :: Dim 0) a)
  (unsqueeze1d (dim :: Dim 0) b)

-- | convenience function, specifying a type-safe 'cat' operation.
cat2d1 :: (All KnownDim '[n,m,m0,m1], m ~ Sing.Sum [m0, m1]) => Tensor '[n, m0] -> Tensor '[n, m1] -> (Tensor '[n, m])
cat2d1 a b = cat a b (dim :: Dim 1)

-- | convenience function, stack two rank-1 tensors along the 1-dimension
stack1d1 :: KnownDim n => Tensor '[n] -> Tensor '[n] -> (Tensor '[n, 2])
stack1d1 a b = cat2d1
  (unsqueeze1d (dim :: Dim 1) a)
  (unsqueeze1d (dim :: Dim 1) b)

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d0
  :: (All KnownDim '[x,y,x0,x1,z], x ~ Sing.Sum [x0, x1])
  => Tensor '[x0, y, z]
  -> Tensor '[x1, y, z]
  -> (Tensor '[x, y, z])
cat3d0 a b = cat a b (dim :: Dim 0)

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d1
  :: (All KnownDim '[x,y,y0,y1,z], y ~ Sing.Sum [y0, y1])
  => Tensor '[x, y0, z]
  -> Tensor '[x, y1, z]
  -> (Tensor '[x, y, z])
cat3d1 a b = cat a b (dim :: Dim 1)

-- | convenience function, specifying a type-safe 'cat' operation.
cat3d2
  :: (All KnownDim '[x,y,z0,z1,z], z ~ Sing.Sum [z0, z1])
  => Tensor '[x, y, z0]
  -> Tensor '[x, y, z1]
  -> (Tensor '[x, y, z])
cat3d2 a b = cat a b (dim :: Dim 2)

-- | Concatenate all tensors in a given list of dynamic tensors along the given dimension.
--
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
catArray
  :: (Dimensions d)
  => NonEmpty Dynamic
  -> Word
  -> Either String (Tensor d)
catArray ts dv = asStatic <$> Dynamic.catArray ts dv

-- | Concatenate all tensors in a given list of dynamic tensors along the given dimension.
-- --
-- -- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- -- last dimension over all input tensors, except if all tensors are empty, then it is 1.
-- catArray0
--   :: forall d ls rs r0 r1 i
--   .  Dimensions d
--   => '([], r0:+rs) ~ Sing.SplitAt i d
--   => (forall _i . [Tensor (_i+:rs)])
--   -> IO (Tensor (r0+:rs))
-- catArray0 ts dv = catArray (asDynamic <$> ts) (dimVal dv)


-- | Concatenate all tensors in a given list of dynamic tensors along the given dimension.
--
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
catArray'
  :: forall d ls rs r0 r1 i
  .  Dimensions d
  => '(ls, r0:+rs) ~ Sing.SplitAt i d
  => d ~ (ls ++ '[r0] ++ rs)
  => (forall _i . NonEmpty (Tensor (ls ++ '[_i] ++ rs)))
  -> Dim i
  -> Either String (Tensor d)
catArray' ts dv = catArray (asDynamic <$> ts) (dimVal dv)

catArray0 :: (Dimensions d, Dimensions d2) => NonEmpty (Tensor d2) -> Either String (Tensor d)
catArray0 ts = catArray (asDynamic <$> ts) 0

{-
catArray_
  :: forall d ls rs out n
  .  All Dimensions '[out]
  => out ~ (rs ++ '[Length '[Tensor d]] ++ ls)
  => '(ls, rs) ~ Sing.SplitAt n d

  => Sing.SList '[Tensor d]
  -> Dim n
  -> IO (Tensor out)
catArray_ ts dv
  = -- fmap asStatic
    catArray
    (asDynamic <$> (fromSing ts :: [Tensor d]))
    (fromIntegral $ dimVal dv)

-- data Sing (z :: [a]) where
--     SNil :: Sing ([] :: [k])
--     SCons :: Sing (n ': n)

singToList :: forall k ks k2 x . Sing.SList '[x] -> [x]
singToList sl = go [] sl
 where
  go :: [x] -> Sing.SList '[x] -> [x]
  -- go acc Sing.SNil = acc
  -- go acc (Sing.SNil :: Sing.SList ('[] :: [x])) = acc
  -- go acc (Sing.SConst :: Sing.Sing '[]) = acc
  go acc (Sing.SCons k ks) = go acc ks

    -- | fromSing (Sing.sNull sl) = reverse acc
    -- | otherwise = go (fromSing (Sing.sHead sl):acc) (Sing.sTail acc)
  -- Sing.SNil = reverse acc
  -- go acc (Sing.SCons sval rest) = go (fromSing sval:acc) rest
-}

-- | Static call to 'Dynamic.onesLike'
onesLike :: forall d . Dimensions d => (Tensor d)
onesLike = asStatic $ Dynamic.onesLike (dims :: Dims d)

-- | Static call to 'Dynamic.zerosLike'
zerosLike :: forall d . Dimensions d => (Tensor d)
zerosLike = asStatic $ Dynamic.zerosLike (dims :: Dims d)




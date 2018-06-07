-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Torch provides MATLAB-like functions for manipulating Tensor objects.
-- Functions fall into several types of categories:
--
--   * Constructors like zeros, ones;
--   * Extractors like diag and triu;
--   * Element-wise mathematical operations like abs and pow;
--   * BLAS operations;
--   * Column or row-wise operations like sum and max;
--   * Matrix-wide operations like trace and norm;
--   * Convolution and cross-correlation operations like conv2;
--   * Basic linear algebra operations like eig;
--   * Logical operations on Tensors.
--
-- Unfortunately the above this comes from the Lua docs. Hasktorch doesn't
-- mimic this exactly and (FIXME) we will have to restructure this module
-- header to reflect these changes.
-------------------------------------------------------------------------------
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math where

import qualified Torch.Sig.Tensor.Math   as Sig
import qualified Torch.Types.TH as TH (IndexStorage)
import qualified Foreign.Marshal as FM
import Numeric.Dimensions
import System.IO.Unsafe


import Torch.Indef.Dynamic.Tensor
import Torch.Indef.Types
import qualified Torch.Indef.Index as Ix


-- | fill a dynamic tensor, inplace, with the given value.
fill_ :: Dynamic -> HsReal -> IO ()
fill_ t v = withDynamicState t $ shuffle2 Sig.c_fill (hs2cReal v)

-- | mutate a tensor, inplace, filling it with zero values.
zero_ :: Dynamic -> IO ()
zero_ t = withDynamicState t Sig.c_zero

-- | mutate a tensor, inplace, resizing the tensor to the given IndexStorage
-- size and replacing its value with zeros.
zeros_ :: Dynamic -> IndexStorage -> IO ()
zeros_ t ix = withDynamicState t Sig.c_zero

-- | mutate a tensor, inplace, resizing the tensor to the same shape as the second tensor argument
-- and replacing the first tensor's values with zeros.
zerosLike_
  :: Dynamic  -- ^ tensor to mutate inplace and replace contents with zeros
  -> Dynamic  -- ^ tensor to extract shape information from.
  -> IO ()
zerosLike_ t0 t1 = with2DynamicState t0 t1 Sig.c_zerosLike

-- | mutate a tensor, inplace, resizing the tensor to the given IndexStorage
-- size and replacing its value with ones.
ones_ :: Dynamic -> TH.IndexStorage -> IO ()
ones_ t0 ix = withDynamicState t0 $ \s' t0' -> Ix.withCPUIxStorage ix $ \ix' ->
  Sig.c_ones s' t0' ix'

-- | mutate a tensor, inplace, resizing the tensor to the same shape as the second tensor argument
-- and replacing the first tensor's values with ones.
onesLike_
  :: Dynamic  -- ^ tensor to mutate inplace and replace contents with ones
  -> Dynamic  -- ^ tensor to extract shape information from.
  -> IO ()
onesLike_ t0 t1 = with2DynamicState t0 t1 Sig.c_onesLike

-- | returns the count of the number of elements in the matrix.
numel :: Dynamic -> IO Integer
numel t = withDynamicState t (fmap fromIntegral .: Sig.c_numel)

-- |
-- @
--   _reshape y x (Ix.newStorage [m, n, k, l, o])
-- @
--
-- Mutates the @y@ dynamic tensor to be reshaped as a @m × n × k × l × o@ tensor whose elements are
-- taken rowwise from @x@, which must have @m * n * k * l * o@ elements. The elements are copied into
-- the new Tensor.
_reshape :: Dynamic -> Dynamic -> TH.IndexStorage -> IO ()
_reshape t0 t1 ix = with2DynamicState t0 t1 $ \s' t0' t1' -> Ix.withCPUIxStorage ix $ \ix' ->
  Sig.c_reshape s' t0' t1' ix'

-- | pure version of '_catArray'
catArray :: [Dynamic] -> DimVal -> IO Dynamic
catArray ts dv = empty >>= \r -> _catArray r ts dv >> pure r

-- | Concatenate all tensors in a given list of dynamic tensors along the given dimension.
--
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_catArray
  :: Dynamic   -- ^ result to mutate
  -> [Dynamic] -- ^ tensors to concatenate
  -> DimVal    -- ^ dimension to concatenate along.
  -> IO ()
_catArray res ds d = withDynamicState res $ \s' r' -> do
  ds' <- FM.newArray =<< mapM (\d -> withForeignPtr (ctensor d) pure) ds
  Sig.c_catArray s' r' ds' (fromIntegral $ length ds) (fromIntegral d)

-- | "Get the lower triangle of a tensor."
--
-- Mutates the first tensor to have the triangular part of the second tensor under the Kth diagonal.
-- where k=0 is the main diagonal, k>0 is above the main diagonal, and k<0 is below the main diagonal.
-- All other elements are set to 0.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_tril :: Dynamic -> Dynamic -> Integer -> IO ()
_tril t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_tril (fromInteger i0)

-- | "Get the upper triangle of a tensor."
--
-- Mutates the first tensor to have the triangular part of the second tensor above the Kth diagonal.
-- where k=0 is the main diagonal, k>0 is above the main diagonal, and k<0 is below the main diagonal.
-- All other elements are set to 0.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_triu :: Dynamic -> Dynamic -> Integer -> IO ()
_triu t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_triu (fromInteger i0)

-- | Concatinate two dynamic tensors along the specified dimension, treating the
-- first argument as the return tensor, to be mutated in-place.
--
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_cat :: Dynamic -> Dynamic -> Dynamic -> DimVal -> IO ()
_cat t0 t1 t2 i = withDynamicState t0 $ \s' t0' ->
  with2DynamicState t1 t2 $ \_ t1' t2' ->
    Sig.c_cat s' t0' t1' t2' (fromIntegral i)

-- | pure version of '_cat'
cat :: Dynamic -> Dynamic -> DimVal -> IO Dynamic
cat t0 t1 dv = empty >>= \r -> _cat r t0 t1 dv >> pure r

-- | Finds and returns a LongTensor corresponding to the subscript indices of all non-zero elements in tensor.
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_nonzero :: IndexDynamic -> Dynamic -> IO ()
_nonzero ix t = withDynamicState t $ \s' t' -> Ix.withDynamicState ix $ \_ ix' -> Sig.c_nonzero s' ix' t'

-- | Returns the trace (sum of the diagonal elements) of a matrix x. This is equal to the sum of the
-- eigenvalues of x.
trace :: Dynamic -> IO HsAccReal
trace t = withDynamicState t (fmap c2hsAccReal .: Sig.c_trace)

-- | mutates a tensor to be an @n × m@ identity matrix with ones on the diagonal and zeros elsewhere.
eye_
  :: Dynamic  -- ^ tensor to mutate inplace
  -> Integer  -- ^ @n@ dimension in an @n × m@ matrix
  -> Integer  -- ^ @m@ dimension in an @n × m@ matrix
  -> IO ()
eye_ t0 l0 l1 = withDynamicState t0 $ \s' t0' -> Sig.c_eye s' t0' (fromIntegral l0) (fromIntegral l1)

-- | identical to a direct C call to the @arange@, or @range@ with special consideration for floating precision types.
_arange :: Dynamic -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
_arange t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' -> Sig.c_arange s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

-- | mutate a Tensor inplace, filling it with values from @min@ to @max@ with @step@. Will make the tensor take a
-- shape of size @floor((y - x) / step) + 1@.
range_
  :: Dynamic    -- ^ tensor to mutate
  -> HsAccReal  -- ^ @min@ value
  -> HsAccReal  -- ^ @max@ value
  -> HsAccReal  -- ^ @step@ size
  -> IO ()
range_ t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' ->
  Sig.c_range s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

-- | pure version of 'range_'
range
  :: Dims (d::[Nat])
  -> HsAccReal
  -> HsAccReal
  -> HsAccReal
  -> IO Dynamic
range d a b c = withInplace (\r -> range_ r a b c) d

-- | create a 'Dynamic' tensor with a given dimension and value
--
-- We can get away 'unsafeDupablePerformIO' this as constant is pure and thread-safe
constant :: Dims (d :: [Nat]) -> HsReal -> Dynamic
constant d v = unsafeDupablePerformIO $ new d >>= \r -> fill_ r v >> pure r
{-# NOINLINE constant #-}

-- | direct call to the C-FFI of @diag@, mutating the first tensor argument with
-- the data from the remaining aruments.
_diag :: Dynamic -> Dynamic -> Int -> IO ()
_diag t0 t1 i0 = with2DynamicState t0 t1 $ \s' t0' t1' -> Sig.c_diag s' t0' t1' (fromIntegral i0)

-- | mutates the tensor inplace and replaces it with the given k-th diagonal,
-- where k=0 is the main diagonal, k>0 is above the main diagonal, and k<0 is
-- below the main diagonal.
diag_ :: Dynamic -> Int -> IO ()
diag_ t d = _diag t t d

-- | returns the k-th diagonal of the input tensor, where k=0 is the main diagonal,
-- k>0 is above the main diagonal, and k<0 is below the main diagonal.
diag :: Dynamic -> Int -> IO Dynamic
diag  t d = withEmpty t $ \r -> _diag r t d

-- | returns a diagonal matrix with diagonal elements constructed from the input tensor
diag1d :: Dynamic -> IO Dynamic
diag1d t = diag t 1

-- | helper function for 'onesLike' and 'zerosLike'
_tenLike
  :: (Dynamic -> Dynamic -> IO ())
  -> Dims (d::[Nat]) -> IO Dynamic
_tenLike _fn d = do
  src <- new d
  shape <- new d
  _fn src shape
  pure src
{-# WARNING _tenLike "this should not be exported outside of hasktorch" #-}

-- | pure version of 'onesLike_'
onesLike :: Dims (d::[Nat]) -> IO Dynamic
onesLike = _tenLike onesLike_

-- | pure version of 'zerosLike_'
zerosLike :: Dims (d::[Nat]) -> IO Dynamic
zerosLike = _tenLike zerosLike_


-- class CPUTensorMath t where
--   match    :: t -> t -> t -> IO (HsReal t)
--   kthvalue :: t -> IndexDynamic t -> t -> Integer -> Int -> IO Int
--   randperm :: t -> Generator t -> Integer -> IO ()



-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Blas
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math.Blas where

import Numeric.Dimensions
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math

import qualified Torch.Indef.Dynamic.Tensor.Math.Blas as Dynamic

-- | Performs a matrix-vector multiplication between @mat@ (2D Tensor) and @vec2@
-- (1D Tensor) and add it to @vec1@.
--
-- Values @v1@ and @v2@ are scalars that multiply @vec1@ and @vec2@ respectively.
-- They are optional in C and we may be able to add this to the API in the future.
--
-- In other words,
--
-- @
--   res = (v1 * vec1) + (v2 * (mat * vec2))
-- @
--
-- Sizes must respect the matrix-multiplication operation: if @mat@ is a @n × m@
-- matrix, @vec2@ must be vector of size @m@ and @vec1@ must be a vector of size
-- @n@.
addmv
  :: (All KnownDim '[r, c])
  => HsReal           -- ^ v1
  -> Tensor '[r]      -- ^ vec1
  -> HsReal           -- ^ v2
  -> Tensor '[r, c]   -- ^ mat
  -> Tensor '[c]      -- ^ vec2
  -> Tensor '[r]      -- ^ res
addmv a b c d e = unsafeDupablePerformIO $ asStatic <$> Dynamic.addmv a (asDynamic b) c (asDynamic d) (asDynamic e)
{-# NOINLINE addmv #-}

-- | Inline version of 'addmv', mutating @vec1@ inplace.
addmv_
  :: (All KnownDim '[r, c])
  => HsReal           -- ^ v1
  -> Tensor '[r]      -- ^ vec1
  -> HsReal           -- ^ v2
  -> Tensor '[r, c]   -- ^ mat
  -> Tensor '[c]      -- ^ vec2
  -> IO ()
addmv_ a b c d e = Dynamic.addmv_ a (asDynamic b) c (asDynamic d) (asDynamic e)

-- | added simplified use of addmv: src1 #> src2
mv
  :: (All KnownDim '[r, c])
  => Tensor '[r, c] -> Tensor '[c] -> Tensor '[r]
mv m v = addmv 0 (constant 0) 1 m v

-- | inline version of 'mv'
(!*) :: (All KnownDim '[r, c]) => Tensor '[r, c] -> Tensor '[c] -> Tensor '[r]
(!*) a b = mv a b

-- | Performs a matrix-matrix multiplication between @mat1@ (2D Tensor) and @mat2@ (2D Tensor).
--
-- Values @v1@ and @v2@ are scalars that multiply @M@ and @mat1 * mat2@ respectively.
-- They are optional in C and we may be able to add this to the API in the future.
--
-- In other words,
--
-- @
--   res = (v1 * M) + (v2 * mat1 * mat2)
-- @
--
-- If @mat1@ is a @n × m@ matrix, @mat2@ a @m × p@ matrix, @M@ must be a @n × p@ matrix.
addmm
  :: All KnownDim '[a, b, c]
  => HsReal                  -- ^ v1
  -> Tensor '[a, c]          -- ^ M
  -> HsReal                  -- ^ v2
  -> Tensor '[a, b]          -- ^ mat1
  -> Tensor '[b, c]          -- ^ mat2
  -> Tensor '[a, c]          -- ^ res
addmm a b c d e = unsafeDupablePerformIO $ asStatic <$> Dynamic.addmm a (asDynamic b) c (asDynamic d) (asDynamic e)
{-# NOINLINE addmm #-}

-- | Inline version of 'addmm', mutating @M@ inplace.
addmm_
  :: All KnownDim '[a, b, c]
  => HsReal                  -- ^ v1
  -> Tensor '[a, c]          -- ^ M
  -> HsReal                  -- ^ v2
  -> Tensor '[a, b]          -- ^ mat1
  -> Tensor '[b, c]          -- ^ mat2
  -> IO ()
addmm_ a b c d e = Dynamic.addmm_ a (asDynamic b) c (asDynamic d) (asDynamic e)

-- | simplified wrapper of 'addmm'
--
-- FIXME: see if we can pass a null pointer in as the constant value (which might eliminate a noop linear pass).
mmult
  :: All KnownDim '[a, b, c]
  => Tensor '[a, b]
  -> Tensor '[b, c]
  -> Tensor '[a, c]
mmult x y = addmm 1 (constant 0) 1 x y

-- | infix 'mmult'
(!*!) :: (All KnownDim '[a, b, c]) => Tensor '[a, b] -> Tensor '[b, c] -> Tensor '[a, c]
(!*!) = mmult


-- | Performs the outer-product between @vec1@ (1D Tensor) and @vec2@
-- (1D Tensor).
--
-- Values @v1@ and @v2@ are scalars that multiply @mat_ij@ and @vec1_i [out] vec2_j@ respectively.
-- They are optional in C and we may be able to add this to the API in the future.
--
-- Thus:
--
-- @
--   res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
-- @
--
-- If @vec1_@ is a vector of size @i@ and @vec2_j@ is a vector of size @j@, then
-- @mat_ij@ must be a matrix of size @i × j@.
addr
  :: All KnownDim '[r, c]
  => HsReal              -- ^ v1
  -> Tensor '[r,c]       -- ^ mat_ij
  -> HsReal              -- ^ v2
  -> Tensor '[r]         -- ^ vec1_i
  -> Tensor '[c]         -- ^ vec2_j
  -> Tensor '[r, c]      -- ^ res_ij
addr a b c d e = unsafeDupablePerformIO $ asStatic <$> Dynamic.addr a (asDynamic b) c (asDynamic d) (asDynamic e)
{-# NOINLINE addr #-}


-- | Inline version of 'addr', mutating @mat_ij@ in-place.
addr_
  :: All KnownDim '[r, c]
  => HsReal              -- ^ v1
  -> Tensor '[r,c]       -- ^ mat_ij -- mutated inplace
  -> HsReal              -- ^ v2
  -> Tensor '[r]         -- ^ vec1_i
  -> Tensor '[c]         -- ^ vec2_j
  -> IO ()
addr_ a b c d e = Dynamic.addr_ a (asDynamic b) c (asDynamic d) (asDynamic e)

-- | 'addr' with the parameters for an outer product filled in.
outer
  :: forall t r c . (All KnownDim '[r, c])
  => Tensor '[r] -> Tensor '[c] -> Tensor '[r, c]
outer v1 v2 = addr 0 (constant 0) 1 v1 v2


-- | Batch matrix-matrix product of matrices stored in @batch1@ and @batch2@,
-- with a reduced add step (all matrix multiplications get accumulated in
-- a single place).
--
-- @batch1@ and @batch2@ must be 3D Tensors each containing the same number
-- of matrices. If @batch1@ is a @b × n × m@ Tensor, @batch2@ a @b × m × p@
-- Tensor, @res@ will be a @n × p@ Tensor.
--
-- In other words,
--
-- @
--   res = (v1 * M) + (v2 * sum(batch1_i * batch2_i, i = 1, b))
-- @
addbmm
  :: All KnownDim '[n,p,b,m]
  => HsReal             -- ^ v1
  -> Tensor '[n, p]     -- ^ M
  -> HsReal             -- ^ v2
  -> Tensor '[b, n, m]  -- ^ batch1_i
  -> Tensor '[b, m, p]  -- ^ batch2_i
  -> Tensor '[n, p]     -- ^ res
addbmm a b c d e = unsafeDupablePerformIO $ asStatic <$> Dynamic.addbmm a (asDynamic b) c (asDynamic d) (asDynamic e)
{-# NOINLINE addbmm #-}

-- | Inline version of 'addbmm', mutating @M@ in-place.
addbmm_
  :: All KnownDim '[n,p,b,m]
  => HsReal             -- ^ v1
  -> Tensor '[n, p]     -- ^ M -- mutated inplace
  -> HsReal             -- ^ v2
  -> Tensor '[b, n, m]  -- ^ batch1_i
  -> Tensor '[b, m, p]  -- ^ batch2_i
  -> IO ()
addbmm_ a b c d e = Dynamic.addbmm_ a (asDynamic b) c (asDynamic d) (asDynamic e)

-- | Batch matrix matrix product of matrices stored in batch1 and batch2, with
-- batch add.
--
-- @batch1@ and @batch2@ must be 3D Tensors each containing the same number of
-- matrices. If @batch1@ is a @b × n × m@ Tensor, @batch2@ a @b × m × p@ Tensor,
-- @res@ will be a @b × n × p@ Tensor.
--
-- In other words,
--
-- @
--   res_i = (v1 * M_i) + (v2 * batch1_i * batch2_i)
-- @
baddbmm
  :: All KnownDim '[n,p,b,m]
  => HsReal              -- ^ v1
  -> Tensor '[b, n, p]   -- ^ M_i
  -> HsReal              -- ^ v2
  -> Tensor '[b, n, m]   -- ^ batch1_i
  -> Tensor '[b, m, p]   -- ^ batch2_i
  -> Tensor '[b, n, p]   -- ^ res_i
baddbmm a b c d e = unsafeDupablePerformIO $ asStatic <$> Dynamic.baddbmm a (asDynamic b) c (asDynamic d) (asDynamic e)
{-# NOINLINE baddbmm #-}

-- | Inline version of 'baddbmm', mutating @M_i@ in-place.
baddbmm_
  :: All KnownDim '[n,p,b,m]
  => HsReal              -- ^ v1
  -> Tensor '[b, n, p]   -- ^ M_i  -- mutated inplace
  -> HsReal              -- ^ v2
  -> Tensor '[b, n, m]   -- ^ batch1_i
  -> Tensor '[b, m, p]   -- ^ batch2_i
  -> IO ()
baddbmm_ a b c d e = Dynamic.baddbmm_ a (asDynamic b) c (asDynamic d) (asDynamic e)


-- | Performs the dot product between two tensors. The number of elements must match: both tensors are
-- seen as a 1D vector.
dot :: All Dimensions '[d,d'] => Tensor d -> Tensor d' -> IO HsAccReal
dot a b = Dynamic.dot (asDynamic a) (asDynamic b)

-- | inline alias of 'dot'
(<.>)
  :: (All Dimensions '[d,d'])
  => Tensor d
  -> Tensor d'
  -> HsAccReal
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}


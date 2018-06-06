-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Blas
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Blas functions.
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Blas
  ( addmv
  , addmm
  , addr
  , addbmm
  , baddbmm

  , addmv_
  , addmm_
  , addr_
  , addbmm_
  , baddbmm_

  , dot
  , (<.>)
  ) where

import Foreign
import GHC.Int

import Data.Void
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Blas as Sig

blasOp
  :: (Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
blasOp fn r a x b y z =
  with2DynamicState r x $ \s' r' x' ->
    with2DynamicState y z $ \_ y' z' ->
      fn s' r' (hs2cReal a) x' (hs2cReal b) y' z'


_addmv   = blasOp Sig.c_addmv
_addmm   = blasOp Sig.c_addmm
_addr    = blasOp Sig.c_addr
_addbmm  = blasOp Sig.c_addbmm
_baddbmm = blasOp Sig.c_baddbmm

-- | Performs the dot product between two tensors. The number of elements must match: both tensors are
-- seen as a 1D vector.
dot :: Dynamic -> Dynamic -> IO HsAccReal
dot a b = with2DynamicState a b $ fmap c2hsAccReal ..: Sig.c_dot

-- class GPUTensorMathBlas t where
--   btrifact :: t -> IntTensor -> IntTensor -> Int -> t -> io ()
--   btrisolve :: t -> t -> t -> IntTensor -> io ()


-- | inline alias of 'dot'
(<.>) :: Dynamic -> Dynamic -> HsAccReal
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}

mkNewFunction
  :: (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
mkNewFunction op a m b x y = withEmpty x $ \r -> op r a m b x y

mkInplaceFunction
  :: (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
mkInplaceFunction op a m b x y = op m a m b x y

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
  :: HsReal    -- ^ v1
  -> Dynamic   -- ^ vec1
  -> HsReal    -- ^ v2
  -> Dynamic   -- ^ mat
  -> Dynamic   -- ^ vec2
  -> IO Dynamic -- ^ res
addmv = mkNewFunction _addmv

-- | Inline version of 'addmv', mutating @vec1@ inplace.
addmv_
  :: HsReal    -- ^ v1
  -> Dynamic   -- ^ vec1
  -> HsReal    -- ^ v2
  -> Dynamic   -- ^ mat
  -> Dynamic   -- ^ vec2
  -> IO ()
addmv_ = mkInplaceFunction _addmv

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
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ M
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ mat1
  -> Dynamic    -- ^ mat2
  -> IO Dynamic -- ^ res
addmm = mkNewFunction _addmm

-- | Inline version of 'addmm', mutating @M@ inplace.
addmm_
  :: HsReal    -- ^ v1
  -> Dynamic   -- ^ M
  -> HsReal    -- ^ v2
  -> Dynamic   -- ^ mat1
  -> Dynamic   -- ^ mat2
  -> IO ()
addmm_ = mkInplaceFunction _addmm

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
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ mat_ij
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ vec1_i
  -> Dynamic    -- ^ vec2_j
  -> IO Dynamic -- ^ res_ij
addr = mkNewFunction _addr

-- | Inline version of 'addr', mutating @mat_ij@ in-place.
addr_
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ mat_ij -- mutated inplace
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ vec1_i
  -> Dynamic    -- ^ vec2_j
  -> IO ()
addr_ = mkInplaceFunction _addr

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
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ M
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ batch1_i
  -> Dynamic    -- ^ batch2_i
  -> IO Dynamic -- ^ res
addbmm  = mkNewFunction     _addbmm

-- | Inline version of 'addbmm', mutating @M@ in-place.
addbmm_
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ M
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ batch1_i
  -> Dynamic    -- ^ batch2_i
  -> IO ()
addbmm_ = mkInplaceFunction _addbmm

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
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ M_i
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ batch1_i
  -> Dynamic    -- ^ batch2_i
  -> IO Dynamic -- ^ res_i
baddbmm  = mkNewFunction     _baddbmm

-- | Inline version of 'baddbmm', mutating @M_i@ in-place.
baddbmm_
  :: HsReal     -- ^ v1
  -> Dynamic    -- ^ M_i
  -> HsReal     -- ^ v2
  -> Dynamic    -- ^ batch1_i
  -> Dynamic    -- ^ batch2_i
  -> IO ()
baddbmm_ = mkInplaceFunction _baddbmm


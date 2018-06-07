-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Lapack
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Lapack
  ( getri  , getri_
  , potrf  , potrf_
  , potri  , potri_
  , potrs  , potrs_
  , qr     , qr_     , _geqrf
  , geev   , geev_   , eig     , eig_
  , syev   , syev_   , symeig  , symeig_
  , gesv   , gesv_
  , gels   , gels_
  , gesvd  , gesvd_
  , gesvd2 , gesvd2_

  , Triangle(..), EigenReturn(..), ComputeSingularValues(..)
  ) where

import GHC.Int

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Lapack (Triangle(..), EigenReturn(..), ComputeSingularValues(..))
import qualified Torch.Indef.Dynamic.Tensor.Math.Lapack as Dynamic

-- |
-- Docs taken from MAGMA documentation at: http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__getri.html
--
-- getri computes the inverse of a matrix using the LU factorization computed by
-- 'getrf'.
--
-- This method inverts @U@ and then computes @inv(A)@ by solving the system
-- @inv(A)*L = inv(U)@ for @inv(A)@.
--
-- Note that it is generally both faster and more accurate to use @gesv@, or
-- @getrf@ and @getrs@, to solve the system @AX = B@, rather than inverting the
-- matrix and multiplying to form @X = inv(A)*B@. Only in special instances
-- should an explicit inverse be computed with this routine.
getri :: Tensor d -> IO (Tensor d')
getri t = asStatic <$> Dynamic.getri (asDynamic t)

-- | inplace version of 'getri'
getri_ :: Tensor d -> IO ()
getri_ t = Dynamic.getri_ (asDynamic t)

-- | Cholesky Decomposition of 2D @tensor A@. The matrix, @A@, has to be a
-- positive-definite and either symmetric or complex Hermitian.
--
-- The factorization has the form
--
-- @
--   A = U**T * U,   if UPLO = 'U', or
--   A = L  * L**T,  if UPLO = 'L',
-- @
--
-- where @U@ is an upper triangular matrix and @L@ is lower triangular.
potrf
  :: Tensor d   -- ^ matrix to decompose
  -> Triangle   -- ^ which triangle should be used.
  -> IO (Tensor d')
potrf s t = asStatic <$> Dynamic.potrf (asDynamic s) t

-- | infix version of 'potrf'.
potrf_
  :: Tensor d  -- ^ matrix to decompose
  -> Triangle  -- ^ which triangle should be used.
  -> IO ()
potrf_ s = Dynamic.potrf_ (asDynamic s)

-- | Returns the inverse of 2D @tensor A@ given its Cholesky decomposition @chol@.
--
-- Square matrix @chol@ should be triangular.
--
-- 'Triangle' specifies matrix @chol@ as either upper or lower triangular.
potri :: Tensor d -> Triangle -> IO (Tensor d')
potri s t = asStatic <$> Dynamic.potri (asDynamic s) t

-- | inplace version of 'potri'.
potri_ :: Tensor d -> Triangle -> IO ()
potri_ s t = Dynamic.potri_ (asDynamic s) t

-- | Returns the solution to linear system @AX = B@ using the Cholesky
-- decomposition @chol@ of 2D tensor @A@.
--
-- Square matrix @chol@ should be triangular; and, righthand side matrix @B@
-- should be of full rank.
--
-- 'Triangle' specifies matrix @chol@ as either upper or lower triangular.
-- /* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
potrs
  :: Tensor d    -- ^ Tensor @B@
  -> Tensor d'   -- ^ Cholesky decomposition @chol@
  -> Triangle   -- ^ which triangle to use (upper or lower)
  -> IO (Tensor d'')
potrs b chol t = asStatic <$> Dynamic.potrs (asDynamic b) (asDynamic chol) t

-- | Inplace version of 'potri'. Mutating tensor B in place.
potrs_
  :: Tensor d    -- ^ Tensor @B@
  -> Tensor d'   -- ^ Cholesky decomposition @chol@
  -> Triangle   -- ^ which triangle to use (upper or lower)
  -> IO ()
potrs_ b chol t = Dynamic.potrs_ (asDynamic b) (asDynamic chol) t


-- | Compute a QR decomposition of the matrix @x@: matrices @q@ and @r@ such that
-- @x = q * r@, with @q@ orthogonal and @r@ upper triangular. This returns the
-- thin (reduced) QR factorization.
--
-- Note that precision may be lost if the magnitudes of the elements of x are
-- large.
--
-- Note also that, while it should always give you a valid decomposition, it may
-- not give you the same one across platforms - it will depend on your LAPACK
-- implementation.
--
-- Note: Irrespective of the original strides, the returned matrix @q@ will be
-- transposed, i.e. with strides @1, m@ instead of @m, 1@.
qr :: Tensor d -> IO (Tensor d', Tensor d'')
qr x = do
  (ra, rb) <- Dynamic.qr (asDynamic x)
  pure (asStatic ra, asStatic rb)

-- | Inplace version of 'qr'
qr_ :: (Tensor d, Tensor d') -> Tensor d'' -> IO ()
qr_ (q, r) x = Dynamic.qr_ (asDynamic q, asDynamic r) (asDynamic x)

-- | This is a low-level function for calling LAPACK directly. You'll generally
-- want to use 'qr' instead.
--
-- Computes a QR decomposition of @a@, but without constructing Q and R as
-- explicit separate matrices. Rather, this directly calls the underlying LAPACK
-- function @?geqrf@ which produces a sequence of "elementary reflectors". See
-- <https://software.intel.com/en-us/node/521004 LAPACK documentation from MKL>
-- for further details and <http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__geqrf.html
-- for MAGMA documentation>.
--
-- Note that, because this is low-level code, hasktorch just calls Torch
-- directly.
_geqrf :: Tensor d -> Tensor d' -> Tensor d'' -> IO ()
_geqrf a b c = Dynamic._geqrf (asDynamic a) (asDynamic b) (asDynamic c)

-- | @(e, V) <- geev A@ returns eigenvalues and eigenvectors of a general real
-- square matrix @A@.
--
-- @A@ and @V@ are @m × m@ matrices and @e@ is an @m@-dimensional vector.
--
-- This function calculates all right eigenvalues (and vectors) of @A@ such that
-- @A = V diag(e) V@.
--
-- The 'EigenReturn' argument defines computation of eigenvectors or eigenvalues
-- only. It determines if only eigenvalues are computed or if both eigenvalues
-- and eigenvectors are computed.
--
-- The eigenvalues returned follow LAPACK convention and are returned as complex
-- (real/imaginary) pairs of numbers (@2 * m@ dimensional tensor).
--
-- Also called the "eig" fuction in torch.
geev
  :: Tensor d                      -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn                  -- ^ whether or not to return eigenvectors.
  -> IO (Tensor d', Maybe (Tensor d''))  -- ^ (e, V) standing for eigenvalues and eigenvectors
geev m er = do
  (r, mt) <- Dynamic.geev (asDynamic m) er
  pure (asStatic r, asStatic <$> mt)

-- | alias to 'geev' to match Torch naming conventions.
eig = geev

-- | In-place version of 'geev'.
--
-- Note: Irrespective of the original strides, the returned matrix @V@ will be
-- transposed, i.e. with strides @1, m@ instead of @m, 1@.
geev_
  :: (Tensor d, Tensor d')  -- ^ (e, V) standing for eigenvalues and eigenvectors
  -> Tensor d''             -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn            -- ^ whether or not to return eigenvectors.
  -> IO ()
geev_ (e, v) m er = Dynamic.geev_ (asDynamic e, asDynamic v) (asDynamic m) er

-- | alias to 'geev_' to match Torch naming conventions.
eig_ = geev_


-- | @(e, V) <- syev A@ returns eigenvalues and eigenvectors of a symmetric real matrix @A@.
--
-- @A@ and @V@ are @m × m@ matrices and @e@ is a @m@-dimensional vector.
--
-- This function calculates all eigenvalues (and vectors) of @A@ such that @A = V diag(e) V@.
--
-- The 'EigenReturn' argument defines computation of eigenvectors or eigenvalues only.
--
-- Since the input matrix @A@ is supposed to be symmetric, only one triangular portion
-- is used. The 'Triangle' argument indicates if this should be the upper or lower triangle.
syev
  :: Tensor d                      -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn                  -- ^ whether or not to return eigenvectors.
  -> Triangle                     -- ^ whether the upper or lower triangle should be used
  -> IO (Tensor d', Maybe (Tensor d''))  -- ^ (e, V) standing for eigenvalues and eigenvectors
syev m er tri = do
  (r, mt) <- Dynamic.syev (asDynamic m) er tri
  pure (asStatic r, asStatic <$> mt)

-- | alias to 'syev' to match Torch naming conventions.
symeig = syev

-- | Inplace version of 'syev'
--
-- Note: Irrespective of the original strides, the returned matrix V will be transposed, i.e. with strides 1, m instead of m, 1.
syev_
  :: (Tensor d, Tensor d')  -- ^ (e, V) standing for eigenvalues and eigenvectors
  -> Tensor d''             -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn         -- ^ whether or not to return eigenvectors.
  -> Triangle            -- ^ whether the upper or lower triangle should be used
  -> IO ()
syev_ (e, v) m er tri = Dynamic.syev_ (asDynamic e, asDynamic v) (asDynamic m) er tri

-- | alias to 'syev' to match Torch naming conventions.
symeig_ = syev_

-- | @ (X, LU) <- gesv B A@ returns the solution of @AX = B@ and @LU@ contains @L@
-- and @U@ factors for @LU@ factorization of @A@.
--
-- @A@ has to be a square and non-singular matrix (a 2D tensor). @A@ and @LU@
-- are @m × m@, @X@ is @m × k@ and @B@ is @m × k@.
gesv
  :: Tensor d                -- ^ @B@
  -> Tensor d'                -- ^ @A@
  -> IO (Tensor d'', Tensor d''')  -- ^ @(X, LU)@
gesv b a = do
  (ra, rb) <- Dynamic.gesv (asDynamic b) (asDynamic a)
  pure (asStatic ra, asStatic rb)

-- | Inplace version of 'gesv'.
--
-- In this case @x@ and @lu@ will be used for temporary storage and returning the result.
--
--   * @x@ will contain the solution @X@.
--   * @lu@ will contain @L@ and @U@ factors for @LU@ factorization of @A@.
--
-- Note: Irrespective of the original strides, the returned matrices @x@ and @lu@ will be transposed,
-- i.e. with strides @1, m@ instead of @m, 1@.
gesv_
  :: (Tensor d, Tensor d')  -- ^ @(X, LU)@
  -> Tensor d''             -- ^ @B@
  -> Tensor d'''           -- ^ @A@
  -> IO ()
gesv_ (x, lu) b a = Dynamic.gesv_ (asDynamic x, asDynamic lu) (asDynamic b) (asDynamic a)

-- | Solution of least squares and least norm problems for a full rank @m × n@ matrix @A@.
--
--   * If @n ≤ m@, then solve @||AX-B||_F@.
--   * If @n > m@, then solve @min ||X||_F@ such that @AX = B@.
--
-- On return, first @n@ rows of @x@ matrix contains the solution and the rest
-- contains residual information. Square root of sum squares of elements of each
-- column of @x@ starting at row @n + 1@ is the residual for corresponding column.
gels :: Tensor d -> Tensor d' -> IO (Tensor d'', Tensor d''')
gels b a = do
  (ra, rb) <- Dynamic.gels (asDynamic b) (asDynamic a)
  pure (asStatic ra, asStatic rb)

-- | Inplace version of 'gels'.
--
-- Note: Irrespective of the original strides, the returned matrices @resb@ and
-- @resa@ will be transposed, i.e. with strides @1, m@ instead of @m, 1@.
gels_
  :: (Tensor d, Tensor d') -- ^ @(resb, resa)@
  -> Tensor d''            -- ^ @matrix b@
  -> Tensor d'''            -- ^ @matrix a@
  -> IO ()
gels_ (a, b) c d = Dynamic.gels_ (asDynamic a, asDynamic b) (asDynamic c) (asDynamic d)


-- | @(U, S, V) <- svd A@ returns the singular value decomposition of a real
-- matrix @A@ of size @n × m@ such that @A = USV'*@.
--
-- @U@ is @n × n@, @S@ is @n × m@ and @V@ is @m × m@.
--
-- The 'ComputeSingularValues' argument represents the number of singular values
-- to be computed. 'SomeSVs' stands for "some" (FIXME: figure out what that means)
-- and 'AllSVs' stands for all.
gesvd
  :: Tensor d
  -> ComputeSingularValues
  -> IO (Tensor d', Tensor d'', Tensor d''')
gesvd m num = do
  (ra, rb, rc) <- Dynamic.gesvd (asDynamic m) num
  pure (asStatic ra, asStatic rb, asStatic rc)

-- | Inplace version of 'gesvd'.
--
-- Note: Irrespective of the original strides, the returned matrix @U@ will be
-- transposed, i.e. with strides @1, n@ instead of @n, 1@.
gesvd_
  :: (Tensor d, Tensor d', Tensor d'') -- ^ (u, s, v)
  -> Tensor d'''                     -- ^ m
  -> ComputeSingularValues       -- ^ Whether to compute all or some of the singular values
  -> IO ()
gesvd_ (u, s, v) m num = Dynamic.gesvd_ (asDynamic u, asDynamic s, asDynamic v) (asDynamic m) num

-- | 'gesvd', computing @A = U*Σ*transpose(V)@.
--
-- NOTE: "'gesvd', computing @A = U*Σ*transpose(V)@." is only inferred documentation. This
-- documentation was made by stites, inferring from the description of the 'gesvd' docs at
-- <https://software.intel.com/en-us/mkl-developer-reference-c-gesvd the intel mkl
-- documentation>.
gesvd2
  :: Tensor d                                 -- ^ m
  -> ComputeSingularValues                   -- ^ Whether to compute all or some of the singular values
  -> IO (Tensor d', Tensor d'', Tensor d''', Tensor d'''') -- ^ (u, s, v, a)
gesvd2 m csv = do
  (ra, rb, rc, rd) <- Dynamic.gesvd2 (asDynamic m) csv
  pure (asStatic ra, asStatic rb, asStatic rc, asStatic rd)

-- | Inplace version of 'gesvd2_'.
gesvd2_
  :: (Tensor d, Tensor d', Tensor d'', Tensor d''') -- ^ (u, s, v, a)
  -> Tensor d''''                              -- ^ m
  -> ComputeSingularValues                -- ^ Whether to compute all or some of the singular values
  -> IO ()
gesvd2_ (u, s, v, a) m = Dynamic.gesvd2_ (asDynamic u, asDynamic s, asDynamic v, asDynamic a) (asDynamic m)


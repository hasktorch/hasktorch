-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Lapack
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Functions in this section are implemented with an interface to LAPACK
-- libraries. If LAPACK libraries are not found during compilation step, then
-- these functions will not be available. Hasktorch has not been tested without
-- LAPACK functionality, so that behaviour is currently undefined for an end
-- user. (FIXME) Someone needs to test LAPACK-less compilation steps.
-------------------------------------------------------------------------------
{-# LANGUAGE LambdaCase #-}
module Torch.Indef.Dynamic.Tensor.Math.Lapack
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

import Data.Coerce
import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Torch.Indef.Dynamic.Tensor (empty)
import qualified Torch.Sig.Tensor.Math.Lapack as Sig

import Torch.Indef.Types

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
getri :: Dynamic -> IO Dynamic
getri t = empty >>= \r -> _getri r t >> pure r

-- | inplace version of 'getri'
getri_ :: Dynamic -> IO ()
getri_ t = _getri t t

-- | C-style version of 'getri' and 'getri_'.
_getri :: Dynamic -> Dynamic -> IO ()
_getri a b = with2DynamicState a b Sig.c_getri

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
  :: Dynamic    -- ^ matrix to decompose
  -> Triangle   -- ^ which triangle should be used.
  -> IO Dynamic
potrf s t = empty >>= \r -> _potrf r s t >> pure r

-- | infix version of 'potrf'.
potrf_
  :: Dynamic   -- ^ matrix to decompose
  -> Triangle  -- ^ which triangle should be used.
  -> IO ()
potrf_ src t = _potrf src src t

-- | C-style version of 'potrf' and 'potrf_' where the first tensor argument is mutated inplace.
_potrf
  :: Dynamic   -- ^ tensor to place the resulting decomposition in.
  -> Dynamic   -- ^ matrix to decompose
  -> Triangle  -- ^ which triangle should be used.
  -> IO ()
_potrf ret src t = triangleArg2C t >>= \c' -> with2DynamicState ret src $ shuffle3 Sig.c_potrf c'

-- | Returns the inverse of 2D @tensor A@ given its Cholesky decomposition @chol@.
--
-- Square matrix @chol@ should be triangular.
--
-- 'Triangle' specifies matrix @chol@ as either upper or lower triangular.
potri :: Dynamic -> Triangle -> IO Dynamic
potri s t = empty >>= \r -> _potri r s t >> pure r

-- | inplace version of 'potri'.
potri_ :: Dynamic -> Triangle -> IO ()
potri_ s t = _potri s s t

-- | C-style mutation of 'potri_' and 'potri'. Should not be exported.
_potri :: Dynamic -> Dynamic -> Triangle -> IO ()
_potri a b v =
  triangleArg2C v >>= \v' ->
    with2DynamicState a b $ shuffle3 Sig.c_potri v'

-- | Returns the solution to linear system @AX = B@ using the Cholesky
-- decomposition @chol@ of 2D tensor @A@.
--
-- Square matrix @chol@ should be triangular; and, righthand side matrix @B@
-- should be of full rank.
--
-- 'Triangle' specifies matrix @chol@ as either upper or lower triangular.
-- /* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
potrs
  :: Dynamic    -- ^ Tensor @B@
  -> Dynamic    -- ^ Cholesky decomposition @chol@
  -> Triangle   -- ^ which triangle to use (upper or lower)
  -> IO Dynamic
potrs b chol t = empty >>= \r -> _potrs r b chol t >> pure r

-- | Inplace version of 'potri'. Mutating tensor B in place.
potrs_
  :: Dynamic    -- ^ Tensor @B@
  -> Dynamic    -- ^ Cholesky decomposition @chol@
  -> Triangle   -- ^ which triangle to use (upper or lower)
  -> IO ()
potrs_ b chol t = _potrs b b chol t

-- | C-style mutation of 'potrs_' and 'potrs'. Should not be exported.
_potrs :: Dynamic -> Dynamic -> Dynamic -> Triangle -> IO ()
_potrs a b c v =
  triangleArg2C v >>= \v' ->
    with3DynamicState a b c $ \s' a' b' c' -> Sig.c_potrs s' a' b' c' v'

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
qr :: Dynamic -> IO (Dynamic, Dynamic)
qr x = (,) <$> empty <*> empty >>= \ret -> qr_ ret x >> pure ret

-- | Inplace version of 'qr'
qr_ :: (Dynamic, Dynamic) -> Dynamic -> IO ()
qr_ (q, r) x = with3DynamicState q r x Sig.c_qr

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
_geqrf :: Dynamic -> Dynamic -> Dynamic -> IO ()
_geqrf a b c = with3DynamicState a b c Sig.c_geqrf

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
  :: Dynamic                      -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn                  -- ^ whether or not to return eigenvectors.
  -> IO (Dynamic, Maybe Dynamic)  -- ^ (e, V) standing for eigenvalues and eigenvectors
geev m er = do
  e <- empty
  v <- empty
  geev_ (e, v) m er
  case er of
    ReturnEigenValues          -> pure (e, Just v)
    ReturnEigenValuesAndVector -> pure (e, Nothing)

-- | alias to 'geev' to match Torch naming conventions.
eig = geev

-- | In-place version of 'geev'.
--
-- Note: Irrespective of the original strides, the returned matrix @V@ will be
-- transposed, i.e. with strides @1, m@ instead of @m, 1@.
geev_
  :: (Dynamic, Dynamic)  -- ^ (e, V) standing for eigenvalues and eigenvectors
  -> Dynamic             -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn         -- ^ whether or not to return eigenvectors.
  -> IO ()
geev_ (e, v) m er =
  eigenArg2C er >>= \er' ->
    with3DynamicState e v m $ \s' e' v' m' -> Sig.c_geev s' e' v' m' er'

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
  :: Dynamic                      -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn                  -- ^ whether or not to return eigenvectors.
  -> Triangle                     -- ^ whether the upper or lower triangle should be used
  -> IO (Dynamic, Maybe Dynamic)  -- ^ (e, V) standing for eigenvalues and eigenvectors
syev m er tri = do
  e <- empty
  v <- empty
  syev_ (e, v) m er tri
  case er of
    ReturnEigenValues          -> pure (e, Just v)
    ReturnEigenValuesAndVector -> pure (e, Nothing)

-- | alias to 'syev' to match Torch naming conventions.
symeig = syev

-- | Inplace version of 'syev'
--
-- Note: Irrespective of the original strides, the returned matrix V will be transposed, i.e. with strides 1, m instead of m, 1.
syev_
  :: (Dynamic, Dynamic)  -- ^ (e, V) standing for eigenvalues and eigenvectors
  -> Dynamic             -- ^ square matrix to get eigen{values/vectors} of.
  -> EigenReturn         -- ^ whether or not to return eigenvectors.
  -> Triangle            -- ^ whether the upper or lower triangle should be used
  -> IO ()
syev_ (e, v) m er tri = do
  er'  <- eigenArg2C er
  tri' <- triangleArg2C tri
  with3DynamicState e v m $ \s' e' v' m' ->
    Sig.c_syev s' e' v' m' er' tri'

-- | alias to 'syev' to match Torch naming conventions.
symeig_ = syev_

-- | @ (X, LU) <- gesv B A@ returns the solution of @AX = B@ and @LU@ contains @L@
-- and @U@ factors for @LU@ factorization of @A@.
--
-- @A@ has to be a square and non-singular matrix (a 2D tensor). @A@ and @LU@
-- are @m × m@, @X@ is @m × k@ and @B@ is @m × k@.
gesv
  :: Dynamic                -- ^ @B@
  -> Dynamic                -- ^ @A@
  -> IO (Dynamic, Dynamic)  -- ^ @(X, LU)@
gesv b a = (,) <$> empty <*> empty >>= \ret -> gesv_ ret b a >> pure ret

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
  :: (Dynamic, Dynamic)  -- ^ @(X, LU)@
  -> Dynamic             -- ^ @B@
  -> Dynamic             -- ^ @A@
  -> IO ()
gesv_ (x, lu) b a =
  with2DynamicState x lu $ \s' x' lu' ->
    with2DynamicState b a $ \_ b' a' ->
      Sig.c_gesv s' x' lu' b' a'

-- | Solution of least squares and least norm problems for a full rank @m × n@ matrix @A@.
--
--   * If @n ≤ m@, then solve @||AX-B||_F@.
--   * If @n > m@, then solve @min ||X||_F@ such that @AX = B@.
--
-- On return, first @n@ rows of @x@ matrix contains the solution and the rest
-- contains residual information. Square root of sum squares of elements of each
-- column of @x@ starting at row @n + 1@ is the residual for corresponding column.
gels :: Dynamic -> Dynamic -> IO (Dynamic, Dynamic)
gels b a = (,) <$> empty <*> empty >>= \ret -> gels_ ret b a >> pure ret

-- | Inplace version of 'gels'.
--
-- Note: Irrespective of the original strides, the returned matrices @resb@ and
-- @resa@ will be transposed, i.e. with strides @1, m@ instead of @m, 1@.
gels_
  :: (Dynamic, Dynamic) -- ^ @(resb, resa)@
  -> Dynamic            -- ^ @matrix b@
  -> Dynamic            -- ^ @matrix a@
  -> IO ()
gels_ (a, b) c d =
  with2DynamicState a b $ \s' a' b' ->
    with2DynamicState c d $ \_ c' d' ->
      Sig.c_gels s' a' b' c' d'


-- | @(U, S, V) <- svd A@ returns the singular value decomposition of a real
-- matrix @A@ of size @n × m@ such that @A = USV'*@.
--
-- @U@ is @n × n@, @S@ is @n × m@ and @V@ is @m × m@.
--
-- The 'ComputeSingularValues' argument represents the number of singular values
-- to be computed. 'SomeSVs' stands for "some" (FIXME: figure out what that means)
-- and 'AllSVs' stands for all.
gesvd
  :: Dynamic
  -> ComputeSingularValues
  -> IO (Dynamic, Dynamic, Dynamic)
gesvd m num = do
  ret <- (,,) <$> empty <*> empty <*> empty
  gesvd_ ret m num
  pure ret

-- | Inplace version of 'gesvd'.
--
-- Note: Irrespective of the original strides, the returned matrix @U@ will be
-- transposed, i.e. with strides @1, n@ instead of @n, 1@.
gesvd_
  :: (Dynamic, Dynamic, Dynamic) -- ^ (u, s, v)
  -> Dynamic                     -- ^ m
  -> ComputeSingularValues       -- ^ Whether to compute all or some of the singular values
  -> IO ()
gesvd_ (u, s, v) m num =
  svArg2C num >>= \num' ->
    with3DynamicState u s v $ \state' u' s' v' ->
      withDynamicState m $ \_ m' ->
        Sig.c_gesvd state' u' s' v' m' num'

-- | 'gesvd', computing @A = U*Σ*transpose(V)@.
--
-- NOTE: "'gesvd', computing @A = U*Σ*transpose(V)@." is only inferred documentation. This
-- documentation was made by stites, inferring from the description of the 'gesvd' docs at
-- <https://software.intel.com/en-us/mkl-developer-reference-c-gesvd the intel mkl
-- documentation>.
gesvd2
  :: Dynamic                                 -- ^ m
  -> ComputeSingularValues                   -- ^ Whether to compute all or some of the singular values
  -> IO (Dynamic, Dynamic, Dynamic, Dynamic) -- ^ (u, s, v, a)
gesvd2 m csv = do
  ret <- (,,,) <$> empty <*> empty <*> empty <*> empty
  gesvd2_ ret m csv
  pure ret

-- | Inplace version of 'gesvd2_'.
gesvd2_
  :: (Dynamic, Dynamic, Dynamic, Dynamic) -- ^ (u, s, v, a)
  -> Dynamic                              -- ^ m
  -> ComputeSingularValues                -- ^ Whether to compute all or some of the singular values
  -> IO ()
gesvd2_ (u, s, v, a) m csv = do
  csv' <- svArg2C csv
  with2DynamicState u s $ \state' u' s' ->
    with2DynamicState v a $ \_ v' a' ->
      withDynamicState m $ \_ m' ->
        Sig.c_gesvd2 state' u' s' v' a' m' csv'

-- ========================================================================= --
-- Helpers

-- | Argument to specify whether the upper or lower triangular decomposition
-- should be used in 'potrf' and 'potrf_'.
data Triangle
  = Upper   -- ^ use upper triangular matrix
  | Lower   -- ^ use lower triangular matrix
  deriving (Eq, Show)

-- | helper function to cast 'Triangle' into C arguments. This should not be exported.
triangleArg2C :: Triangle -> IO (Ptr CChar)
triangleArg2C = \case
  Upper -> newCString "U"
  Lower -> newCString "L"

-- | Argument to be passed to 'geev', 'syev', and their inplace variants.
-- Determines if the a function should only compute eigenvalues or both
-- eigenvalues and eigenvectors.
data EigenReturn
  = ReturnEigenValues
  | ReturnEigenValuesAndVector
  deriving (Eq, Show)

-- | helper function to cast 'Triangle' into C arguments. This should not be exported.
eigenArg2C :: EigenReturn -> IO (Ptr CChar)
eigenArg2C = \case
  ReturnEigenValues          -> newCString "N"
  ReturnEigenValuesAndVector -> newCString "V"

-- | Represents the number of singular values to be computed in 'gesvd' and 'gesvd2'.
-- While fairly opaque about how many values are computed, Torch says we either compute
-- "some" or all of the values.
data ComputeSingularValues
  = SomeSVs
  | AllSVs

-- | helper function to cast 'ComputeSingularValues into C arguments. This should not be exported.
svArg2C :: ComputeSingularValues -> IO (Ptr CChar)
svArg2C = \case
  SomeSVs -> newCString "S"
  AllSVs  -> newCString "A"

{-
class CPUTensorMathLapack t where

  -- |
  -- copied from the Lua docs. See:
  -- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#x-torchtrtrsresb-resa-b-a--u-or-l--n-or-t--n-or-u
  --
  -- X = torch.trtrs(B, A) returns the solution of AX = B where A is upper-triangular.
  --
  -- A has to be a square, triangular, non-singular matrix (2D Tensor). A and
  -- resa are m × m, X and B are m × k. (To be very precise: A does not have to
  -- be triangular and non-singular, rather only its upper or lower triangle
  -- will be taken into account and that part has to be non-singular.)
  --
  -- The function has several options:
  --
  --   * uplo ('U' or 'L') specifies whether A is upper or lower triangular; the
  --     default value is 'U'.
  --
  --   * trans ('N' or 'T') specifies the system of equations: 'N' for A * X = B
  --     (no transpose), or 'T' for A^T * X = B (transpose); the default value is 'N'.
  --
  --   * diag ('N' or 'U') 'U' specifies that A is unit triangular, i.e., it has
  --     ones on its diagonal; 'N' specifies that A is not (necessarily) unit
  --     triangular; the default value is 'N'.
  --
  -- If resb and resa are given, then they will be used for temporary storage
  -- and returning the result. resb will contain the solution X.
  --
  -- Note: Irrespective of the original strides, the returned matrices resb and
  -- resa will be transposed, i.e. with strides 1, m instead of m, 1.
  _trtrs     :: t -> t -> t -> t -> [Int8] -> [Int8] -> [Int8] -> IO ()

  _orgqr     :: t -> t -> t -> IO ()
  _ormqr     :: t -> t -> t -> t -> [Int8] -> [Int8] -> IO ()
  _pstrf     :: t -> Int.DynTensor -> t -> [Int8] -> HsReal t -> IO ()
  _btrifact  :: t -> Int.DynTensor -> Int.DynTensor -> Int32 -> t -> IO ()
  _btrisolve :: t -> t -> t -> Int.DynTensor -> IO ()
-}

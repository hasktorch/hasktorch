{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Core.Tensor.Static.DoubleLapack (
  tds_gesv,
  -- tds_trtrs,
  tds_gels,
  -- tds_syev,
  -- tds_geev,
  -- tds_gesvd,
  -- tds_gesvd2,
  tds_getri,
  tds_potrf,
  tds_potrs,
  tds_potri,
  tds_qr,
  tds_geqrf,
  tds_orgqr,
  -- tds_ormqr,
  -- tds_pstrf,
  -- tds_btrifact,
  -- tds_btrisolve

  ) where


import Control.Monad.Managed
import Data.Singletons
import Data.Singletons.TypeLits
import Foreign (Ptr)
import Foreign.C.String
import Foreign.C.Types (CLong, CDouble, CInt, CChar)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import Torch.Raw.Tensor.Generic
import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dim
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorLapack
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath

data UpperLower = Upper | Lower deriving (Eq, Show)

toChar :: UpperLower -> IO CString
toChar Upper = newCString "U"
toChar Lower = newCString "L"

-- X, LU = gesv(B, A) returns the solution of AX = B and LU contains L and
-- U factors for LU factorization of A. A has to be a square and non-singular
-- matrix (2D Tensor). A and LU are m × m, X is m × k and B is m × k. If resb
-- and resa are given, then they will be used for temporary storage and
-- returning the result.
-- - resa will contain L and U factors for LU factorization of A.
-- - resb will contain the solution X.
-- Note: Irrespective of the original strides, the returned matrices resb and
-- resa will be transposed, i.e. with strides 1, m instead of m, 1.
-- TH_API void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
tds_gesv :: KnownNatDim d => TDS '[d] -> TDS '[d, d] -> (TDS '[d, d], TDS '[d, d])
tds_gesv b a = unsafePerformIO $ do
  let (rb, ra) = (tds_new, tds_new)
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_gesv prb pra pb pa
  pure (rb, ra)
{-# NOINLINE tds_gesv #-}

-- TH_API void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag);

-- gels([resb, resa,] b, a)
-- Solution of least squares and least norm problems for a full rank m × n
-- matrix A.
-- - If n ≤ m, then solve ||AX-B||_F.
-- - If n > m , then solve min ||X||_F s.t. AX = B.
-- On return, first n rows of x matrix contains the solution and the rest
-- contains residual information. Square root of sum squares of elements of each
-- column of x starting at row n + 1 is the residual for corresponding column.
-- Note: Irrespective of the original strides, the returned matrices resb and
-- resa will be transposed, i.e. with strides 1, m instead of m, 1.
-- TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
tds_gels
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> TDS d4 -> (TDS d1, TDS d2)
tds_gels b a = unsafePerformIO $ do
  let (rb, ra) = (tds_new, tds_new)
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $c_THDoubleTensor_gels prb pra pb pa
  pure (rb, ra)
{-# NOINLINE tds_gels #-}

-- TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);

-- TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);

-- TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu);

-- TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);

-- TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a);
tds_getri a = do
  let ra = tds_new
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_getri pra pa
  pure ra

-- Cholesky Decomposition of 2D Tensor A. The matrix A has to be a positive-definite and either symmetric or complex Hermitian.
-- The factorization has the form
--  A = U**T * U,   if UPLO = 'U', or
--  A = L  * L**T,  if UPLO = 'L',
-- where U is an upper triangular matrix and L is lower triangular.
-- The optional character uplo = {'U', 'L'} specifies whether the upper or lower triangulardecomposition should be returned. By default, uplo = 'U'.
-- U = torch.potrf(A, 'U') returns the upper triangular Cholesky decomposition of A.
-- L = torch.potrf(A, 'L') returns the lower triangular Cholesky decomposition of A.
-- TH_API void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo);
tds_potrf :: KnownNatDim d => TDS '[d, d] -> UpperLower -> TDS '[d, d]
tds_potrf a ul = unsafePerformIO $ do
  let ra = tds_new
  ulC <- toChar ul
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_potrf pra pa ulC
  pure ra
{-# NOINLINE tds_potrf #-}

-- Returns the solution to linear system AX = B using the Cholesky decomposition
-- chol of 2D Tensor A. Square matrix chol should be triangular; and, righthand
-- side matrix B should be of full rank. Optional character ul = Upper / Lower
-- specifies matrix chol as either upper or lower triangular
-- TH_API void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);

tds_potrs
  :: SingDimensions d1
  => TDS d2 -> TDS d3 -> UpperLower -> TDS d1
tds_potrs b a ul = unsafePerformIO $ do
  let rb = tds_new
  ulC <- toChar ul
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_potrs prb pb pa ulC
  pure rb
{-# NOINLINE tds_potrs #-}

-- TH_API void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);
-- Returns the inverse of 2D Tensor A given its Cholesky decomposition chol.
-- Square matrix chol should be triangular.
-- ul specifies matrix chol as either upper or lower triangular

tds_potri :: SingDimensions d1 => TDS d2 -> UpperLower -> TDS d1
tds_potri a ul = unsafePerformIO $ do
  let ra = tds_new
  ulC <- toChar ul
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_potri pra pa ulC
  pure ra
{-# NOINLINE tds_potri #-}

-- Compute a QR decomposition of the matrix x: matrices q and r such that x = q
-- * r, with q orthogonal and r upper triangular. This returns the thin
-- (reduced) QR factorization. Note that precision may be lost if the magnitudes
-- of the elements of x are large. Note also that, while it should always give
-- you a valid decomposition, it may not give you the same one across platforms
-- - it will depend on your LAPACK implementation. Note: Irrespective of the
-- original strides, the returned matrix q will be transposed, i.e. with strides
-- 1, m instead of m, 1.
-- TH_API void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);

tds_qr
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> IO (TDS d1, TDS d2)
tds_qr a = do
  let (rq, rr) = (tds_new, tds_new)
  runManaged $ do
    prq <- managed $ withForeignPtr (tdsTensor rq)
    prr <- managed $ withForeignPtr (tdsTensor rr)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_THDoubleTensor_qr prq prr pa
  pure (rq, rr)
{-# NOINLINE tds_qr #-}

-- This is a low-level function for calling LAPACK directly. You'll generally
-- want to use torch.qr() instead. Computes a QR decomposition of a, but without
-- constructing Q and R as explicit separate matrices. Rather, this directly
-- calls the underlying LAPACK function ?geqrf which produces a sequence of
-- 'elementary reflectors'. See LAPACK documentation for further details.
-- TH_API void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
tds_geqrf
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> IO (TDS d1, TDS d2)
tds_geqrf a = do
  let (ra, rtau) = (tds_new, tds_new)
  withForeignPtr (tdsTensor ra)
    (\pra ->
      withForeignPtr (tdsTensor rtau)
        (\prtau ->
           withForeignPtr (tdsTensor a)
             (\pa ->
                 c_THDoubleTensor_geqrf pra prtau pa
             )
        )
    )
  pure (ra, rtau)
{-# NOINLINE tds_geqrf #-}

-- This is a low-level function for calling LAPACK directly. You'll generally
-- want to use torch.qr() instead. Constructs a Q matrix from a sequence of
-- elementary reflectors, such as that given by torch.geqrf. See LAPACK
-- documentation for further details.
-- TH_API void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
tds_orgqr
  :: SingDimensions d1
  => TDS d2 -> TDS d3 -> TDS d1
tds_orgqr a tau = unsafePerformIO $ do
  let ra = tds_new
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    ptau <- managed $ withForeignPtr (tdsTensor tau)
    liftIO $ c_THDoubleTensor_orgqr pra pa ptau
  pure ra
{-# NOINLINE tds_orgqr #-}

-- TH_API void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans);

-- TH_API void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol);

-- TH_API void THTensor_(btrifact)(THTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THTensor *a);
-- TH_API void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots);

ct = do
  let x = (tds_fromList [10.0,8.0,2.0, 3.0, 27.0, 8.0, 4.0, 10.0, 3.0] :: TDS [3,3])
  tds_p x
  let y = tds_potrf x Upper
  tds_p y
  --tds_p (y !*! (tds_trans y))
  tds_p ((tds_trans y) !*! y)


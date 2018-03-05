module Torch.Core.Tensor.Static.Lapack
  ( gesv
  , trtrs
  , gels
  , syev
  , geev
  , gesvd
  , gesvd2
  , getri
  , potrf
  , potrs
  , potri
  , qr
  , geqrf
  , orgqr
  , ormqr
  , pstrf
  , btrifact
  , btrisolve
  ) where


import Data.Singletons
import Data.Singletons.TypeLits
-- import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )

import Torch.Raw.Tensor.Generic
import Torch.Core.Tensor.Types
import Torch.Dimensions
import Torch.Types.TH
import Torch.FFI.TH.Double.Tensor
import Torch.FFI.TH.Double.TensorMath
import Torch.FFI.TH.Double.TensorLapack
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
gesv :: KnownNatDim d => TDS '[d] -> TDS '[d, d] -> (TDS '[d, d], TDS '[d, d])
gesv b a = unsafePerformIO $ do
  let (rb, ra) = (new, new)
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_gesv prb pra pb pa
  pure (rb, ra)
{-# NOINLINE gesv #-}

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
gels
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> TDS d4 -> (TDS d1, TDS d2)
gels b a = unsafePerformIO $ do
  let (rb, ra) = (new, new)
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $c_Torch.FFI.TH.Double.Tensor_gels prb pra pb pa
  pure (rb, ra)
{-# NOINLINE gels #-}

-- TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);

-- TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);

-- TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu);

-- TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);

-- TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a);
getri a = do
  let ra = new
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_getri pra pa
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
potrf :: KnownNatDim d => TDS '[d, d] -> UpperLower -> TDS '[d, d]
potrf a ul = unsafePerformIO $ do
  let ra = new
  ulC <- toChar ul
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_potrf pra pa ulC
  pure ra
{-# NOINLINE potrf #-}

-- Returns the solution to linear system AX = B using the Cholesky decomposition
-- chol of 2D Tensor A. Square matrix chol should be triangular; and, righthand
-- side matrix B should be of full rank. Optional character ul = Upper / Lower
-- specifies matrix chol as either upper or lower triangular
-- TH_API void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);

potrs
  :: SingDimensions d1
  => TDS d2 -> TDS d3 -> UpperLower -> TDS d1
potrs b a ul = unsafePerformIO $ do
  let rb = new
  ulC <- toChar ul
  runManaged $ do
    prb <- managed $ withForeignPtr (tdsTensor rb)
    pb <- managed $ withForeignPtr (tdsTensor b)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_potrs prb pb pa ulC
  pure rb
{-# NOINLINE potrs #-}

-- TH_API void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);
-- Returns the inverse of 2D Tensor A given its Cholesky decomposition chol.
-- Square matrix chol should be triangular.
-- ul specifies matrix chol as either upper or lower triangular

potri :: SingDimensions d1 => TDS d2 -> UpperLower -> TDS d1
potri a ul = unsafePerformIO $ do
  let ra = new
  ulC <- toChar ul
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_potri pra pa ulC
  pure ra
{-# NOINLINE potri #-}

-- Compute a QR decomposition of the matrix x: matrices q and r such that x = q
-- * r, with q orthogonal and r upper triangular. This returns the thin
-- (reduced) QR factorization. Note that precision may be lost if the magnitudes
-- of the elements of x are large. Note also that, while it should always give
-- you a valid decomposition, it may not give you the same one across platforms
-- - it will depend on your LAPACK implementation. Note: Irrespective of the
-- original strides, the returned matrix q will be transposed, i.e. with strides
-- 1, m instead of m, 1.
-- TH_API void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);

qr
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> IO (TDS d1, TDS d2)
qr a = do
  let (rq, rr) = (new, new)
  runManaged $ do
    prq <- managed $ withForeignPtr (tdsTensor rq)
    prr <- managed $ withForeignPtr (tdsTensor rr)
    pa <- managed $ withForeignPtr (tdsTensor a)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_qr prq prr pa
  pure (rq, rr)
{-# NOINLINE qr #-}

-- This is a low-level function for calling LAPACK directly. You'll generally
-- want to use torch.qr() instead. Computes a QR decomposition of a, but without
-- constructing Q and R as explicit separate matrices. Rather, this directly
-- calls the underlying LAPACK function ?geqrf which produces a sequence of
-- 'elementary reflectors'. See LAPACK documentation for further details.
-- TH_API void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
geqrf
  :: (SingDimensions d1, SingDimensions d2)
  => TDS d3 -> IO (TDS d1, TDS d2)
geqrf a = do
  let (ra, rtau) = (new, new)
  withForeignPtr (tdsTensor ra)
    (\pra ->
      withForeignPtr (tdsTensor rtau)
        (\prtau ->
           withForeignPtr (tdsTensor a)
             (\pa ->
                 c_Torch.FFI.TH.Double.Tensor_geqrf pra prtau pa
             )
        )
    )
  pure (ra, rtau)
{-# NOINLINE geqrf #-}

-- This is a low-level function for calling LAPACK directly. You'll generally
-- want to use torch.qr() instead. Constructs a Q matrix from a sequence of
-- elementary reflectors, such as that given by torch.geqrf. See LAPACK
-- documentation for further details.
-- TH_API void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
orgqr
  :: SingDimensions d1
  => TDS d2 -> TDS d3 -> TDS d1
orgqr a tau = unsafePerformIO $ do
  let ra = new
  runManaged $ do
    pra <- managed $ withForeignPtr (tdsTensor ra)
    pa <- managed $ withForeignPtr (tdsTensor a)
    ptau <- managed $ withForeignPtr (tdsTensor tau)
    liftIO $ c_Torch.FFI.TH.Double.Tensor_orgqr pra pa ptau
  pure ra
{-# NOINLINE orgqr #-}

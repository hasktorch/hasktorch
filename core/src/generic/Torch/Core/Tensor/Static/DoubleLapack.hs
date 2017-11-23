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
  -- tds_potrf,
  -- tds_potrs,
  -- tds_potri,
  tds_qr,
  tds_geqrf,
  tds_orgqr,
  -- tds_ormqr,
  -- tds_pstrf,
  -- tds_btrifact,
  -- tds_btrisolve

  ) where


import Data.Singletons
import Data.Singletons.TypeLits
import Foreign (Ptr)
import Foreign.C.Types (CLong, CDouble, CInt)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Double
import Torch.Core.Tensor.Long
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorLapack
import Torch.Core.Tensor.Static.Double


-- TH_API void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
tds_gesv b a = do
  let (rb, ra) = (tds_new, tds_new)
  withForeignPtr (tdsTensor rb)
    (\prb ->
      withForeignPtr (tdsTensor ra)
        (\pra ->
           withForeignPtr (tdsTensor b)
             (\pb ->
                withForeignPtr (tdsTensor a)
                  (\pa ->
                     c_THDoubleTensor_gesv prb pra pb pa
                  )
             )
        )
    )
  pure (rb, ra)

-- TH_API void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag);

-- TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
tds_gels b a = do
  let (rb, ra) = (tds_new, tds_new)
  withForeignPtr (tdsTensor rb)
    (\prb ->
      withForeignPtr (tdsTensor ra)
        (\pra ->
           withForeignPtr (tdsTensor b)
             (\pb ->
                withForeignPtr (tdsTensor a)
                  (\pa ->
                     c_THDoubleTensor_gels prb pra pb pa
                  )
             )
        )
    )
  pure (rb, ra)



-- TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);
-- TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);

-- TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu);
-- TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);

-- TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a);
tds_getri a = do
  let ra = tds_new
  withForeignPtr (tdsTensor ra)
    (\pra ->
        withForeignPtr (tdsTensor a)
          (\pa ->
              c_THDoubleTensor_getri pra pa
          )
    )
  pure ra

-- TH_API void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo);
-- TH_API void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);
-- TH_API void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);

-- TH_API void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);
tds_qr a = do
  let (rq, rr) = (tds_new, tds_new)
  withForeignPtr (tdsTensor rq)
    (\prq ->
      withForeignPtr (tdsTensor rr)
        (\prr ->
           withForeignPtr (tdsTensor a)
             (\pa ->
                 c_THDoubleTensor_qr prq prr pa
             )
        )
    )
  pure (rq, rr)

-- TH_API void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
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

-- TH_API void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
tds_orgqr a tau = do
  let ra = tds_new
  withForeignPtr (tdsTensor ra)
    (\pra ->
      withForeignPtr (tdsTensor a)
        (\pa ->
           withForeignPtr (tdsTensor tau)
             (\ptau ->
                 c_THDoubleTensor_orgqr pra pa ptau
             )
        )
    )
  pure ra




-- TH_API void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans);
-- TH_API void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol);

-- TH_API void THTensor_(btrifact)(THTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THTensor *a);
-- TH_API void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots);





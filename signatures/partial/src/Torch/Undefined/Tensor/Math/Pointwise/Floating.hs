module Torch.Undefined.Tensor.Math.Pointwise.Floating where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global

c_cinv         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_cinv         = undefined
c_sigmoid      :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_sigmoid      = undefined
c_log          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_log          = undefined
c_lgamma       :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_lgamma       = undefined
c_log1p        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_log1p        = undefined
c_exp          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_exp          = undefined
c_cos          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_cos          = undefined
c_acos         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_acos         = undefined
c_cosh         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_cosh         = undefined
c_sin          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_sin          = undefined
c_asin         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_asin         = undefined
c_sinh         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_sinh         = undefined
c_tan          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_tan          = undefined
c_atan         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_atan         = undefined
c_atan2        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_atan2        = undefined
c_tanh         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_tanh         = undefined
c_erf          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_erf          = undefined
c_erfinv       :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_erfinv       = undefined
c_pow          :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CReal -> IO ()
c_pow          = undefined
c_tpow         :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> IO ()
c_tpow         = undefined
c_sqrt         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_sqrt         = undefined
c_rsqrt        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_rsqrt        = undefined
c_ceil         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_ceil         = undefined
c_floor        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_floor        = undefined
c_round        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_round        = undefined
c_trunc        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_trunc        = undefined
c_frac         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_frac         = undefined
c_lerp         :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> CReal -> IO ()
c_lerp         = undefined


{-
c_rand         :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr CIndexStorage -> IO ()
c_randn        :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr CIndexStorage -> IO ()
-}

{-
UNKNOWN, but in TH
-- c_histc        :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CLLong -> CReal -> CReal -> IO ()
-- c_bhistc       :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CLLong -> CReal -> CReal -> IO ()
-}

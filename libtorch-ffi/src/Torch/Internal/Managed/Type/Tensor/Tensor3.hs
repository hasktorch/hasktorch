
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Tensor.Tensor3 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Tensor.Tensor3 as Unmanaged





tensor___iand___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___iand___t = cast2 Unmanaged.tensor___iand___t

tensor_bitwise_or_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or_s = cast2 Unmanaged.tensor_bitwise_or_s

tensor_bitwise_or_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or_t = cast2 Unmanaged.tensor_bitwise_or_t

tensor_bitwise_or__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or__s = cast2 Unmanaged.tensor_bitwise_or__s

tensor_bitwise_or__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or__t = cast2 Unmanaged.tensor_bitwise_or__t

tensor___or___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___or___s = cast2 Unmanaged.tensor___or___s

tensor___or___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___or___t = cast2 Unmanaged.tensor___or___t

tensor___ior___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ior___s = cast2 Unmanaged.tensor___ior___s

tensor___ior___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ior___t = cast2 Unmanaged.tensor___ior___t

tensor_bitwise_xor_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor_s = cast2 Unmanaged.tensor_bitwise_xor_s

tensor_bitwise_xor_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor_t = cast2 Unmanaged.tensor_bitwise_xor_t

tensor_bitwise_xor__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor__s = cast2 Unmanaged.tensor_bitwise_xor__s

tensor_bitwise_xor__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor__t = cast2 Unmanaged.tensor_bitwise_xor__t

tensor___xor___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___xor___s = cast2 Unmanaged.tensor___xor___s

tensor___xor___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___xor___t = cast2 Unmanaged.tensor___xor___t

tensor___ixor___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ixor___s = cast2 Unmanaged.tensor___ixor___s

tensor___ixor___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ixor___t = cast2 Unmanaged.tensor___ixor___t

tensor___lshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___lshift___s = cast2 Unmanaged.tensor___lshift___s

tensor___lshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___lshift___t = cast2 Unmanaged.tensor___lshift___t

tensor___ilshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ilshift___s = cast2 Unmanaged.tensor___ilshift___s

tensor___ilshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ilshift___t = cast2 Unmanaged.tensor___ilshift___t

tensor___rshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___rshift___s = cast2 Unmanaged.tensor___rshift___s

tensor___rshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___rshift___t = cast2 Unmanaged.tensor___rshift___t

tensor___irshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___irshift___s = cast2 Unmanaged.tensor___irshift___s

tensor___irshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___irshift___t = cast2 Unmanaged.tensor___irshift___t

tensor_lgamma_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lgamma_ = cast1 Unmanaged.tensor_lgamma_

tensor_atan2__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan2__t = cast2 Unmanaged.tensor_atan2__t

tensor_tril__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_tril__l = cast2 Unmanaged.tensor_tril__l

tensor_triu__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_triu__l = cast2 Unmanaged.tensor_triu__l

tensor_digamma_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_digamma_ = cast1 Unmanaged.tensor_digamma_

tensor_polygamma__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_polygamma__l = cast2 Unmanaged.tensor_polygamma__l

tensor_renorm__sls
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> Int64
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_renorm__sls = cast4 Unmanaged.tensor_renorm__sls

tensor_pow__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_pow__s = cast2 Unmanaged.tensor_pow__s

tensor_pow__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_pow__t = cast2 Unmanaged.tensor_pow__t

tensor_lerp__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lerp__ts = cast3 Unmanaged.tensor_lerp__ts

tensor_lerp__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lerp__tt = cast3 Unmanaged.tensor_lerp__tt

tensor_fmod__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_fmod__s = cast2 Unmanaged.tensor_fmod__s

tensor_fmod__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmod__t = cast2 Unmanaged.tensor_fmod__t

tensor_remainder__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_remainder__s = cast2 Unmanaged.tensor_remainder__s

tensor_remainder__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_remainder__t = cast2 Unmanaged.tensor_remainder__t

tensor_addbmm__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addbmm__ttss = cast5 Unmanaged.tensor_addbmm__ttss

tensor_addbmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addbmm_ttss = cast5 Unmanaged.tensor_addbmm_ttss

tensor_addcdiv__tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcdiv__tts = cast4 Unmanaged.tensor_addcdiv__tts

tensor_random__llG
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__llG = cast4 Unmanaged.tensor_random__llG

tensor_random__lG
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__lG = cast3 Unmanaged.tensor_random__lG

tensor_random__G
  :: ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__G = cast2 Unmanaged.tensor_random__G

tensor_uniform__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_uniform__ddG = cast4 Unmanaged.tensor_uniform__ddG

tensor_normal__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_normal__ddG = cast4 Unmanaged.tensor_normal__ddG

tensor_cauchy__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_cauchy__ddG = cast4 Unmanaged.tensor_cauchy__ddG

tensor_log_normal__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_log_normal__ddG = cast4 Unmanaged.tensor_log_normal__ddG

tensor_exponential__dG
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_exponential__dG = cast3 Unmanaged.tensor_exponential__dG

tensor_geometric__dG
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_geometric__dG = cast3 Unmanaged.tensor_geometric__dG

tensor_diag_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diag_l = cast2 Unmanaged.tensor_diag_l

tensor_triu_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_triu_l = cast2 Unmanaged.tensor_triu_l

tensor_tril_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_tril_l = cast2 Unmanaged.tensor_tril_l

tensor_trace
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trace = cast1 Unmanaged.tensor_trace

tensor_ne_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ne_s = cast2 Unmanaged.tensor_ne_s

tensor_ne_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ne_t = cast2 Unmanaged.tensor_ne_t

tensor_not_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_not_equal_s = cast2 Unmanaged.tensor_not_equal_s

tensor_not_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_not_equal_t = cast2 Unmanaged.tensor_not_equal_t

tensor_not_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_not_equal__s = cast2 Unmanaged.tensor_not_equal__s

tensor_not_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_not_equal__t = cast2 Unmanaged.tensor_not_equal__t

tensor_eq_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_eq_s = cast2 Unmanaged.tensor_eq_s

tensor_eq_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_eq_t = cast2 Unmanaged.tensor_eq_t

tensor_ge_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ge_s = cast2 Unmanaged.tensor_ge_s

tensor_ge_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ge_t = cast2 Unmanaged.tensor_ge_t

tensor_greater_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_equal_s = cast2 Unmanaged.tensor_greater_equal_s

tensor_greater_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_equal_t = cast2 Unmanaged.tensor_greater_equal_t

tensor_greater_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_equal__s = cast2 Unmanaged.tensor_greater_equal__s

tensor_greater_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_equal__t = cast2 Unmanaged.tensor_greater_equal__t

tensor_le_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_le_s = cast2 Unmanaged.tensor_le_s

tensor_le_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_le_t = cast2 Unmanaged.tensor_le_t

tensor_less_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_equal_s = cast2 Unmanaged.tensor_less_equal_s

tensor_less_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_equal_t = cast2 Unmanaged.tensor_less_equal_t

tensor_less_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_equal__s = cast2 Unmanaged.tensor_less_equal__s

tensor_less_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_equal__t = cast2 Unmanaged.tensor_less_equal__t

tensor_gt_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_gt_s = cast2 Unmanaged.tensor_gt_s

tensor_gt_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gt_t = cast2 Unmanaged.tensor_gt_t

tensor_greater_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_s = cast2 Unmanaged.tensor_greater_s

tensor_greater_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_t = cast2 Unmanaged.tensor_greater_t

tensor_greater__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater__s = cast2 Unmanaged.tensor_greater__s

tensor_greater__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater__t = cast2 Unmanaged.tensor_greater__t

tensor_lt_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lt_s = cast2 Unmanaged.tensor_lt_s

tensor_lt_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lt_t = cast2 Unmanaged.tensor_lt_t

tensor_less_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_s = cast2 Unmanaged.tensor_less_s

tensor_less_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_t = cast2 Unmanaged.tensor_less_t

tensor_less__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less__s = cast2 Unmanaged.tensor_less__s

tensor_less__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less__t = cast2 Unmanaged.tensor_less__t

tensor_take_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_take_t = cast2 Unmanaged.tensor_take_t

tensor_index_select_lt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_select_lt = cast3 Unmanaged.tensor_index_select_lt

tensor_index_select_nt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_select_nt = cast3 Unmanaged.tensor_index_select_nt

tensor_masked_select_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_select_t = cast2 Unmanaged.tensor_masked_select_t

tensor_nonzero
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nonzero = cast1 Unmanaged.tensor_nonzero

tensor_nonzero_numpy
  :: ForeignPtr Tensor
  -> IO (ForeignPtr TensorList)
tensor_nonzero_numpy = cast1 Unmanaged.tensor_nonzero_numpy

tensor_gather_ltb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_gather_ltb = cast4 Unmanaged.tensor_gather_ltb

tensor_gather_ntb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_gather_ntb = cast4 Unmanaged.tensor_gather_ntb

tensor_addcmul_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcmul_tts = cast4 Unmanaged.tensor_addcmul_tts

tensor_addcmul__tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcmul__tts = cast4 Unmanaged.tensor_addcmul__tts

tensor_addcdiv_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcdiv_tts = cast4 Unmanaged.tensor_addcdiv_tts

tensor_lstsq_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_lstsq_t = cast2 Unmanaged.tensor_lstsq_t

tensor_triangular_solve_tbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_triangular_solve_tbbb = cast5 Unmanaged.tensor_triangular_solve_tbbb

tensor_symeig_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_symeig_bb = cast3 Unmanaged.tensor_symeig_bb

tensor_eig_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_eig_b = cast2 Unmanaged.tensor_eig_b

tensor_svd_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
tensor_svd_bb = cast3 Unmanaged.tensor_svd_bb

tensor_cholesky_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_b = cast2 Unmanaged.tensor_cholesky_b

tensor_cholesky_solve_tb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_solve_tb = cast3 Unmanaged.tensor_cholesky_solve_tb

tensor_solve_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_solve_t = cast2 Unmanaged.tensor_solve_t

tensor_cholesky_inverse_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_inverse_b = cast2 Unmanaged.tensor_cholesky_inverse_b

tensor_qr_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_qr_b = cast2 Unmanaged.tensor_qr_b

tensor_geqrf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_geqrf = cast1 Unmanaged.tensor_geqrf

tensor_orgqr_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_orgqr_t = cast2 Unmanaged.tensor_orgqr_t

tensor_ormqr_ttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_ormqr_ttbb = cast5 Unmanaged.tensor_ormqr_ttbb

tensor_lu_solve_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lu_solve_tt = cast3 Unmanaged.tensor_lu_solve_tt

tensor_multinomial_lbG
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_multinomial_lbG = cast4 Unmanaged.tensor_multinomial_lbG

tensor_lgamma
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lgamma = cast1 Unmanaged.tensor_lgamma

tensor_digamma
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_digamma = cast1 Unmanaged.tensor_digamma

tensor_polygamma_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_polygamma_l = cast2 Unmanaged.tensor_polygamma_l

tensor_erfinv
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfinv = cast1 Unmanaged.tensor_erfinv

tensor_erfinv_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfinv_ = cast1 Unmanaged.tensor_erfinv_

tensor_i0
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_i0 = cast1 Unmanaged.tensor_i0

tensor_i0_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_i0_ = cast1 Unmanaged.tensor_i0_

tensor_sign
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign = cast1 Unmanaged.tensor_sign

tensor_sign_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign_ = cast1 Unmanaged.tensor_sign_

tensor_signbit
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_signbit = cast1 Unmanaged.tensor_signbit

tensor_dist_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_dist_ts = cast3 Unmanaged.tensor_dist_ts

tensor_atan2_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan2_t = cast2 Unmanaged.tensor_atan2_t

tensor_lerp_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lerp_ts = cast3 Unmanaged.tensor_lerp_ts

tensor_lerp_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lerp_tt = cast3 Unmanaged.tensor_lerp_tt

tensor_histc_lss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_histc_lss = cast4 Unmanaged.tensor_histc_lss

tensor_fmod_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_fmod_s = cast2 Unmanaged.tensor_fmod_s

tensor_fmod_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmod_t = cast2 Unmanaged.tensor_fmod_t

tensor_hypot_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_hypot_t = cast2 Unmanaged.tensor_hypot_t

tensor_hypot__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_hypot__t = cast2 Unmanaged.tensor_hypot__t

tensor_nextafter_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nextafter_t = cast2 Unmanaged.tensor_nextafter_t

tensor_nextafter__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nextafter__t = cast2 Unmanaged.tensor_nextafter__t

tensor_remainder_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_remainder_s = cast2 Unmanaged.tensor_remainder_s

tensor_remainder_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_remainder_t = cast2 Unmanaged.tensor_remainder_t

tensor_min
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min = cast1 Unmanaged.tensor_min

tensor_max
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max = cast1 Unmanaged.tensor_max

tensor_maximum_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_maximum_t = cast2 Unmanaged.tensor_maximum_t

tensor_max_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max_t = cast2 Unmanaged.tensor_max_t

tensor_minimum_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_minimum_t = cast2 Unmanaged.tensor_minimum_t

tensor_min_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min_t = cast2 Unmanaged.tensor_min_t

tensor_median
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_median = cast1 Unmanaged.tensor_median

tensor_sort_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_lb = cast3 Unmanaged.tensor_sort_lb

tensor_sort_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_nb = cast3 Unmanaged.tensor_sort_nb

tensor_argsort_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_argsort_lb = cast3 Unmanaged.tensor_argsort_lb

tensor_argsort_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_argsort_nb = cast3 Unmanaged.tensor_argsort_nb

tensor_topk_llbb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_topk_llbb = cast5 Unmanaged.tensor_topk_llbb

tensor_all
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_all = cast1 Unmanaged.tensor_all

tensor_any
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_any = cast1 Unmanaged.tensor_any

tensor_renorm_sls
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> Int64
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_renorm_sls = cast4 Unmanaged.tensor_renorm_sls

tensor_unfold_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unfold_lll = cast4 Unmanaged.tensor_unfold_lll

tensor_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_equal_t = cast2 Unmanaged.tensor_equal_t

tensor_pow_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_pow_t = cast2 Unmanaged.tensor_pow_t

tensor_alias
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_alias = cast1 Unmanaged.tensor_alias


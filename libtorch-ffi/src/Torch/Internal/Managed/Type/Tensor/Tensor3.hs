
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





tensor___ior___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ior___s = _cast2 Unmanaged.tensor___ior___s

tensor___ior___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ior___t = _cast2 Unmanaged.tensor___ior___t

tensor_bitwise_xor_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor_s = _cast2 Unmanaged.tensor_bitwise_xor_s

tensor_bitwise_xor_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor_t = _cast2 Unmanaged.tensor_bitwise_xor_t

tensor_bitwise_xor__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor__s = _cast2 Unmanaged.tensor_bitwise_xor__s

tensor_bitwise_xor__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_xor__t = _cast2 Unmanaged.tensor_bitwise_xor__t

tensor___xor___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___xor___s = _cast2 Unmanaged.tensor___xor___s

tensor___xor___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___xor___t = _cast2 Unmanaged.tensor___xor___t

tensor___ixor___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ixor___s = _cast2 Unmanaged.tensor___ixor___s

tensor___ixor___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ixor___t = _cast2 Unmanaged.tensor___ixor___t

tensor___lshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___lshift___s = _cast2 Unmanaged.tensor___lshift___s

tensor___lshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___lshift___t = _cast2 Unmanaged.tensor___lshift___t

tensor___ilshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___ilshift___s = _cast2 Unmanaged.tensor___ilshift___s

tensor___ilshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___ilshift___t = _cast2 Unmanaged.tensor___ilshift___t

tensor_bitwise_left_shift_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_left_shift_t = _cast2 Unmanaged.tensor_bitwise_left_shift_t

tensor_bitwise_left_shift__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_left_shift__t = _cast2 Unmanaged.tensor_bitwise_left_shift__t

tensor_bitwise_left_shift_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_left_shift_s = _cast2 Unmanaged.tensor_bitwise_left_shift_s

tensor_bitwise_left_shift__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_left_shift__s = _cast2 Unmanaged.tensor_bitwise_left_shift__s

tensor___rshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___rshift___s = _cast2 Unmanaged.tensor___rshift___s

tensor___rshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___rshift___t = _cast2 Unmanaged.tensor___rshift___t

tensor___irshift___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___irshift___s = _cast2 Unmanaged.tensor___irshift___s

tensor___irshift___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___irshift___t = _cast2 Unmanaged.tensor___irshift___t

tensor_bitwise_right_shift_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_right_shift_t = _cast2 Unmanaged.tensor_bitwise_right_shift_t

tensor_bitwise_right_shift__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_right_shift__t = _cast2 Unmanaged.tensor_bitwise_right_shift__t

tensor_bitwise_right_shift_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_right_shift_s = _cast2 Unmanaged.tensor_bitwise_right_shift_s

tensor_bitwise_right_shift__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_right_shift__s = _cast2 Unmanaged.tensor_bitwise_right_shift__s

tensor_tril__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_tril__l = _cast2 Unmanaged.tensor_tril__l

tensor_triu__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_triu__l = _cast2 Unmanaged.tensor_triu__l

tensor_digamma_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_digamma_ = _cast1 Unmanaged.tensor_digamma_

tensor_lerp__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lerp__ts = _cast3 Unmanaged.tensor_lerp__ts

tensor_lerp__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lerp__tt = _cast3 Unmanaged.tensor_lerp__tt

tensor_addbmm__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addbmm__ttss = _cast5 Unmanaged.tensor_addbmm__ttss

tensor_addbmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addbmm_ttss = _cast5 Unmanaged.tensor_addbmm_ttss

tensor_random__llG
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__llG = _cast4 Unmanaged.tensor_random__llG

tensor_random__lG
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__lG = _cast3 Unmanaged.tensor_random__lG

tensor_random__G
  :: ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__G = _cast2 Unmanaged.tensor_random__G

tensor_uniform__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_uniform__ddG = _cast4 Unmanaged.tensor_uniform__ddG

tensor_cauchy__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_cauchy__ddG = _cast4 Unmanaged.tensor_cauchy__ddG

tensor_log_normal__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_log_normal__ddG = _cast4 Unmanaged.tensor_log_normal__ddG

tensor_exponential__dG
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_exponential__dG = _cast3 Unmanaged.tensor_exponential__dG

tensor_geometric__dG
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_geometric__dG = _cast3 Unmanaged.tensor_geometric__dG

tensor_diag_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diag_l = _cast2 Unmanaged.tensor_diag_l

tensor_cross_tl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_cross_tl = _cast3 Unmanaged.tensor_cross_tl

tensor_triu_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_triu_l = _cast2 Unmanaged.tensor_triu_l

tensor_tril_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_tril_l = _cast2 Unmanaged.tensor_tril_l

tensor_trace
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trace = _cast1 Unmanaged.tensor_trace

tensor_ne_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ne_s = _cast2 Unmanaged.tensor_ne_s

tensor_ne_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ne_t = _cast2 Unmanaged.tensor_ne_t

tensor_ne__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ne__s = _cast2 Unmanaged.tensor_ne__s

tensor_ne__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ne__t = _cast2 Unmanaged.tensor_ne__t

tensor_not_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_not_equal_s = _cast2 Unmanaged.tensor_not_equal_s

tensor_not_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_not_equal_t = _cast2 Unmanaged.tensor_not_equal_t

tensor_not_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_not_equal__s = _cast2 Unmanaged.tensor_not_equal__s

tensor_not_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_not_equal__t = _cast2 Unmanaged.tensor_not_equal__t

tensor_eq_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_eq_s = _cast2 Unmanaged.tensor_eq_s

tensor_eq_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_eq_t = _cast2 Unmanaged.tensor_eq_t

tensor_ge_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ge_s = _cast2 Unmanaged.tensor_ge_s

tensor_ge_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ge_t = _cast2 Unmanaged.tensor_ge_t

tensor_ge__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ge__s = _cast2 Unmanaged.tensor_ge__s

tensor_ge__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ge__t = _cast2 Unmanaged.tensor_ge__t

tensor_greater_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_equal_s = _cast2 Unmanaged.tensor_greater_equal_s

tensor_greater_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_equal_t = _cast2 Unmanaged.tensor_greater_equal_t

tensor_greater_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_equal__s = _cast2 Unmanaged.tensor_greater_equal__s

tensor_greater_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_equal__t = _cast2 Unmanaged.tensor_greater_equal__t

tensor_le_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_le_s = _cast2 Unmanaged.tensor_le_s

tensor_le_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_le_t = _cast2 Unmanaged.tensor_le_t

tensor_le__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_le__s = _cast2 Unmanaged.tensor_le__s

tensor_le__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_le__t = _cast2 Unmanaged.tensor_le__t

tensor_less_equal_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_equal_s = _cast2 Unmanaged.tensor_less_equal_s

tensor_less_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_equal_t = _cast2 Unmanaged.tensor_less_equal_t

tensor_less_equal__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_equal__s = _cast2 Unmanaged.tensor_less_equal__s

tensor_less_equal__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_equal__t = _cast2 Unmanaged.tensor_less_equal__t

tensor_gt_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_gt_s = _cast2 Unmanaged.tensor_gt_s

tensor_gt_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gt_t = _cast2 Unmanaged.tensor_gt_t

tensor_gt__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_gt__s = _cast2 Unmanaged.tensor_gt__s

tensor_gt__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gt__t = _cast2 Unmanaged.tensor_gt__t

tensor_greater_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater_s = _cast2 Unmanaged.tensor_greater_s

tensor_greater_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater_t = _cast2 Unmanaged.tensor_greater_t

tensor_greater__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_greater__s = _cast2 Unmanaged.tensor_greater__s

tensor_greater__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_greater__t = _cast2 Unmanaged.tensor_greater__t

tensor_lt_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lt_s = _cast2 Unmanaged.tensor_lt_s

tensor_lt_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lt_t = _cast2 Unmanaged.tensor_lt_t

tensor_lt__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lt__s = _cast2 Unmanaged.tensor_lt__s

tensor_lt__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lt__t = _cast2 Unmanaged.tensor_lt__t

tensor_less_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less_s = _cast2 Unmanaged.tensor_less_s

tensor_less_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less_t = _cast2 Unmanaged.tensor_less_t

tensor_less__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_less__s = _cast2 Unmanaged.tensor_less__s

tensor_less__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_less__t = _cast2 Unmanaged.tensor_less__t

tensor_take_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_take_t = _cast2 Unmanaged.tensor_take_t

tensor_take_along_dim_tl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_take_along_dim_tl = _cast3 Unmanaged.tensor_take_along_dim_tl

tensor_index_select_lt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_select_lt = _cast3 Unmanaged.tensor_index_select_lt

tensor_index_select_nt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_select_nt = _cast3 Unmanaged.tensor_index_select_nt

tensor_masked_select_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_select_t = _cast2 Unmanaged.tensor_masked_select_t

tensor_nonzero
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nonzero = _cast1 Unmanaged.tensor_nonzero

tensor_nonzero_numpy
  :: ForeignPtr Tensor
  -> IO (ForeignPtr TensorList)
tensor_nonzero_numpy = _cast1 Unmanaged.tensor_nonzero_numpy

tensor_argwhere
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_argwhere = _cast1 Unmanaged.tensor_argwhere

tensor_gather_ltb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_gather_ltb = _cast4 Unmanaged.tensor_gather_ltb

tensor_gather_ntb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_gather_ntb = _cast4 Unmanaged.tensor_gather_ntb

tensor_addcmul_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcmul_tts = _cast4 Unmanaged.tensor_addcmul_tts

tensor_addcmul__tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcmul__tts = _cast4 Unmanaged.tensor_addcmul__tts

tensor_addcdiv_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcdiv_tts = _cast4 Unmanaged.tensor_addcdiv_tts

tensor_addcdiv__tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addcdiv__tts = _cast4 Unmanaged.tensor_addcdiv__tts

tensor_lstsq_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_lstsq_t = _cast2 Unmanaged.tensor_lstsq_t

tensor_triangular_solve_tbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_triangular_solve_tbbb = _cast5 Unmanaged.tensor_triangular_solve_tbbb

tensor_linalg_solve_triangular_tbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_linalg_solve_triangular_tbbb = _cast5 Unmanaged.tensor_linalg_solve_triangular_tbbb

tensor_symeig_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_symeig_bb = _cast3 Unmanaged.tensor_symeig_bb

tensor_eig_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_eig_b = _cast2 Unmanaged.tensor_eig_b

tensor_svd_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
tensor_svd_bb = _cast3 Unmanaged.tensor_svd_bb

tensor_swapaxes_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_swapaxes_ll = _cast3 Unmanaged.tensor_swapaxes_ll

tensor_swapaxes__ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_swapaxes__ll = _cast3 Unmanaged.tensor_swapaxes__ll

tensor_swapdims_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_swapdims_ll = _cast3 Unmanaged.tensor_swapdims_ll

tensor_swapdims__ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_swapdims__ll = _cast3 Unmanaged.tensor_swapdims__ll

tensor_cholesky_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_b = _cast2 Unmanaged.tensor_cholesky_b

tensor_cholesky_solve_tb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_solve_tb = _cast3 Unmanaged.tensor_cholesky_solve_tb

tensor_solve_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_solve_t = _cast2 Unmanaged.tensor_solve_t

tensor_cholesky_inverse_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_cholesky_inverse_b = _cast2 Unmanaged.tensor_cholesky_inverse_b

tensor_qr_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_qr_b = _cast2 Unmanaged.tensor_qr_b

tensor_geqrf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_geqrf = _cast1 Unmanaged.tensor_geqrf

tensor_orgqr_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_orgqr_t = _cast2 Unmanaged.tensor_orgqr_t

tensor_ormqr_ttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_ormqr_ttbb = _cast5 Unmanaged.tensor_ormqr_ttbb

tensor_lu_solve_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lu_solve_tt = _cast3 Unmanaged.tensor_lu_solve_tt

tensor_multinomial_lbG
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_multinomial_lbG = _cast4 Unmanaged.tensor_multinomial_lbG

tensor_lgamma_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lgamma_ = _cast1 Unmanaged.tensor_lgamma_

tensor_lgamma
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lgamma = _cast1 Unmanaged.tensor_lgamma

tensor_digamma
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_digamma = _cast1 Unmanaged.tensor_digamma

tensor_polygamma__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_polygamma__l = _cast2 Unmanaged.tensor_polygamma__l

tensor_erfinv
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfinv = _cast1 Unmanaged.tensor_erfinv

tensor_erfinv_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfinv_ = _cast1 Unmanaged.tensor_erfinv_

tensor_i0
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_i0 = _cast1 Unmanaged.tensor_i0

tensor_i0_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_i0_ = _cast1 Unmanaged.tensor_i0_

tensor_sign
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign = _cast1 Unmanaged.tensor_sign

tensor_sign_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign_ = _cast1 Unmanaged.tensor_sign_

tensor_signbit
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_signbit = _cast1 Unmanaged.tensor_signbit

tensor_dist_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_dist_ts = _cast3 Unmanaged.tensor_dist_ts

tensor_atan2__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan2__t = _cast2 Unmanaged.tensor_atan2__t

tensor_atan2_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan2_t = _cast2 Unmanaged.tensor_atan2_t

tensor_arctan2_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_arctan2_t = _cast2 Unmanaged.tensor_arctan2_t

tensor_arctan2__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_arctan2__t = _cast2 Unmanaged.tensor_arctan2__t

tensor_lerp_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lerp_ts = _cast3 Unmanaged.tensor_lerp_ts

tensor_lerp_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lerp_tt = _cast3 Unmanaged.tensor_lerp_tt

tensor_histc_lss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_histc_lss = _cast4 Unmanaged.tensor_histc_lss

tensor_histogram_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_histogram_ttb = _cast4 Unmanaged.tensor_histogram_ttb

tensor_histogram_latb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr (StdVector CDouble)
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_histogram_latb = _cast5 Unmanaged.tensor_histogram_latb

tensor_fmod_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_fmod_s = _cast2 Unmanaged.tensor_fmod_s

tensor_fmod__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_fmod__s = _cast2 Unmanaged.tensor_fmod__s

tensor_fmod_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmod_t = _cast2 Unmanaged.tensor_fmod_t

tensor_fmod__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmod__t = _cast2 Unmanaged.tensor_fmod__t

tensor_hypot_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_hypot_t = _cast2 Unmanaged.tensor_hypot_t

tensor_hypot__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_hypot__t = _cast2 Unmanaged.tensor_hypot__t

tensor_igamma_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_igamma_t = _cast2 Unmanaged.tensor_igamma_t

tensor_igamma__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_igamma__t = _cast2 Unmanaged.tensor_igamma__t

tensor_igammac_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_igammac_t = _cast2 Unmanaged.tensor_igammac_t

tensor_igammac__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_igammac__t = _cast2 Unmanaged.tensor_igammac__t

tensor_nextafter_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nextafter_t = _cast2 Unmanaged.tensor_nextafter_t

tensor_nextafter__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nextafter__t = _cast2 Unmanaged.tensor_nextafter__t

tensor_remainder_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_remainder_s = _cast2 Unmanaged.tensor_remainder_s

tensor_remainder__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_remainder__s = _cast2 Unmanaged.tensor_remainder__s

tensor_remainder_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_remainder_t = _cast2 Unmanaged.tensor_remainder_t

tensor_remainder__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_remainder__t = _cast2 Unmanaged.tensor_remainder__t

tensor_min
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min = _cast1 Unmanaged.tensor_min

tensor_fmin_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmin_t = _cast2 Unmanaged.tensor_fmin_t

tensor_max
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max = _cast1 Unmanaged.tensor_max

tensor_fmax_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fmax_t = _cast2 Unmanaged.tensor_fmax_t

tensor_maximum_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_maximum_t = _cast2 Unmanaged.tensor_maximum_t

tensor_max_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max_t = _cast2 Unmanaged.tensor_max_t

tensor_minimum_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_minimum_t = _cast2 Unmanaged.tensor_minimum_t

tensor_min_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min_t = _cast2 Unmanaged.tensor_min_t

tensor_quantile_tlbs
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_quantile_tlbs = _cast5 Unmanaged.tensor_quantile_tlbs

tensor_quantile_dlbs
  :: ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> CBool
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_quantile_dlbs = _cast5 Unmanaged.tensor_quantile_dlbs

tensor_nanquantile_tlbs
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_nanquantile_tlbs = _cast5 Unmanaged.tensor_nanquantile_tlbs

tensor_nanquantile_dlbs
  :: ForeignPtr Tensor
  -> CDouble
  -> Int64
  -> CBool
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_nanquantile_dlbs = _cast5 Unmanaged.tensor_nanquantile_dlbs

tensor_sort_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_lb = _cast3 Unmanaged.tensor_sort_lb

tensor_sort_blb
  :: ForeignPtr Tensor
  -> CBool
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_blb = _cast4 Unmanaged.tensor_sort_blb

tensor_sort_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_nb = _cast3 Unmanaged.tensor_sort_nb

tensor_sort_bnb
  :: ForeignPtr Tensor
  -> CBool
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_sort_bnb = _cast4 Unmanaged.tensor_sort_bnb

tensor_msort
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_msort = _cast1 Unmanaged.tensor_msort

tensor_argsort_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_argsort_lb = _cast3 Unmanaged.tensor_argsort_lb

tensor_argsort_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_argsort_nb = _cast3 Unmanaged.tensor_argsort_nb

tensor_topk_llbb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_topk_llbb = _cast5 Unmanaged.tensor_topk_llbb

tensor_all
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_all = _cast1 Unmanaged.tensor_all

tensor_any
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_any = _cast1 Unmanaged.tensor_any

tensor_renorm_sls
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> Int64
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_renorm_sls = _cast4 Unmanaged.tensor_renorm_sls

tensor_renorm__sls
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> Int64
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_renorm__sls = _cast4 Unmanaged.tensor_renorm__sls

tensor_unfold_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unfold_lll = _cast4 Unmanaged.tensor_unfold_lll

tensor_equal_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_equal_t = _cast2 Unmanaged.tensor_equal_t

tensor_pow_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_pow_t = _cast2 Unmanaged.tensor_pow_t

tensor_pow_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_pow_s = _cast2 Unmanaged.tensor_pow_s

tensor_pow__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_pow__s = _cast2 Unmanaged.tensor_pow__s

tensor_pow__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_pow__t = _cast2 Unmanaged.tensor_pow__t

tensor_float_power_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_float_power_t = _cast2 Unmanaged.tensor_float_power_t

tensor_float_power_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_float_power_s = _cast2 Unmanaged.tensor_float_power_s

tensor_float_power__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_float_power__s = _cast2 Unmanaged.tensor_float_power__s

tensor_float_power__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_float_power__t = _cast2 Unmanaged.tensor_float_power__t

tensor_normal__ddG
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_normal__ddG = _cast4 Unmanaged.tensor_normal__ddG

tensor_alias
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_alias = _cast1 Unmanaged.tensor_alias

tensor_isfinite
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isfinite = _cast1 Unmanaged.tensor_isfinite

tensor_isinf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isinf = _cast1 Unmanaged.tensor_isinf

tensor_record_stream_s
  :: ForeignPtr Tensor
  -> ForeignPtr Stream
  -> IO (())
tensor_record_stream_s = _cast2 Unmanaged.tensor_record_stream_s

tensor_isposinf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isposinf = _cast1 Unmanaged.tensor_isposinf

tensor_isneginf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isneginf = _cast1 Unmanaged.tensor_isneginf

-- tensor_special_polygamma_t
--   :: ForeignPtr Tensor
--   -> ForeignPtr Tensor
--   -> IO (ForeignPtr Tensor)
-- tensor_special_polygamma_t = _cast2 Unmanaged.tensor_special_polygamma_t

tensor_det
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_det = _cast1 Unmanaged.tensor_det

tensor_inner_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_inner_t = _cast2 Unmanaged.tensor_inner_t

tensor_outer_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_outer_t = _cast2 Unmanaged.tensor_outer_t

tensor_ger_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ger_t = _cast2 Unmanaged.tensor_ger_t


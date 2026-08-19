
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Tensor.Tensor1 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Tensor.Tensor1 as Unmanaged





tensor_cummin_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_cummin_n = cast2 Unmanaged.tensor_cummin_n

tensor_cumprod_ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumprod_ls = cast3 Unmanaged.tensor_cumprod_ls

tensor_cumprod__ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumprod__ls = cast3 Unmanaged.tensor_cumprod__ls

tensor_cumprod_ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumprod_ns = cast3 Unmanaged.tensor_cumprod_ns

tensor_cumprod__ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumprod__ns = cast3 Unmanaged.tensor_cumprod__ns

tensor_cumsum_ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumsum_ls = cast3 Unmanaged.tensor_cumsum_ls

tensor_cumsum__ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumsum__ls = cast3 Unmanaged.tensor_cumsum__ls

tensor_cumsum_ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumsum_ns = cast3 Unmanaged.tensor_cumsum_ns

tensor_cumsum__ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_cumsum__ns = cast3 Unmanaged.tensor_cumsum__ns

tensor_diag_embed_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diag_embed_lll = cast4 Unmanaged.tensor_diag_embed_lll

tensor_diagflat_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diagflat_l = cast2 Unmanaged.tensor_diagflat_l

tensor_diagonal_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diagonal_lll = cast4 Unmanaged.tensor_diagonal_lll

tensor_diagonal_nnnl
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diagonal_nnnl = cast5 Unmanaged.tensor_diagonal_nnnl

tensor_fill_diagonal__sb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_fill_diagonal__sb = cast3 Unmanaged.tensor_fill_diagonal__sb

tensor_diff_lltt
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_diff_lltt = cast5 Unmanaged.tensor_diff_lltt

tensor_div_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_div_t = cast2 Unmanaged.tensor_div_t

tensor_div__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_div__t = cast2 Unmanaged.tensor_div__t

tensor_div_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_div_ts = cast3 Unmanaged.tensor_div_ts

tensor_div__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_div__ts = cast3 Unmanaged.tensor_div__ts

tensor_div_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_div_s = cast2 Unmanaged.tensor_div_s

tensor_div__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_div__s = cast2 Unmanaged.tensor_div__s

tensor_div_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_div_ss = cast3 Unmanaged.tensor_div_ss

tensor_div__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_div__ss = cast3 Unmanaged.tensor_div__ss

tensor_divide_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_divide_t = cast2 Unmanaged.tensor_divide_t

tensor_divide__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_divide__t = cast2 Unmanaged.tensor_divide__t

tensor_divide_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_divide_s = cast2 Unmanaged.tensor_divide_s

tensor_divide__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_divide__s = cast2 Unmanaged.tensor_divide__s

tensor_divide_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_divide_ts = cast3 Unmanaged.tensor_divide_ts

tensor_divide__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_divide__ts = cast3 Unmanaged.tensor_divide__ts

tensor_divide_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_divide_ss = cast3 Unmanaged.tensor_divide_ss

tensor_divide__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_divide__ss = cast3 Unmanaged.tensor_divide__ss

tensor_true_divide_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_true_divide_t = cast2 Unmanaged.tensor_true_divide_t

tensor_true_divide__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_true_divide__t = cast2 Unmanaged.tensor_true_divide__t

tensor_true_divide_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_true_divide_s = cast2 Unmanaged.tensor_true_divide_s

tensor_true_divide__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_true_divide__s = cast2 Unmanaged.tensor_true_divide__s

tensor_dot_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_dot_t = cast2 Unmanaged.tensor_dot_t

tensor_vdot_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_vdot_t = cast2 Unmanaged.tensor_vdot_t

tensor_new_empty_lo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_empty_lo = cast3 Unmanaged.tensor_new_empty_lo

tensor_new_empty_strided_llo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_empty_strided_llo = cast4 Unmanaged.tensor_new_empty_strided_llo

tensor_new_full_lso
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_full_lso = cast4 Unmanaged.tensor_new_full_lso

tensor_new_zeros_lo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_zeros_lo = cast3 Unmanaged.tensor_new_zeros_lo

tensor_new_ones_lo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_ones_lo = cast3 Unmanaged.tensor_new_ones_lo

tensor_resize__lM
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_resize__lM = cast3 Unmanaged.tensor_resize__lM

tensor_erf
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erf = cast1 Unmanaged.tensor_erf

tensor_erf_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erf_ = cast1 Unmanaged.tensor_erf_

tensor_erfc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfc = cast1 Unmanaged.tensor_erfc

tensor_erfc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_erfc_ = cast1 Unmanaged.tensor_erfc_

tensor_exp
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_exp = cast1 Unmanaged.tensor_exp

tensor_exp_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_exp_ = cast1 Unmanaged.tensor_exp_

tensor_exp2
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_exp2 = cast1 Unmanaged.tensor_exp2

tensor_exp2_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_exp2_ = cast1 Unmanaged.tensor_exp2_

tensor_expm1
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_expm1 = cast1 Unmanaged.tensor_expm1

tensor_expm1_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_expm1_ = cast1 Unmanaged.tensor_expm1_

tensor_expand_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_expand_lb = cast3 Unmanaged.tensor_expand_lb

tensor_expand_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_expand_as_t = cast2 Unmanaged.tensor_expand_as_t

tensor_flatten_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_flatten_ll = cast3 Unmanaged.tensor_flatten_ll

tensor_flatten_lln
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_flatten_lln = cast4 Unmanaged.tensor_flatten_lln

tensor_flatten_nnn
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_flatten_nnn = cast4 Unmanaged.tensor_flatten_nnn

tensor_flatten_Nn
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_flatten_Nn = cast3 Unmanaged.tensor_flatten_Nn

tensor_unflatten_ll
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_unflatten_ll = cast3 Unmanaged.tensor_unflatten_ll

tensor_unflatten_nlN
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr IntArray
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
tensor_unflatten_nlN = cast4 Unmanaged.tensor_unflatten_nlN

tensor_fill__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_fill__s = cast2 Unmanaged.tensor_fill__s

tensor_fill__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fill__t = cast2 Unmanaged.tensor_fill__t

tensor_floor
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_floor = cast1 Unmanaged.tensor_floor

tensor_floor_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_floor_ = cast1 Unmanaged.tensor_floor_

tensor_floor_divide_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_floor_divide_t = cast2 Unmanaged.tensor_floor_divide_t

tensor_floor_divide__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_floor_divide__t = cast2 Unmanaged.tensor_floor_divide__t

tensor_floor_divide_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_floor_divide_s = cast2 Unmanaged.tensor_floor_divide_s

tensor_floor_divide__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_floor_divide__s = cast2 Unmanaged.tensor_floor_divide__s

tensor_frac
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_frac = cast1 Unmanaged.tensor_frac

tensor_frac_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_frac_ = cast1 Unmanaged.tensor_frac_

tensor_gcd_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gcd_t = cast2 Unmanaged.tensor_gcd_t

tensor_gcd__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gcd__t = cast2 Unmanaged.tensor_gcd__t

tensor_lcm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lcm_t = cast2 Unmanaged.tensor_lcm_t

tensor_lcm__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lcm__t = cast2 Unmanaged.tensor_lcm__t

tensor_index_l
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> IO (ForeignPtr Tensor)
tensor_index_l = cast2 Unmanaged.tensor_index_l

tensor_index_copy__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_copy__ltt = cast4 Unmanaged.tensor_index_copy__ltt

tensor_index_copy_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_copy_ltt = cast4 Unmanaged.tensor_index_copy_ltt

tensor_index_copy__ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_copy__ntt = cast4 Unmanaged.tensor_index_copy__ntt

tensor_index_copy_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_copy_ntt = cast4 Unmanaged.tensor_index_copy_ntt

tensor_index_put__ltb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_put__ltb = cast4 Unmanaged.tensor_index_put__ltb

tensor_index_put_ltb
  :: ForeignPtr Tensor
  -> ForeignPtr (C10List (C10Optional Tensor))
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_put_ltb = cast4 Unmanaged.tensor_index_put_ltb

tensor_isclose_tddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_isclose_tddb = cast5 Unmanaged.tensor_isclose_tddb

tensor_isnan
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isnan = cast1 Unmanaged.tensor_isnan

tensor_is_distributed
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_distributed = cast1 Unmanaged.tensor_is_distributed

tensor_is_floating_point
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_floating_point = cast1 Unmanaged.tensor_is_floating_point

tensor_is_complex
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_complex = cast1 Unmanaged.tensor_is_complex

tensor_is_conj
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_conj = cast1 Unmanaged.tensor_is_conj

tensor__is_zerotensor
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor__is_zerotensor = cast1 Unmanaged.tensor__is_zerotensor

tensor_is_neg
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_neg = cast1 Unmanaged.tensor_is_neg

tensor_isreal
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_isreal = cast1 Unmanaged.tensor_isreal

tensor_is_nonzero
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_nonzero = cast1 Unmanaged.tensor_is_nonzero

tensor_is_same_size_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_is_same_size_t = cast2 Unmanaged.tensor_is_same_size_t

tensor_is_signed
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_signed = cast1 Unmanaged.tensor_is_signed

tensor_is_inference
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_inference = cast1 Unmanaged.tensor_is_inference

tensor_kron_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_kron_t = cast2 Unmanaged.tensor_kron_t

tensor_kthvalue_llb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_kthvalue_llb = cast4 Unmanaged.tensor_kthvalue_llb

tensor_kthvalue_lnb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_kthvalue_lnb = cast4 Unmanaged.tensor_kthvalue_lnb

tensor_nan_to_num_ddd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_nan_to_num_ddd = cast4 Unmanaged.tensor_nan_to_num_ddd

tensor_nan_to_num__ddd
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_nan_to_num__ddd = cast4 Unmanaged.tensor_nan_to_num__ddd

tensor_ldexp_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ldexp_t = cast2 Unmanaged.tensor_ldexp_t

tensor_ldexp__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ldexp__t = cast2 Unmanaged.tensor_ldexp__t

tensor_log
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log = cast1 Unmanaged.tensor_log

tensor_log_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log_ = cast1 Unmanaged.tensor_log_

tensor_log10
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log10 = cast1 Unmanaged.tensor_log10

tensor_log10_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log10_ = cast1 Unmanaged.tensor_log10_

tensor_log1p
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log1p = cast1 Unmanaged.tensor_log1p

tensor_log1p_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log1p_ = cast1 Unmanaged.tensor_log1p_

tensor_log2
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log2 = cast1 Unmanaged.tensor_log2

tensor_log2_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_log2_ = cast1 Unmanaged.tensor_log2_

tensor_logaddexp_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logaddexp_t = cast2 Unmanaged.tensor_logaddexp_t

tensor_logaddexp2_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logaddexp2_t = cast2 Unmanaged.tensor_logaddexp2_t

tensor_xlogy_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_xlogy_t = cast2 Unmanaged.tensor_xlogy_t

tensor_xlogy_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_xlogy_s = cast2 Unmanaged.tensor_xlogy_s

tensor_xlogy__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_xlogy__t = cast2 Unmanaged.tensor_xlogy__t

tensor_xlogy__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_xlogy__s = cast2 Unmanaged.tensor_xlogy__s

tensor_log_softmax_ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_log_softmax_ls = cast3 Unmanaged.tensor_log_softmax_ls

tensor_log_softmax_ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_log_softmax_ns = cast3 Unmanaged.tensor_log_softmax_ns

tensor_logcumsumexp_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_logcumsumexp_l = cast2 Unmanaged.tensor_logcumsumexp_l

tensor_logcumsumexp_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_logcumsumexp_n = cast2 Unmanaged.tensor_logcumsumexp_n

tensor_logsumexp_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_logsumexp_lb = cast3 Unmanaged.tensor_logsumexp_lb

tensor_logsumexp_Nb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_logsumexp_Nb = cast3 Unmanaged.tensor_logsumexp_Nb

tensor_matmul_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_matmul_t = cast2 Unmanaged.tensor_matmul_t

tensor_matrix_power_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_matrix_power_l = cast2 Unmanaged.tensor_matrix_power_l

tensor_matrix_exp
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_matrix_exp = cast1 Unmanaged.tensor_matrix_exp

tensor_aminmax_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_aminmax_lb = cast3 Unmanaged.tensor_aminmax_lb

tensor_max_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_max_lb = cast3 Unmanaged.tensor_max_lb

tensor_max_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_max_nb = cast3 Unmanaged.tensor_max_nb

tensor_amax_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_amax_lb = cast3 Unmanaged.tensor_amax_lb

tensor_mean_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_mean_s = cast2 Unmanaged.tensor_mean_s

tensor_mean_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_mean_lbs = cast4 Unmanaged.tensor_mean_lbs

tensor_mean_Nbs
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_mean_Nbs = cast4 Unmanaged.tensor_mean_Nbs

tensor_nanmean_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_nanmean_lbs = cast4 Unmanaged.tensor_nanmean_lbs

tensor_median
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_median = cast1 Unmanaged.tensor_median

tensor_median_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_median_lb = cast3 Unmanaged.tensor_median_lb

tensor_median_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_median_nb = cast3 Unmanaged.tensor_median_nb

tensor_nanmedian
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_nanmedian = cast1 Unmanaged.tensor_nanmedian

tensor_nanmedian_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_nanmedian_lb = cast3 Unmanaged.tensor_nanmedian_lb

tensor_nanmedian_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_nanmedian_nb = cast3 Unmanaged.tensor_nanmedian_nb

tensor_min_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_min_lb = cast3 Unmanaged.tensor_min_lb

tensor_min_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_min_nb = cast3 Unmanaged.tensor_min_nb

tensor_amin_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_amin_lb = cast3 Unmanaged.tensor_amin_lb

tensor_mm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mm_t = cast2 Unmanaged.tensor_mm_t

tensor_mode_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_mode_lb = cast3 Unmanaged.tensor_mode_lb

tensor_mode_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_mode_nb = cast3 Unmanaged.tensor_mode_nb

tensor_mul_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mul_t = cast2 Unmanaged.tensor_mul_t

tensor_mul__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mul__t = cast2 Unmanaged.tensor_mul__t

tensor_mul_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_mul_s = cast2 Unmanaged.tensor_mul_s

tensor_mul__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_mul__s = cast2 Unmanaged.tensor_mul__s

tensor_multiply_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_multiply_t = cast2 Unmanaged.tensor_multiply_t

tensor_multiply__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_multiply__t = cast2 Unmanaged.tensor_multiply__t

tensor_multiply_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_multiply_s = cast2 Unmanaged.tensor_multiply_s

tensor_multiply__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_multiply__s = cast2 Unmanaged.tensor_multiply__s

tensor_mv_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mv_t = cast2 Unmanaged.tensor_mv_t

tensor_mvlgamma_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_mvlgamma_l = cast2 Unmanaged.tensor_mvlgamma_l

tensor_mvlgamma__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_mvlgamma__l = cast2 Unmanaged.tensor_mvlgamma__l

tensor_narrow_copy_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_narrow_copy_lll = cast4 Unmanaged.tensor_narrow_copy_lll

tensor_narrow_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_narrow_lll = cast4 Unmanaged.tensor_narrow_lll

tensor_narrow_ltl
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_narrow_ltl = cast4 Unmanaged.tensor_narrow_ltl

tensor_permute_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_permute_l = cast2 Unmanaged.tensor_permute_l

tensor_numpy_T
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_numpy_T = cast1 Unmanaged.tensor_numpy_T

tensor_matrix_H
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_matrix_H = cast1 Unmanaged.tensor_matrix_H

tensor_mT
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mT = cast1 Unmanaged.tensor_mT

tensor_mH
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_mH = cast1 Unmanaged.tensor_mH

tensor_adjoint
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_adjoint = cast1 Unmanaged.tensor_adjoint

tensor_is_pinned_D
  :: ForeignPtr Tensor
  -> DeviceType
  -> IO (CBool)
tensor_is_pinned_D = cast2 Unmanaged.tensor_is_pinned_D

tensor_pin_memory_D
  :: ForeignPtr Tensor
  -> DeviceType
  -> IO (ForeignPtr Tensor)
tensor_pin_memory_D = cast2 Unmanaged.tensor_pin_memory_D

tensor_pinverse_d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_pinverse_d = cast2 Unmanaged.tensor_pinverse_d

tensor_rad2deg
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_rad2deg = cast1 Unmanaged.tensor_rad2deg

tensor_rad2deg_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_rad2deg_ = cast1 Unmanaged.tensor_rad2deg_

tensor_deg2rad
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_deg2rad = cast1 Unmanaged.tensor_deg2rad

tensor_deg2rad_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_deg2rad_ = cast1 Unmanaged.tensor_deg2rad_

tensor_ravel
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ravel = cast1 Unmanaged.tensor_ravel

tensor_reciprocal
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_reciprocal = cast1 Unmanaged.tensor_reciprocal

tensor_reciprocal_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_reciprocal_ = cast1 Unmanaged.tensor_reciprocal_

tensor_neg
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_neg = cast1 Unmanaged.tensor_neg

tensor_neg_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_neg_ = cast1 Unmanaged.tensor_neg_

tensor_negative
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_negative = cast1 Unmanaged.tensor_negative

tensor_negative_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_negative_ = cast1 Unmanaged.tensor_negative_

tensor_repeat_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_repeat_l = cast2 Unmanaged.tensor_repeat_l

tensor_repeat_interleave_tll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_repeat_interleave_tll = cast4 Unmanaged.tensor_repeat_interleave_tll

tensor_repeat_interleave_lll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_repeat_interleave_lll = cast4 Unmanaged.tensor_repeat_interleave_lll

tensor_reshape_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_reshape_l = cast2 Unmanaged.tensor_reshape_l

tensor__reshape_alias_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor__reshape_alias_ll = cast3 Unmanaged.tensor__reshape_alias_ll

tensor_reshape_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_reshape_as_t = cast2 Unmanaged.tensor_reshape_as_t

tensor_round
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_round = cast1 Unmanaged.tensor_round

tensor_round_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_round_ = cast1 Unmanaged.tensor_round_

tensor_round_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_round_l = cast2 Unmanaged.tensor_round_l

tensor_round__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_round__l = cast2 Unmanaged.tensor_round__l

tensor_relu
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_relu = cast1 Unmanaged.tensor_relu

tensor_relu_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_relu_ = cast1 Unmanaged.tensor_relu_

tensor_prelu_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_prelu_t = cast2 Unmanaged.tensor_prelu_t

tensor_hardshrink_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_hardshrink_s = cast2 Unmanaged.tensor_hardshrink_s

tensor_hardshrink_backward_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_hardshrink_backward_ts = cast3 Unmanaged.tensor_hardshrink_backward_ts

tensor_rsqrt
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_rsqrt = cast1 Unmanaged.tensor_rsqrt

tensor_rsqrt_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_rsqrt_ = cast1 Unmanaged.tensor_rsqrt_

tensor_select_nl
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_select_nl = cast3 Unmanaged.tensor_select_nl

tensor_select_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_select_ll = cast3 Unmanaged.tensor_select_ll

tensor_sigmoid
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sigmoid = cast1 Unmanaged.tensor_sigmoid

tensor_sigmoid_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sigmoid_ = cast1 Unmanaged.tensor_sigmoid_

tensor_logit_d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_logit_d = cast2 Unmanaged.tensor_logit_d

tensor_logit__d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_logit__d = cast2 Unmanaged.tensor_logit__d

tensor_sin
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin = cast1 Unmanaged.tensor_sin

tensor_sin_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin_ = cast1 Unmanaged.tensor_sin_

tensor_sinc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinc = cast1 Unmanaged.tensor_sinc

tensor_sinc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinc_ = cast1 Unmanaged.tensor_sinc_

tensor_sinh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinh = cast1 Unmanaged.tensor_sinh


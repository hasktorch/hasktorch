
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Tensor where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged



newTensor
  :: IO (ForeignPtr Tensor)
newTensor = cast0 Unmanaged.newTensor

newTensor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
newTensor_t = cast1 Unmanaged.newTensor_t





tensor_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dim = cast1 Unmanaged.tensor_dim

tensor_storage_offset
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_storage_offset = cast1 Unmanaged.tensor_storage_offset

tensor_defined
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_defined = cast1 Unmanaged.tensor_defined

tensor_reset
  :: ForeignPtr Tensor
  -> IO (())
tensor_reset = cast1 Unmanaged.tensor_reset

tensor__assign__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__assign__t = cast2 Unmanaged.tensor__assign__t

tensor_is_same_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_is_same_t = cast2 Unmanaged.tensor_is_same_t

tensor_use_count
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_use_count = cast1 Unmanaged.tensor_use_count

tensor_weak_use_count
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_weak_use_count = cast1 Unmanaged.tensor_weak_use_count

tensor_toString
  :: ForeignPtr Tensor
  -> IO (ForeignPtr StdString)
tensor_toString = cast1 Unmanaged.tensor_toString

tensor_sizes
  :: ForeignPtr Tensor
  -> IO (ForeignPtr IntArray)
tensor_sizes = cast1 Unmanaged.tensor_sizes

tensor_strides
  :: ForeignPtr Tensor
  -> IO (ForeignPtr IntArray)
tensor_strides = cast1 Unmanaged.tensor_strides

tensor_ndimension
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_ndimension = cast1 Unmanaged.tensor_ndimension

tensor_is_contiguous
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_contiguous = cast1 Unmanaged.tensor_is_contiguous

tensor_is_non_overlapping_and_dense
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_non_overlapping_and_dense = cast1 Unmanaged.tensor_is_non_overlapping_and_dense

tensor_nbytes
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_nbytes = cast1 Unmanaged.tensor_nbytes

tensor_numel
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_numel = cast1 Unmanaged.tensor_numel

tensor_itemsize
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_itemsize = cast1 Unmanaged.tensor_itemsize

tensor_element_size
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_element_size = cast1 Unmanaged.tensor_element_size

tensor_scalar_type
  :: ForeignPtr Tensor
  -> IO (ScalarType)
tensor_scalar_type = cast1 Unmanaged.tensor_scalar_type

tensor_has_storage
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_has_storage = cast1 Unmanaged.tensor_has_storage

tensor_storage
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Storage)
tensor_storage = cast1 Unmanaged.tensor_storage

tensor_is_alias_of_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_is_alias_of_t = cast2 Unmanaged.tensor_is_alias_of_t

tensor_toType_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_toType_s = cast2 Unmanaged.tensor_toType_s

tensor_toBackend_B
  :: ForeignPtr Tensor
  -> Backend
  -> IO (ForeignPtr Tensor)
tensor_toBackend_B = cast2 Unmanaged.tensor_toBackend_B

tensor_is_variable
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_variable = cast1 Unmanaged.tensor_is_variable

tensor_layout
  :: ForeignPtr Tensor
  -> IO (Layout)
tensor_layout = cast1 Unmanaged.tensor_layout

tensor_get_device
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_get_device = cast1 Unmanaged.tensor_get_device

tensor_is_cuda
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_cuda = cast1 Unmanaged.tensor_is_cuda

tensor_is_hip
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_hip = cast1 Unmanaged.tensor_is_hip

tensor_is_sparse
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_sparse = cast1 Unmanaged.tensor_is_sparse

tensor_is_mkldnn
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_mkldnn = cast1 Unmanaged.tensor_is_mkldnn

tensor_is_quantized
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_quantized = cast1 Unmanaged.tensor_is_quantized

tensor_has_names
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_has_names = cast1 Unmanaged.tensor_has_names

tensor_options
  :: ForeignPtr Tensor
  -> IO (ForeignPtr TensorOptions)
tensor_options = cast1 Unmanaged.tensor_options

tensor_data_ptr
  :: ForeignPtr Tensor
  -> IO (Ptr ())
tensor_data_ptr = cast1 Unmanaged.tensor_data_ptr

tensor_item_int64_t
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_item_int64_t = cast1 Unmanaged.tensor_item_int64_t

tensor_item_float
  :: ForeignPtr Tensor
  -> IO (CFloat)
tensor_item_float = cast1 Unmanaged.tensor_item_float

tensor_item_double
  :: ForeignPtr Tensor
  -> IO (CDouble)
tensor_item_double = cast1 Unmanaged.tensor_item_double

tensor_print
  :: ForeignPtr Tensor
  -> IO (())
tensor_print = cast1 Unmanaged.tensor_print

tensor__iadd__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (())
tensor__iadd__t = cast2 Unmanaged.tensor__iadd__t

tensor__iadd__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (())
tensor__iadd__s = cast2 Unmanaged.tensor__iadd__s

tensor__isub__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (())
tensor__isub__t = cast2 Unmanaged.tensor__isub__t

tensor__isub__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (())
tensor__isub__s = cast2 Unmanaged.tensor__isub__s

tensor__imul__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (())
tensor__imul__t = cast2 Unmanaged.tensor__imul__t

tensor__imul__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (())
tensor__imul__s = cast2 Unmanaged.tensor__imul__s

tensor__idiv__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (())
tensor__idiv__t = cast2 Unmanaged.tensor__idiv__t

tensor__idiv__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (())
tensor__idiv__s = cast2 Unmanaged.tensor__idiv__s

tensor__at__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor__at__s = cast2 Unmanaged.tensor__at__s

tensor__at__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__at__t = cast2 Unmanaged.tensor__at__t

tensor__at__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor__at__l = cast2 Unmanaged.tensor__at__l

tensor_cpu
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cpu = cast1 Unmanaged.tensor_cpu

tensor_cuda
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cuda = cast1 Unmanaged.tensor_cuda

tensor_hip
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_hip = cast1 Unmanaged.tensor_hip

tensor_set_requires_grad_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_set_requires_grad_b = cast2 Unmanaged.tensor_set_requires_grad_b

tensor_requires_grad
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_requires_grad = cast1 Unmanaged.tensor_requires_grad

tensor_grad
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_grad = cast1 Unmanaged.tensor_grad

tensor_backward_tbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (())
tensor_backward_tbb = cast4 Unmanaged.tensor_backward_tbb

tensor_set_data_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (())
tensor_set_data_t = cast2 Unmanaged.tensor_set_data_t

tensor_data
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_data = cast1 Unmanaged.tensor_data

tensor_is_leaf
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_leaf = cast1 Unmanaged.tensor_is_leaf

tensor_output_nr
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_output_nr = cast1 Unmanaged.tensor_output_nr

tensor__version
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__version = cast1 Unmanaged.tensor__version

tensor_requires_grad__b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_requires_grad__b = cast2 Unmanaged.tensor_requires_grad__b

tensor_retain_grad
  :: ForeignPtr Tensor
  -> IO (())
tensor_retain_grad = cast1 Unmanaged.tensor_retain_grad

tensor_align_to_N
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
tensor_align_to_N = cast2 Unmanaged.tensor_align_to_N

tensor_align_to_Nl
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_align_to_Nl = cast3 Unmanaged.tensor_align_to_Nl

tensor_align_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_align_as_t = cast2 Unmanaged.tensor_align_as_t

tensor_refine_names_N
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
tensor_refine_names_N = cast2 Unmanaged.tensor_refine_names_N

tensor_unflatten_nlN
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr IntArray
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
tensor_unflatten_nlN = cast4 Unmanaged.tensor_unflatten_nlN

tensor_unflatten_llN
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr DimnameList
  -> IO (ForeignPtr Tensor)
tensor_unflatten_llN = cast4 Unmanaged.tensor_unflatten_llN

tensor_abs
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_abs = cast1 Unmanaged.tensor_abs

tensor_abs_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_abs_ = cast1 Unmanaged.tensor_abs_

tensor_angle
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_angle = cast1 Unmanaged.tensor_angle

tensor_conj
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_conj = cast1 Unmanaged.tensor_conj

tensor_acos
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_acos = cast1 Unmanaged.tensor_acos

tensor_acos_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_acos_ = cast1 Unmanaged.tensor_acos_

tensor_add_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_add_ts = cast3 Unmanaged.tensor_add_ts

tensor_add__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_add__ts = cast3 Unmanaged.tensor_add__ts

tensor_add_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_add_ss = cast3 Unmanaged.tensor_add_ss

tensor_add__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_add__ss = cast3 Unmanaged.tensor_add__ss

tensor_addmv_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmv_ttss = cast5 Unmanaged.tensor_addmv_ttss

tensor_addmv__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmv__ttss = cast5 Unmanaged.tensor_addmv__ttss

tensor_addr_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addr_ttss = cast5 Unmanaged.tensor_addr_ttss

tensor_addr__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addr__ttss = cast5 Unmanaged.tensor_addr__ttss

tensor_all_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_all_lb = cast3 Unmanaged.tensor_all_lb

tensor_all_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_all_nb = cast3 Unmanaged.tensor_all_nb

tensor_allclose_tddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (CBool)
tensor_allclose_tddb = cast5 Unmanaged.tensor_allclose_tddb

tensor_any_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_any_lb = cast3 Unmanaged.tensor_any_lb

tensor_any_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_any_nb = cast3 Unmanaged.tensor_any_nb

tensor_asin
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_asin = cast1 Unmanaged.tensor_asin

tensor_asin_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_asin_ = cast1 Unmanaged.tensor_asin_

tensor_atan
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan = cast1 Unmanaged.tensor_atan

tensor_atan_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atan_ = cast1 Unmanaged.tensor_atan_

tensor_baddbmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_baddbmm_ttss = cast5 Unmanaged.tensor_baddbmm_ttss

tensor_baddbmm__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_baddbmm__ttss = cast5 Unmanaged.tensor_baddbmm__ttss

tensor_bernoulli_p
  :: ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_bernoulli_p = cast2 Unmanaged.tensor_bernoulli_p

tensor_bernoulli__tp
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_bernoulli__tp = cast3 Unmanaged.tensor_bernoulli__tp

tensor_bernoulli__dp
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_bernoulli__dp = cast3 Unmanaged.tensor_bernoulli__dp

tensor_bernoulli_dp
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_bernoulli_dp = cast3 Unmanaged.tensor_bernoulli_dp

tensor_bincount_tl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_bincount_tl = cast3 Unmanaged.tensor_bincount_tl

tensor_bitwise_not
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_not = cast1 Unmanaged.tensor_bitwise_not

tensor_bitwise_not_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_not_ = cast1 Unmanaged.tensor_bitwise_not_

tensor_logical_not
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_not = cast1 Unmanaged.tensor_logical_not

tensor_logical_not_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_not_ = cast1 Unmanaged.tensor_logical_not_

tensor_logical_xor_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_xor_t = cast2 Unmanaged.tensor_logical_xor_t

tensor_logical_xor__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_xor__t = cast2 Unmanaged.tensor_logical_xor__t

tensor_logical_and_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_and_t = cast2 Unmanaged.tensor_logical_and_t

tensor_logical_and__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_and__t = cast2 Unmanaged.tensor_logical_and__t

tensor_logical_or_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_or_t = cast2 Unmanaged.tensor_logical_or_t

tensor_logical_or__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logical_or__t = cast2 Unmanaged.tensor_logical_or__t

tensor_bmm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bmm_t = cast2 Unmanaged.tensor_bmm_t

tensor_ceil
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ceil = cast1 Unmanaged.tensor_ceil

tensor_ceil_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ceil_ = cast1 Unmanaged.tensor_ceil_

tensor_chunk_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_chunk_ll = cast3 Unmanaged.tensor_chunk_ll

tensor_clamp_max_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_clamp_max_s = cast2 Unmanaged.tensor_clamp_max_s

tensor_clamp_max__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_clamp_max__s = cast2 Unmanaged.tensor_clamp_max__s

tensor_clamp_min_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_clamp_min_s = cast2 Unmanaged.tensor_clamp_min_s

tensor_clamp_min__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_clamp_min__s = cast2 Unmanaged.tensor_clamp_min__s

tensor_contiguous
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_contiguous = cast1 Unmanaged.tensor_contiguous

tensor_copy__tb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_copy__tb = cast3 Unmanaged.tensor_copy__tb

tensor_cos
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cos = cast1 Unmanaged.tensor_cos

tensor_cos_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cos_ = cast1 Unmanaged.tensor_cos_

tensor_cosh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cosh = cast1 Unmanaged.tensor_cosh

tensor_cosh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_cosh_ = cast1 Unmanaged.tensor_cosh_

tensor_cummax_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_cummax_l = cast2 Unmanaged.tensor_cummax_l

tensor_cummax_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_cummax_n = cast2 Unmanaged.tensor_cummax_n

tensor_cummin_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_cummin_l = cast2 Unmanaged.tensor_cummin_l

tensor_cummin_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_cummin_n = cast2 Unmanaged.tensor_cummin_n

tensor_det
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_det = cast1 Unmanaged.tensor_det

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

tensor_dot_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_dot_t = cast2 Unmanaged.tensor_dot_t

tensor_new_empty_lo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_empty_lo = cast3 Unmanaged.tensor_new_empty_lo

tensor_new_full_lso
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_full_lso = cast4 Unmanaged.tensor_new_full_lso

tensor_resize__l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_resize__l = cast2 Unmanaged.tensor_resize__l

tensor_new_zeros_lo
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
tensor_new_zeros_lo = cast3 Unmanaged.tensor_new_zeros_lo

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

tensor_ger_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ger_t = cast2 Unmanaged.tensor_ger_t

tensor_fft_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_fft_lb = cast3 Unmanaged.tensor_fft_lb

tensor_ifft_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_ifft_lb = cast3 Unmanaged.tensor_ifft_lb

tensor_rfft_lbb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_rfft_lbb = cast4 Unmanaged.tensor_rfft_lbb

tensor_irfft_lbbl
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> CBool
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_irfft_lbbl = cast5 Unmanaged.tensor_irfft_lbbl

tensor_index_l
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
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
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_put__ltb = cast4 Unmanaged.tensor_index_put__ltb

tensor_index_put_ltb
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_put_ltb = cast4 Unmanaged.tensor_index_put_ltb

tensor_inverse
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_inverse = cast1 Unmanaged.tensor_inverse

tensor_isclose_tddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_isclose_tddb = cast5 Unmanaged.tensor_isclose_tddb

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

tensor_logdet
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_logdet = cast1 Unmanaged.tensor_logdet

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

tensor_max_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_max_lb = cast3 Unmanaged.tensor_max_lb

tensor_max_values_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_max_values_lb = cast3 Unmanaged.tensor_max_values_lb

tensor_max_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_max_nb = cast3 Unmanaged.tensor_max_nb

tensor_max_values_Nb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_max_values_Nb = cast3 Unmanaged.tensor_max_values_Nb

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

tensor_min_lb
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_min_lb = cast3 Unmanaged.tensor_min_lb

tensor_min_values_lb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_min_values_lb = cast3 Unmanaged.tensor_min_values_lb

tensor_min_nb
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_min_nb = cast3 Unmanaged.tensor_min_nb

tensor_min_values_Nb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_min_values_Nb = cast3 Unmanaged.tensor_min_values_Nb

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

tensor_is_pinned
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_pinned = cast1 Unmanaged.tensor_is_pinned

tensor_pin_memory
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_pin_memory = cast1 Unmanaged.tensor_pin_memory

tensor_pinverse_d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_pinverse_d = cast2 Unmanaged.tensor_pinverse_d

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

tensor_repeat_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_repeat_l = cast2 Unmanaged.tensor_repeat_l

tensor_reshape_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_reshape_l = cast2 Unmanaged.tensor_reshape_l

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

tensor_prelu_backward_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_prelu_backward_tt = cast3 Unmanaged.tensor_prelu_backward_tt

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

tensor_sin
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin = cast1 Unmanaged.tensor_sin

tensor_sin_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin_ = cast1 Unmanaged.tensor_sin_

tensor_sinh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinh = cast1 Unmanaged.tensor_sinh

tensor_sinh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinh_ = cast1 Unmanaged.tensor_sinh_

tensor_detach
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_detach = cast1 Unmanaged.tensor_detach

tensor_detach_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_detach_ = cast1 Unmanaged.tensor_detach_

tensor_size_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (Int64)
tensor_size_l = cast2 Unmanaged.tensor_size_l

tensor_size_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (Int64)
tensor_size_n = cast2 Unmanaged.tensor_size_n

tensor_slice_llll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_slice_llll = cast5 Unmanaged.tensor_slice_llll

tensor_slogdet
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_slogdet = cast1 Unmanaged.tensor_slogdet

tensor_smm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_smm_t = cast2 Unmanaged.tensor_smm_t

tensor_split_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_split_ll = cast3 Unmanaged.tensor_split_ll

tensor_split_with_sizes_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_split_with_sizes_ll = cast3 Unmanaged.tensor_split_with_sizes_ll

tensor_squeeze
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_squeeze = cast1 Unmanaged.tensor_squeeze

tensor_squeeze_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_squeeze_l = cast2 Unmanaged.tensor_squeeze_l

tensor_squeeze_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_squeeze_n = cast2 Unmanaged.tensor_squeeze_n

tensor_squeeze_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_squeeze_ = cast1 Unmanaged.tensor_squeeze_

tensor_squeeze__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_squeeze__l = cast2 Unmanaged.tensor_squeeze__l

tensor_squeeze__n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_squeeze__n = cast2 Unmanaged.tensor_squeeze__n

tensor_sspaddmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sspaddmm_ttss = cast5 Unmanaged.tensor_sspaddmm_ttss

tensor_stride_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (Int64)
tensor_stride_l = cast2 Unmanaged.tensor_stride_l

tensor_stride_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (Int64)
tensor_stride_n = cast2 Unmanaged.tensor_stride_n

tensor_sum_to_size_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_sum_to_size_l = cast2 Unmanaged.tensor_sum_to_size_l

tensor_sqrt
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sqrt = cast1 Unmanaged.tensor_sqrt

tensor_sqrt_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sqrt_ = cast1 Unmanaged.tensor_sqrt_

tensor_square
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_square = cast1 Unmanaged.tensor_square

tensor_square_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_square_ = cast1 Unmanaged.tensor_square_

tensor_std_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_b = cast2 Unmanaged.tensor_std_b

tensor_std_lbb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_lbb = cast4 Unmanaged.tensor_std_lbb

tensor_std_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_Nbb = cast4 Unmanaged.tensor_std_Nbb

tensor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_t = cast1 Unmanaged.tensor_t

tensor_t_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_t_ = cast1 Unmanaged.tensor_t_

tensor_tan
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tan = cast1 Unmanaged.tensor_tan

tensor_tan_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tan_ = cast1 Unmanaged.tensor_tan_

tensor_tanh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tanh = cast1 Unmanaged.tensor_tanh

tensor_tanh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tanh_ = cast1 Unmanaged.tensor_tanh_

tensor_transpose_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_transpose_ll = cast3 Unmanaged.tensor_transpose_ll

tensor_transpose_nn
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_transpose_nn = cast3 Unmanaged.tensor_transpose_nn

tensor_transpose__ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_transpose__ll = cast3 Unmanaged.tensor_transpose__ll

tensor_flip_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_flip_l = cast2 Unmanaged.tensor_flip_l

tensor_roll_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_roll_ll = cast3 Unmanaged.tensor_roll_ll

tensor_rot90_ll
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_rot90_ll = cast3 Unmanaged.tensor_rot90_ll

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

tensor_trunc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc = cast1 Unmanaged.tensor_trunc

tensor_trunc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc_ = cast1 Unmanaged.tensor_trunc_

tensor_type_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_type_as_t = cast2 Unmanaged.tensor_type_as_t

tensor_unsqueeze_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unsqueeze_l = cast2 Unmanaged.tensor_unsqueeze_l

tensor_unsqueeze__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unsqueeze__l = cast2 Unmanaged.tensor_unsqueeze__l

tensor_var_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_b = cast2 Unmanaged.tensor_var_b

tensor_var_lbb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_lbb = cast4 Unmanaged.tensor_var_lbb

tensor_var_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_Nbb = cast4 Unmanaged.tensor_var_Nbb

tensor_view_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_view_as_t = cast2 Unmanaged.tensor_view_as_t

tensor_where_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_where_tt = cast3 Unmanaged.tensor_where_tt

tensor_norm_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_norm_s = cast2 Unmanaged.tensor_norm_s

tensor_clone
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_clone = cast1 Unmanaged.tensor_clone

tensor_resize_as__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_resize_as__t = cast2 Unmanaged.tensor_resize_as__t

tensor_pow_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_pow_s = cast2 Unmanaged.tensor_pow_s

tensor_zero_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_zero_ = cast1 Unmanaged.tensor_zero_

tensor_sub_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub_ts = cast3 Unmanaged.tensor_sub_ts

tensor_sub__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub__ts = cast3 Unmanaged.tensor_sub__ts

tensor_sub_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub_ss = cast3 Unmanaged.tensor_sub_ss

tensor_sub__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub__ss = cast3 Unmanaged.tensor_sub__ss

tensor_addmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmm_ttss = cast5 Unmanaged.tensor_addmm_ttss

tensor_addmm__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmm__ttss = cast5 Unmanaged.tensor_addmm__ttss

tensor_sparse_resize__lll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_sparse_resize__lll = cast4 Unmanaged.tensor_sparse_resize__lll

tensor_sparse_resize_and_clear__lll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_sparse_resize_and_clear__lll = cast4 Unmanaged.tensor_sparse_resize_and_clear__lll

tensor_sparse_mask_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sparse_mask_t = cast2 Unmanaged.tensor_sparse_mask_t

tensor_to_dense
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_to_dense = cast1 Unmanaged.tensor_to_dense

tensor_sparse_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_sparse_dim = cast1 Unmanaged.tensor_sparse_dim

tensor__dimI
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__dimI = cast1 Unmanaged.tensor__dimI

tensor_dense_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dense_dim = cast1 Unmanaged.tensor_dense_dim

tensor__dimV
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__dimV = cast1 Unmanaged.tensor__dimV

tensor__nnz
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__nnz = cast1 Unmanaged.tensor__nnz

tensor_coalesce
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_coalesce = cast1 Unmanaged.tensor_coalesce

tensor_is_coalesced
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_coalesced = cast1 Unmanaged.tensor_is_coalesced

tensor__indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__indices = cast1 Unmanaged.tensor__indices

tensor__values
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__values = cast1 Unmanaged.tensor__values

tensor__coalesced__b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor__coalesced__b = cast2 Unmanaged.tensor__coalesced__b

tensor_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_indices = cast1 Unmanaged.tensor_indices

tensor_values
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_values = cast1 Unmanaged.tensor_values

tensor_unbind_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unbind_l = cast2 Unmanaged.tensor_unbind_l

tensor_unbind_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr TensorList)
tensor_unbind_n = cast2 Unmanaged.tensor_unbind_n

tensor_to_sparse_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_l = cast2 Unmanaged.tensor_to_sparse_l

tensor_to_sparse
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_to_sparse = cast1 Unmanaged.tensor_to_sparse

tensor_to_mkldnn
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_to_mkldnn = cast1 Unmanaged.tensor_to_mkldnn

tensor_dequantize
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_dequantize = cast1 Unmanaged.tensor_dequantize

tensor_q_scale
  :: ForeignPtr Tensor
  -> IO (CDouble)
tensor_q_scale = cast1 Unmanaged.tensor_q_scale

tensor_q_zero_point
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_q_zero_point = cast1 Unmanaged.tensor_q_zero_point

tensor_q_per_channel_scales
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_q_per_channel_scales = cast1 Unmanaged.tensor_q_per_channel_scales

tensor_q_per_channel_zero_points
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_q_per_channel_zero_points = cast1 Unmanaged.tensor_q_per_channel_zero_points

tensor_q_per_channel_axis
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_q_per_channel_axis = cast1 Unmanaged.tensor_q_per_channel_axis

tensor_int_repr
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_int_repr = cast1 Unmanaged.tensor_int_repr

tensor_qscheme
  :: ForeignPtr Tensor
  -> IO (QScheme)
tensor_qscheme = cast1 Unmanaged.tensor_qscheme

tensor_to_obb
  :: ForeignPtr Tensor
  -> ForeignPtr TensorOptions
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_to_obb = cast4 Unmanaged.tensor_to_obb

tensor_to_Dsbb
  :: ForeignPtr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_to_Dsbb = cast5 Unmanaged.tensor_to_Dsbb

tensor_to_sbb
  :: ForeignPtr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_to_sbb = cast4 Unmanaged.tensor_to_sbb

tensor_to_tbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_to_tbb = cast4 Unmanaged.tensor_to_tbb

tensor_set__S
  :: ForeignPtr Tensor
  -> ForeignPtr Storage
  -> IO (ForeignPtr Tensor)
tensor_set__S = cast2 Unmanaged.tensor_set__S

tensor_set__Slll
  :: ForeignPtr Tensor
  -> ForeignPtr Storage
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_set__Slll = cast5 Unmanaged.tensor_set__Slll

tensor_set__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_set__t = cast2 Unmanaged.tensor_set__t

tensor_set_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_set_ = cast1 Unmanaged.tensor_set_

tensor_is_set_to_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_is_set_to_t = cast2 Unmanaged.tensor_is_set_to_t

tensor_masked_fill__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_masked_fill__ts = cast3 Unmanaged.tensor_masked_fill__ts

tensor_masked_fill_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_masked_fill_ts = cast3 Unmanaged.tensor_masked_fill_ts

tensor_masked_fill__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_fill__tt = cast3 Unmanaged.tensor_masked_fill__tt

tensor_masked_fill_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_fill_tt = cast3 Unmanaged.tensor_masked_fill_tt

tensor_masked_scatter__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_scatter__tt = cast3 Unmanaged.tensor_masked_scatter__tt

tensor_masked_scatter_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_scatter_tt = cast3 Unmanaged.tensor_masked_scatter_tt

tensor_view_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_view_l = cast2 Unmanaged.tensor_view_l

tensor_put__ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_put__ttb = cast4 Unmanaged.tensor_put__ttb

tensor_index_add__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_add__ltt = cast4 Unmanaged.tensor_index_add__ltt

tensor_index_add_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_add_ltt = cast4 Unmanaged.tensor_index_add_ltt

tensor_index_add_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_add_ntt = cast4 Unmanaged.tensor_index_add_ntt

tensor_index_fill__lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill__lts = cast4 Unmanaged.tensor_index_fill__lts

tensor_index_fill_lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill_lts = cast4 Unmanaged.tensor_index_fill_lts

tensor_index_fill__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill__ltt = cast4 Unmanaged.tensor_index_fill__ltt

tensor_index_fill_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill_ltt = cast4 Unmanaged.tensor_index_fill_ltt

tensor_index_fill__nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill__nts = cast4 Unmanaged.tensor_index_fill__nts

tensor_index_fill__ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill__ntt = cast4 Unmanaged.tensor_index_fill__ntt

tensor_index_fill_nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill_nts = cast4 Unmanaged.tensor_index_fill_nts

tensor_index_fill_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill_ntt = cast4 Unmanaged.tensor_index_fill_ntt

tensor_scatter__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltt = cast4 Unmanaged.tensor_scatter__ltt

tensor_scatter_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltt = cast4 Unmanaged.tensor_scatter_ltt

tensor_scatter__lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter__lts = cast4 Unmanaged.tensor_scatter__lts

tensor_scatter_lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter_lts = cast4 Unmanaged.tensor_scatter_lts

tensor_scatter_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_ntt = cast4 Unmanaged.tensor_scatter_ntt

tensor_scatter_nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter_nts = cast4 Unmanaged.tensor_scatter_nts

tensor_scatter_add__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add__ltt = cast4 Unmanaged.tensor_scatter_add__ltt

tensor_scatter_add_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ltt = cast4 Unmanaged.tensor_scatter_add_ltt

tensor_scatter_add_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ntt = cast4 Unmanaged.tensor_scatter_add_ntt

tensor_lt__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_lt__s = cast2 Unmanaged.tensor_lt__s

tensor_lt__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_lt__t = cast2 Unmanaged.tensor_lt__t

tensor_gt__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_gt__s = cast2 Unmanaged.tensor_gt__s

tensor_gt__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_gt__t = cast2 Unmanaged.tensor_gt__t

tensor_le__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_le__s = cast2 Unmanaged.tensor_le__s

tensor_le__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_le__t = cast2 Unmanaged.tensor_le__t

tensor_ge__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ge__s = cast2 Unmanaged.tensor_ge__s

tensor_ge__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ge__t = cast2 Unmanaged.tensor_ge__t

tensor_eq__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_eq__s = cast2 Unmanaged.tensor_eq__s

tensor_eq__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_eq__t = cast2 Unmanaged.tensor_eq__t

tensor_ne__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_ne__s = cast2 Unmanaged.tensor_ne__s

tensor_ne__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ne__t = cast2 Unmanaged.tensor_ne__t

tensor_bitwise_and_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and_s = cast2 Unmanaged.tensor_bitwise_and_s

tensor_bitwise_and_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and_t = cast2 Unmanaged.tensor_bitwise_and_t

tensor_bitwise_and__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and__s = cast2 Unmanaged.tensor_bitwise_and__s

tensor_bitwise_and__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and__t = cast2 Unmanaged.tensor_bitwise_and__t

tensor___and___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___and___s = cast2 Unmanaged.tensor___and___s

tensor___and___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___and___t = cast2 Unmanaged.tensor___and___t

tensor___iand___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___iand___s = cast2 Unmanaged.tensor___iand___s

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

tensor_random__llp
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__llp = cast4 Unmanaged.tensor_random__llp

tensor_random__lp
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__lp = cast3 Unmanaged.tensor_random__lp

tensor_random__p
  :: ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_random__p = cast2 Unmanaged.tensor_random__p

tensor_uniform__ddp
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_uniform__ddp = cast4 Unmanaged.tensor_uniform__ddp

tensor_normal__ddp
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_normal__ddp = cast4 Unmanaged.tensor_normal__ddp

tensor_cauchy__ddp
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_cauchy__ddp = cast4 Unmanaged.tensor_cauchy__ddp

tensor_log_normal__ddp
  :: ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_log_normal__ddp = cast4 Unmanaged.tensor_log_normal__ddp

tensor_exponential__dp
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_exponential__dp = cast3 Unmanaged.tensor_exponential__dp

tensor_geometric__dp
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_geometric__dp = cast3 Unmanaged.tensor_geometric__dp

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

tensor_multinomial_lbp
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
tensor_multinomial_lbp = cast4 Unmanaged.tensor_multinomial_lbp

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

tensor_sign
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign = cast1 Unmanaged.tensor_sign

tensor_sign_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sign_ = cast1 Unmanaged.tensor_sign_

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

tensor_min_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min_t = cast2 Unmanaged.tensor_min_t

tensor_min
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_min = cast1 Unmanaged.tensor_min

tensor_max_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max_t = cast2 Unmanaged.tensor_max_t

tensor_max
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_max = cast1 Unmanaged.tensor_max

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




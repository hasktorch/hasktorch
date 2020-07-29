
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Tensor.Tensor0 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Tensor.Tensor0 as Unmanaged





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

tensor_names
  :: ForeignPtr Tensor
  -> IO (ForeignPtr DimnameList)
tensor_names = cast1 Unmanaged.tensor_names

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

tensor_is_vulkan
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_vulkan = cast1 Unmanaged.tensor_is_vulkan

tensor_is_quantized
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_quantized = cast1 Unmanaged.tensor_is_quantized

tensor_is_meta
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_meta = cast1 Unmanaged.tensor_is_meta

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

tensor_vulkan
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_vulkan = cast1 Unmanaged.tensor_vulkan

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

tensor_absolute
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_absolute = cast1 Unmanaged.tensor_absolute

tensor_absolute_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_absolute_ = cast1 Unmanaged.tensor_absolute_

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

tensor_acosh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_acosh = cast1 Unmanaged.tensor_acosh

tensor_acosh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_acosh_ = cast1 Unmanaged.tensor_acosh_

tensor_asinh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_asinh = cast1 Unmanaged.tensor_asinh

tensor_asinh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_asinh_ = cast1 Unmanaged.tensor_asinh_

tensor_atanh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atanh = cast1 Unmanaged.tensor_atanh

tensor_atanh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_atanh_ = cast1 Unmanaged.tensor_atanh_

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


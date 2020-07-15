
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Tensor.Tensor2 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.Tensor.Tensor2 as Unmanaged



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

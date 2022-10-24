
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





tensor_select_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_select_ll = _cast3 Unmanaged.tensor_select_ll

tensor_sigmoid
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sigmoid = _cast1 Unmanaged.tensor_sigmoid

tensor_sigmoid_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sigmoid_ = _cast1 Unmanaged.tensor_sigmoid_

tensor_logit_d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_logit_d = _cast2 Unmanaged.tensor_logit_d

tensor_logit__d
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
tensor_logit__d = _cast2 Unmanaged.tensor_logit__d

tensor_sin
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin = _cast1 Unmanaged.tensor_sin

tensor_sin_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sin_ = _cast1 Unmanaged.tensor_sin_

tensor_sinc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinc = _cast1 Unmanaged.tensor_sinc

tensor_sinc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinc_ = _cast1 Unmanaged.tensor_sinc_

tensor_sinh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinh = _cast1 Unmanaged.tensor_sinh

tensor_sinh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sinh_ = _cast1 Unmanaged.tensor_sinh_

tensor_detach
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_detach = _cast1 Unmanaged.tensor_detach

tensor_detach_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_detach_ = _cast1 Unmanaged.tensor_detach_

tensor_size_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (Int64)
tensor_size_n = _cast2 Unmanaged.tensor_size_n

tensor_slice_llll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_slice_llll = _cast5 Unmanaged.tensor_slice_llll

tensor_slice_scatter_tllll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_slice_scatter_tllll = _cast6 Unmanaged.tensor_slice_scatter_tllll

tensor_select_scatter_tll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_select_scatter_tll = _cast4 Unmanaged.tensor_select_scatter_tll

tensor_diagonal_scatter_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diagonal_scatter_tlll = _cast5 Unmanaged.tensor_diagonal_scatter_tlll

tensor_slogdet
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_slogdet = _cast1 Unmanaged.tensor_slogdet

tensor_smm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_smm_t = _cast2 Unmanaged.tensor_smm_t

tensor_softmax_ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_softmax_ls = _cast3 Unmanaged.tensor_softmax_ls

tensor_softmax_ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_softmax_ns = _cast3 Unmanaged.tensor_softmax_ns

tensor_unsafe_split_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unsafe_split_ll = _cast3 Unmanaged.tensor_unsafe_split_ll

tensor_split_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_split_ll = _cast3 Unmanaged.tensor_split_ll

tensor_unsafe_split_with_sizes_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unsafe_split_with_sizes_ll = _cast3 Unmanaged.tensor_unsafe_split_with_sizes_ll

tensor_split_with_sizes_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_split_with_sizes_ll = _cast3 Unmanaged.tensor_split_with_sizes_ll

tensor_squeeze
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_squeeze = _cast1 Unmanaged.tensor_squeeze

tensor_squeeze_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_squeeze_l = _cast2 Unmanaged.tensor_squeeze_l

tensor_squeeze_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_squeeze_n = _cast2 Unmanaged.tensor_squeeze_n

tensor_squeeze_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_squeeze_ = _cast1 Unmanaged.tensor_squeeze_

tensor_squeeze__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_squeeze__l = _cast2 Unmanaged.tensor_squeeze__l

tensor_squeeze__n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_squeeze__n = _cast2 Unmanaged.tensor_squeeze__n

tensor_sspaddmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sspaddmm_ttss = _cast5 Unmanaged.tensor_sspaddmm_ttss

tensor_stft_llltbbb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_stft_llltbbb = _cast8 Unmanaged.tensor_stft_llltbbb

tensor_istft_llltbbblb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_istft_llltbbblb = _cast10 Unmanaged.tensor_istft_llltbbblb

tensor_stride_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (Int64)
tensor_stride_n = _cast2 Unmanaged.tensor_stride_n

tensor_sum_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_s = _cast2 Unmanaged.tensor_sum_s

tensor_sum_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_lbs = _cast4 Unmanaged.tensor_sum_lbs

tensor_sum_Nbs
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_Nbs = _cast4 Unmanaged.tensor_sum_Nbs

tensor_nansum_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_nansum_s = _cast2 Unmanaged.tensor_nansum_s

tensor_nansum_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_nansum_lbs = _cast4 Unmanaged.tensor_nansum_lbs

tensor_sum_to_size_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_sum_to_size_l = _cast2 Unmanaged.tensor_sum_to_size_l

tensor_sqrt
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sqrt = _cast1 Unmanaged.tensor_sqrt

tensor_sqrt_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sqrt_ = _cast1 Unmanaged.tensor_sqrt_

tensor_square
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_square = _cast1 Unmanaged.tensor_square

tensor_square_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_square_ = _cast1 Unmanaged.tensor_square_

tensor_std_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_b = _cast2 Unmanaged.tensor_std_b

tensor_std_lbb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_lbb = _cast4 Unmanaged.tensor_std_lbb

tensor_std_llb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_llb = _cast4 Unmanaged.tensor_std_llb

tensor_std_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_Nbb = _cast4 Unmanaged.tensor_std_Nbb

tensor_std_Nlb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_Nlb = _cast4 Unmanaged.tensor_std_Nlb

tensor_prod_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_s = _cast2 Unmanaged.tensor_prod_s

tensor_prod_lbs
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_lbs = _cast4 Unmanaged.tensor_prod_lbs

tensor_prod_nbs
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_nbs = _cast4 Unmanaged.tensor_prod_nbs

tensor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_t = _cast1 Unmanaged.tensor_t

tensor_t_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_t_ = _cast1 Unmanaged.tensor_t_

tensor_tan
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tan = _cast1 Unmanaged.tensor_tan

tensor_tan_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tan_ = _cast1 Unmanaged.tensor_tan_

tensor_tanh
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tanh = _cast1 Unmanaged.tensor_tanh

tensor_tanh_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_tanh_ = _cast1 Unmanaged.tensor_tanh_

tensor_tile_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_tile_l = _cast2 Unmanaged.tensor_tile_l

tensor_transpose_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_transpose_ll = _cast3 Unmanaged.tensor_transpose_ll

tensor_transpose_nn
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
tensor_transpose_nn = _cast3 Unmanaged.tensor_transpose_nn

tensor_transpose__ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_transpose__ll = _cast3 Unmanaged.tensor_transpose__ll

tensor_flip_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_flip_l = _cast2 Unmanaged.tensor_flip_l

tensor_fliplr
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fliplr = _cast1 Unmanaged.tensor_fliplr

tensor_flipud
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_flipud = _cast1 Unmanaged.tensor_flipud

tensor_roll_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_roll_ll = _cast3 Unmanaged.tensor_roll_ll

tensor_rot90_ll
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_rot90_ll = _cast3 Unmanaged.tensor_rot90_ll

tensor_trunc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc = _cast1 Unmanaged.tensor_trunc

tensor_trunc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc_ = _cast1 Unmanaged.tensor_trunc_

tensor_fix
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fix = _cast1 Unmanaged.tensor_fix

tensor_fix_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fix_ = _cast1 Unmanaged.tensor_fix_

tensor_type_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_type_as_t = _cast2 Unmanaged.tensor_type_as_t

tensor_unsqueeze_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unsqueeze_l = _cast2 Unmanaged.tensor_unsqueeze_l

tensor_unsqueeze__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_unsqueeze__l = _cast2 Unmanaged.tensor_unsqueeze__l

tensor_var_b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_b = _cast2 Unmanaged.tensor_var_b

tensor_var_lbb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_lbb = _cast4 Unmanaged.tensor_var_lbb

tensor_var_llb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_llb = _cast4 Unmanaged.tensor_var_llb

tensor_var_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_Nbb = _cast4 Unmanaged.tensor_var_Nbb

tensor_var_Nlb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_Nlb = _cast4 Unmanaged.tensor_var_Nlb

tensor_view_as_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_view_as_t = _cast2 Unmanaged.tensor_view_as_t

tensor_where_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_where_tt = _cast3 Unmanaged.tensor_where_tt

tensor_norm_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_ss = _cast3 Unmanaged.tensor_norm_ss

tensor_norm_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_norm_s = _cast2 Unmanaged.tensor_norm_s

tensor_norm_slbs
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_slbs = _cast5 Unmanaged.tensor_norm_slbs

tensor_norm_slb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_norm_slb = _cast4 Unmanaged.tensor_norm_slb

tensor_norm_sNbs
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_sNbs = _cast5 Unmanaged.tensor_norm_sNbs

tensor_norm_sNb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_norm_sNb = _cast4 Unmanaged.tensor_norm_sNb

tensor_frexp
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_frexp = _cast1 Unmanaged.tensor_frexp

tensor_clone_M
  :: ForeignPtr Tensor
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_clone_M = _cast2 Unmanaged.tensor_clone_M

tensor_positive
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_positive = _cast1 Unmanaged.tensor_positive

tensor_resize_as__tM
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_resize_as__tM = _cast3 Unmanaged.tensor_resize_as__tM

tensor_zero_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_zero_ = _cast1 Unmanaged.tensor_zero_

tensor_sub_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub_ts = _cast3 Unmanaged.tensor_sub_ts

tensor_sub__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub__ts = _cast3 Unmanaged.tensor_sub__ts

tensor_sub_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub_ss = _cast3 Unmanaged.tensor_sub_ss

tensor_sub__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_sub__ss = _cast3 Unmanaged.tensor_sub__ss

tensor_subtract_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract_ts = _cast3 Unmanaged.tensor_subtract_ts

tensor_subtract__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract__ts = _cast3 Unmanaged.tensor_subtract__ts

tensor_subtract_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract_ss = _cast3 Unmanaged.tensor_subtract_ss

tensor_subtract__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract__ss = _cast3 Unmanaged.tensor_subtract__ss

tensor_heaviside_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_heaviside_t = _cast2 Unmanaged.tensor_heaviside_t

tensor_heaviside__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_heaviside__t = _cast2 Unmanaged.tensor_heaviside__t

tensor_addmm_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmm_ttss = _cast5 Unmanaged.tensor_addmm_ttss

tensor_addmm__ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_addmm__ttss = _cast5 Unmanaged.tensor_addmm__ttss

tensor_sparse_resize__lll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_sparse_resize__lll = _cast4 Unmanaged.tensor_sparse_resize__lll

tensor_sparse_resize_and_clear__lll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_sparse_resize_and_clear__lll = _cast4 Unmanaged.tensor_sparse_resize_and_clear__lll

tensor_sparse_mask_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_sparse_mask_t = _cast2 Unmanaged.tensor_sparse_mask_t

tensor_to_dense_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_to_dense_s = _cast2 Unmanaged.tensor_to_dense_s

tensor_sparse_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_sparse_dim = _cast1 Unmanaged.tensor_sparse_dim

tensor__dimI
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__dimI = _cast1 Unmanaged.tensor__dimI

tensor_dense_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dense_dim = _cast1 Unmanaged.tensor_dense_dim

tensor__dimV
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__dimV = _cast1 Unmanaged.tensor__dimV

tensor__nnz
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor__nnz = _cast1 Unmanaged.tensor__nnz

tensor_coalesce
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_coalesce = _cast1 Unmanaged.tensor_coalesce

tensor_is_coalesced
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_coalesced = _cast1 Unmanaged.tensor_is_coalesced

tensor__indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__indices = _cast1 Unmanaged.tensor__indices

tensor__values
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__values = _cast1 Unmanaged.tensor__values

tensor__coalesced__b
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor__coalesced__b = _cast2 Unmanaged.tensor__coalesced__b

tensor_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_indices = _cast1 Unmanaged.tensor_indices

tensor_values
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_values = _cast1 Unmanaged.tensor_values

tensor_crow_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_crow_indices = _cast1 Unmanaged.tensor_crow_indices

tensor_col_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_col_indices = _cast1 Unmanaged.tensor_col_indices

tensor_unbind_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unbind_l = _cast2 Unmanaged.tensor_unbind_l

tensor_unbind_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (ForeignPtr TensorList)
tensor_unbind_n = _cast2 Unmanaged.tensor_unbind_n

tensor_to_sparse_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_l = _cast2 Unmanaged.tensor_to_sparse_l

tensor_to_sparse
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_to_sparse = _cast1 Unmanaged.tensor_to_sparse

tensor_to_mkldnn_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_to_mkldnn_s = _cast2 Unmanaged.tensor_to_mkldnn_s

tensor_dequantize
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_dequantize = _cast1 Unmanaged.tensor_dequantize

tensor_q_scale
  :: ForeignPtr Tensor
  -> IO (CDouble)
tensor_q_scale = _cast1 Unmanaged.tensor_q_scale

tensor_q_zero_point
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_q_zero_point = _cast1 Unmanaged.tensor_q_zero_point

tensor_q_per_channel_scales
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_q_per_channel_scales = _cast1 Unmanaged.tensor_q_per_channel_scales

tensor_q_per_channel_zero_points
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_q_per_channel_zero_points = _cast1 Unmanaged.tensor_q_per_channel_zero_points

tensor_q_per_channel_axis
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_q_per_channel_axis = _cast1 Unmanaged.tensor_q_per_channel_axis

tensor_int_repr
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_int_repr = _cast1 Unmanaged.tensor_int_repr

tensor_qscheme
  :: ForeignPtr Tensor
  -> IO (QScheme)
tensor_qscheme = _cast1 Unmanaged.tensor_qscheme

tensor__autocast_to_reduced_precision_bbss
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> ScalarType
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor__autocast_to_reduced_precision_bbss = _cast5 Unmanaged.tensor__autocast_to_reduced_precision_bbss

tensor__autocast_to_full_precision_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor__autocast_to_full_precision_bb = _cast3 Unmanaged.tensor__autocast_to_full_precision_bb

tensor_to_obbM
  :: ForeignPtr Tensor
  -> ForeignPtr TensorOptions
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_obbM = _cast5 Unmanaged.tensor_to_obbM

tensor_to_DsbbM
  :: ForeignPtr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_DsbbM = _cast6 Unmanaged.tensor_to_DsbbM

tensor_to_sbbM
  :: ForeignPtr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_sbbM = _cast5 Unmanaged.tensor_to_sbbM

tensor_to_tbbM
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_tbbM = _cast5 Unmanaged.tensor_to_tbbM

tensor_item
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Scalar)
tensor_item = _cast1 Unmanaged.tensor_item

tensor_set__S
  :: ForeignPtr Tensor
  -> ForeignPtr Storage
  -> IO (ForeignPtr Tensor)
tensor_set__S = _cast2 Unmanaged.tensor_set__S

tensor_set__Slll
  :: ForeignPtr Tensor
  -> ForeignPtr Storage
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_set__Slll = _cast5 Unmanaged.tensor_set__Slll

tensor_set__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_set__t = _cast2 Unmanaged.tensor_set__t

tensor_set_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_set_ = _cast1 Unmanaged.tensor_set_

tensor_is_set_to_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (CBool)
tensor_is_set_to_t = _cast2 Unmanaged.tensor_is_set_to_t

tensor_masked_fill__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_masked_fill__ts = _cast3 Unmanaged.tensor_masked_fill__ts

tensor_masked_fill_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_masked_fill_ts = _cast3 Unmanaged.tensor_masked_fill_ts

tensor_masked_fill__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_fill__tt = _cast3 Unmanaged.tensor_masked_fill__tt

tensor_masked_fill_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_fill_tt = _cast3 Unmanaged.tensor_masked_fill_tt

tensor_masked_scatter__tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_scatter__tt = _cast3 Unmanaged.tensor_masked_scatter__tt

tensor_masked_scatter_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_masked_scatter_tt = _cast3 Unmanaged.tensor_masked_scatter_tt

tensor_view_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_view_l = _cast2 Unmanaged.tensor_view_l

tensor_view_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_view_s = _cast2 Unmanaged.tensor_view_s

tensor_put__ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_put__ttb = _cast4 Unmanaged.tensor_put__ttb

tensor_put_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_put_ttb = _cast4 Unmanaged.tensor_put_ttb

tensor_index_add__ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add__ltts = _cast5 Unmanaged.tensor_index_add__ltts

tensor_index_add_ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add_ltts = _cast5 Unmanaged.tensor_index_add_ltts

tensor_index_add_ntts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add_ntts = _cast5 Unmanaged.tensor_index_add_ntts

tensor_index_fill__lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill__lts = _cast4 Unmanaged.tensor_index_fill__lts

tensor_index_fill_lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill_lts = _cast4 Unmanaged.tensor_index_fill_lts

tensor_index_fill__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill__ltt = _cast4 Unmanaged.tensor_index_fill__ltt

tensor_index_fill_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill_ltt = _cast4 Unmanaged.tensor_index_fill_ltt

tensor_index_fill__nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill__nts = _cast4 Unmanaged.tensor_index_fill__nts

tensor_index_fill__ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill__ntt = _cast4 Unmanaged.tensor_index_fill__ntt

tensor_index_fill_nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_fill_nts = _cast4 Unmanaged.tensor_index_fill_nts

tensor_index_fill_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_index_fill_ntt = _cast4 Unmanaged.tensor_index_fill_ntt

tensor_scatter_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltt = _cast4 Unmanaged.tensor_scatter_ltt

tensor_scatter__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltt = _cast4 Unmanaged.tensor_scatter__ltt

tensor_scatter_lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter_lts = _cast4 Unmanaged.tensor_scatter_lts

tensor_scatter__lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter__lts = _cast4 Unmanaged.tensor_scatter__lts

tensor_scatter_ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltts = _cast5 Unmanaged.tensor_scatter_ltts

tensor_scatter__ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltts = _cast5 Unmanaged.tensor_scatter__ltts

tensor_scatter_ltss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltss = _cast5 Unmanaged.tensor_scatter_ltss

tensor_scatter__ltss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltss = _cast5 Unmanaged.tensor_scatter__ltss

tensor_scatter_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_ntt = _cast4 Unmanaged.tensor_scatter_ntt

tensor_scatter_nts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter_nts = _cast4 Unmanaged.tensor_scatter_nts

tensor_scatter_add_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ltt = _cast4 Unmanaged.tensor_scatter_add_ltt

tensor_scatter_add__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add__ltt = _cast4 Unmanaged.tensor_scatter_add__ltt

tensor_scatter_add_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ntt = _cast4 Unmanaged.tensor_scatter_add_ntt

tensor_scatter_reduce_ltsl
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_scatter_reduce_ltsl = _cast5 Unmanaged.tensor_scatter_reduce_ltsl

tensor_eq__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_eq__s = _cast2 Unmanaged.tensor_eq__s

tensor_eq__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_eq__t = _cast2 Unmanaged.tensor_eq__t

tensor_bitwise_and_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and_s = _cast2 Unmanaged.tensor_bitwise_and_s

tensor_bitwise_and_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and_t = _cast2 Unmanaged.tensor_bitwise_and_t

tensor_bitwise_and__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and__s = _cast2 Unmanaged.tensor_bitwise_and__s

tensor_bitwise_and__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_and__t = _cast2 Unmanaged.tensor_bitwise_and__t

tensor___and___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___and___s = _cast2 Unmanaged.tensor___and___s

tensor___and___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___and___t = _cast2 Unmanaged.tensor___and___t

tensor___iand___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___iand___s = _cast2 Unmanaged.tensor___iand___s

tensor___iand___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___iand___t = _cast2 Unmanaged.tensor___iand___t

tensor_bitwise_or_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or_s = _cast2 Unmanaged.tensor_bitwise_or_s

tensor_bitwise_or_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or_t = _cast2 Unmanaged.tensor_bitwise_or_t

tensor_bitwise_or__s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or__s = _cast2 Unmanaged.tensor_bitwise_or__s

tensor_bitwise_or__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_bitwise_or__t = _cast2 Unmanaged.tensor_bitwise_or__t

tensor___or___s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor___or___s = _cast2 Unmanaged.tensor___or___s

tensor___or___t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor___or___t = _cast2 Unmanaged.tensor___or___t


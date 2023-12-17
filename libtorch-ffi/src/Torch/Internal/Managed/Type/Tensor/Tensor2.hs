
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

tensor_slice_scatter_tllll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_slice_scatter_tllll = cast6 Unmanaged.tensor_slice_scatter_tllll

tensor_select_scatter_tll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_select_scatter_tll = cast4 Unmanaged.tensor_select_scatter_tll

tensor_diagonal_scatter_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_diagonal_scatter_tlll = cast5 Unmanaged.tensor_diagonal_scatter_tlll

tensor_as_strided_scatter_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_as_strided_scatter_tlll = cast5 Unmanaged.tensor_as_strided_scatter_tlll

tensor_smm_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_smm_t = cast2 Unmanaged.tensor_smm_t

tensor_softmax_ls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_softmax_ls = cast3 Unmanaged.tensor_softmax_ls

tensor_softmax_ns
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_softmax_ns = cast3 Unmanaged.tensor_softmax_ns

tensor_unsafe_split_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unsafe_split_ll = cast3 Unmanaged.tensor_unsafe_split_ll

tensor_split_ll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_split_ll = cast3 Unmanaged.tensor_split_ll

-- tensor_split_ll
--   :: ForeignPtr Tensor
--   -> ForeignPtr IntArray
--   -> Int64
--   -> IO (ForeignPtr TensorList)
-- tensor_split_ll = cast3 Unmanaged.tensor_split_ll

tensor_unsafe_split_with_sizes_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr TensorList)
tensor_unsafe_split_with_sizes_ll = cast3 Unmanaged.tensor_unsafe_split_with_sizes_ll

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

-- tensor_squeeze_l
--   :: ForeignPtr Tensor
--   -> ForeignPtr IntArray
--   -> IO (ForeignPtr Tensor)
-- tensor_squeeze_l = cast2 Unmanaged.tensor_squeeze_l

tensor_squeeze_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_squeeze_ = cast1 Unmanaged.tensor_squeeze_

tensor_squeeze__l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_squeeze__l = cast2 Unmanaged.tensor_squeeze__l

-- tensor_squeeze__l
--   :: ForeignPtr Tensor
--   -> ForeignPtr IntArray
--   -> IO (ForeignPtr Tensor)
-- tensor_squeeze__l = cast2 Unmanaged.tensor_squeeze__l

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
tensor_stft_llltbbb = cast8 Unmanaged.tensor_stft_llltbbb

tensor_stft_llltbsbbb
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> CBool
  -> ForeignPtr StdString
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_stft_llltbsbbb = cast10 Unmanaged.tensor_stft_llltbsbbb

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
tensor_istft_llltbbblb = cast10 Unmanaged.tensor_istft_llltbbblb

tensor_stride_n
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> IO (Int64)
tensor_stride_n = cast2 Unmanaged.tensor_stride_n

tensor_sum_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_s = cast2 Unmanaged.tensor_sum_s

tensor_sum_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_lbs = cast4 Unmanaged.tensor_sum_lbs

tensor_sum_Nbs
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_sum_Nbs = cast4 Unmanaged.tensor_sum_Nbs

tensor_nansum_lbs
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_nansum_lbs = cast4 Unmanaged.tensor_nansum_lbs

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

tensor_std_llb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_llb = cast4 Unmanaged.tensor_std_llb

tensor_std_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_Nbb = cast4 Unmanaged.tensor_std_Nbb

tensor_std_Nlb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_std_Nlb = cast4 Unmanaged.tensor_std_Nlb

tensor_prod_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_s = cast2 Unmanaged.tensor_prod_s

tensor_prod_lbs
  :: ForeignPtr Tensor
  -> Int64
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_lbs = cast4 Unmanaged.tensor_prod_lbs

tensor_prod_nbs
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_prod_nbs = cast4 Unmanaged.tensor_prod_nbs

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

tensor_tile_l
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_tile_l = cast2 Unmanaged.tensor_tile_l

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

tensor_fliplr
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fliplr = cast1 Unmanaged.tensor_fliplr

tensor_flipud
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_flipud = cast1 Unmanaged.tensor_flipud

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

tensor__nested_tensor_size
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__nested_tensor_size = cast1 Unmanaged.tensor__nested_tensor_size

tensor__nested_tensor_strides
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor__nested_tensor_strides = cast1 Unmanaged.tensor__nested_tensor_strides

-- tensor__nested_tensor_offsets
--   :: ForeignPtr Tensor
--   -> IO (ForeignPtr IntArray)
-- tensor__nested_tensor_offsets = cast1 Unmanaged.tensor__nested_tensor_offsets

tensor_trunc
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc = cast1 Unmanaged.tensor_trunc

tensor_trunc_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_trunc_ = cast1 Unmanaged.tensor_trunc_

tensor_fix
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fix = cast1 Unmanaged.tensor_fix

tensor_fix_
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_fix_ = cast1 Unmanaged.tensor_fix_

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

tensor_var_llb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_llb = cast4 Unmanaged.tensor_var_llb

tensor_var_Nbb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_Nbb = cast4 Unmanaged.tensor_var_Nbb

tensor_var_Nlb
  :: ForeignPtr Tensor
  -> ForeignPtr DimnameList
  -> Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_var_Nlb = cast4 Unmanaged.tensor_var_Nlb

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

tensor_where_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_where_ts = cast3 Unmanaged.tensor_where_ts

tensor_norm_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_ss = cast3 Unmanaged.tensor_norm_ss

tensor_norm_s
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_norm_s = cast2 Unmanaged.tensor_norm_s

tensor_norm_slbs
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr IntArray
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_slbs = cast5 Unmanaged.tensor_norm_slbs

tensor_norm_slb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_norm_slb = cast4 Unmanaged.tensor_norm_slb

tensor_norm_sNbs
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> CBool
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_norm_sNbs = cast5 Unmanaged.tensor_norm_sNbs

tensor_norm_sNb
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr DimnameList
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_norm_sNb = cast4 Unmanaged.tensor_norm_sNb

tensor_frexp
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
tensor_frexp = cast1 Unmanaged.tensor_frexp

tensor_clone_M
  :: ForeignPtr Tensor
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_clone_M = cast2 Unmanaged.tensor_clone_M

tensor_positive
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_positive = cast1 Unmanaged.tensor_positive

tensor_resize_as__tM
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_resize_as__tM = cast3 Unmanaged.tensor_resize_as__tM

tensor_resize_as_sparse__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_resize_as_sparse__t = cast2 Unmanaged.tensor_resize_as_sparse__t

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

tensor_subtract_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract_ts = cast3 Unmanaged.tensor_subtract_ts

tensor_subtract__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract__ts = cast3 Unmanaged.tensor_subtract__ts

tensor_subtract_ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract_ss = cast3 Unmanaged.tensor_subtract_ss

tensor_subtract__ss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_subtract__ss = cast3 Unmanaged.tensor_subtract__ss

tensor_heaviside_t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_heaviside_t = cast2 Unmanaged.tensor_heaviside_t

tensor_heaviside__t
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_heaviside__t = cast2 Unmanaged.tensor_heaviside__t

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

tensor__addmm_activation_ttssb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor__addmm_activation_ttssb = cast6 Unmanaged.tensor__addmm_activation_ttssb

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

tensor_to_dense_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_to_dense_s = cast2 Unmanaged.tensor_to_dense_s

tensor__to_dense_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor__to_dense_s = cast2 Unmanaged.tensor__to_dense_s

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

tensor_crow_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_crow_indices = cast1 Unmanaged.tensor_crow_indices

tensor_col_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_col_indices = cast1 Unmanaged.tensor_col_indices

tensor_ccol_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_ccol_indices = cast1 Unmanaged.tensor_ccol_indices

tensor_row_indices
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_row_indices = cast1 Unmanaged.tensor_row_indices

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

tensor_to_sparse_Lll
  :: ForeignPtr Tensor
  -> Layout
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_Lll = cast4 Unmanaged.tensor_to_sparse_Lll

tensor_to_sparse_csr_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_csr_l = cast2 Unmanaged.tensor_to_sparse_csr_l

tensor_to_sparse_csc_l
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_csc_l = cast2 Unmanaged.tensor_to_sparse_csc_l

tensor_to_sparse_bsr_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_bsr_ll = cast3 Unmanaged.tensor_to_sparse_bsr_ll

tensor_to_sparse_bsc_ll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
tensor_to_sparse_bsc_ll = cast3 Unmanaged.tensor_to_sparse_bsc_ll

tensor_to_mkldnn_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_to_mkldnn_s = cast2 Unmanaged.tensor_to_mkldnn_s

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

tensor__autocast_to_reduced_precision_bbss
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> ScalarType
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor__autocast_to_reduced_precision_bbss = cast5 Unmanaged.tensor__autocast_to_reduced_precision_bbss

tensor__autocast_to_full_precision_bb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor__autocast_to_full_precision_bb = cast3 Unmanaged.tensor__autocast_to_full_precision_bb

tensor_to_obbM
  :: ForeignPtr Tensor
  -> ForeignPtr TensorOptions
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_obbM = cast5 Unmanaged.tensor_to_obbM

tensor_to_DsbbM
  :: ForeignPtr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_DsbbM = cast6 Unmanaged.tensor_to_DsbbM

tensor_to_sbbM
  :: ForeignPtr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_sbbM = cast5 Unmanaged.tensor_to_sbbM

tensor_to_tbbM
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (ForeignPtr Tensor)
tensor_to_tbbM = cast5 Unmanaged.tensor_to_tbbM

tensor_item
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Scalar)
tensor_item = cast1 Unmanaged.tensor_item

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

tensor_set__tlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
tensor_set__tlll = cast5 Unmanaged.tensor_set__tlll

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

tensor_view_s
  :: ForeignPtr Tensor
  -> ScalarType
  -> IO (ForeignPtr Tensor)
tensor_view_s = cast2 Unmanaged.tensor_view_s

tensor_put__ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_put__ttb = cast4 Unmanaged.tensor_put__ttb

tensor_put_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_put_ttb = cast4 Unmanaged.tensor_put_ttb

tensor_index_add__ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add__ltts = cast5 Unmanaged.tensor_index_add__ltts

tensor_index_add_ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add_ltts = cast5 Unmanaged.tensor_index_add_ltts

tensor_index_add_ntts
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_index_add_ntts = cast5 Unmanaged.tensor_index_add_ntts

tensor_index_reduce__lttsb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_reduce__lttsb = cast6 Unmanaged.tensor_index_reduce__lttsb

tensor_index_reduce_lttsb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_index_reduce_lttsb = cast6 Unmanaged.tensor_index_reduce_lttsb

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

tensor_scatter_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltt = cast4 Unmanaged.tensor_scatter_ltt

tensor_scatter__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltt = cast4 Unmanaged.tensor_scatter__ltt

tensor_scatter_lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter_lts = cast4 Unmanaged.tensor_scatter_lts

tensor_scatter__lts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
tensor_scatter__lts = cast4 Unmanaged.tensor_scatter__lts

tensor_scatter_ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltts = cast5 Unmanaged.tensor_scatter_ltts

tensor_scatter__ltts
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltts = cast5 Unmanaged.tensor_scatter__ltts

tensor_scatter_ltss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter_ltss = cast5 Unmanaged.tensor_scatter_ltss

tensor_scatter__ltss
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
tensor_scatter__ltss = cast5 Unmanaged.tensor_scatter__ltss

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

tensor_scatter_add_ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ltt = cast4 Unmanaged.tensor_scatter_add_ltt

tensor_scatter_add__ltt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add__ltt = cast4 Unmanaged.tensor_scatter_add__ltt

tensor_scatter_add_ntt
  :: ForeignPtr Tensor
  -> ForeignPtr Dimname
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_scatter_add_ntt = cast4 Unmanaged.tensor_scatter_add_ntt

tensor_scatter_reduce_lttsb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_scatter_reduce_lttsb = cast6 Unmanaged.tensor_scatter_reduce_lttsb

tensor_scatter_reduce__lttsb
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr StdString
  -> CBool
  -> IO (ForeignPtr Tensor)
tensor_scatter_reduce__lttsb = cast6 Unmanaged.tensor_scatter_reduce__lttsb

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


{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor2 where


import qualified Data.Map                         as Map
import           Foreign
import           Foreign.C.String
import           Foreign.C.Types
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C
import           Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }



C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"
C.include "<ATen/TensorOperators.h>"
C.include "<vector>"



tensor_size_l
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
tensor_size_l _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).size(
    $(int64_t _dim));
  }|]

tensor_size_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
tensor_size_n _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).size(
    *$(at::Dimname* _dim));
  }|]

tensor_slice_llll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_slice_llll _obj _dim _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).slice(
    $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)
  , $(int64_t _step)));
  }|]

tensor_slogdet
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_slogdet _obj =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).slogdet(
    ));
  }|]

tensor_smm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_smm_t _obj _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).smm(
    *$(at::Tensor* _mat2)));
  }|]

tensor_split_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
tensor_split_ll _obj _split_size _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).split(
    $(int64_t _split_size)
  , $(int64_t _dim)));
  }|]

tensor_split_with_sizes_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr TensorList)
tensor_split_with_sizes_ll _obj _split_sizes _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).split_with_sizes(
    *$(std::vector<int64_t>* _split_sizes)
  , $(int64_t _dim)));
  }|]

tensor_squeeze
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_squeeze _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze(
    ));
  }|]

tensor_squeeze_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_squeeze_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze(
    $(int64_t _dim)));
  }|]

tensor_squeeze_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_squeeze_n _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze(
    *$(at::Dimname* _dim)));
  }|]

tensor_squeeze_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_squeeze_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze_(
    ));
  }|]

tensor_squeeze__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_squeeze__l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze_(
    $(int64_t _dim)));
  }|]

tensor_squeeze__n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_squeeze__n _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze_(
    *$(at::Dimname* _dim)));
  }|]

tensor_sspaddmm_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_sspaddmm_ttss _obj _mat1 _mat2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sspaddmm(
    *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_stride_l
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
tensor_stride_l _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).stride(
    $(int64_t _dim));
  }|]

tensor_stride_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
tensor_stride_n _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).stride(
    *$(at::Dimname* _dim));
  }|]

tensor_sum_to_size_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_sum_to_size_l _obj _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum_to_size(
    *$(std::vector<int64_t>* _size)));
  }|]

tensor_sqrt
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sqrt _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sqrt(
    ));
  }|]

tensor_sqrt_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sqrt_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sqrt_(
    ));
  }|]

tensor_square
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_square _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).square(
    ));
  }|]

tensor_square_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_square_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).square_(
    ));
  }|]

tensor_std_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_std_b _obj _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).std(
    $(bool _unbiased)));
  }|]

tensor_std_lbb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_std_lbb _obj _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).std(
    *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

tensor_std_Nbb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_std_Nbb _obj _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).std(
    *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

tensor_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_t _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).t(
    ));
  }|]

tensor_t_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_t_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).t_(
    ));
  }|]

tensor_tan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_tan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tan(
    ));
  }|]

tensor_tan_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_tan_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tan_(
    ));
  }|]

tensor_tanh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_tanh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tanh(
    ));
  }|]

tensor_tanh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_tanh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tanh_(
    ));
  }|]

tensor_transpose_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_transpose_ll _obj _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).transpose(
    $(int64_t _dim0)
  , $(int64_t _dim1)));
  }|]

tensor_transpose_nn
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_transpose_nn _obj _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).transpose(
    *$(at::Dimname* _dim0)
  , *$(at::Dimname* _dim1)));
  }|]

tensor_transpose__ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_transpose__ll _obj _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).transpose_(
    $(int64_t _dim0)
  , $(int64_t _dim1)));
  }|]

tensor_flip_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_flip_l _obj _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flip(
    *$(std::vector<int64_t>* _dims)));
  }|]

tensor_fliplr
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fliplr _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fliplr(
    ));
  }|]

tensor_flipud
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_flipud _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flipud(
    ));
  }|]

tensor_roll_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_roll_ll _obj _shifts _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).roll(
    *$(std::vector<int64_t>* _shifts)
  , *$(std::vector<int64_t>* _dims)));
  }|]

tensor_rot90_ll
  :: Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_rot90_ll _obj _k _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rot90(
    $(int64_t _k)
  , *$(std::vector<int64_t>* _dims)));
  }|]

tensor_true_divide_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_true_divide_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).true_divide(
    *$(at::Tensor* _other)));
  }|]

tensor_true_divide__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_true_divide__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).true_divide_(
    *$(at::Tensor* _other)));
  }|]

tensor_true_divide_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_true_divide_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).true_divide(
    *$(at::Scalar* _other)));
  }|]

tensor_true_divide__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_true_divide__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).true_divide_(
    *$(at::Scalar* _other)));
  }|]

tensor_trunc
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_trunc _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).trunc(
    ));
  }|]

tensor_trunc_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_trunc_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).trunc_(
    ));
  }|]

tensor_type_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_type_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).type_as(
    *$(at::Tensor* _other)));
  }|]

tensor_unsqueeze_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_unsqueeze_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unsqueeze(
    $(int64_t _dim)));
  }|]

tensor_unsqueeze__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_unsqueeze__l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unsqueeze_(
    $(int64_t _dim)));
  }|]

tensor_var_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_var_b _obj _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).var(
    $(bool _unbiased)));
  }|]

tensor_var_lbb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_var_lbb _obj _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).var(
    *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

tensor_var_Nbb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_var_Nbb _obj _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).var(
    *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

tensor_view_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_view_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).view_as(
    *$(at::Tensor* _other)));
  }|]

tensor_where_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_where_tt _obj _condition _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).where(
    *$(at::Tensor* _condition)
  , *$(at::Tensor* _other)));
  }|]

tensor_norm_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_norm_s _obj _p =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)));
  }|]

tensor_clone
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clone _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clone(
    ));
  }|]

tensor_resize_as__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_resize_as__t _obj _the_template =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_as_(
    *$(at::Tensor* _the_template)));
  }|]

tensor_pow_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_pow_s _obj _exponent =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pow(
    *$(at::Scalar* _exponent)));
  }|]

tensor_zero_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_zero_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).zero_(
    ));
  }|]

tensor_sub_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_sub_ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sub(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_sub__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_sub__ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sub_(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_sub_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_sub_ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sub(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_sub__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_sub__ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sub_(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmm_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmm_ttss _obj _mat1 _mat2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmm(
    *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmm__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmm__ttss _obj _mat1 _mat2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmm_(
    *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_sparse_resize__lll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_sparse_resize__lll _obj _size _sparse_dim _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sparse_resize_(
    *$(std::vector<int64_t>* _size)
  , $(int64_t _sparse_dim)
  , $(int64_t _dense_dim)));
  }|]

tensor_sparse_resize_and_clear__lll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_sparse_resize_and_clear__lll _obj _size _sparse_dim _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sparse_resize_and_clear_(
    *$(std::vector<int64_t>* _size)
  , $(int64_t _sparse_dim)
  , $(int64_t _dense_dim)));
  }|]

tensor_sparse_mask_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sparse_mask_t _obj _mask =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sparse_mask(
    *$(at::Tensor* _mask)));
  }|]

tensor_to_dense
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_dense _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_dense(
    ));
  }|]

tensor_sparse_dim
  :: Ptr Tensor
  -> IO (Int64)
tensor_sparse_dim _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).sparse_dim(
    );
  }|]

tensor__dimI
  :: Ptr Tensor
  -> IO (Int64)
tensor__dimI _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj))._dimI(
    );
  }|]

tensor_dense_dim
  :: Ptr Tensor
  -> IO (Int64)
tensor_dense_dim _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).dense_dim(
    );
  }|]

tensor__dimV
  :: Ptr Tensor
  -> IO (Int64)
tensor__dimV _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj))._dimV(
    );
  }|]

tensor__nnz
  :: Ptr Tensor
  -> IO (Int64)
tensor__nnz _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj))._nnz(
    );
  }|]

tensor_coalesce
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_coalesce _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).coalesce(
    ));
  }|]

tensor_is_coalesced
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_coalesced _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_coalesced(
    );
  }|]

tensor__indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._indices(
    ));
  }|]

tensor__values
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__values _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._values(
    ));
  }|]

tensor__coalesced__b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor__coalesced__b _obj _coalesced =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._coalesced_(
    $(bool _coalesced)));
  }|]

tensor_indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).indices(
    ));
  }|]

tensor_values
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_values _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).values(
    ));
  }|]

tensor_unbind_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr TensorList)
tensor_unbind_l _obj _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).unbind(
    $(int64_t _dim)));
  }|]

tensor_unbind_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr TensorList)
tensor_unbind_n _obj _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).unbind(
    *$(at::Dimname* _dim)));
  }|]

tensor_to_sparse_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_l _obj _sparse_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse(
    $(int64_t _sparse_dim)));
  }|]

tensor_to_sparse
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_sparse _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse(
    ));
  }|]

tensor_to_mkldnn
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_mkldnn _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_mkldnn(
    ));
  }|]

tensor_dequantize
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_dequantize _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).dequantize(
    ));
  }|]

tensor_q_scale
  :: Ptr Tensor
  -> IO (CDouble)
tensor_q_scale _obj =
  [C.throwBlock| double { return (*$(at::Tensor* _obj)).q_scale(
    );
  }|]

tensor_q_zero_point
  :: Ptr Tensor
  -> IO (Int64)
tensor_q_zero_point _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).q_zero_point(
    );
  }|]

tensor_q_per_channel_scales
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_q_per_channel_scales _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).q_per_channel_scales(
    ));
  }|]

tensor_q_per_channel_zero_points
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_q_per_channel_zero_points _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).q_per_channel_zero_points(
    ));
  }|]

tensor_q_per_channel_axis
  :: Ptr Tensor
  -> IO (Int64)
tensor_q_per_channel_axis _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).q_per_channel_axis(
    );
  }|]

tensor_int_repr
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_int_repr _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).int_repr(
    ));
  }|]

tensor_qscheme
  :: Ptr Tensor
  -> IO (QScheme)
tensor_qscheme _obj =
  [C.throwBlock| at::QScheme { return (*$(at::Tensor* _obj)).qscheme(
    );
  }|]

tensor_to_obb
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_obb _obj _options _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::TensorOptions* _options)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_Dsbb
  :: Ptr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_Dsbb _obj _device _dtype _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::DeviceType _device)
  , $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_sbb
  :: Ptr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_sbb _obj _dtype _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_tbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_tbb _obj _other _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::Tensor* _other)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_set__S
  :: Ptr Tensor
  -> Ptr Storage
  -> IO (Ptr Tensor)
tensor_set__S _obj _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_(
    *$(at::Storage* _source)));
  }|]

tensor_set__Slll
  :: Ptr Tensor
  -> Ptr Storage
  -> Int64
  -> Ptr IntArray
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_set__Slll _obj _source _storage_offset _size _stride =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_(
    *$(at::Storage* _source)
  , $(int64_t _storage_offset)
  , *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)));
  }|]

tensor_set__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_set__t _obj _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_(
    *$(at::Tensor* _source)));
  }|]

tensor_set_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_set_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_(
    ));
  }|]

tensor_is_set_to_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_set_to_t _obj _tensor =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_set_to(
    *$(at::Tensor* _tensor));
  }|]

tensor_masked_fill__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_masked_fill__ts _obj _mask _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_fill_(
    *$(at::Tensor* _mask)
  , *$(at::Scalar* _value)));
  }|]

tensor_masked_fill_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_masked_fill_ts _obj _mask _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_fill(
    *$(at::Tensor* _mask)
  , *$(at::Scalar* _value)));
  }|]

tensor_masked_fill__tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_masked_fill__tt _obj _mask _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_fill_(
    *$(at::Tensor* _mask)
  , *$(at::Tensor* _value)));
  }|]

tensor_masked_fill_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_masked_fill_tt _obj _mask _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_fill(
    *$(at::Tensor* _mask)
  , *$(at::Tensor* _value)));
  }|]

tensor_masked_scatter__tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_masked_scatter__tt _obj _mask _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_scatter_(
    *$(at::Tensor* _mask)
  , *$(at::Tensor* _source)));
  }|]

tensor_masked_scatter_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_masked_scatter_tt _obj _mask _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_scatter(
    *$(at::Tensor* _mask)
  , *$(at::Tensor* _source)));
  }|]

tensor_view_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_view_l _obj _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).view(
    *$(std::vector<int64_t>* _size)));
  }|]

tensor_put__ttb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_put__ttb _obj _index _source _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).put_(
    *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , $(bool _accumulate)));
  }|]

tensor_index_add__ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_add__ltt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_add_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_add_ltt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_add_ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_add_ntt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_fill__lts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_fill__lts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_index_fill_lts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_fill_lts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_index_fill__ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_fill__ltt _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _value)));
  }|]

tensor_index_fill_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_fill_ltt _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _value)));
  }|]

tensor_index_fill__nts
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_fill__nts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill_(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_index_fill__ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_fill__ntt _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill_(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _value)));
  }|]

tensor_index_fill_nts
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_fill_nts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_index_fill_ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_fill_ntt _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_fill(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _value)));
  }|]

tensor_scatter__ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter__ltt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_scatter_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter_ltt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_scatter__lts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_scatter__lts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_scatter_lts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_scatter_lts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_scatter_ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter_ntt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_scatter_nts
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_scatter_nts _obj _dim _index _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)));
  }|]

tensor_scatter_add__ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter_add__ltt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_add_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_scatter_add_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter_add_ltt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_add(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_scatter_add_ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_scatter_add_ntt _obj _dim _index _src =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_add(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)));
  }|]

tensor_lt__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_lt__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lt_(
    *$(at::Scalar* _other)));
  }|]

tensor_lt__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lt__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lt_(
    *$(at::Tensor* _other)));
  }|]

tensor_gt__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_gt__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gt_(
    *$(at::Scalar* _other)));
  }|]

tensor_gt__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_gt__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gt_(
    *$(at::Tensor* _other)));
  }|]

tensor_le__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_le__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).le_(
    *$(at::Scalar* _other)));
  }|]

tensor_le__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_le__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).le_(
    *$(at::Tensor* _other)));
  }|]

tensor_ge__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_ge__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ge_(
    *$(at::Scalar* _other)));
  }|]

tensor_ge__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ge__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ge_(
    *$(at::Tensor* _other)));
  }|]

tensor_eq__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_eq__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).eq_(
    *$(at::Scalar* _other)));
  }|]

tensor_eq__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_eq__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).eq_(
    *$(at::Tensor* _other)));
  }|]


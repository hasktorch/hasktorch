
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor2 where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }



C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"
C.include "<ATen/TensorOperators.h>"
C.include "<vector>"



tensor_sinh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sinh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sinh_(
    ));
  }|]

tensor_detach
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_detach _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).detach(
    ));
  }|]

tensor_detach_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_detach_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).detach_(
    ));
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

tensor_slice_scatter_tllll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_slice_scatter_tllll _obj _src _dim _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).slice_scatter(
    *$(at::Tensor* _src)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)
  , $(int64_t _step)));
  }|]

tensor_select_scatter_tll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_select_scatter_tll _obj _src _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).select_scatter(
    *$(at::Tensor* _src)
  , $(int64_t _dim)
  , $(int64_t _index)));
  }|]

tensor_diagonal_scatter_tlll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_diagonal_scatter_tlll _obj _src _offset _dim1 _dim2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diagonal_scatter(
    *$(at::Tensor* _src)
  , $(int64_t _offset)
  , $(int64_t _dim1)
  , $(int64_t _dim2)));
  }|]

tensor_as_strided_scatter_tlll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_as_strided_scatter_tlll _obj _src _size _stride _storage_offset =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).as_strided_scatter(
    *$(at::Tensor* _src)
  , *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)
  , $(int64_t _storage_offset)));
  }|]

tensor_smm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_smm_t _obj _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).smm(
    *$(at::Tensor* _mat2)));
  }|]

tensor_softmax_ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_softmax_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).softmax(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_softmax_ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_softmax_ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).softmax(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_unsafe_split_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
tensor_unsafe_split_ll _obj _split_size _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).unsafe_split(
    $(int64_t _split_size)
  , $(int64_t _dim)));
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

-- tensor_split_ll
--   :: Ptr Tensor
--   -> Ptr IntArray
--   -> Int64
--   -> IO (Ptr TensorList)
-- tensor_split_ll _obj _split_size _dim =
--   [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).split(
--     *$(std::vector<int64_t>* _split_size)
--   , $(int64_t _dim)));
--   }|]

tensor_unsafe_split_with_sizes_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr TensorList)
tensor_unsafe_split_with_sizes_ll _obj _split_sizes _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).unsafe_split_with_sizes(
    *$(std::vector<int64_t>* _split_sizes)
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

-- tensor_squeeze_l
--   :: Ptr Tensor
--   -> Ptr IntArray
--   -> IO (Ptr Tensor)
-- tensor_squeeze_l _obj _dim =
--   [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze(
--     *$(std::vector<int64_t>* _dim)));
--   }|]

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

-- tensor_squeeze__l
--   :: Ptr Tensor
--   -> Ptr IntArray
--   -> IO (Ptr Tensor)
-- tensor_squeeze__l _obj _dim =
--   [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).squeeze_(
--     *$(std::vector<int64_t>* _dim)));
--   }|]

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

tensor_stft_llltbbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_stft_llltbbb _obj _n_fft _hop_length _win_length _window _normalized _onesided _return_complex =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).stft(
    $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _normalized)
  , $(bool _onesided)
  , $(bool _return_complex)));
  }|]

tensor_stft_llltbsbbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> Ptr StdString
  -> CBool
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_stft_llltbsbbb _obj _n_fft _hop_length _win_length _window _center _pad_mode _normalized _onesided _return_complex =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).stft(
    $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)
  , *$(std::string* _pad_mode)
  , $(bool _normalized)
  , $(bool _onesided)
  , $(bool _return_complex)));
  }|]

tensor_istft_llltbbblb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_istft_llltbbblb _obj _n_fft _hop_length _win_length _window _center _normalized _onesided _length _return_complex =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).istft(
    $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)
  , $(bool _normalized)
  , $(bool _onesided)
  , $(int64_t _length)
  , $(bool _return_complex)));
  }|]

tensor_stride_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
tensor_stride_n _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).stride(
    *$(at::Dimname* _dim));
  }|]

tensor_sum_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_sum_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    $(at::ScalarType _dtype)));
  }|]

tensor_sum_lbs
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_sum_lbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_sum_Nbs
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_sum_Nbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_nansum_lbs
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_nansum_lbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nansum(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
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

tensor_std_llb
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_std_llb _obj _dim _correction _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).std(
    *$(std::vector<int64_t>* _dim)
  , $(int64_t _correction)
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

tensor_std_Nlb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_std_Nlb _obj _dim _correction _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).std(
    *$(std::vector<at::Dimname>* _dim)
  , $(int64_t _correction)
  , $(bool _keepdim)));
  }|]

tensor_prod_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_prod_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    $(at::ScalarType _dtype)));
  }|]

tensor_prod_lbs
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_prod_lbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    $(int64_t _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_prod_nbs
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_prod_nbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
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

tensor_tile_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_tile_l _obj _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tile(
    *$(std::vector<int64_t>* _dims)));
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

tensor__nested_tensor_size
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__nested_tensor_size _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._nested_tensor_size(
    ));
  }|]

tensor__nested_tensor_strides
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__nested_tensor_strides _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._nested_tensor_strides(
    ));
  }|]

-- tensor__nested_tensor_offsets
--   :: Ptr Tensor
--   -> IO (Ptr IntArray)
-- tensor__nested_tensor_offsets _obj =
--   [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>((*$(at::Tensor* _obj))._nested_tensor_offsets(
--     ).vec());
--   }|]

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

tensor_fix
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fix _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fix(
    ));
  }|]

tensor_fix_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fix_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fix_(
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

tensor_var_llb
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_var_llb _obj _dim _correction _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).var(
    *$(std::vector<int64_t>* _dim)
  , $(int64_t _correction)
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

tensor_var_Nlb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_var_Nlb _obj _dim _correction _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).var(
    *$(std::vector<at::Dimname>* _dim)
  , $(int64_t _correction)
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
tensor_where_tt _obj _self _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).where(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _other)));
  }|]

tensor_where_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_where_ts _obj _self _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).where(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _other)));
  }|]

tensor_norm_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_norm_ss _obj _p _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)
  , $(at::ScalarType _dtype)));
  }|]

tensor_norm_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_norm_s _obj _p =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)));
  }|]

tensor_norm_slbs
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_norm_slbs _obj _p _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_norm_slb
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_norm_slb _obj _p _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_norm_sNbs
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr DimnameList
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_norm_sNbs _obj _p _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_norm_sNb
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
tensor_norm_sNb _obj _p _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).norm(
    *$(at::Scalar* _p)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_frexp
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_frexp _obj =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).frexp(
    ));
  }|]

tensor_clone_M
  :: Ptr Tensor
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_clone_M _obj _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clone(
    $(at::MemoryFormat _memory_format)));
  }|]

tensor_positive
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_positive _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).positive(
    ));
  }|]

tensor_resize_as__tM
  :: Ptr Tensor
  -> Ptr Tensor
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_resize_as__tM _obj _the_template _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_as_(
    *$(at::Tensor* _the_template)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_resize_as_sparse__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_resize_as_sparse__t _obj _the_template =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_as_sparse_(
    *$(at::Tensor* _the_template)));
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

tensor_subtract_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_subtract_ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).subtract(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_subtract__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_subtract__ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).subtract_(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_subtract_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_subtract_ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).subtract(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_subtract__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_subtract__ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).subtract_(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_heaviside_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_heaviside_t _obj _values =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).heaviside(
    *$(at::Tensor* _values)));
  }|]

tensor_heaviside__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_heaviside__t _obj _values =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).heaviside_(
    *$(at::Tensor* _values)));
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

tensor__addmm_activation_ttssb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> CBool
  -> IO (Ptr Tensor)
tensor__addmm_activation_ttssb _obj _mat1 _mat2 _beta _alpha _use_gelu =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._addmm_activation(
    *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)
  , $(bool _use_gelu)));
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

tensor_to_dense_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_to_dense_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_dense(
    $(at::ScalarType _dtype)));
  }|]

tensor__to_dense_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor__to_dense_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._to_dense(
    $(at::ScalarType _dtype)));
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

tensor_crow_indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_crow_indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).crow_indices(
    ));
  }|]

tensor_col_indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_col_indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).col_indices(
    ));
  }|]

tensor_ccol_indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ccol_indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ccol_indices(
    ));
  }|]

tensor_row_indices
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_row_indices _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).row_indices(
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

tensor_to_sparse_Lll
  :: Ptr Tensor
  -> Layout
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_Lll _obj _layout _blocksize _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse(
    $(at::Layout _layout)
  , *$(std::vector<int64_t>* _blocksize)
  , $(int64_t _dense_dim)));
  }|]

tensor_to_sparse_csr_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_csr_l _obj _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse_csr(
    $(int64_t _dense_dim)));
  }|]

tensor_to_sparse_csc_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_csc_l _obj _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse_csc(
    $(int64_t _dense_dim)));
  }|]

tensor_to_sparse_bsr_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_bsr_ll _obj _blocksize _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse_bsr(
    *$(std::vector<int64_t>* _blocksize)
  , $(int64_t _dense_dim)));
  }|]

tensor_to_sparse_bsc_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_to_sparse_bsc_ll _obj _blocksize _dense_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_sparse_bsc(
    *$(std::vector<int64_t>* _blocksize)
  , $(int64_t _dense_dim)));
  }|]

tensor_to_mkldnn_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_to_mkldnn_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_mkldnn(
    $(at::ScalarType _dtype)));
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

tensor__autocast_to_reduced_precision_bbss
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> ScalarType
  -> ScalarType
  -> IO (Ptr Tensor)
tensor__autocast_to_reduced_precision_bbss _obj _cuda_enabled _cpu_enabled _cuda_dtype _cpu_dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._autocast_to_reduced_precision(
    $(bool _cuda_enabled)
  , $(bool _cpu_enabled)
  , $(at::ScalarType _cuda_dtype)
  , $(at::ScalarType _cpu_dtype)));
  }|]

tensor__autocast_to_full_precision_bb
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor__autocast_to_full_precision_bb _obj _cuda_enabled _cpu_enabled =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._autocast_to_full_precision(
    $(bool _cuda_enabled)
  , $(bool _cpu_enabled)));
  }|]

tensor_to_obbM
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_to_obbM _obj _options _non_blocking _copy _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::TensorOptions* _options)
  , $(bool _non_blocking)
  , $(bool _copy)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_to_DsbbM
  :: Ptr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_to_DsbbM _obj _device _dtype _non_blocking _copy _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::DeviceType _device)
  , $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_to_sbbM
  :: Ptr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_to_sbbM _obj _dtype _non_blocking _copy _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_to_tbbM
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_to_tbbM _obj _other _non_blocking _copy _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::Tensor* _other)
  , $(bool _non_blocking)
  , $(bool _copy)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_item
  :: Ptr Tensor
  -> IO (Ptr Scalar)
tensor_item _obj =
  [C.throwBlock| at::Scalar* { return new at::Scalar((*$(at::Tensor* _obj)).item(
    ));
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

tensor_set__tlll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_set__tlll _obj _source _storage_offset _size _stride =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_(
    *$(at::Tensor* _source)
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

tensor_view_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_view_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).view(
    $(at::ScalarType _dtype)));
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

tensor_put_ttb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_put_ttb _obj _index _source _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).put(
    *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , $(bool _accumulate)));
  }|]

tensor_index_add__ltts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_add__ltts _obj _dim _index _source _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_index_add_ltts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_add_ltts _obj _dim _index _source _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_index_add_ntts
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_index_add_ntts _obj _dim _index _source _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_add(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_index_reduce__lttsb
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_reduce__lttsb _obj _dim _index _source _reduce _include_self =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_reduce_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , *$(std::string* _reduce)
  , $(bool _include_self)));
  }|]

tensor_index_reduce_lttsb
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_reduce_lttsb _obj _dim _index _source _reduce _include_self =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_reduce(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)
  , *$(std::string* _reduce)
  , $(bool _include_self)));
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

tensor_scatter_ltts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_scatter_ltts _obj _dim _index _src _reduce =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)
  , *$(std::string* _reduce)));
  }|]

tensor_scatter__ltts
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_scatter__ltts _obj _dim _index _src _reduce =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)
  , *$(std::string* _reduce)));
  }|]

tensor_scatter_ltss
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_scatter_ltss _obj _dim _index _value _reduce =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)
  , *$(std::string* _reduce)));
  }|]

tensor_scatter__ltss
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_scatter__ltss _obj _dim _index _value _reduce =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Scalar* _value)
  , *$(std::string* _reduce)));
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

tensor_scatter_reduce_lttsb
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> CBool
  -> IO (Ptr Tensor)
tensor_scatter_reduce_lttsb _obj _dim _index _src _reduce _include_self =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_reduce(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)
  , *$(std::string* _reduce)
  , $(bool _include_self)));
  }|]

tensor_scatter_reduce__lttsb
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> CBool
  -> IO (Ptr Tensor)
tensor_scatter_reduce__lttsb _obj _dim _index _src _reduce _include_self =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).scatter_reduce_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _src)
  , *$(std::string* _reduce)
  , $(bool _include_self)));
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

tensor_bitwise_and_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_and_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_and(
    *$(at::Scalar* _other)));
  }|]

tensor_bitwise_and_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_and_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_and(
    *$(at::Tensor* _other)));
  }|]

tensor_bitwise_and__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_and__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_and_(
    *$(at::Scalar* _other)));
  }|]

tensor_bitwise_and__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_and__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_and_(
    *$(at::Tensor* _other)));
  }|]

tensor___and___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___and___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__and__(
    *$(at::Scalar* _other)));
  }|]

tensor___and___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___and___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__and__(
    *$(at::Tensor* _other)));
  }|]

tensor___iand___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___iand___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__iand__(
    *$(at::Scalar* _other)));
  }|]

tensor___iand___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___iand___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__iand__(
    *$(at::Tensor* _other)));
  }|]

tensor_bitwise_or_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_or_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_or(
    *$(at::Scalar* _other)));
  }|]


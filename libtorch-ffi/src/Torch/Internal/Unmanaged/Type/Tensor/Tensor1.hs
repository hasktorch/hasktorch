
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor1 where


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



tensor_cummin_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummin_n _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummin(
    *$(at::Dimname* _dim)));
  }|]

tensor_cumprod_ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumprod_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumprod(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumprod__ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumprod__ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumprod_(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumprod_ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumprod_ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumprod(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumprod__ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumprod__ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumprod_(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumsum_ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumsum_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumsum(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumsum__ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumsum__ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumsum_(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumsum_ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumsum_ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumsum(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_cumsum__ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_cumsum__ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumsum_(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_diag_embed_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_diag_embed_lll _obj _offset _dim1 _dim2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diag_embed(
    $(int64_t _offset)
  , $(int64_t _dim1)
  , $(int64_t _dim2)));
  }|]

tensor_diagflat_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_diagflat_l _obj _offset =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diagflat(
    $(int64_t _offset)));
  }|]

tensor_diagonal_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_diagonal_lll _obj _offset _dim1 _dim2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diagonal(
    $(int64_t _offset)
  , $(int64_t _dim1)
  , $(int64_t _dim2)));
  }|]

tensor_diagonal_nnnl
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Dimname
  -> Ptr Dimname
  -> Int64
  -> IO (Ptr Tensor)
tensor_diagonal_nnnl _obj _outdim _dim1 _dim2 _offset =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diagonal(
    *$(at::Dimname* _outdim)
  , *$(at::Dimname* _dim1)
  , *$(at::Dimname* _dim2)
  , $(int64_t _offset)));
  }|]

tensor_fill_diagonal__sb
  :: Ptr Tensor
  -> Ptr Scalar
  -> CBool
  -> IO (Ptr Tensor)
tensor_fill_diagonal__sb _obj _fill_value _wrap =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fill_diagonal_(
    *$(at::Scalar* _fill_value)
  , $(bool _wrap)));
  }|]

tensor_diff_lltt
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_diff_lltt _obj _n _dim _prepend _append =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diff(
    $(int64_t _n)
  , $(int64_t _dim)
  , *$(at::Tensor* _prepend)
  , *$(at::Tensor* _append)));
  }|]

tensor_div_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_div_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div(
    *$(at::Tensor* _other)));
  }|]

tensor_div__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_div__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div_(
    *$(at::Tensor* _other)));
  }|]

tensor_div_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_div_ts _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div(
    *$(at::Tensor* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_div__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_div__ts _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div_(
    *$(at::Tensor* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_div_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_div_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div(
    *$(at::Scalar* _other)));
  }|]

tensor_div__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_div__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div_(
    *$(at::Scalar* _other)));
  }|]

tensor_div_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_div_ss _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div(
    *$(at::Scalar* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_div__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_div__ss _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).div_(
    *$(at::Scalar* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_divide_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_divide_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide(
    *$(at::Tensor* _other)));
  }|]

tensor_divide__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_divide__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide_(
    *$(at::Tensor* _other)));
  }|]

tensor_divide_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_divide_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide(
    *$(at::Scalar* _other)));
  }|]

tensor_divide__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_divide__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide_(
    *$(at::Scalar* _other)));
  }|]

tensor_divide_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_divide_ts _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide(
    *$(at::Tensor* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_divide__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_divide__ts _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide_(
    *$(at::Tensor* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_divide_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_divide_ss _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide(
    *$(at::Scalar* _other)
  , *$(std::string* _rounding_mode)));
  }|]

tensor_divide__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr StdString
  -> IO (Ptr Tensor)
tensor_divide__ss _obj _other _rounding_mode =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).divide_(
    *$(at::Scalar* _other)
  , *$(std::string* _rounding_mode)));
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

tensor_dot_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_dot_t _obj _tensor =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).dot(
    *$(at::Tensor* _tensor)));
  }|]

tensor_vdot_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_vdot_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).vdot(
    *$(at::Tensor* _other)));
  }|]

tensor_new_empty_lo
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
tensor_new_empty_lo _obj _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).new_empty(
    *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

tensor_new_empty_strided_llo
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
tensor_new_empty_strided_llo _obj _size _stride _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).new_empty_strided(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)
  , *$(at::TensorOptions* _options)));
  }|]

tensor_new_full_lso
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr Scalar
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
tensor_new_full_lso _obj _size _fill_value _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).new_full(
    *$(std::vector<int64_t>* _size)
  , *$(at::Scalar* _fill_value)
  , *$(at::TensorOptions* _options)));
  }|]

tensor_new_zeros_lo
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
tensor_new_zeros_lo _obj _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).new_zeros(
    *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

tensor_new_ones_lo
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
tensor_new_ones_lo _obj _size _options =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).new_ones(
    *$(std::vector<int64_t>* _size)
  , *$(at::TensorOptions* _options)));
  }|]

tensor_resize__lM
  :: Ptr Tensor
  -> Ptr IntArray
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_resize__lM _obj _size _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_(
    *$(std::vector<int64_t>* _size)
  , $(at::MemoryFormat _memory_format)));
  }|]

tensor_erf
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erf _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erf(
    ));
  }|]

tensor_erf_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erf_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erf_(
    ));
  }|]

tensor_erfc
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erfc _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erfc(
    ));
  }|]

tensor_erfc_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erfc_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erfc_(
    ));
  }|]

tensor_exp
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_exp _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).exp(
    ));
  }|]

tensor_exp_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_exp_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).exp_(
    ));
  }|]

tensor_exp2
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_exp2 _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).exp2(
    ));
  }|]

tensor_exp2_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_exp2_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).exp2_(
    ));
  }|]

tensor_expm1
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_expm1 _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).expm1(
    ));
  }|]

tensor_expm1_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_expm1_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).expm1_(
    ));
  }|]

tensor_expand_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_expand_lb _obj _size _implicit =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).expand(
    *$(std::vector<int64_t>* _size)
  , $(bool _implicit)));
  }|]

tensor_expand_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_expand_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).expand_as(
    *$(at::Tensor* _other)));
  }|]

tensor_flatten_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_flatten_ll _obj _start_dim _end_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flatten(
    $(int64_t _start_dim)
  , $(int64_t _end_dim)));
  }|]

tensor_flatten_lln
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_flatten_lln _obj _start_dim _end_dim _out_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flatten(
    $(int64_t _start_dim)
  , $(int64_t _end_dim)
  , *$(at::Dimname* _out_dim)));
  }|]

tensor_flatten_nnn
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Dimname
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_flatten_nnn _obj _start_dim _end_dim _out_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flatten(
    *$(at::Dimname* _start_dim)
  , *$(at::Dimname* _end_dim)
  , *$(at::Dimname* _out_dim)));
  }|]

tensor_flatten_Nn
  :: Ptr Tensor
  -> Ptr DimnameList
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_flatten_Nn _obj _dims _out_dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).flatten(
    *$(std::vector<at::Dimname>* _dims)
  , *$(at::Dimname* _out_dim)));
  }|]

tensor_unflatten_ll
  :: Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_unflatten_ll _obj _dim _sizes =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unflatten(
    $(int64_t _dim)
  , *$(std::vector<int64_t>* _sizes)));
  }|]

tensor_unflatten_nlN
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr IntArray
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_unflatten_nlN _obj _dim _sizes _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unflatten(
    *$(at::Dimname* _dim)
  , *$(std::vector<int64_t>* _sizes)
  , *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_fill__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_fill__s _obj _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fill_(
    *$(at::Scalar* _value)));
  }|]

tensor_fill__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fill__t _obj _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fill_(
    *$(at::Tensor* _value)));
  }|]

tensor_floor
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_floor _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor(
    ));
  }|]

tensor_floor_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_floor_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor_(
    ));
  }|]

tensor_floor_divide_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_floor_divide_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor_divide(
    *$(at::Tensor* _other)));
  }|]

tensor_floor_divide__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_floor_divide__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor_divide_(
    *$(at::Tensor* _other)));
  }|]

tensor_floor_divide_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_floor_divide_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor_divide(
    *$(at::Scalar* _other)));
  }|]

tensor_floor_divide__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_floor_divide__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).floor_divide_(
    *$(at::Scalar* _other)));
  }|]

tensor_frac
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_frac _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).frac(
    ));
  }|]

tensor_frac_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_frac_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).frac_(
    ));
  }|]

tensor_gcd_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_gcd_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gcd(
    *$(at::Tensor* _other)));
  }|]

tensor_gcd__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_gcd__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gcd_(
    *$(at::Tensor* _other)));
  }|]

tensor_lcm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lcm_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lcm(
    *$(at::Tensor* _other)));
  }|]

tensor_lcm__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lcm__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lcm_(
    *$(at::Tensor* _other)));
  }|]

tensor_index_l
  :: Ptr Tensor
  -> Ptr (C10List (C10Optional Tensor))
  -> IO (Ptr Tensor)
tensor_index_l _obj _indices =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index(
    *$(c10::List<c10::optional<at::Tensor>>* _indices)));
  }|]

tensor_index_copy__ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_copy__ltt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_copy_(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_copy_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_copy_ltt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_copy(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_copy__ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_copy__ntt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_copy_(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_copy_ntt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_copy_ntt _obj _dim _index _source =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_copy(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , *$(at::Tensor* _source)));
  }|]

tensor_index_put__ltb
  :: Ptr Tensor
  -> Ptr (C10List (C10Optional Tensor))
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_put__ltb _obj _indices _values _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_put_(
    *$(c10::List<c10::optional<at::Tensor>>* _indices)
  , *$(at::Tensor* _values)
  , $(bool _accumulate)));
  }|]

tensor_index_put_ltb
  :: Ptr Tensor
  -> Ptr (C10List (C10Optional Tensor))
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_put_ltb _obj _indices _values _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_put(
    *$(c10::List<c10::optional<at::Tensor>>* _indices)
  , *$(at::Tensor* _values)
  , $(bool _accumulate)));
  }|]

tensor_isclose_tddb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (Ptr Tensor)
tensor_isclose_tddb _obj _other _rtol _atol _equal_nan =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).isclose(
    *$(at::Tensor* _other)
  , $(double _rtol)
  , $(double _atol)
  , $(bool _equal_nan)));
  }|]

tensor_isnan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_isnan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).isnan(
    ));
  }|]

tensor_is_distributed
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_distributed _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_distributed(
    );
  }|]

tensor_is_floating_point
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_floating_point _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_floating_point(
    );
  }|]

tensor_is_complex
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_complex _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_complex(
    );
  }|]

tensor_is_conj
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_conj _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_conj(
    );
  }|]

tensor__is_zerotensor
  :: Ptr Tensor
  -> IO (CBool)
tensor__is_zerotensor _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj))._is_zerotensor(
    );
  }|]

tensor_is_neg
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_neg _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_neg(
    );
  }|]

tensor_isreal
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_isreal _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).isreal(
    ));
  }|]

tensor_is_nonzero
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_nonzero _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_nonzero(
    );
  }|]

tensor_is_same_size_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_same_size_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_same_size(
    *$(at::Tensor* _other));
  }|]

tensor_is_signed
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_signed _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_signed(
    );
  }|]

tensor_is_inference
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_inference _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_inference(
    );
  }|]

tensor_kron_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_kron_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).kron(
    *$(at::Tensor* _other)));
  }|]

tensor_kthvalue_llb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_kthvalue_llb _obj _k _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).kthvalue(
    $(int64_t _k)
  , $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_kthvalue_lnb
  :: Ptr Tensor
  -> Int64
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_kthvalue_lnb _obj _k _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).kthvalue(
    $(int64_t _k)
  , *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_nan_to_num_ddd
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (Ptr Tensor)
tensor_nan_to_num_ddd _obj _nan _posinf _neginf =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nan_to_num(
    $(double _nan)
  , $(double _posinf)
  , $(double _neginf)));
  }|]

tensor_nan_to_num__ddd
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> CDouble
  -> IO (Ptr Tensor)
tensor_nan_to_num__ddd _obj _nan _posinf _neginf =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nan_to_num_(
    $(double _nan)
  , $(double _posinf)
  , $(double _neginf)));
  }|]

tensor_ldexp_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ldexp_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ldexp(
    *$(at::Tensor* _other)));
  }|]

tensor_ldexp__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ldexp__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ldexp_(
    *$(at::Tensor* _other)));
  }|]

tensor_log
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log(
    ));
  }|]

tensor_log_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log_(
    ));
  }|]

tensor_log10
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log10 _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log10(
    ));
  }|]

tensor_log10_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log10_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log10_(
    ));
  }|]

tensor_log1p
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log1p _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log1p(
    ));
  }|]

tensor_log1p_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log1p_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log1p_(
    ));
  }|]

tensor_log2
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log2 _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log2(
    ));
  }|]

tensor_log2_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_log2_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log2_(
    ));
  }|]

tensor_logaddexp_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logaddexp_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logaddexp(
    *$(at::Tensor* _other)));
  }|]

tensor_logaddexp2_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logaddexp2_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logaddexp2(
    *$(at::Tensor* _other)));
  }|]

tensor_xlogy_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_xlogy_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).xlogy(
    *$(at::Tensor* _other)));
  }|]

tensor_xlogy_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_xlogy_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).xlogy(
    *$(at::Scalar* _other)));
  }|]

tensor_xlogy__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_xlogy__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).xlogy_(
    *$(at::Tensor* _other)));
  }|]

tensor_xlogy__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_xlogy__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).xlogy_(
    *$(at::Scalar* _other)));
  }|]

tensor_log_softmax_ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_log_softmax_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log_softmax(
    $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_log_softmax_ns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_log_softmax_ns _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log_softmax(
    *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_logcumsumexp_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_logcumsumexp_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logcumsumexp(
    $(int64_t _dim)));
  }|]

tensor_logcumsumexp_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
tensor_logcumsumexp_n _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logcumsumexp(
    *$(at::Dimname* _dim)));
  }|]

tensor_logsumexp_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_logsumexp_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logsumexp(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_logsumexp_Nb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
tensor_logsumexp_Nb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logsumexp(
    *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_matmul_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_matmul_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).matmul(
    *$(at::Tensor* _other)));
  }|]

tensor_matrix_power_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_matrix_power_l _obj _n =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).matrix_power(
    $(int64_t _n)));
  }|]

tensor_matrix_exp
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_matrix_exp _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).matrix_exp(
    ));
  }|]

tensor_aminmax_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_aminmax_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).aminmax(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_max_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_max_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).max(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_max_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_max_nb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).max(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_amax_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_amax_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).amax(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_mean_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_mean_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    $(at::ScalarType _dtype)));
  }|]

tensor_mean_lbs
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_mean_lbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_mean_Nbs
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_mean_Nbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_nanmean_lbs
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_nanmean_lbs _obj _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nanmean(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_median
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_median _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).median(
    ));
  }|]

tensor_median_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_median_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).median(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_median_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_median_nb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).median(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_nanmedian
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_nanmedian _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nanmedian(
    ));
  }|]

tensor_nanmedian_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_nanmedian_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).nanmedian(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_nanmedian_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_nanmedian_nb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).nanmedian(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_min_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_min_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).min(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_min_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_min_nb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).min(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_amin_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_amin_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).amin(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_mm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mm_t _obj _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mm(
    *$(at::Tensor* _mat2)));
  }|]

tensor_mode_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_mode_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).mode(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_mode_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_mode_nb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).mode(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_mul_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mul_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mul(
    *$(at::Tensor* _other)));
  }|]

tensor_mul__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mul__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mul_(
    *$(at::Tensor* _other)));
  }|]

tensor_mul_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_mul_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mul(
    *$(at::Scalar* _other)));
  }|]

tensor_mul__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_mul__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mul_(
    *$(at::Scalar* _other)));
  }|]

tensor_multiply_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_multiply_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).multiply(
    *$(at::Tensor* _other)));
  }|]

tensor_multiply__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_multiply__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).multiply_(
    *$(at::Tensor* _other)));
  }|]

tensor_multiply_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_multiply_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).multiply(
    *$(at::Scalar* _other)));
  }|]

tensor_multiply__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_multiply__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).multiply_(
    *$(at::Scalar* _other)));
  }|]

tensor_mv_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mv_t _obj _vec =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mv(
    *$(at::Tensor* _vec)));
  }|]

tensor_mvlgamma_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_mvlgamma_l _obj _p =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mvlgamma(
    $(int64_t _p)));
  }|]

tensor_mvlgamma__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_mvlgamma__l _obj _p =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mvlgamma_(
    $(int64_t _p)));
  }|]

tensor_narrow_copy_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_narrow_copy_lll _obj _dim _start _length =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).narrow_copy(
    $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _length)));
  }|]

tensor_narrow_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_narrow_lll _obj _dim _start _length =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).narrow(
    $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _length)));
  }|]

tensor_narrow_ltl
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_narrow_ltl _obj _dim _start _length =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).narrow(
    $(int64_t _dim)
  , *$(at::Tensor* _start)
  , $(int64_t _length)));
  }|]

tensor_permute_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_permute_l _obj _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).permute(
    *$(std::vector<int64_t>* _dims)));
  }|]

tensor_numpy_T
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_numpy_T _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).numpy_T(
    ));
  }|]

tensor_matrix_H
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_matrix_H _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).matrix_H(
    ));
  }|]

tensor_mT
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mT _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mT(
    ));
  }|]

tensor_mH
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mH _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mH(
    ));
  }|]

tensor_adjoint
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_adjoint _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).adjoint(
    ));
  }|]

tensor_is_pinned_D
  :: Ptr Tensor
  -> DeviceType
  -> IO (CBool)
tensor_is_pinned_D _obj _device =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_pinned(
    $(at::DeviceType _device));
  }|]

tensor_pin_memory_D
  :: Ptr Tensor
  -> DeviceType
  -> IO (Ptr Tensor)
tensor_pin_memory_D _obj _device =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pin_memory(
    $(at::DeviceType _device)));
  }|]

tensor_pinverse_d
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
tensor_pinverse_d _obj _rcond =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pinverse(
    $(double _rcond)));
  }|]

tensor_rad2deg
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_rad2deg _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rad2deg(
    ));
  }|]

tensor_rad2deg_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_rad2deg_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rad2deg_(
    ));
  }|]

tensor_deg2rad
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_deg2rad _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).deg2rad(
    ));
  }|]

tensor_deg2rad_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_deg2rad_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).deg2rad_(
    ));
  }|]

tensor_ravel
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ravel _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ravel(
    ));
  }|]

tensor_reciprocal
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_reciprocal _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reciprocal(
    ));
  }|]

tensor_reciprocal_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_reciprocal_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reciprocal_(
    ));
  }|]

tensor_neg
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_neg _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).neg(
    ));
  }|]

tensor_neg_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_neg_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).neg_(
    ));
  }|]

tensor_negative
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_negative _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).negative(
    ));
  }|]

tensor_negative_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_negative_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).negative_(
    ));
  }|]

tensor_repeat_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_repeat_l _obj _repeats =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).repeat(
    *$(std::vector<int64_t>* _repeats)));
  }|]

tensor_repeat_interleave_tll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_repeat_interleave_tll _obj _repeats _dim _output_size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).repeat_interleave(
    *$(at::Tensor* _repeats)
  , $(int64_t _dim)
  , $(int64_t _output_size)));
  }|]

tensor_repeat_interleave_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_repeat_interleave_lll _obj _repeats _dim _output_size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).repeat_interleave(
    $(int64_t _repeats)
  , $(int64_t _dim)
  , $(int64_t _output_size)));
  }|]

tensor_reshape_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_reshape_l _obj _shape =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reshape(
    *$(std::vector<int64_t>* _shape)));
  }|]

tensor__reshape_alias_ll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor__reshape_alias_ll _obj _size _stride =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._reshape_alias(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)));
  }|]

tensor_reshape_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_reshape_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reshape_as(
    *$(at::Tensor* _other)));
  }|]

tensor_round
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_round _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).round(
    ));
  }|]

tensor_round_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_round_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).round_(
    ));
  }|]

tensor_round_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_round_l _obj _decimals =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).round(
    $(int64_t _decimals)));
  }|]

tensor_round__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_round__l _obj _decimals =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).round_(
    $(int64_t _decimals)));
  }|]

tensor_relu
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_relu _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).relu(
    ));
  }|]

tensor_relu_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_relu_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).relu_(
    ));
  }|]

tensor_prelu_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_prelu_t _obj _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prelu(
    *$(at::Tensor* _weight)));
  }|]

tensor_hardshrink_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_hardshrink_s _obj _lambd =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).hardshrink(
    *$(at::Scalar* _lambd)));
  }|]

tensor_hardshrink_backward_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_hardshrink_backward_ts _obj _self _lambd =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).hardshrink_backward(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _lambd)));
  }|]

tensor_rsqrt
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_rsqrt _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rsqrt(
    ));
  }|]

tensor_rsqrt_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_rsqrt_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rsqrt_(
    ));
  }|]

tensor_select_nl
  :: Ptr Tensor
  -> Ptr Dimname
  -> Int64
  -> IO (Ptr Tensor)
tensor_select_nl _obj _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).select(
    *$(at::Dimname* _dim)
  , $(int64_t _index)));
  }|]

tensor_select_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_select_ll _obj _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).select(
    $(int64_t _dim)
  , $(int64_t _index)));
  }|]

tensor_sigmoid
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sigmoid _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sigmoid(
    ));
  }|]

tensor_sigmoid_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sigmoid_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sigmoid_(
    ));
  }|]

tensor_logit_d
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
tensor_logit_d _obj _eps =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logit(
    $(double _eps)));
  }|]

tensor_logit__d
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
tensor_logit__d _obj _eps =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logit_(
    $(double _eps)));
  }|]

tensor_sin
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sin _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sin(
    ));
  }|]

tensor_sin_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sin_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sin_(
    ));
  }|]

tensor_sinc
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sinc _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sinc(
    ));
  }|]

tensor_sinc_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sinc_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sinc_(
    ));
  }|]

tensor_sinh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sinh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sinh(
    ));
  }|]


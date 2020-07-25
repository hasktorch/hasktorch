
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor3 where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/Tensor.h>"
C.include "<vector>"

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

tensor_bitwise_or_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_or_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_or(
    *$(at::Tensor* _other)));
  }|]

tensor_bitwise_or__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_or__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_or_(
    *$(at::Scalar* _other)));
  }|]

tensor_bitwise_or__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_or__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_or_(
    *$(at::Tensor* _other)));
  }|]

tensor___or___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___or___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__or__(
    *$(at::Scalar* _other)));
  }|]

tensor___or___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___or___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__or__(
    *$(at::Tensor* _other)));
  }|]

tensor___ior___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___ior___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ior__(
    *$(at::Scalar* _other)));
  }|]

tensor___ior___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___ior___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ior__(
    *$(at::Tensor* _other)));
  }|]

tensor_bitwise_xor_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_xor_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_xor(
    *$(at::Scalar* _other)));
  }|]

tensor_bitwise_xor_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_xor_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_xor(
    *$(at::Tensor* _other)));
  }|]

tensor_bitwise_xor__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_bitwise_xor__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_xor_(
    *$(at::Scalar* _other)));
  }|]

tensor_bitwise_xor__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_xor__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_xor_(
    *$(at::Tensor* _other)));
  }|]

tensor___xor___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___xor___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__xor__(
    *$(at::Scalar* _other)));
  }|]

tensor___xor___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___xor___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__xor__(
    *$(at::Tensor* _other)));
  }|]

tensor___ixor___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___ixor___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ixor__(
    *$(at::Scalar* _other)));
  }|]

tensor___ixor___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___ixor___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ixor__(
    *$(at::Tensor* _other)));
  }|]

tensor___lshift___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___lshift___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__lshift__(
    *$(at::Scalar* _other)));
  }|]

tensor___lshift___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___lshift___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__lshift__(
    *$(at::Tensor* _other)));
  }|]

tensor___ilshift___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___ilshift___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ilshift__(
    *$(at::Scalar* _other)));
  }|]

tensor___ilshift___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___ilshift___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__ilshift__(
    *$(at::Tensor* _other)));
  }|]

tensor___rshift___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___rshift___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__rshift__(
    *$(at::Scalar* _other)));
  }|]

tensor___rshift___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___rshift___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__rshift__(
    *$(at::Tensor* _other)));
  }|]

tensor___irshift___s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor___irshift___s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__irshift__(
    *$(at::Scalar* _other)));
  }|]

tensor___irshift___t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor___irshift___t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__irshift__(
    *$(at::Tensor* _other)));
  }|]

tensor_lgamma_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lgamma_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lgamma_(
    ));
  }|]

tensor_atan2__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan2__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan2_(
    *$(at::Tensor* _other)));
  }|]

tensor_tril__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_tril__l _obj _diagonal =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tril_(
    $(int64_t _diagonal)));
  }|]

tensor_triu__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_triu__l _obj _diagonal =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).triu_(
    $(int64_t _diagonal)));
  }|]

tensor_digamma_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_digamma_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).digamma_(
    ));
  }|]

tensor_polygamma__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_polygamma__l _obj _n =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).polygamma_(
    $(int64_t _n)));
  }|]

tensor_renorm__sls
  :: Ptr Tensor
  -> Ptr Scalar
  -> Int64
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_renorm__sls _obj _p _dim _maxnorm =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).renorm_(
    *$(at::Scalar* _p)
  , $(int64_t _dim)
  , *$(at::Scalar* _maxnorm)));
  }|]

tensor_pow__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_pow__s _obj _exponent =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pow_(
    *$(at::Scalar* _exponent)));
  }|]

tensor_pow__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_pow__t _obj _exponent =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pow_(
    *$(at::Tensor* _exponent)));
  }|]

tensor_lerp__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_lerp__ts _obj _end _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lerp_(
    *$(at::Tensor* _end)
  , *$(at::Scalar* _weight)));
  }|]

tensor_lerp__tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lerp__tt _obj _end _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lerp_(
    *$(at::Tensor* _end)
  , *$(at::Tensor* _weight)));
  }|]

tensor_fmod__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_fmod__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fmod_(
    *$(at::Scalar* _other)));
  }|]

tensor_fmod__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fmod__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fmod_(
    *$(at::Tensor* _other)));
  }|]

tensor_remainder__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_remainder__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).remainder_(
    *$(at::Scalar* _other)));
  }|]

tensor_remainder__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_remainder__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).remainder_(
    *$(at::Tensor* _other)));
  }|]

tensor_addbmm__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addbmm__ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addbmm_(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addbmm_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addbmm_ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addbmm(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addcdiv__tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addcdiv__tts _obj _tensor1 _tensor2 _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addcdiv_(
    *$(at::Tensor* _tensor1)
  , *$(at::Tensor* _tensor2)
  , *$(at::Scalar* _value)));
  }|]

tensor_random__llp
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_random__llp _obj _from _to _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).random_(
    $(int64_t _from)
  , $(int64_t _to)
  , $(at::Generator * _generator)));
  }|]

tensor_random__lp
  :: Ptr Tensor
  -> Int64
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_random__lp _obj _to _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).random_(
    $(int64_t _to)
  , $(at::Generator * _generator)));
  }|]

tensor_random__p
  :: Ptr Tensor
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_random__p _obj _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).random_(
    $(at::Generator * _generator)));
  }|]

tensor_uniform__ddp
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_uniform__ddp _obj _from _to _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).uniform_(
    $(double _from)
  , $(double _to)
  , $(at::Generator * _generator)));
  }|]

tensor_normal__ddp
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_normal__ddp _obj _mean _std _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).normal_(
    $(double _mean)
  , $(double _std)
  , $(at::Generator * _generator)));
  }|]

tensor_cauchy__ddp
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_cauchy__ddp _obj _median _sigma _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cauchy_(
    $(double _median)
  , $(double _sigma)
  , $(at::Generator * _generator)));
  }|]

tensor_log_normal__ddp
  :: Ptr Tensor
  -> CDouble
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_log_normal__ddp _obj _mean _std _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log_normal_(
    $(double _mean)
  , $(double _std)
  , $(at::Generator * _generator)));
  }|]

tensor_exponential__dp
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_exponential__dp _obj _lambd _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).exponential_(
    $(double _lambd)
  , $(at::Generator * _generator)));
  }|]

tensor_geometric__dp
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_geometric__dp _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).geometric_(
    $(double _p)
  , $(at::Generator * _generator)));
  }|]

tensor_diag_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_diag_l _obj _diagonal =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).diag(
    $(int64_t _diagonal)));
  }|]

tensor_triu_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_triu_l _obj _diagonal =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).triu(
    $(int64_t _diagonal)));
  }|]

tensor_tril_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_tril_l _obj _diagonal =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).tril(
    $(int64_t _diagonal)));
  }|]

tensor_trace
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_trace _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).trace(
    ));
  }|]

tensor_ne_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_ne_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ne(
    *$(at::Scalar* _other)));
  }|]

tensor_ne_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ne_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ne(
    *$(at::Tensor* _other)));
  }|]

tensor_eq_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_eq_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).eq(
    *$(at::Scalar* _other)));
  }|]

tensor_eq_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_eq_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).eq(
    *$(at::Tensor* _other)));
  }|]

tensor_ge_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_ge_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ge(
    *$(at::Scalar* _other)));
  }|]

tensor_ge_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ge_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ge(
    *$(at::Tensor* _other)));
  }|]

tensor_le_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_le_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).le(
    *$(at::Scalar* _other)));
  }|]

tensor_le_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_le_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).le(
    *$(at::Tensor* _other)));
  }|]

tensor_gt_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_gt_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gt(
    *$(at::Scalar* _other)));
  }|]

tensor_gt_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_gt_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gt(
    *$(at::Tensor* _other)));
  }|]

tensor_lt_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_lt_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lt(
    *$(at::Scalar* _other)));
  }|]

tensor_lt_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lt_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lt(
    *$(at::Tensor* _other)));
  }|]

tensor_take_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_take_t _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).take(
    *$(at::Tensor* _index)));
  }|]

tensor_index_select_lt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_select_lt _obj _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_select(
    $(int64_t _dim)
  , *$(at::Tensor* _index)));
  }|]

tensor_index_select_nt
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_index_select_nt _obj _dim _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_select(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)));
  }|]

tensor_masked_select_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_masked_select_t _obj _mask =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).masked_select(
    *$(at::Tensor* _mask)));
  }|]

tensor_nonzero
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_nonzero _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).nonzero(
    ));
  }|]

tensor_nonzero_numpy
  :: Ptr Tensor
  -> IO (Ptr TensorList)
tensor_nonzero_numpy _obj =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).nonzero_numpy(
    ));
  }|]

tensor_gather_ltb
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_gather_ltb _obj _dim _index _sparse_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gather(
    $(int64_t _dim)
  , *$(at::Tensor* _index)
  , $(bool _sparse_grad)));
  }|]

tensor_gather_ntb
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_gather_ntb _obj _dim _index _sparse_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).gather(
    *$(at::Dimname* _dim)
  , *$(at::Tensor* _index)
  , $(bool _sparse_grad)));
  }|]

tensor_addcmul_tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addcmul_tts _obj _tensor1 _tensor2 _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addcmul(
    *$(at::Tensor* _tensor1)
  , *$(at::Tensor* _tensor2)
  , *$(at::Scalar* _value)));
  }|]

tensor_addcmul__tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addcmul__tts _obj _tensor1 _tensor2 _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addcmul_(
    *$(at::Tensor* _tensor1)
  , *$(at::Tensor* _tensor2)
  , *$(at::Scalar* _value)));
  }|]

tensor_addcdiv_tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addcdiv_tts _obj _tensor1 _tensor2 _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addcdiv(
    *$(at::Tensor* _tensor1)
  , *$(at::Tensor* _tensor2)
  , *$(at::Scalar* _value)));
  }|]

tensor_lstsq_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_lstsq_t _obj _A =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).lstsq(
    *$(at::Tensor* _A)));
  }|]

tensor_triangular_solve_tbbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_triangular_solve_tbbb _obj _A _upper _transpose _unitriangular =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).triangular_solve(
    *$(at::Tensor* _A)
  , $(bool _upper)
  , $(bool _transpose)
  , $(bool _unitriangular)));
  }|]

tensor_symeig_bb
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_symeig_bb _obj _eigenvectors _upper =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).symeig(
    $(bool _eigenvectors)
  , $(bool _upper)));
  }|]

tensor_eig_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_eig_b _obj _eigenvectors =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).eig(
    $(bool _eigenvectors)));
  }|]

tensor_svd_bb
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor,Tensor)))
tensor_svd_bb _obj _some _compute_uv =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).svd(
    $(bool _some)
  , $(bool _compute_uv)));
  }|]

tensor_cholesky_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_cholesky_b _obj _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cholesky(
    $(bool _upper)));
  }|]

tensor_cholesky_solve_tb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_cholesky_solve_tb _obj _input2 _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cholesky_solve(
    *$(at::Tensor* _input2)
  , $(bool _upper)));
  }|]

tensor_solve_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_solve_t _obj _A =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).solve(
    *$(at::Tensor* _A)));
  }|]

tensor_cholesky_inverse_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_cholesky_inverse_b _obj _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cholesky_inverse(
    $(bool _upper)));
  }|]

tensor_qr_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_qr_b _obj _some =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).qr(
    $(bool _some)));
  }|]

tensor_geqrf
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_geqrf _obj =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).geqrf(
    ));
  }|]

tensor_orgqr_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_orgqr_t _obj _input2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).orgqr(
    *$(at::Tensor* _input2)));
  }|]

tensor_ormqr_ttbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_ormqr_ttbb _obj _input2 _input3 _left _transpose =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ormqr(
    *$(at::Tensor* _input2)
  , *$(at::Tensor* _input3)
  , $(bool _left)
  , $(bool _transpose)));
  }|]

tensor_lu_solve_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lu_solve_tt _obj _LU_data _LU_pivots =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lu_solve(
    *$(at::Tensor* _LU_data)
  , *$(at::Tensor* _LU_pivots)));
  }|]

tensor_multinomial_lbp
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_multinomial_lbp _obj _num_samples _replacement _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).multinomial(
    $(int64_t _num_samples)
  , $(bool _replacement)
  , $(at::Generator * _generator)));
  }|]

tensor_lgamma
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lgamma _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lgamma(
    ));
  }|]

tensor_digamma
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_digamma _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).digamma(
    ));
  }|]

tensor_polygamma_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_polygamma_l _obj _n =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).polygamma(
    $(int64_t _n)));
  }|]

tensor_erfinv
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erfinv _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erfinv(
    ));
  }|]

tensor_erfinv_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erfinv_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erfinv_(
    ));
  }|]

tensor_sign
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sign _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sign(
    ));
  }|]

tensor_sign_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sign_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sign_(
    ));
  }|]

tensor_dist_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_dist_ts _obj _other _p =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).dist(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _p)));
  }|]

tensor_atan2_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan2_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan2(
    *$(at::Tensor* _other)));
  }|]

tensor_lerp_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_lerp_ts _obj _end _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lerp(
    *$(at::Tensor* _end)
  , *$(at::Scalar* _weight)));
  }|]

tensor_lerp_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_lerp_tt _obj _end _weight =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).lerp(
    *$(at::Tensor* _end)
  , *$(at::Tensor* _weight)));
  }|]

tensor_histc_lss
  :: Ptr Tensor
  -> Int64
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_histc_lss _obj _bins _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).histc(
    $(int64_t _bins)
  , *$(at::Scalar* _min)
  , *$(at::Scalar* _max)));
  }|]

tensor_fmod_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_fmod_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fmod(
    *$(at::Scalar* _other)));
  }|]

tensor_fmod_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_fmod_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fmod(
    *$(at::Tensor* _other)));
  }|]

tensor_remainder_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_remainder_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).remainder(
    *$(at::Scalar* _other)));
  }|]

tensor_remainder_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_remainder_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).remainder(
    *$(at::Tensor* _other)));
  }|]

tensor_min_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_min_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).min(
    *$(at::Tensor* _other)));
  }|]

tensor_min
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_min _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).min(
    ));
  }|]

tensor_max_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_max_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).max(
    *$(at::Tensor* _other)));
  }|]

tensor_max
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_max _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).max(
    ));
  }|]

tensor_median
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_median _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).median(
    ));
  }|]

tensor_sort_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_sort_lb _obj _dim _descending =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).sort(
    $(int64_t _dim)
  , $(bool _descending)));
  }|]

tensor_sort_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_sort_nb _obj _dim _descending =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).sort(
    *$(at::Dimname* _dim)
  , $(bool _descending)));
  }|]

tensor_argsort_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_argsort_lb _obj _dim _descending =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argsort(
    $(int64_t _dim)
  , $(bool _descending)));
  }|]

tensor_argsort_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
tensor_argsort_nb _obj _dim _descending =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argsort(
    *$(at::Dimname* _dim)
  , $(bool _descending)));
  }|]

tensor_topk_llbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_topk_llbb _obj _k _dim _largest _sorted =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).topk(
    $(int64_t _k)
  , $(int64_t _dim)
  , $(bool _largest)
  , $(bool _sorted)));
  }|]

tensor_all
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_all _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).all(
    ));
  }|]

tensor_any
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_any _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).any(
    ));
  }|]

tensor_renorm_sls
  :: Ptr Tensor
  -> Ptr Scalar
  -> Int64
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_renorm_sls _obj _p _dim _maxnorm =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).renorm(
    *$(at::Scalar* _p)
  , $(int64_t _dim)
  , *$(at::Scalar* _maxnorm)));
  }|]

tensor_unfold_lll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
tensor_unfold_lll _obj _dimension _size _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unfold(
    $(int64_t _dimension)
  , $(int64_t _size)
  , $(int64_t _step)));
  }|]

tensor_equal_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_equal_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).equal(
    *$(at::Tensor* _other));
  }|]

tensor_pow_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_pow_t _obj _exponent =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pow(
    *$(at::Tensor* _exponent)));
  }|]

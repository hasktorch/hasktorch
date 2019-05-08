
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.Tensor where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"



newTensor
  :: IO (Ptr Tensor)
newTensor  =
  [C.throwBlock| at::Tensor* { return new at::Tensor(
    );
  }|]

newTensor_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
newTensor_t _x =
  [C.throwBlock| at::Tensor* { return new at::Tensor(
    *$(at::Tensor* _x));
  }|]



deleteTensor :: Ptr Tensor -> IO ()
deleteTensor object = [C.throwBlock| void { delete $(at::Tensor* object);}|]

instance CppObject Tensor where
  fromPtr ptr = newForeignPtr ptr (deleteTensor ptr)



tensor_dim
  :: Ptr Tensor
  -> IO (Int64)
tensor_dim _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).dim(
    );
  }|]

tensor_storage_offset
  :: Ptr Tensor
  -> IO (Int64)
tensor_storage_offset _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).storage_offset(
    );
  }|]

tensor_defined
  :: Ptr Tensor
  -> IO (CBool)
tensor_defined _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).defined(
    );
  }|]

tensor_reset
  :: Ptr Tensor
  -> IO (())
tensor_reset _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).reset(
    );
  }|]

tensor__assign__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__assign__t _obj _x =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))=(
    *$(at::Tensor* _x));
  }|]

tensor_is_same_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_same_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_same(
    *$(at::Tensor* _other));
  }|]

tensor_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_use_count _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).use_count(
    );
  }|]

tensor_weak_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_weak_use_count _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).weak_use_count(
    );
  }|]

tensor_sizes
  :: Ptr Tensor
  -> IO (Ptr IntArray)
tensor_sizes _obj =
  [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>((*$(at::Tensor* _obj)).sizes(
    ).vec());
  }|]

tensor_strides
  :: Ptr Tensor
  -> IO (Ptr IntArray)
tensor_strides _obj =
  [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>((*$(at::Tensor* _obj)).strides(
    ).vec());
  }|]

tensor_ndimension
  :: Ptr Tensor
  -> IO (Int64)
tensor_ndimension _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).ndimension(
    );
  }|]

tensor_is_contiguous
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_contiguous _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_contiguous(
    );
  }|]

tensor_nbytes
  :: Ptr Tensor
  -> IO (CSize)
tensor_nbytes _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).nbytes(
    );
  }|]

tensor_itemsize
  :: Ptr Tensor
  -> IO (CSize)
tensor_itemsize _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).itemsize(
    );
  }|]

tensor_element_size
  :: Ptr Tensor
  -> IO (CSize)
tensor_element_size _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).element_size(
    );
  }|]

tensor_scalar_type
  :: Ptr Tensor
  -> IO (ScalarType)
tensor_scalar_type _obj =
  [C.throwBlock| at::ScalarType { return (*$(at::Tensor* _obj)).scalar_type(
    );
  }|]

tensor_has_storage
  :: Ptr Tensor
  -> IO (CBool)
tensor_has_storage _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).has_storage(
    );
  }|]

tensor_is_alias_of_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_alias_of_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_alias_of(
    *$(at::Tensor* _other));
  }|]

tensor_copy__tb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_copy__tb _obj _src _non_blocking =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copy_(
    *$(at::Tensor* _src)
  , $(bool _non_blocking)));
  }|]

tensor_toType_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_toType_s _obj _t =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).toType(
    $(at::ScalarType _t)));
  }|]

tensor_is_variable
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_variable _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_variable(
    );
  }|]

tensor_get_device
  :: Ptr Tensor
  -> IO (Int64)
tensor_get_device _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).get_device(
    );
  }|]

tensor_is_cuda
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_cuda _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_cuda(
    );
  }|]

tensor_is_hip
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_hip _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_hip(
    );
  }|]

tensor_is_sparse
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_sparse _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_sparse(
    );
  }|]

tensor_options
  :: Ptr Tensor
  -> IO (Ptr TensorOptions)
tensor_options _obj =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::Tensor* _obj)).options(
    ));
  }|]

tensor_item_int64_t
  :: Ptr Tensor
  -> IO (Int64)
tensor_item_int64_t _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).item<int64_t>(
    );
  }|]

tensor_item_float
  :: Ptr Tensor
  -> IO (CFloat)
tensor_item_float _obj =
  [C.throwBlock| float { return (*$(at::Tensor* _obj)).item<float>(
    );
  }|]

tensor_item_double
  :: Ptr Tensor
  -> IO (CDouble)
tensor_item_double _obj =
  [C.throwBlock| double { return (*$(at::Tensor* _obj)).item<double>(
    );
  }|]

tensor_print
  :: Ptr Tensor
  -> IO (())
tensor_print _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).print(
    );
  }|]

tensor__iadd__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__iadd__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))+=(
    *$(at::Tensor* _other));
  }|]

tensor__iadd__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__iadd__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))+=(
    *$(at::Scalar* _other));
  }|]

tensor__isub__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__isub__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))-=(
    *$(at::Tensor* _other));
  }|]

tensor__isub__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__isub__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))-=(
    *$(at::Scalar* _other));
  }|]

tensor__imul__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__imul__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))*=(
    *$(at::Tensor* _other));
  }|]

tensor__imul__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__imul__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))*=(
    *$(at::Scalar* _other));
  }|]

tensor__idiv__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor__idiv__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))/=(
    *$(at::Tensor* _other)));
  }|]

tensor__idiv__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor__idiv__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))/=(
    *$(at::Scalar* _other)));
  }|]

tensor__at__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor__at__s _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    *$(at::Scalar* _index))]);
  }|]

tensor__at__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor__at__t _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    *$(at::Tensor* _index))]);
  }|]

tensor__at__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor__at__l _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    $(int64_t _index))]);
  }|]

tensor_cpu
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cpu _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cpu(
    ));
  }|]

tensor_cuda
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cuda _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cuda(
    ));
  }|]

tensor_hip
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_hip _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).hip(
    ));
  }|]

tensor_set_requires_grad_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_set_requires_grad_b _obj _requires_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_requires_grad(
    $(bool _requires_grad)));
  }|]

tensor_requires_grad
  :: Ptr Tensor
  -> IO (CBool)
tensor_requires_grad _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).requires_grad(
    );
  }|]

tensor_grad
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_grad _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).grad(
    ));
  }|]

tensor_set_data_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor_set_data_t _obj _new_data =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).set_data(
    *$(at::Tensor* _new_data));
  }|]

tensor_backward
  :: Ptr Tensor
  -> IO (())
tensor_backward _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).backward(
    );
  }|]

tensor_abs
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_abs _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).abs(
    ));
  }|]

tensor_abs_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_abs_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).abs_(
    ));
  }|]

tensor_acos
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acos _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acos(
    ));
  }|]

tensor_acos_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acos_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acos_(
    ));
  }|]

tensor_add_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add_ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add__ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add_(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add_ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add__ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add_(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmv_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmv_ttss _obj _mat _vec _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmv(
    *$(at::Tensor* _mat)
  , *$(at::Tensor* _vec)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmv__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmv__ttss _obj _mat _vec _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmv_(
    *$(at::Tensor* _mat)
  , *$(at::Tensor* _vec)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addr_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addr_ttss _obj _vec1 _vec2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addr(
    *$(at::Tensor* _vec1)
  , *$(at::Tensor* _vec2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addr__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addr__ttss _obj _vec1 _vec2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addr_(
    *$(at::Tensor* _vec1)
  , *$(at::Tensor* _vec2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_all_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_all_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).all(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_allclose_tddb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (CBool)
tensor_allclose_tddb _obj _other _rtol _atol _equal_nan =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).allclose(
    *$(at::Tensor* _other)
  , $(double _rtol)
  , $(double _atol)
  , $(bool _equal_nan));
  }|]

tensor_any_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_any_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).any(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_argmax_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_argmax_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmax(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_argmax
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_argmax _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmax(
    ));
  }|]

tensor_argmin_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_argmin_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmin(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_argmin
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_argmin _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmin(
    ));
  }|]

tensor_asin
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asin _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asin(
    ));
  }|]

tensor_asin_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asin_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asin_(
    ));
  }|]

tensor_atan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan(
    ));
  }|]

tensor_atan_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan_(
    ));
  }|]

tensor_baddbmm_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_baddbmm_ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).baddbmm(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_baddbmm__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_baddbmm__ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).baddbmm_(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_bernoulli_p
  :: Ptr Tensor
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli_p _obj _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli(
    $(at::Generator * _generator)));
  }|]

tensor_bernoulli__tp
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli__tp _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli_(
    *$(at::Tensor* _p)
  , $(at::Generator * _generator)));
  }|]

tensor_bernoulli__dp
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli__dp _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli_(
    $(double _p)
  , $(at::Generator * _generator)));
  }|]

tensor_bernoulli_dp
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli_dp _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli(
    $(double _p)
  , $(at::Generator * _generator)));
  }|]

tensor_bincount_tl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_bincount_tl _obj _weights _minlength =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bincount(
    *$(at::Tensor* _weights)
  , $(int64_t _minlength)));
  }|]

tensor_bmm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bmm_t _obj _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bmm(
    *$(at::Tensor* _mat2)));
  }|]

tensor_ceil
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ceil _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ceil(
    ));
  }|]

tensor_ceil_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ceil_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ceil_(
    ));
  }|]

tensor_clamp_max_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_max_s _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max(
    *$(at::Scalar* _max)));
  }|]

tensor_clamp_max__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_max__s _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max_(
    *$(at::Scalar* _max)));
  }|]

tensor_clamp_min_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_min_s _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min(
    *$(at::Scalar* _min)));
  }|]

tensor_clamp_min__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_min__s _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min_(
    *$(at::Scalar* _min)));
  }|]

tensor_contiguous
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_contiguous _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).contiguous(
    ));
  }|]

tensor_cos
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cos _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cos(
    ));
  }|]

tensor_cos_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cos_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cos_(
    ));
  }|]

tensor_cosh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cosh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cosh(
    ));
  }|]

tensor_cosh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cosh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cosh_(
    ));
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

tensor_cumsum_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_cumsum_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumsum(
    $(int64_t _dim)));
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

tensor_cumprod_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_cumprod_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cumprod(
    $(int64_t _dim)));
  }|]

tensor_det
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_det _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).det(
    ));
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

tensor_dot_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_dot_t _obj _tensor =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).dot(
    *$(at::Tensor* _tensor)));
  }|]

tensor_resize__l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_resize__l _obj _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_(
    *$(std::vector<int64_t>* _size)));
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

tensor_ger_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ger_t _obj _vec2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ger(
    *$(at::Tensor* _vec2)));
  }|]

tensor_fft_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_fft_lb _obj _signal_ndim _normalized =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).fft(
    $(int64_t _signal_ndim)
  , $(bool _normalized)));
  }|]

tensor_ifft_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_ifft_lb _obj _signal_ndim _normalized =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ifft(
    $(int64_t _signal_ndim)
  , $(bool _normalized)));
  }|]

tensor_rfft_lbb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_rfft_lbb _obj _signal_ndim _normalized _onesided =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rfft(
    $(int64_t _signal_ndim)
  , $(bool _normalized)
  , $(bool _onesided)));
  }|]

tensor_irfft_lbbl
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> CBool
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_irfft_lbbl _obj _signal_ndim _normalized _onesided _signal_sizes =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).irfft(
    $(int64_t _signal_ndim)
  , $(bool _normalized)
  , $(bool _onesided)
  , *$(std::vector<int64_t>* _signal_sizes)));
  }|]

tensor_index_l
  :: Ptr Tensor
  -> Ptr TensorList
  -> IO (Ptr Tensor)
tensor_index_l _obj _indices =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index(
    *$(std::vector<at::Tensor>* _indices)));
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

tensor_index_put__ltb
  :: Ptr Tensor
  -> Ptr TensorList
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_put__ltb _obj _indices _values _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_put_(
    *$(std::vector<at::Tensor>* _indices)
  , *$(at::Tensor* _values)
  , $(bool _accumulate)));
  }|]

tensor_index_put_ltb
  :: Ptr Tensor
  -> Ptr TensorList
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_index_put_ltb _obj _indices _values _accumulate =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_put(
    *$(std::vector<at::Tensor>* _indices)
  , *$(at::Tensor* _values)
  , $(bool _accumulate)));
  }|]

tensor_inverse
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_inverse _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).inverse(
    ));
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

tensor_kthvalue_llb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_kthvalue_llb _obj _k _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).kthvalue(
    $(int64_t _k)
  , $(int64_t _dim)
  , $(bool _keepdim)));
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

tensor_logdet
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logdet _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logdet(
    ));
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

tensor_log_softmax_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_log_softmax_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).log_softmax(
    $(int64_t _dim)));
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

tensor_max_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_max_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).max(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_max_values_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_max_values_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).max_values(
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

tensor_mean
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mean _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    ));
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

tensor_mean_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_mean_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_mean_ls
  :: Ptr Tensor
  -> Ptr IntArray
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_mean_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mean(
    *$(std::vector<int64_t>* _dim)
  , $(at::ScalarType _dtype)));
  }|]

tensor_median_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_median_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).median(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_min_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_min_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).min(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_min_values_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_min_values_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).min_values(
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
  -> IO (Ptr (Tensor,Tensor))
tensor_mode_lb _obj _dim _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).mode(
    $(int64_t _dim)
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

tensor_permute_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_permute_l _obj _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).permute(
    *$(std::vector<int64_t>* _dims)));
  }|]

tensor_pin_memory
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_pin_memory _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pin_memory(
    ));
  }|]

tensor_pinverse_d
  :: Ptr Tensor
  -> CDouble
  -> IO (Ptr Tensor)
tensor_pinverse_d _obj _rcond =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).pinverse(
    $(double _rcond)));
  }|]

tensor_repeat_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_repeat_l _obj _repeats =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).repeat(
    *$(std::vector<int64_t>* _repeats)));
  }|]

tensor_reshape_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_reshape_l _obj _shape =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reshape(
    *$(std::vector<int64_t>* _shape)));
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

tensor_prelu_backward_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr (Tensor,Tensor))
tensor_prelu_backward_tt _obj _grad_output _weight =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).prelu_backward(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _weight)));
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
tensor_hardshrink_backward_ts _obj _grad_out _lambd =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).hardshrink_backward(
    *$(at::Tensor* _grad_out)
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

tensor_sinh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sinh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sinh(
    ));
  }|]

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

tensor_size_l
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
tensor_size_l _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).size(
    $(int64_t _dim));
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
  -> IO (Ptr (Tensor,Tensor))
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

tensor_softmax_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_softmax_l _obj _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).softmax(
    $(int64_t _dim)));
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

tensor_sum_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_sum_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    $(at::ScalarType _dtype)));
  }|]

tensor_sum
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sum _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    ));
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

tensor_sum_lb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
tensor_sum_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

tensor_sum_ls
  :: Ptr Tensor
  -> Ptr IntArray
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_sum_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sum(
    *$(std::vector<int64_t>* _dim)
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

tensor_prod_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_prod_s _obj _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    $(at::ScalarType _dtype)));
  }|]

tensor_prod
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_prod _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    ));
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

tensor_prod_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_prod_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_prod_ls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_prod_ls _obj _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).prod(
    $(int64_t _dim)
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

tensor_sparse_mask_r
  :: Ptr Tensor
  -> Ptr SparseTensorRef
  -> IO (Ptr Tensor)
tensor_sparse_mask_r _obj _mask =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sparse_mask(
    *$(at::SparseTensorRef* _mask)));
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

tensor_numel
  :: Ptr Tensor
  -> IO (Int64)
tensor_numel _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).numel(
    );
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

tensor_item
  :: Ptr Tensor
  -> IO (Ptr Scalar)
tensor_item _obj =
  [C.throwBlock| at::Scalar* { return new at::Scalar((*$(at::Tensor* _obj)).item(
    ));
  }|]

tensor_data_ptr
  :: Ptr Tensor
  -> IO (Ptr ())
tensor_data_ptr _obj =
  [C.throwBlock| void * { return (*$(at::Tensor* _obj)).data_ptr(
    );
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

tensor_ne__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_ne__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ne_(
    *$(at::Scalar* _other)));
  }|]

tensor_ne__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ne__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ne_(
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

tensor_erfinv_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_erfinv_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).erfinv_(
    ));
  }|]

tensor_frac_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_frac_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).frac_(
    ));
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

tensor_reciprocal_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_reciprocal_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reciprocal_(
    ));
  }|]

tensor_neg_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_neg_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).neg_(
    ));
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

tensor_sign_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sign_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sign_(
    ));
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

tensor_cross_tl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_cross_tl _obj _other _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cross(
    *$(at::Tensor* _other)
  , $(int64_t _dim)));
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

tensor_gels_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr (Tensor,Tensor))
tensor_gels_t _obj _A =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).gels(
    *$(at::Tensor* _A)));
  }|]

tensor_trtrs_tbbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_trtrs_tbbb _obj _A _upper _transpose _unitriangular =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).trtrs(
    *$(at::Tensor* _A)
  , $(bool _upper)
  , $(bool _transpose)
  , $(bool _unitriangular)));
  }|]

tensor_symeig_bb
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_symeig_bb _obj _eigenvectors _upper =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).symeig(
    $(bool _eigenvectors)
  , $(bool _upper)));
  }|]

tensor_eig_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_eig_b _obj _eigenvectors =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).eig(
    $(bool _eigenvectors)));
  }|]

tensor_svd_bb
  :: Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr (Tensor,Tensor,Tensor))
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
  -> IO (Ptr (Tensor,Tensor))
tensor_solve_t _obj _A =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).solve(
    *$(at::Tensor* _A)));
  }|]

tensor_potri_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_potri_b _obj _upper =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).potri(
    $(bool _upper)));
  }|]

tensor_pstrf_bs
  :: Ptr Tensor
  -> CBool
  -> Ptr Scalar
  -> IO (Ptr (Tensor,Tensor))
tensor_pstrf_bs _obj _upper _tol =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).pstrf(
    $(bool _upper)
  , *$(at::Scalar* _tol)));
  }|]

tensor_qr
  :: Ptr Tensor
  -> IO (Ptr (Tensor,Tensor))
tensor_qr _obj =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).qr(
    ));
  }|]

tensor_geqrf
  :: Ptr Tensor
  -> IO (Ptr (Tensor,Tensor))
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

tensor_btrifact_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
tensor_btrifact_b _obj _pivot =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).btrifact(
    $(bool _pivot)));
  }|]

tensor_btrifact_with_info_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (Tensor,Tensor,Tensor))
tensor_btrifact_with_info_b _obj _pivot =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).btrifact_with_info(
    $(bool _pivot)));
  }|]

tensor_btrisolve_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_btrisolve_tt _obj _LU_data _LU_pivots =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).btrisolve(
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

tensor_frac
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_frac _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).frac(
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

tensor_reciprocal
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_reciprocal _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).reciprocal(
    ));
  }|]

tensor_neg
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_neg _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).neg(
    ));
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

tensor_sign
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sign _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sign(
    ));
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
  -> IO (Ptr (Tensor,Tensor))
tensor_sort_lb _obj _dim _descending =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).sort(
    $(int64_t _dim)
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

tensor_topk_llbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> CBool
  -> CBool
  -> IO (Ptr (Tensor,Tensor))
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

tensor_alias
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_alias _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).alias(
    ));
  }|]


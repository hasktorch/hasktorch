
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor0 where


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

C.include "<ATen/Functions.h>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/TensorOperators.h>"
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
  -> IO (Ptr Tensor)
tensor__assign__t _obj _x =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))=(
    *$(at::Tensor* _x)));
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

tensor_toString
  :: Ptr Tensor
  -> IO (Ptr StdString)
tensor_toString _obj =
  [C.throwBlock| std::string* { return new std::string((*$(at::Tensor* _obj)).toString(
    ));
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

tensor_is_non_overlapping_and_dense
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_non_overlapping_and_dense _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_non_overlapping_and_dense(
    );
  }|]

tensor_nbytes
  :: Ptr Tensor
  -> IO (CSize)
tensor_nbytes _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).nbytes(
    );
  }|]

tensor_numel
  :: Ptr Tensor
  -> IO (Int64)
tensor_numel _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).numel(
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
  -> IO (Int64)
tensor_element_size _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).element_size(
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

tensor_storage
  :: Ptr Tensor
  -> IO (Ptr Storage)
tensor_storage _obj =
  [C.throwBlock| at::Storage* { return new at::Storage((*$(at::Tensor* _obj)).storage(
    ));
  }|]

tensor_is_alias_of_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_alias_of_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_alias_of(
    *$(at::Tensor* _other));
  }|]

tensor_toType_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_toType_s _obj _t =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).toType(
    $(at::ScalarType _t)));
  }|]

tensor_toBackend_B
  :: Ptr Tensor
  -> Backend
  -> IO (Ptr Tensor)
tensor_toBackend_B _obj _b =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).toBackend(
    $(at::Backend _b)));
  }|]

tensor_is_variable
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_variable _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_variable(
    );
  }|]

tensor_layout
  :: Ptr Tensor
  -> IO (Layout)
tensor_layout _obj =
  [C.throwBlock| at::Layout { return (*$(at::Tensor* _obj)).layout(
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

tensor_is_mkldnn
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_mkldnn _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_mkldnn(
    );
  }|]

tensor_is_quantized
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_quantized _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_quantized(
    );
  }|]

tensor_has_names
  :: Ptr Tensor
  -> IO (CBool)
tensor_has_names _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).has_names(
    );
  }|]

tensor_options
  :: Ptr Tensor
  -> IO (Ptr TensorOptions)
tensor_options _obj =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::Tensor* _obj)).options(
    ));
  }|]

tensor_data_ptr
  :: Ptr Tensor
  -> IO (Ptr ())
tensor_data_ptr _obj =
  [C.throwBlock| void * { return (*$(at::Tensor* _obj)).data_ptr(
    );
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
  -> IO (())
tensor__idiv__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))/=(
    *$(at::Tensor* _other));
  }|]

tensor__idiv__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__idiv__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))/=(
    *$(at::Scalar* _other));
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

tensor_backward_tbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (())
tensor_backward_tbb _obj _gradient _keep_graph _create_graph =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).backward(
    *$(at::Tensor* _gradient)
  , $(bool _keep_graph)
  , $(bool _create_graph));
  }|]

tensor_set_data_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor_set_data_t _obj _new_data =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).set_data(
    *$(at::Tensor* _new_data));
  }|]

tensor_data
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_data _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).data(
    ));
  }|]

tensor_is_leaf
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_leaf _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_leaf(
    );
  }|]

tensor_output_nr
  :: Ptr Tensor
  -> IO (Int64)
tensor_output_nr _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).output_nr(
    );
  }|]

tensor__version
  :: Ptr Tensor
  -> IO (Int64)
tensor__version _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj))._version(
    );
  }|]

tensor_requires_grad__b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_requires_grad__b _obj __requires_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).requires_grad_(
    $(bool __requires_grad)));
  }|]

tensor_retain_grad
  :: Ptr Tensor
  -> IO (())
tensor_retain_grad _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).retain_grad(
    );
  }|]

tensor_align_to_N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_align_to_N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_to(
    *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_align_to_Nl
  :: Ptr Tensor
  -> Ptr DimnameList
  -> Int64
  -> IO (Ptr Tensor)
tensor_align_to_Nl _obj _order _ellipsis_idx =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_to(
    *$(std::vector<at::Dimname>* _order)
  , $(int64_t _ellipsis_idx)));
  }|]

tensor_align_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_align_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_as(
    *$(at::Tensor* _other)));
  }|]

tensor_refine_names_N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_refine_names_N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).refine_names(
    *$(std::vector<at::Dimname>* _names)));
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

tensor_unflatten_llN
  :: Ptr Tensor
  -> Int64
  -> Ptr IntArray
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_unflatten_llN _obj _dim _sizes _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).unflatten(
    $(int64_t _dim)
  , *$(std::vector<int64_t>* _sizes)
  , *$(std::vector<at::Dimname>* _names)));
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

tensor_angle
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_angle _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).angle(
    ));
  }|]

tensor_conj
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_conj _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).conj(
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

tensor_all_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
tensor_all_nb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).all(
    *$(at::Dimname* _dim)
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

tensor_any_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
tensor_any_nb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).any(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
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

tensor_bitwise_not
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_not _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_not(
    ));
  }|]

tensor_bitwise_not_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_not_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_not_(
    ));
  }|]

tensor_logical_not
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_not _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_not(
    ));
  }|]

tensor_logical_not_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_not_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_not_(
    ));
  }|]

tensor_logical_xor_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_xor_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_xor(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_xor__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_xor__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_xor_(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_and_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_and_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_and(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_and__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_and__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_and_(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_or_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_or_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_or(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_or__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_or__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_or_(
    *$(at::Tensor* _other)));
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

tensor_chunk_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
tensor_chunk_ll _obj _chunks _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).chunk(
    $(int64_t _chunks)
  , $(int64_t _dim)));
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

tensor_cummax_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummax_l _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummax(
    $(int64_t _dim)));
  }|]

tensor_cummax_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummax_n _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummax(
    *$(at::Dimname* _dim)));
  }|]

tensor_cummin_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummin_l _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummin(
    $(int64_t _dim)));
  }|]

tensor_cummin_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummin_n _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummin(
    *$(at::Dimname* _dim)));
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
